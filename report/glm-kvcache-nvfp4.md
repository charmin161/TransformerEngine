可以。按 **vLLM 0.26.0** 的源码看，你这个实验不应该从普通 `attention.py` / `flash_attn` 路径找，而应该直接盯住 GLM-5.2 在 vLLM 中复用的 **DeepSeek-V3.2 NVIDIA DSA/MLA 路径**。vLLM 0.26.0 的 `DeepseekV32ForCausalLM` 明确用于 DeepSeek-V3.2 以及复用 DSA 的架构，例如 GLM-5.2；每层最终创建的是 `DeepseekV32Attention`。

先给结论：

> **如果你的目标是：把实际写入 KV cache 的数据做一次 NVFP4 fake-quant/QDQ，然后后续 attention 使用这个被扰动的数据，那么最合适的位置就是**
>
> `vllm/models/deepseek_v32/nvidia/kernels.py`
>
> 的 `_fused_norm_rope_kernel()` 中，`pid == 1` 分支里，**RMSNorm 和 RoPE 完成之后、FP8 KV cache 写入之前**。
>
> **不需要改 FlashInfer backend。**

这也是我建议你第一版实验采用的方案。

---

## 一、先弄清楚 GLM-5.2 这里的“KV”到底是什么

这是最重要的一点。

GLM-5.2 在这条路径上使用的是 **MLA**，所以 KV cache 不是传统形式：

```text
K : [tokens, num_kv_heads, head_dim]
V : [tokens, num_kv_heads, head_dim]
```

而是保存：

```text
[kv_c, k_pe]

kv_c : compressed/latent K/V
k_pe : decoupled RoPE key
```

vLLM 自己的 MLA 文档把这个过程写得非常清楚：`kv_c` 是 K/V 共用的 latent，逻辑上的 `k_nope` 和 `v` 都可以由 `kv_c` 分别乘 `W_UK`、`W_UV` 得到；decode 的 data-movement-friendly 路径甚至直接让 attention 用 `[kv_c, k_pe]` 做 K，并用 `kv_c` 做 V，最后再通过 `W_UV` 恢复。

vLLM 0.26.0 实际代码也是：

```python
ql_nope = torch.bmm(q_nope, self.W_UK_T)
...
attn_out, _ = self.impl.forward_mqa(
    mqa_q_arg, kv_cache, attn_metadata, self
)
...
torch.bmm(x, self.W_UV, out=out)
```

也就是说，**你量化 `kv_c`，实际上会同时影响逻辑 K_nope 和 V**。

因此后面我说的“量化 KV cache”，默认指的是：

```text
kv_c_normed + k_pe_roped
```

而不是显式展开出来的传统 K、V。

---

# 二、你的实际调用链

vLLM 0.26.0 下可以按下面这条链找：

```text
vllm/models/deepseek_v32/nvidia/model.py

DeepseekV32DecoderLayer.forward()
    ↓
self.self_attn(...)
    ↓

vllm/models/deepseek_v32/nvidia/attention.py

DeepseekV32Attention.forward()
    ↓
fused_qkv_a_proj()
    ↓
q_c, kv_c, k_pe
    ↓
_fused_attention()
    ↓
fused_norm_rope(...)
    ↓

vllm/models/deepseek_v32/nvidia/kernels.py

_fused_norm_rope_kernel()
    ↓
pid == 1
    ↓
KV RMSNorm
    ↓
K RoPE
    ↓
【你就在这里做 NVFP4 fake quant】
    ↓
FP8 E4M3 cache write
    ↓

attention.py
self.impl.forward_mqa(...)
    ↓

FLASHINFER_MLA_SPARSE
    ↓
FlashInfer TRTLLM sparse MLA kernel
```

`DeepseekV32Attention.forward()` 首先把 fused projection 的输出拆成 `q_c / kv_c / k_pe`，然后进入 `_fused_attention()`。

真正的 cache 写入发生在 `fused_norm_rope()` 调用的 Triton kernel 里面；随后同一次 attention 又从 `self.kv_cache` 取出 cache，交给 `self.impl.forward_mqa()`。

---

# 三、第一步：先加 print，确认你确实走到这条路径

我建议先不要改量化。

先打开：

```text
vllm/models/deepseek_v32/nvidia/attention.py
```

找到：

```python
class DeepseekV32Attention(MLAAttention):
```

在 `__init__()` 中，大概这段后面：

```python
self._fp8_query = self.impl.supports_quant_query_input

...

self._fp8_kv_needs_view = self.kv_cache_dtype != "fp8_ds_mla"
```

源码大约对应 251～275 行。这里明确区分了 FlashInfer sparse FP8 路径和 FlashMLA `fp8_ds_mla` 路径。

临时加：

```python
print(
    "[KV DEBUG] "
    f"layer_id={layer_id}, "
    f"prefix={prefix}, "
    f"impl={type(self.impl).__module__}.{type(self.impl).__name__}, "
    f"kv_cache_dtype={self.kv_cache_dtype}, "
    f"fp8_query={self._fp8_query}, "
    f"fp8_kv_needs_view={self._fp8_kv_needs_view}",
    flush=True,
)
```

你用 TP8，所以它会打印很多次，没关系。你甚至可以先限制：

```python
if layer_id == 0:
    print(
        "[KV DEBUG] "
        f"impl={type(self.impl).__module__}.{type(self.impl).__name__}, "
        f"kv_cache_dtype={self.kv_cache_dtype}, "
        f"fp8_query={self._fp8_query}",
        flush=True,
    )
```

如果你使用 NVIDIA 官方推荐：

```bash
--kv-cache-dtype fp8_e4m3
```

NVIDIA 当前模型卡给出的 vLLM 命令确实就是这个配置。需要注意的是模型 checkpoint 自身的 NVFP4 只覆盖 transformer MoE expert 中 linear operator 的权重和激活，**并没有把 KV cache 做 NVFP4**。([Hugging Face][1])

如果是 SM100/B200 类环境，vLLM 0.26.0 的 `FLASHINFER_MLA_SPARSE` 正好支持 compute capability 10.x 和 `fp8_e4m3`，其实现类是：

```text
FlashInferMLASparseImpl
```



所以你大概率看到类似：

```text
[KV DEBUG] impl=...FlashInferMLASparseImpl,
kv_cache_dtype=fp8_e4m3,
fp8_query=True
```

**以你实际打印结果为准。**

---

# 四、再确认 cache-write kernel 确实执行了

接着打开：

```text
vllm/models/deepseek_v32/nvidia/kernels.py
```

找到：

```python
def fused_norm_rope(
```

大概 346 行附近。这个 Python wrapper 最终会 launch：

```python
_fused_norm_rope_kernel[(4, num_tokens)](
    ...
)
```

源码中 launch 在约 449 行。

为了确认，可以临时在 `fused_norm_rope()` 开头加入：

```python
print(
    "[KV DEBUG] fused_norm_rope called: "
    f"kv_c.shape={tuple(kv_c.shape)}, "
    f"k_pe.shape={tuple(k_pe.shape)}, "
    f"kv_cache_dtype={mla_kv_cache_dtype}",
    flush=True,
)
```

这个会爆很多输出，所以确认一次之后就删掉。

看到：

```text
[KV DEBUG] fused_norm_rope called ...
kv_cache_dtype=fp8_e4m3
```

就说明你已经定位到真正的 KV cache 写入路径。

---

# 五、真正应该插入 NVFP4 fake quant 的地方

现在来到最核心的位置。

文件：

```text
vllm/models/deepseek_v32/nvidia/kernels.py
```

函数：

```python
@triton.jit
def _fused_norm_rope_kernel(...):
```

找到：

```python
elif pid == 1:
```

这里源码是：

```python
elif pid == 1:
    # KV RMS Norm + KV RoPE + MLA concat_and_cache.

    kv_block = tl.arange(0, KV_DIM)

    kv_c = tl.load(
        kv_ptr + tok_idx * kv_stride + kv_block
    )

    kv_c_rms_w = tl.load(
        kv_rms_norm_w_ptr + kv_block
    )

    kv_c = _rms_norm(
        kv_c,
        kv_c_rms_w,
        kv_rms_eps,
        KV_DIM
    )

    # KV RoPE
    ...
    x1 = ...
    x2 = ...

    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin

    # MLA concat_and_cache
```

源码对应位置就是 175～200 行。这里的 `kv_c` 已经完成 RMSNorm，`r1/r2` 则是完成 RoPE 后的 `k_pe`。

**你的 fake quant 最好就加在这里：**

```python
kv_c = _rms_norm(
    kv_c,
    kv_c_rms_w,
    kv_rms_eps,
    KV_DIM
)

...

r1 = x1 * cos - x2 * sin
r2 = x2 * cos + x1 * sin


# =========================================
# ADD NVFP4 FAKE QUANT HERE
# =========================================

kv_c = nvfp4_fake_quant(kv_c, ...)

# 如果你希望整个 MLA KV cache 都做 NVFP4，
# k_pe 也是 K 的一部分，因此也处理：
r1 = nvfp4_fake_quant(r1, ...)
r2 = nvfp4_fake_quant(r2, ...)


# =========================================

# MLA concat_and_cache
if mla_cache_entry_stride == 0:
    return
```

然后原来的代码不动。

---

## 六、为什么一定是这里，而不是 `attention.py` 中的 `kv_c`

这是一个很容易犯的错误。

在：

```text
attention.py
DeepseekV32Attention.forward()
```

这里确实已经有：

```python
q_c, kv_c, k_pe = qkv_lora.split(...)
```



你可能自然会想写：

```python
kv_c = fake_quant_nvfp4(kv_c)
k_pe = fake_quant_nvfp4(k_pe)
```

但是**我不推荐**。

因为此时：

```text
kv_c
   ↓
还没有 RMSNorm

k_pe
   ↓
还没有 RoPE
```

而真正进入 cache 的是：

```text
RMSNorm(kv_c)
RoPE(k_pe)
```

kernel 注释本身就明确写着 cache 的布局是：

```text
MLA KV cache
=
concat(
    kv_c_normed,
    k_pe_roped
)
```



如果你的实验题目是：

> 对实际 KV cache 数据进行 NVFP4 fake quant

那么应该量化 **post-RMSNorm/post-RoPE** 数据，而不是 projection 刚产生的原始 latent。

---

# 七、官方 `fp8_e4m3` 下，你的完整数据路径会变成什么

NVIDIA 官方配置本身是：

```bash
--kv-cache-dtype fp8_e4m3
```

([Hugging Face][1])

所以你插入 fake quant 后实际上是：

```text
hidden states
   ↓
fused_qkv_a_proj
   ↓
kv_c
   ↓
RMSNorm
   ↓
NVFP4 quant
   ↓
NVFP4 dequant
   ↓
FP8 E4M3 quant
   ↓
KV cache
   ↓
FlashInfer Sparse MLA
```

`k_pe` 则是：

```text
k_pe
 ↓
RoPE
 ↓
NVFP4 quant
 ↓
NVFP4 dequant
 ↓
FP8 E4M3 quant
 ↓
KV cache
```

因为官方 FP8 cache 的代码就在后面：

```python
if MLA_CACHE_FP8:
    scale = tl.load(mla_cache_scale_ptr)

    kv_c_fp8 = (
        kv_c.to(tl.float32) / scale
    ).to(tl.float8e4nv)

    tl.store(dst + kv_block, kv_c_fp8)
```

以及：

```python
tl.store(
    dst + KV_DIM + dim_off * 2,
    (r1 / scale).to(tl.float8e4nv)
)

tl.store(
    dst + KV_DIM + dim_off * 2 + 1,
    (r2 / scale).to(tl.float8e4nv)
)
```



所以这项实验严格来说测的是：

```text
baseline:
BF16 → FP8 KV cache

experiment:
BF16 → NVFP4 QDQ → FP8 KV cache
```

我认为第一阶段应该就这么做，因为它**保留 NVIDIA 官方推理配置不变**，只额外注入 NVFP4 quantization error。

---

# 八、你的 fake quant 如果是普通 PyTorch 代码，要特别注意

这里可能是你实际修改时最大的障碍。

我上面推荐的位置位于：

```python
@triton.jit
def _fused_norm_rope_kernel(...)
```

所以此时：

```python
kv_c
r1
r2
```

都是 **Triton kernel 内部 register values**。

假如你现有代码是：

```python
def nvfp4_fake_quant(x: torch.Tensor):
    ...
```

那么不能直接：

```python
kv_c = nvfp4_fake_quant(kv_c)
```

因为这里不是普通 `torch.Tensor` Python execution。

这种情况下，最小且正确的做法是把你 NVFP4 fake quant 的核心逻辑写成：

```python
@triton.jit
def _nvfp4_fake_quant(...):
    ...
```

然后：

```python
kv_c = _nvfp4_fake_quant(kv_c, ...)
r1 = _nvfp4_fake_quant(r1, ...)
r2 = _nvfp4_fake_quant(r2, ...)
```

**不要为了方便直接在 `attention.py` 的原始 `kv_c` 上调用 PyTorch fake quant**，否则你改变的是 pre-RMSNorm 数据，实验定义已经发生变化。

如果你把你现在的 NVFP4 fake-quant 函数发给我，我可以下一步直接按照这个 Triton kernel 的 tensor/block 结构告诉你怎么嵌进去。

---

# 九、需要改 FlashInfer backend 吗？

**你当前这个目的不需要。**

原因很简单。

cache 写完以后，`attention.py` 中：

```python
kv_cache = self.kv_cache

if self._fp8_kv_needs_view:
    kv_cache = kv_cache.view(torch.float8_e4m3fn)

...

attn_out, _ = self.impl.forward_mqa(
    mqa_q_arg,
    kv_cache,
    attn_metadata,
    self
)
```



而 `FlashInferMLASparseImpl.forward_mqa()` 继续直接把这个 cache 传给：

```python
trtllm_batch_decode_with_kv_cache_mla(
    query=query,
    kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
    ...
)
```



因此：

```text
你修改 cache write
       ↓
self.kv_cache 已经是被 NVFP4 QDQ 扰动后的数据
       ↓
forward_mqa
       ↓
FlashInfer
```

已经能够完整达到目的。

为了验证“读端”也确实走到这里，你可以临时在：

```text
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
```

的：

```python
class FlashInferMLASparseImpl
```

里面：

```python
def forward_mqa(
```

最开头加：

```python
print(
    "[KV DEBUG] FlashInferMLASparseImpl.forward_mqa: "
    f"q={q.shape if isinstance(q, torch.Tensor) else 'tuple'}, "
    f"kv_cache_shape={kv_c_and_k_pe_cache.shape}, "
    f"kv_cache_dtype={kv_c_and_k_pe_cache.dtype}",
    flush=True,
)
```

只用于确认，然后删掉。

---

# 十、什么时候你才真的需要改 backend

有一种情况和上面完全不同：

> 你不是想量化 MLA cache 中的 `kv_c + k_pe`，而是想量化**逻辑意义上真正展开后的 per-head K 和 V**。

也就是你真正想研究：

```text
kv_c
 ↓ W_UK
K_nope
 ↓
NVFP4 fake quant

kv_c
 ↓ W_UV
V
 ↓
NVFP4 fake quant
```

这种情况下，**就不能只改 cache-write kernel**。

因为 decode 的 optimized MLA 根本不会把完整：

```text
K_nope [seq, heads, dim]
V      [seq, heads, dim]
```

materialize 出来。

它利用吸收矩阵的方式：

```text
Q_nope × W_UK^T
       ↓
Q_latent

Q_latent × kv_c
```

以及：

```text
attention(..., V=kv_c)
      ↓
latent output
      ↓
× W_UV
```

直接避开 K/V 展开。这个正是 MLA decode 节省 KV 带宽的核心。

这时你就必须改：

```text
FlashInfer sparse MLA backend
```

甚至继续进入 FlashInfer/TRTLLM kernel，把逻辑 K/V quantization 插到内部矩阵运算中。

**如果只是研究“NVFP4 KV cache 对精度的影响”，我不建议走这条路。** 因为这已经不是“KV cache fake quant”了，而是在改变 MLA 内部等价变换后的计算精度定义。

---

# 十一、还有一个东西先不要碰：Indexer K cache

GLM-5.2 是 DSA sparse attention，所以除了 MLA KV cache，还有另外一份：

```text
Indexer K cache
```

它的作用不是 attention V，而是决定：

```text
当前 query
   ↓
哪些历史 token 是 top-k
   ↓
真正参与 sparse attention
```

`kernels.py` 里紧接着的：

```python
elif pid == 0:
```

就是在处理：

```text
Index K
→ LayerNorm
→ RoPE
→ FP8 quant
→ indexer K cache
```

最后调用：

```python
_fp8_quant_and_cache_write(
    result,
    ...
)
```



而后面的 `sparse_attn_indexer()` 就会使用这份 cache 生成 top-k。

**第一轮实验不要改这里。**

否则你的结果同时包含：

```text
① NVFP4 导致 attention K/V 数值变化
+
② NVFP4 导致 sparse top-k token 选择变化
```

两种影响混在一起，很难解释结果。

建议第一轮：

```text
Indexer K cache
    保持原样 FP8

MLA KV cache
    NVFP4 fake quant + 官方 FP8 storage
```

如果之后你专门想研究：

> Rubin adaptive compression / 低比特 indexer 是否会改变 DSA top-k selection

再在这里加入第二套 fake quant：

```python
result = ...

# 第二阶段实验再加
result = nvfp4_fake_quant(result, ...)

_fp8_quant_and_cache_write(
    result,
    ...
)
```

这样实验变量最干净。

---

# 十二、所以我建议你第一版只改这两个文件

可以把修改范围压缩到非常小：

1. `vllm/models/deepseek_v32/nvidia/attention.py`：**只加 debug print**，确认 `DeepseekV32Attention`、实际 `self.impl`、`kv_cache_dtype`，并确认 `self.impl.forward_mqa()` 是最终消费 cache 的地方。
2. `vllm/models/deepseek_v32/nvidia/kernels.py`：**真正的 NVFP4 fake quant 放这里**，具体就是 `_fused_norm_rope_kernel()` 的 `pid == 1`，在 `kv_c` RMSNorm 和 `k_pe` RoPE 完成之后、现有 FP8 cache write 之前。

**暂时不要改：**

```text
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
```

除了临时加 `print`。

你最终想得到的修改结构基本就是：

```python
elif pid == 1:

    # -------------------------
    # Existing vLLM
    # -------------------------
    kv_c = ...
    kv_c = _rms_norm(...)

    ...

    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin


    # =========================
    # YOUR CODE START
    # =========================

    kv_c = _nvfp4_fake_quant(kv_c, ...)

    # 如果整个 cache entry 都做 NVFP4
    r1 = _nvfp4_fake_quant(r1, ...)
    r2 = _nvfp4_fake_quant(r2, ...)

    # =========================
    # YOUR CODE END
    # =========================


    # -------------------------
    # Existing vLLM FP8 cache
    # -------------------------
    if MLA_CACHE_DS_MLA:
        ...
    else:
        ...
        if MLA_CACHE_FP8:
            scale = tl.load(mla_cache_scale_ptr)

            kv_c_fp8 = (
                kv_c.to(tl.float32) / scale
            ).to(tl.float8e4nv)

            tl.store(...)

            tl.store(... (r1 / scale).to(tl.float8e4nv))
            tl.store(... (r2 / scale).to(tl.float8e4nv))
```

**这就是我认为你当前最正确、最小改动的插入位置。**

另外还有一个后续实验我认为值得单独做：官方 `fp8_e4m3` 会让你的实验变成 **NVFP4 QDQ → FP8 cache**。如果你之后想严格测“**只有 NVFP4 quantization error，不叠加 FP8 cache error**”，我可以继续帮你把 vLLM 0.26.0 这条 DSA 路径改成 **BF16 storage + NVFP4 QDQ**。这需要同时处理 `DeepseekV32Attention` 里对 quantized KV cache 的 assert、`_fp8_query` 和 `_fp8_kv_needs_view`，比现在多几处，但可以做到，而且实验解释会更纯。

[1]: https://huggingface.co/nvidia/GLM-5.2-NVFP4 "nvidia/GLM-5.2-NVFP4 · Hugging Face"
