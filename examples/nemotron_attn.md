你看得对。前一条回答里提到的 `torch_native_backend.py` **不是 Nemotron-3-Ultra 官方默认调用链的一部分**，而是我建议你为了先做数值仿真，主动切换过去的实验后端。默认在 B200 上，SGLang 会优先选择 `trtllm_mha`，所以你继续向下跟踪时看到的是 FlashInfer 函数。当前 SGLang 文档也明确写明：Blackwell 上的 MHA 默认选择 `trtllm_mha`；`triton`、`torch_native`、`flashinfer` 等是可显式选择的其他后端。([SGLang Documentation][1])

## 1. `torch_native_backend.py` 究竟在哪里接入

启动时：

```bash
--attention-backend torch_native
```

会在：

```text
python/sglang/srt/layers/attention/attention_registry.py
```

中命中：

```python
@register_attention_backend("torch_native")
def create_torch_native_backend(runner):
    from sglang.srt.layers.attention.torch_native_backend import (
        TorchNativeAttnBackend,
    )
    return TorchNativeAttnBackend(runner)
```

同一个 registry 中还分别注册了：

```text
triton       -> TritonAttnBackend
torch_native -> TorchNativeAttnBackend
trtllm_mha   -> TRTLLMHAAttnBackend
flashinfer   -> FlashInferAttnBackend
fa4          -> FlashAttentionBackend
```

SGLang `v0.5.13` 的注册代码就是如此。([GitHub][2])

Nemotron-3-Ultra 又是 Mamba-2 与 full attention 混合模型，所以注册完 full-attention backend 后，还会包上一层：

```text
HybridLinearAttnBackend
├─ full_attn_backend
└─ linear_attn_backend = Mamba2AttnBackend
```

主模型的 full-attention layer ID 走 `full_attn_backend`；Mamba 层走 `Mamba2AttnBackend`。SGLang 直接从模型配置的 `full_attention_layer_ids` 构造这套路由。([GitHub][2])

因此实际情况是：

```text
--attention-backend trtllm_mha
    full attention -> TRTLLMHAAttnBackend -> FlashInfer TRT-LLM MHA

--attention-backend triton
    full attention -> TritonAttnBackend -> SGLang Triton kernels

--attention-backend torch_native
    full attention -> TorchNativeAttnBackend -> PyTorch SDPA

--mamba-backend flashinfer
    Mamba-2 layer -> FlashInfer Mamba kernel
```

尤其要注意：**即使 full attention 已经切到 `torch_native` 或 `triton`，只要 `--mamba-backend flashinfer`，日志和调用栈里仍然会出现 FlashInfer。**那是 Mamba-2，不一定是你要修改的 12 个 full-attention 层。

### 最可靠的运行时确认方法

可以临时在 `attention_registry.py` 的：

```python
return HybridLinearAttnBackend(
    full_attn_backend, linear_attn_backend, full_attn_layers
)
```

之前增加：

```python
logger.warning(
    "ATTN DEBUG: full=%s, linear=%s, full_layers=%s",
    type(full_attn_backend).__name__,
    type(linear_attn_backend).__name__,
    full_attn_layers,
)
```

然后你会看到类似：

```text
# 官方默认
full=TRTLLMHAAttnBackend
linear=Mamba2AttnBackend

# 切换 Triton 后
full=TritonAttnBackend
linear=Mamba2AttnBackend

# 切换 torch_native 后
full=TorchNativeAttnBackend
linear=Mamba2AttnBackend
```

---

# 2. 能不能直接修改 FlashInfer 源码再重新编译

答案要分成两个 FlashInfer 路径。

## 2.1 当前 `trtllm_mha` 路径：不能只改普通 FlashInfer 源码就实现

SGLang `TRTLLMHAAttnBackend` 的底部调用是：

```python
# decode
flashinfer.decode.trtllm_batch_decode_with_kv_cache(...)

# prefill / extend
flashinfer.prefill.trtllm_batch_context_with_kv_cache(...)
```

SGLang `v0.5.13` 中就是这一调用方式。([GitHub][3])

问题在于，B200 是 SM100，走的是 TRT-LLM-GEN FMHA 路径。FlashInfer 的 module generator 会编译：

```text
trtllm_fmha_kernel_launcher.cu
fmhaReduction.cu
trtllm_sage_quant.cu
```

但真正的 FMHA 主体由 `TRTLLM_GEN_FMHA` 的预编译 CUBIN artifact 提供。换言之，公开源码主要负责参数准备、launcher 和 reduction，真正执行：

```text
QK
softmax
PV
```

的核心内核并不完整地以可编辑 CUDA 源码形式出现在这个路径里。

所以：

```text
修改 flashinfer/prefill.py                 不够
修改 flashinfer/decode.py                  不够
修改 trtllm_mha_backend.py                 不够
重新 pip install FlashInfer                也不够
```

这些操作最多改变调用参数和调度，不能在预编译 CUBIN 内部的 `QK -> softmax` 之间插入：

```python
每 4 个 score 选 2 个
其他设为 -inf
```

除非你：

1. 用自己的 CUDA/CuTe kernel 完整替换 TRTLLM-GEN FMHA；
2. 获得并修改对应 TensorRT-LLM FMHA kernel 的生成源码；
3. 修改 launcher，让它不再调用原 CUBIN，而是调用你自己的 kernel。

这已经相当于实现一个新 attention backend，而不是简单重编译 FlashInfer。

## 2.2 普通 `flashinfer` JIT 路径：可以修改，但难度仍高

如果改为：

```bash
--attention-backend flashinfer
```

FlashInfer 的普通 FA2、FA3、CUTLASS、CuTe DSL 等路径中，确实有大量可从源码 JIT 编译的 kernel。FlashInfer 官方也支持 custom attention variants 和 editable source installation；修改 JIT kernel 后，清理缓存即可触发重新编译。([GitHub][4])

源码安装方式大致是：

```bash
# 先查当前版本
python3 - <<'PY'
import importlib.metadata as md
import flashinfer

print("flashinfer-python:", md.version("flashinfer-python"))
print("flashinfer path:", flashinfer.__file__)
PY

# 必须使用与当前 SGLang 镜像匹配的 FlashInfer tag
git clone --recursive https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
git checkout v<上面查到的版本>

export FLASHINFER_CUDA_ARCH_LIST="10.0a"
export MAX_JOBS=8

python3 -m pip install \
  --no-build-isolation \
  -e . \
  -v

rm -rf ~/.cache/flashinfer

export FLASHINFER_JIT_VERBOSE=1
export FLASHINFER_LOGLEVEL=3
```

FlashInfer 官方开发说明指出，editable 安装下 kernel 源码修改通常可由 JIT 自动检测并重新编译。([GitHub][4])

但仍有一个关键问题：

> 你的 4 选 2 不是普通的静态 mask，也不是逐元素 `score_mod`。

每个 score 是否保留，依赖同一个 quartet 中另外三个 score 的值。因此需要在 kernel 的 score tile 已经形成之后，执行横向比较：

```text
score fragment
→ reshape/group by 4
→ top-2
→ mask other 2
→ online softmax
```

这通常需要修改 FlashInfer attention 主循环中的 score fragment，而不是只增加一个额外参数。对于 CuTe/CUTLASS kernel，开发复杂度明显高于 Triton。

另外，若安装了：

```text
flashinfer-cubin
flashinfer-jit-cache
```

某些 API 会优先使用预编译产物。只有切换到普通 JIT FlashInfer 路径时，才考虑卸载：

```bash
pip uninstall -y flashinfer-cubin flashinfer-jit-cache
rm -rf ~/.cache/flashinfer
```

**不要在继续使用 `trtllm_mha` 时卸载它们**，因为 TRTLLM-GEN 路径本身可能依赖相应 CUBIN artifact。

---

# 3. 最适合你当前实验的后端：Triton

我建议你不要先改 FlashInfer，而是直接切换：

```bash
--attention-backend triton
```

原因是：

1. QK score 在 SGLang 源码中显式可见；
2. prefill 和 decode 都能修改；
3. 不需要手动编译 wheel，第一次运行由 Triton JIT 编译；
4. 当前 SGLang 支持矩阵中，Triton 支持 MHA、FP8/FP4 KV cache、speculative decoding 和 sliding window；只是 page size 大于 1 不是原生实现。([SGLang Documentation][1])
5. `TritonAttnBackend` 直接导入公开的两个 kernel 文件：

```text
triton_ops/decode_attention.py
triton_ops/extend_attention.py
```

([GitHub][5])

## 推荐的第一轮启动配置

第一轮只做 2:4 对模型精度的影响，不要同时引入 MTP、FP8 KV、prefix cache 和 CUDA Graph：

```bash
python3 -m sglang.launch_server \
  --model-path /wireless/public/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --tp 4 \
  --ep 4 \
  --attention-backend triton \
  --mamba-backend triton \
  --page-size 1 \
  --kv-cache-dtype bf16 \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size -1 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-overlap-schedule \
  --trust-remote-code
```

不同 SGLang 版本的 Mamba 参数名可能是：

```text
--mamba-scheduler-strategy extra_buffer
```

或旧别名：

```text
--mamba-radix-cache-strategy extra_buffer
```

以当前环境的帮助信息为准：

```bash
python3 -m sglang.launch_server --help \
  | grep -E "mamba.*strategy|attention-backend|page-size"
```

这里把 `--mamba-backend` 也切到 Triton，只是为了避免调用栈中继续出现 FlashInfer，方便追踪。它不影响 full-attention 的 2:4 语义。

---

# 4. Triton 路径中应修改的准确位置

## 4.1 Prefill / extend

文件：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py
```

其中 prefix KV 部分大致是：

```python
qk = tl.dot(q.to(k.dtype), k)
qk *= sm_scale * k_scale

# logit cap / temperature 等变换

qk = tl.where(final_mask, qk, float("-inf"))

# 在这里插入 4 选 2

row_max = tl.max(qk, 1)
p = tl.exp(qk - n_e_max[:, None])
acc += tl.dot(p.to(v.dtype), v)
```

对应的 QK、mask、online softmax 和 PV 在源码中是连续的。

当前 extend token 部分还有第二处：

```python
qk = tl.dot(q, k, out_dtype=tl.float32)
...
qk = tl.where(final_mask, qk, float("-inf"))

# 同样在这里插入 4 选 2

row_max = tl.max(qk, 1)
```

两处都必须改，否则：

```text
已有 prefix token 使用 dense attention
当前新增 token 使用 2:4
```

或者反过来，语义会不一致。

## 4.2 Decode

文件：

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py
```

Nemotron-3-Ultra 是 GQA，因此主要走：

```python
_fwd_grouped_kernel_stage1
```

而不是普通 MHA kernel。

核心位置是：

```python
qk = tl.dot(q_k, k)
qk *= sm_scale_withk

# logit cap / temperature

qk = tl.where(valid_mask, qk, float("-inf"))

# 在这里做每行、每四列 top-2

n_e_max = tl.maximum(tl.max(qk, 1), e_max)
p = tl.exp(qk - n_e_max[:, None])
acc += tl.dot(p.to(v.dtype), v)
```

---

# 5. 4 选 2 的 Triton 核心逻辑

在 Triton tile 内，`qk` 通常是：

```text
prefill: [BLOCK_M, BLOCK_N]
decode:  [BLOCK_H, BLOCK_N]
```

可以按最后一维 reshape：

```text
[ROWS, BLOCK_N]
    ↓
[ROWS, BLOCK_N / 4, 4]
```

然后找两次 argmax：

```python
@triton.jit
def mask_top2_of4(
    qk,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    qk4 = tl.reshape(qk, (ROWS, BLOCK_N // 4, 4))

    idx0 = tl.argmax(
        qk4,
        axis=2,
        tie_break_left=True,
    )

    lanes = tl.arange(0, 4)
    keep0 = lanes[None, None, :] == idx0[:, :, None]

    qk4_without_max = tl.where(
        keep0,
        float("-inf"),
        qk4,
    )

    idx1 = tl.argmax(
        qk4_without_max,
        axis=2,
        tie_break_left=True,
    )

    keep1 = lanes[None, None, :] == idx1[:, :, None]
    keep = keep0 | keep1

    qk4 = tl.where(
        keep,
        qk4,
        float("-inf"),
    )

    return tl.reshape(qk4, (ROWS, BLOCK_N))
```

不同 Triton 版本可能要求用：

```python
tl.expand_dims(...)
```

代替 `[:, :, None]`，但算法就是这个结构。

调用顺序必须是：

```python
qk = standard_score_transform(qk)
qk = tl.where(causal_padding_mask, qk, -inf)

if ENABLE_2OF4:
    qk = mask_top2_of4(qk, ...)

row_max = tl.max(qk, axis=1)
```

不是先 top-2 再 causal mask，否则可能把未来 token 选进 top-2，之后再 mask 掉，导致 quartet 中只剩一个有效 token。

---

# 6. 最好增加编译期实验开关，不要直接写死

建议在 `triton_backend.py` 中读取：

```python
import os

self.enable_attention_2of4 = (
    os.getenv("SGLANG_ATTN_2OF4", "0") == "1"
)
```

然后把它传给：

```python
self.extend_attention_fwd(
    ...,
    enable_2of4=self.enable_attention_2of4,
)

self.decode_attention_fwd(
    ...,
    enable_2of4=self.enable_attention_2of4,
)
```

在 Triton kernel 中声明：

```python
ENABLE_2OF4: tl.constexpr
```

然后：

```python
if ENABLE_2OF4:
    qk = mask_top2_of4(...)
```

这样 Triton 会分别编译：

```text
dense variant
2:4 variant
```

A/B 测试只需：

```bash
# Dense
export SGLANG_ATTN_2OF4=0

# 2:4
export SGLANG_ATTN_2OF4=1
```

源码改动后通常由 Triton 的 source hash 自动触发重新编译；不确定时可以清缓存：

```bash
rm -rf ~/.triton/cache
```

---

# 7. 首轮实验必须规避的边界问题

## Prefix 与 extend 之间可能跨 quartet

假设：

```text
prefix_len = 6
```

按绝对 key position 分组时存在：

```text
[4, 5, 6, 7]
```

但 SGLang extend kernel 可能把：

```text
4,5 作为 prefix 处理
6,7 作为 extend 处理
```

如果两个循环各自 top-2，同一 quartet 最终可能保留 4 个。

所以第一轮建议：

```text
--disable-radix-cache
--chunked-prefill-size -1
无 MTP
完整 prompt 一次 prefill
```

让初始 prefill 的 `prefix_len=0`，先验证基础语义。

正式支持 prefix reuse 时，需要专门处理：

```text
prefix 尾部不足 4 个的位置
+
extend 开头的位置
```

把它们组合后再选 top-2。

## 分组必须按逻辑 key position

Kernel 中同时存在：

```text
offs_n  = 逻辑 token 位置
kv_loc  = KV cache 物理地址
```

2:4 分组必须按：

```python
offs_n // 4
```

而不是：

```python
kv_loc // 4
```

否则 Radix cache 的物理分配变化会改变模型输出。

---

# 8. 各后端的实际选择建议

| 后端                  |      是否容易看到 QK score | 适合用途             | 主要问题                     |
| ------------------- | -------------------: | ---------------- | ------------------------ |
| `torch_native`      | 需要把 SDPA 改成显式 matmul | 最快验证数学正确性        | 极慢、显存大、不支持 MTP           |
| `triton`            |                    是 | **当前最推荐**        | 需要处理 prefix 边界和两个 kernel |
| `flashinfer` 普通 JIT |  可以，但在 CUDA/CuTe 内核中 | 后续做高性能实现         | 修改难度较大                   |
| `trtllm_mha`        |          否，核心是 CUBIN | 保留生产基线           | 无法简单插入 score top-2       |
| `fa4`               |                理论上可以 | 高性能 Blackwell 路径 | CuTe/FlashAttention 代码复杂 |

## 推荐实施顺序

```text
第一阶段
SGLang Triton：
QK 后置 -inf，验证任务精度、perplexity、logits 差异

第二阶段
SGLang Triton：
只加载每组被选中的两个 V，近似评估 V 流量下降

第三阶段
FlashInfer custom JIT / 自定义 CuTe kernel：
实现真正融合的 top-2 + sparse PV

第四阶段
与原始 trtllm_mha 对比生产吞吐
```

最后还要区分两种“性能”：

* 只把一半 score 写成 `-inf`，能够测**模型质量、任务准确率和输出变化**；
* 现有 kernel 后面的 `tl.dot(p, v)` 仍是 dense PV，所以通常不能测出 Rubin 的硬件加速，甚至会因为多了 top-2 选择而变慢；
* 要评估计算性能，必须让后半段真的只读取两个 V，并跳过另外两个 PV 乘加。

[1]: https://docs.sglang.io/docs/advanced_features/attention_backend?utm_source=chatgpt.com "Attention Backend - SGLang Documentation"
[2]: https://github.com/sgl-project/sglang/blob/v0.5.13/python/sglang/srt/layers/attention/attention_registry.py "sglang/python/sglang/srt/layers/attention/attention_registry.py at v0.5.13 · sgl-project/sglang · GitHub"
[3]: https://raw.githubusercontent.com/sgl-project/sglang/v0.5.13/python/sglang/srt/layers/attention/trtllm_mha_backend.py "raw.githubusercontent.com"
[4]: https://github.com/flashinfer-ai/flashinfer/blob/main/CLAUDE.md?utm_source=chatgpt.com "flashinfer/CLAUDE.md at main · flashinfer-ai/flashinfer · GitHub"
[5]: https://github.com/sgl-project/sglang/blob/v0.5.13/python/sglang/srt/layers/attention/triton_backend.py "sglang/python/sglang/srt/layers/attention/triton_backend.py at v0.5.13 · sgl-project/sglang · GitHub"
