**结合你这条命令、上一条报错确认的实际 backend，以及 vLLM 0.29.0 源码，你当前的主 attention 并不是“NVFP4 QK”，而是：**

> **Attention 投影权重和主要中间张量使用 BF16；RMSNorm、RoPE 在 FP32 寄存器中计算；进入 decode attention 前，最终 Q 被量化为普通 E4M3 FP8，KV cache 也以 E4M3 FP8 存储。Decode 的 QK 使用 FP8 Q、FP8 K 操作数。**  
> **你此前添加的 K 侧 NVFP4 伪量化，发生在原生 FP8 缓存量化之前，并没有把物理 KV cache 改成 NVFP4。** [GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/layers/quantization/modelopt.py)

这里尤其需要澄清：**NVIDIA 这个 NVFP4 checkpoint 把 attention 的线性投影排除在 NVFP4 权重量化之外，但又单独配置了 FP8 KV cache。** 因此，“模型名带 NVFP4”“Attention 权重精度”“QK 操作数精度”“KV cache 精度”是四件不同的事。[Hugging Face](https://huggingface.co/nvidia/GLM-5.2-NVFP4/raw/main/config.json)

下面先审计原生流程，再叠加你已经加入的 K 伪量化。核对依据是官方 **v0.29.0 源码与 NVIDIA 模型配置**，不是读取了你服务器的本地目录；你的日志则已经实证确认了 `FlashInferMLASparseImpl.forward_mqa()` 入口处 Q 为 `torch.float8_e4m3fn`。粘贴的文本 (1)

---

## 一、你的启动参数实际决定了什么精度

### 1. 没有指定 `--dtype`：主要计算 dtype 为 BF16

官方模型配置中写的是：

```json
"dtype": "bfloat16"
```

你没有覆盖 `--dtype`，vLLM 按自动解析采用 BF16。因此，非量化权重、通常的 hidden states，以及多数 attention 中间张量以 BF16 保存。**这不意味着已量化的 MoE 权重会被整体转换成 BF16。** [Hugging Face](https://huggingface.co/nvidia/GLM-5.2-NVFP4/raw/main/config.json)

### 2. 没有指定 `--kv-cache-dtype`：这里实际解析成 FP8，而不是 BF16

这次能把原因明确到配置解析函数，不只是根据报错反推。

模型配置包含：

```json
"quantization_config": {
    "quant_algo": "NVFP4",
    "kv_cache_scheme": {
        "dynamic": false,
        "num_bits": 8,
        "type": "float"
    },
    "quant_method": "modelopt"
}
```

`vllm/utils/torch_utils.py` 中，解析过程是：

```text
--kv-cache-dtype 未指定
        ↓
初始为 auto
        ↓
resolve_kv_cache_dtype_string()
        ↓
读取模型 quantization_config.kv_cache_scheme
        ↓
ModelOpt + static 8-bit float
        ↓
fp8_e4m3
```

**所以你这条命令没有写 FP8，并不代表没有启用 FP8 KV cache；FP8 来自 checkpoint 的配置。** [Hugging Face](https://huggingface.co/nvidia/GLM-5.2-NVFP4/raw/main/config.json)

### 3. `--quantization modelopt` 不会把所有 attention 投影强制变成 NVFP4

官方配置的 `ignore` 覆盖了 attention 模块，例如：

```text
model.layers.3.self_attn*
model.layers.4.self_attn*
...
model.layers.77.self_attn*
```

前几层则通过更大范围的整层排除覆盖。vLLM 的 `ModelOptQuantConfigBase.get_quant_method()` 对被排除的线性层返回：

```python
UnquantizedLinearMethod()
```

因此，按照这份 checkpoint 配置，**`fused_qkv_a_proj`、`q_b_proj`、`kv_b_proj`、`o_proj` 的权重是 BF16 路径，不是 W4A4 NVFP4 GEMM。** [Hugging Face](https://huggingface.co/nvidia/GLM-5.2-NVFP4/raw/main/config.json)

为什么 attention 被排除，cache 仍然量化？因为源码里**先处理 `Attention/MLAAttention` 的 KV-cache 量化方法，再处理线性层的排除规则**。两者确实分开管理。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/layers/quantization/modelopt.py)

---

## 二、先确定你这次 TP=8 的维度

模型的主要参数是：

```text
hidden_size          = 6144
q_lora_rank          = 2048
kv_lora_rank         = 512
qk_nope_head_dim     = 192
qk_rope_head_dim     = 64
v_head_dim           = 256
num_attention_heads = 64
```

你的 **TP 总大小是 8**，所以每个 rank 的主 attention head 数为：

\[
H_{\mathrm{local}}=64/8=8
\]

`--nnodes 2` 表示跨两个节点部署这个并行配置，不能再把 TP 乘二。下面用 `T` 表示当前一步的 query token 数。主 Q 的关键形状为：

```text
Q 压缩表示：       [T, 2048]
Q 升维后：         [T, 8, 256]
Q NoPE：           [T, 8, 192]
Q RoPE：           [T, 8, 64]
吸收投影后的 NoPE：[T, 8, 512]
最终 decode Q：    [T, 8, 576]
```

这些维度由模型配置和 `DeepseekV32Attention` 中的 TP 划分、投影与 BMM 决定。[Hugging Face](https://huggingface.co/nvidia/GLM-5.2-NVFP4/raw/main/config.json)

---

## 三、Q 形成过程：逐步精度

主要入口：

```text
vllm/models/deepseek_v32/attention.py
    DeepseekV32Attention.forward()
```

### Q 路径总览

```text
hidden_states [T,6144]                         BF16
          │
          │ fused_qkv_a_proj，BF16 权重
          ▼
q_c [T,2048]                                  BF16
          │
          │ RMSNorm：FP32 计算，写回 BF16
          ▼
q_c_normed [T,2048]                            BF16
          │
          │ q_b_proj，BF16 权重
          ▼
q [T,8,256]                                   BF16
          │
          ├── q_nope [T,8,192]                 BF16
          │        │
          │        │ BMM × W_UK_T [8,192,512]
          │        ▼
          │   ql_nope [T,8,512]                BF16
          │        │
          │        │ 转 FP32，除以本层 q_scale
          │        ▼
          │   ql_nope_fp8                      E4M3 FP8
          │
          └── q_pe [T,8,64]                    BF16
                   │
                   │ 转 FP32，执行 RoPE
                   │ 再除以同一个 q_scale
                   ▼
              q_pe_rotated_fp8                E4M3 FP8

最终打包：
Q_mla [T,8,576]                                E4M3 FP8
```

前半部分来自模型 forward，最后的 RoPE、缩放与 FP8 打包来自 `fused_q()` 的实现。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/attention.py)

### 1. `fused_qkv_a_proj`：BF16 输入、BF16 权重、BF16 输出

这个融合线性层一次生成：

```text
q_c  : [T,2048]
kv_c : [T,512]
k_pe : [T,64]
```

三部分都来自同一个 BF16 输出张量的切片。融合投影不是说这三部分已经进行了 Q/K 的最终量化，只是减少了投影调用。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/attention.py)

### 2. Q 的 RMSNorm：FP32 计算，BF16 落地

`_fused_norm_rope_kernel` 的 `pid == 2` 分支调用 `_rms_norm()`：

| 环节 | 精度 |
|---|---|
| 读取 `q_c`、RMSNorm 权重 | BF16 |
| 平方、归约求均值、`rsqrt`、归一化与乘权重 | FP32 |
| 写入 `q_c_out` | BF16 |

**“RMSNorm 用 FP32 计算”不代表后面的 `q_b_proj` 接收 FP32。** 写回 BF16 buffer 时已经发生了一次 BF16 舍入。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/common/kernels.py)

### 3. `q_b_proj` 与 MLA 吸收投影：BF16 操作数

`q_b_proj` 输出 BF16 的 `[T,8,256]`，随后拆成 192 维 NoPE 与 64 维 PE。

NoPE 接着执行：

\[
Q^{L}_{t,h}=Q^{N}_{t,h}W^{UK}_{h}
\]

对应：

```text
[T,8,192] × [8,192,512] → [T,8,512]
```

这里的 `W_UK_T` 也是 BF16。`MLAAttention.process_weights_after_loading()` 明确为这些 BMM 准备 FP16/BF16 权重副本；你的模型 activation dtype 是 BF16，所以使用 BF16 副本。输出 `ql_nope` 仍是 BF16。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/attention.py)

**关于这些 BF16 GEMM 的累加：**输入和输出是 BF16 可以确认，但不能据此写成“内部每一步都是 BF16”。PyTorch 通常采用 FP32 中间累加，同时允许部分 BF16 reduced-precision reduction；是否发生中间截断，还受运行时设置和实际 GEMM kernel 影响。你的启动命令不足以证明所有归约步骤都保持完整 FP32。[GitHub](https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/notes/numerical_accuracy.md)

### 4. 最终 Q 在 `fused_q()` 中变成普通 FP8

源码中的控制关系是：

```text
量化 KV cache
    + backend 支持量化 Q
        ↓
_fp8_query = True
        ↓
fused_q(..., quantize_mqa=True)
```

FlashInfer sparse MLA backend 明确要求 FP8 cache 配套 FP8 query，因此这里启用最终 Q 的 FP8 转换。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/attention.py)

它使用的是：

\[
Q_8=\operatorname{E4M3}\!\left(\frac{Q_{\mathrm{MLA}}}{s_Q}\right)
\]

这里的 \(s_Q\) 是本层 `_q_scale` 标量，**不是每 32 个数重新计算一个 E8M0 scale**。因此这是普通的 scaled E4M3 FP8，**不是 MXFP8**。CuTeDSL 实现也明确将 BF16 NoPE 转 FP32、缩放后转 E4M3；PE 则在 FP32 中旋转、缩放后直接转 E4M3。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py)

还有一个精度细节：默认 RoPE 的 sin/cos 先计算，再按模型 dtype 保存为 BF16；kernel 读取后转成 FP32参与旋转。**转换为 FP32 不会恢复 sin/cos 缓存先前舍入掉的精度。** [GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/layers/rotary_embedding/base.py)

---

## 四、K 形成过程与 KV cache 精度

### 1. 原生 K 路径

Decode 使用的是 MLA 的压缩表示：

\[
K^{MLA}_{j}=[c^{KV}_{j},k^{R}_{j}]
\]

其中 latent 部分为 512 维，RoPE 部分为 64 维。缓存里不是每个 head 一份展开后的 192 维 K。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/layers/attention/mla_attention.py)

| 步骤 | 张量/数据 | 实际精度 |
|---|---|---|
| A 投影输出 | `kv_c [T,512]` | BF16 |
| A 投影输出 | `k_pe [T,64]` | BF16 |
| KV RMSNorm | 归约、归一化、乘权重后的 `kv_c` 寄存器值 | FP32 |
| K RoPE | `r1`、`r2` 旋转结果 | FP32 |
| 可选的 prefill 输出副本 | `kv_c_out`、`k_pe_out` | BF16 |
| 主 MLA cache 写入 | 512 维 latent + 64 维 RoPE | **全部 E4M3 FP8** |

这里最容易看错的是最后两行：**prefill 用的 BF16 输出副本，与真正的持久化 FP8 cache 是两份不同用途的数据。** 原生 cache 分支直接从 FP32 寄存器值缩放、转换到 FP8，并不是必须先绕一遍 BF16 输出副本。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/common/kernels.py)

### 2. 原生 FP8 cache 写入使用同一个 K scale

这条分支的数值含义是：

\[
C_8=\operatorname{E4M3}\left(c^{KV}/s_K\right)
\]

\[
R_8=\operatorname{E4M3}\left(k^{R}/s_K\right)
\]

\[
K_{\text{cache}}=[C_8,R_8]
\]

其中 \(s_K\) 来自本层 `_k_scale`，512 维 latent 和 64 维 RoPE 共用它。**这不是 per-16 NVFP4 block scale，也不是 per-32 MXFP8 scale。** [GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/common/kernels.py)

缓存的逻辑布局是：

```text
[num_blocks, block_size, 576]

每个 token：
[ 512 个 latent E4M3 | 64 个 RoPE E4M3 ]
```

单看数值载荷，每个 token、每层是：

\[
576\times 1\text{ byte}=576\text{ bytes}
\]

不包括分页、索引等其他开销。底层 tensor 有时显示为 `torch.uint8`，进入 attention 前再 `.view(torch.float8_e4m3fn)`；这是用字节保存 FP8 bit pattern，**不是 INT8 数值运算**。[GitHub](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.6.18/flashinfer/mla/_core.py)

**你这条路径也不是 `fp8_ds_mla` 的“FP8 latent + BF16 RoPE”混合布局。** 源码有那条独立分支，但你目前普通 `fp8_e4m3` 路径中，RoPE 的 64 维同样转成 FP8。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/common/kernels.py)

### 3. 你的 NVFP4 伪量化叠加在哪里？

按照你贴出的插入位置，实际流程是：

```text
kv_c：
BF16 投影输出
    → FP32 RMSNorm
    → 你的 NVFP4 量化/反量化
    → FP32 寄存器结果
    → 原生除以 _k_scale
    → E4M3 FP8 cache

k_pe：
BF16 投影输出
    → FP32 RoPE
    → 你的 r1/r2 NVFP4 量化/反量化
    → FP32 寄存器结果
    → 原生除以 _k_scale
    → E4M3 FP8 cache
```

这里“伪量化后仍是 FP32”由你函数末尾的：

```python
return ... .to(x.dtype)
```

以及插入点的输入 dtype 决定：**此时 `_rms_norm` 返回值、`r1`、`r2` 都已经是 FP32 寄存器值。** 后续原生代码仍执行 FP8 转换。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/common/kernels.py)

所以你此前实验的缓存数值更准确地写成：

\[
K_{\text{cache}}
=
\operatorname{E4M3}
\left(
\frac{\mathcal F_{\mathrm{NVFP4}}(K)}{s_K}
\right)
\]

其中 \(\mathcal F_{\mathrm{NVFP4}}\) 表示你自己的量化—反量化过程。

**你的 `global_scale=1.0` 与原生 `_k_scale` 不是同一个 scale。** 前者服务于自定义 NVFP4 伪量化，后者服务于最终 FP8 缓存；前者等于 1，不代表后者等于 1。后者的具体数值需要看 checkpoint 加载后的 scale，命令本身不能给出。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/layers/quantization/kv_cache.py)

---

## 五、Decode 的 QK、softmax 和输出究竟是什么精度？

### 1. QK 的两个输入：明确都是 E4M3 FP8

在你的实际 backend 中：

```text
Q：
[T,8,576]，torch.float8_e4m3fn

K：
从 FP8 MLA cache 取出的 [历史 token,576]
torch.float8_e4m3fn
```

`forward_mqa()` 将它们交给：

```text
flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla()
```

因此，**这是原生 FP8 Q × FP8 K 的 attention 路径，不是 BF16 Q × BF16 K，也不是 FP8 Q × 原生 NVFP4 K。** 你上一条报错也直接证实了 Q 的实际 dtype。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py) 粘贴的文本 (1)

### 2. score 的缩放因子

vLLM 传给这个 kernel 的两个主要系数为：

```text
bmm1_scale = attention_scale × q_scale × k_scale
bmm2_scale = k_scale
```

按当前模型默认 RoPE 配置：

\[
\alpha=\frac{1}{\sqrt{192+64}}=\frac1{16}
\]

所以，忽略有限精度舍入、仅表达数值语义：

\[
S_{t,h,j}
=
\frac{s_Qs_K}{16}
\sum_{d=0}^{575}
Q_{8,t,h,d}K_{8,j,d}
\]

**这里的 attention scale 仍是 \(1/16\)，不是 \(1/\sqrt{576}\)。** 576 是吸收投影后的归约维度，不是重新定义模型 attention scale 的依据。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py)

### 3. 不能把“FP8 QK”理解成 score 和 softmax 都是 FP8

目前能从调用链明确核实的是：

| 部分 | 已核实的精度 |
|---|---|
| QK 的 Q 操作数 | E4M3 FP8 |
| QK 的 K 操作数 | E4M3 FP8 |
| 原生 Q/K scale | FP32 标量 |
| 默认 softmax 累加版本 | **FP32 accumulator** |
| LSE，若请求输出 | FP32 |
| attention 返回的 latent output | BF16 |

vLLM 0.29.0 的官方依赖固定到 FlashInfer 0.6.18；该版本函数默认 `use_fp16_softmax=None`，文档明确对应标准 **FP32-accumulator softmax** cubin。vLLM 的这个调用没有启用 FP16 softmax。输出默认分配为 BF16。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/requirements/cuda.txt)

**但 QK 内每条 MMA 指令、分段累加和跨 CTA 归约是否全程保持完整 FP32，不能仅凭上述 Python dtype 就下结论。** 我已核对到 FlashInfer 的 C++ launcher 和按 dtype 选择原生 kernel 的 runner；它们继续调用具体 cubin，没有在这层暴露所有中间舍入细节。因此，这里明确确认的是 **FP8 操作数路径与默认 FP32 softmax 累加**，不是声称已经反汇编验证了你机器上每一级 QK 累加。[GitHub](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.6.18/csrc/trtllm_fmha_kernel_launcher.cu)

同理，也不能把 softmax 的 FP32 累加直接等同于“传给后续 PV MMA 的 P 一定始终以 FP32 保存”。

### 4. PV 与输出投影

MLA 的 latent cache 前 512 维同时承担 V 的作用。数值语义为：

\[
O^{L}=s_K\cdot P\,C_8
\]

所以 `bmm2_scale` 使用 `_k_scale`，而不是另存一份展开后的 BF16 V cache。随后：

```text
attention latent output [T,8,512]     BF16
    │
    │ BMM × W_UV [8,512,256]          BF16 操作数
    ▼
[T,8,256]                            BF16
    │
    │ o_proj，BF16 权重
    ▼
输出 hidden states                    BF16
```

**这也说明你量化 `kv_c` 会同时影响 QK 和 PV，不是只改变 score 的 K。** [GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py)

---

## 六、Prefill 不能直接套用上面的 Decode 结论

这条实现存在分支，**不是所有 prefill 都是 BF16 QK，也不是所有 prefill 都是 FP8 QK。**

| 实际选择的执行分支 | QK 归约维度 | Q/K 操作数 |
|---|---:|---|
| Decode 使用 sparse MQA | 576 | FP8 / FP8 |
| Prefill 使用展开后的 dense MHA | 256 | BF16 / BF16 |
| Prefill 仍使用 sparse MQA | 576 | FP8 / FP8 |

原因是：符合 dense MHA 条件时，模型重新组织 BF16 的原始 Q，并通过 BF16 `kv_b_proj` 将 latent 展开成 K/V；否则仍使用吸收投影后的 Q 与 FP8 latent cache。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/attention.py)

源码中 dense MHA 的条件之一是：

```text
prefill_max_seq_len <= index_topk
```

你的模型 `index_topk=2048`，但这还取决于是否构造了对应 prefill metadata、实际调度分类等。**不能简化成“输入不超过 2048，就无条件使用 BF16”。** 此外，当前 FP8 cache 配置会使另一条长上下文 masked-MHA 优化分支不可用。[Hugging Face](https://huggingface.co/nvidia/GLM-5.2-NVFP4/raw/main/config.json)

对于 dense-MHA prefill，当前新 token 可以使用 `kv_c_out/k_pe_out` 的 BF16 副本；历史 prefix 若来自 FP8 cache，则需要先反量化再展开。**反量化后的 BF16 不会消除历史 FP8 舍入误差。** [GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/layers/attention/mla_attention.py)

---

## 七、Indexer 的 Q/K 是另一套精度，不要和主 attention 混在一起

你日志中的：

```text
use_fp4_cache=False
```

说的是 **DSA indexer cache**，不是主 MLA cache，也不是你自定义 NVFP4 伪量化的开关。粘贴的文本 (1)

Indexer 在这条实现中的主要流程是：

```text
Indexer 投影：BF16
    ↓
Indexer K 的 LayerNorm、Q/K 的 RoPE：FP32 计算
    ↓
Indexer Q/K：E4M3 FP8
    ↓
Indexer score / top-k
    ↓
给主 attention 提供选中的历史 token 索引
```

它的 head dimension 是 **128**，K cache 保存 128 个 FP8 值与 FP32 scale；Q 也按整个 128 维向量求 scale，scale 的影响会合入 indexer 权重。即使 scale 取二次幂，这也**不等于你想做的主 Q 每 32 维一组的 MXFP8**。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py)

---

## 最后把你“加入 Q MXFP8 之前”的真实基线收束起来

### 没有任何自定义 Q/K 伪量化的原生基线

```text
Attention 权重：BF16

Q：
BF16 投影
→ FP32 RMSNorm / BF16 写回
→ BF16 Q 升维与吸收投影
→ FP32 RoPE / 缩放
→ 普通 E4M3 FP8

K：
BF16 投影
→ FP32 RMSNorm / RoPE
→ 普通 E4M3 FP8 cache

Decode：
FP8 Q × FP8 K
→ 默认 FP32 softmax 累加
→ BF16 latent output
→ BF16 输出投影
```

### 你已经加了 K NVFP4 伪量化、尚未加 Q MXFP8 的基线

```text
Q：
仍是上面的普通 E4M3 FP8

K：
BF16 投影
→ FP32 RMSNorm / RoPE
→ 你的 NVFP4 量化—反量化
→ 再做原生 E4M3 FP8 量化
→ FP8 cache

Decode：
仍执行 FP8 Q × FP8 K 路径
但 K 的数值已经受到前面的 NVFP4 伪量化扰动
PV 使用的 latent V 同样受到影响
```

这两条基线的区别来自你新增的 K 数值变换，而**不是物理 cache dtype 或原生 QK kernel 变成了 NVFP4**。[GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/models/deepseek_v32/common/kernels.py)

**所以，你下一步研究 MXFP8×NVFP4 操作数量化误差时，真正要替换的是现有的“普通 FP8 Q 量化”和“普通 FP8 cache 存储路径”，不能只在它们后面追加伪量化，再把结果当作独立的 MXFP8×NVFP4 实验。** 同时，若为实验改成 BF16 载体，还应把对照组也改成相同载体，避免将原生 FP8 移除的影响与新量化方案的影响混在一起。
