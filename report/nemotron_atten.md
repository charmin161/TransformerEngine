## 结论

对 NVIDIA 官方推荐的 SGLang `v0.5.13` 路径，**attention score 并不在 `nemotron_h.py` 中显式出现**。实际调用链是：

```text
NemotronHAttention.forward
    └─ QKV projection + split
       └─ RadixAttention.forward
          └─ HybridLinearAttnBackend
             ├─ Mamba 层 → Mamba2 backend
             └─ Full-attention 层 → TRTLLMHAAttnBackend
                 ├─ prefill/extend
                 │   └─ flashinfer.prefill.trtllm_batch_context_with_kv_cache
                 └─ decode
                     └─ flashinfer.decode.trtllm_batch_decode_with_kv_cache
                         └─ TRT-LLM FMHA 编译内核/CUBIN
```

所以有三个不同层次的改法：

| 目标                   | 应修改的位置                          | 能回答什么                            |
| -------------------- | ------------------------------- | -------------------------------- |
| 先看模型精度损失             | `torch_native_backend.py`       | 可靠模拟 4 选 2，但不能评估 Rubin 加速        |
| 看普通 GPU kernel 开销    | SGLang Triton attention kernels | 能测选择、mask、减少部分 V 访问的成本           |
| 真正模拟 Rubin sparse PV | FlashInfer/TRT-LLM FMHA 内核      | 才可能模拟压缩 metadata 和 Sparse MMA 性能 |

最合理的顺序是先做第一种，确认模型质量，再做 Triton，最后才考虑 FlashInfer/TRT-LLM 内核。

---

# 1. Nemotron-3-Ultra 的相关模型结构

官方模型卡给出的结构是 550B 总参数、55B 激活参数，采用 Mamba-2、LatentMoE、选择性 full attention 和 MTP 的混合架构；QKV 与 attention projection 不是主要的 NVFP4 线性层，而是保留为 BF16 或 MXFP8 路径。([Hugging Face][1])

我按官方 `config.json` 的 `layers_block_type` 计数，主干为：

```text
108 个主干 block
├─ 48 × Mamba-2
├─ 48 × MoE
└─ 12 × Full Attention
```

12 个 full-attention block 的 0-based layer ID 为：

```python
NEMOTRON_ULTRA_ATTN_LAYERS = {
    7, 14, 23, 32, 39, 48,
    57, 64, 73, 82, 89, 98,
}
```

Attention 的主要配置是：

```text
hidden_size          = 8192
num_attention_heads  = 64
num_key_value_heads  = 2
head_dim              = 128
GQA ratio             = 32 Q heads / KV head
```

这些值来自官方模型配置；上述 block 数量和 layer ID 是对 `layers_block_type` 的直接计数。([Hugging Face][2])

官方 B200 部署示例固定使用：

```text
lmsysorg/sglang:v0.5.13
TP = 4
EP = 4
attention backend = trtllm_mha
mamba backend = flashinfer
```

而且默认生产配置还打开了 EAGLE/MTP speculative decoding。([Hugging Face][3])

在 TP=4 时，SGLang 对每个 rank 分配：

```text
16 个 Q heads
1 个 KV head
```

由于全模型只有 2 个 KV heads、但 TP 为 4，KV head 会在 TP ranks 间复制。每个 rank 的 attention kernel 因而看到本地 GQA 比率 `16:1`；全局仍是 `32:1`。SGLang 的 head 切分与 KV 复制逻辑位于 `NemotronHAttention.__init__`。

---

# 2. Attention 在 SGLang 中具体在哪里

以下均以 NVIDIA 官方镜像对应的 SGLang `v0.5.13` 为准。

## 2.1 模型层只生成 Q、K、V

文件：

```text
python/sglang/srt/models/nemotron_h.py
```

`NemotronHAttention.forward()` 做的事情只有：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split(...)
attn_output = self.attn.forward(q, k, v, forward_batch)
output, _ = self.o_proj(attn_output)
```

即模型文件里没有 `Q @ K.T`，也没有显式 softmax。

## 2.2 `RadixAttention` 将 Q/K/V 交给 backend

文件：

```text
python/sglang/srt/layers/radix_attention.py
```

`RadixAttention.forward()` 对 K/V 做 reshape、缓存写入准备，然后调用：

```python
get_attn_backend().forward(...)
```

实际 attention 运算已经下沉到 backend。

## 2.3 Nemotron 使用 hybrid backend

文件：

```text
python/sglang/srt/layers/attention/attention_registry.py
```

对于 Mamba-2 hybrid 模型，SGLang 会构造：

```text
HybridLinearAttnBackend(
    full_attn_backend=<由 --attention-backend 指定>,
    linear_attn_backend=Mamba2AttnBackend,
    full_attn_layers=config.full_attention_layer_ids,
)
```

因此 `--attention-backend torch_native` 或 `triton` 只会替换那 12 个 full-attention block，Mamba-2 仍然走 Mamba backend。

## 2.4 官方 B200 路径中的真正 attention kernel

文件：

```text
python/sglang/srt/layers/attention/trtllm_mha_backend.py
```

Decode 调用：

```python
flashinfer.decode.trtllm_batch_decode_with_kv_cache(...)
```

Prefill/extend 调用：

```python
flashinfer.prefill.trtllm_batch_context_with_kv_cache(...)
```

所以，官方路径下的：

```text
QK
causal mask
softmax
PV
```

全部融合在 FlashInfer 的 TRT-LLM FMHA kernel 中。

FlashInfer 当前公开源码也将两个主要部分明确称为：

```text
BMM1 = QKᵀ
BMM2 = softmax(QKᵀ) × V
```

更深一层，FlashInfer 的 `get_trtllm_gen_prefill_module()` 会构建并加载 TRT-LLM FMHA module。

其生成器并不包含完整的 FMHA CUDA kernel 源码，而是编译 launcher/reduction，并加载 TRT-LLM FMHA CUBIN artifact：

```text
trtllm_fmha_kernel_launcher.cu
fmhaReduction.cu
TRTLLM_GEN_FMHA CUBIN artifact
```

这意味着：**直接改 SGLang 的 `trtllm_mha_backend.py`，只能改变传参，无法在 QK 与 softmax 之间插入 Python 逻辑。**

---

# 3. 你要模拟的 2:4 attention 应如何定义

建议先固定一个明确且可复现的定义：

对于每个：

```text
request
query head
query token
```

沿逻辑 key-position 维度分组：

```text
keys [0,1,2,3]    → 保留 score 最大的两个
keys [4,5,6,7]    → 保留 score 最大的两个
keys [8,9,10,11]  → 保留 score 最大的两个
...
```

然后：

```python
未保留的 score = -inf
```

完整顺序应为：

```text
QK
→ scale / model-specific score transform
→ causal、padding、sliding-window mask
→ 每 4 个有效 logits 选最大的 2 个
→ 其余设为 -inf
→ softmax
→ PV
```

注意四点。

第一，应保留**数值最大的两个**，而不是绝对值最大的两个。对于 softmax，`-20` 即使绝对值很大，仍然几乎没有贡献。

第二，2:4 分组沿的是 **K/token 维度**，不是 128 维的 head dimension。

第三，GQA 下每个 query head 必须单独选择。不能让共享同一 KV head 的 16 个本地 Q heads 共用一份 mask。

第四，paged KV cache 下必须按逻辑位置 `offs_n` 分组，不能按物理 cache slot `kv_loc` 分组。否则 mask 会受 radix-cache 分配方式影响，相同 prompt 可能因为物理 cache 布局不同而产生不同输出。

---

# 4. 最快、最可靠的质量仿真：修改 `torch_native_backend.py`

文件：

```text
python/sglang/srt/layers/attention/torch_native_backend.py
```

现有 extend 路径最终调用 PyTorch SDPA。

Decode 路径也是同样的 SDPA。

SDPA 不暴露 score，所以需要在实验分支中将其替换为显式：

```text
QK → mask → 2:4 → softmax → PV
```

## 4.1 可直接使用的 reference attention

在文件顶部增加：

```python
import math
import os

import torch
import torch.nn.functional as F
```

在 `TorchNativeAttnBackend` 中增加：

```python
@staticmethod
def _manual_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None,
    causal: bool,
    query_offset: int,
    sliding_window_size: int | None,
    prune_2of4: bool,
    q_chunk_size: int,
) -> torch.Tensor:
    """
    Args:
        q: [Hq, Q, Dq]
        k: [Hkv, K, Dq]
        v: [Hkv, K, Dv]

    Returns:
        [Hq, Q, Dv]

    2:4 is applied independently for every:
        [KV head, Q-head-within-GQA-group, query position]
    along the logical K/token dimension.
    """
    hq, q_len, qk_dim = q.shape
    hkv, kv_len, k_dim = k.shape

    if qk_dim != k_dim:
        raise ValueError(f"Q/K head dimensions differ: {qk_dim} vs {k_dim}")
    if hq % hkv != 0:
        raise ValueError(f"Hq={hq} must be divisible by Hkv={hkv}")

    gqa_group_size = hq // hkv
    value_dim = v.shape[-1]

    # SGLang/PyTorch GQA mapping uses contiguous Q-head groups for each KV head.
    q_grouped = q.reshape(hkv, gqa_group_size, q_len, qk_dim)

    if scale is None:
        scale = 1.0 / math.sqrt(qk_dim)

    # Converting BF16 Q/K values to FP32 preserves their quantized values,
    # while giving FP32 accumulation for the reference path.
    k_transposed = k.float().unsqueeze(1).transpose(-2, -1)
    v_grouped = v.float().unsqueeze(1)

    k_pos = torch.arange(kv_len, device=q.device)
    output_chunks: list[torch.Tensor] = []

    for q_start in range(0, q_len, q_chunk_size):
        q_end = min(q_start + q_chunk_size, q_len)
        q_chunk = q_grouped[:, :, q_start:q_end, :]

        # [Hkv, G, Qc, D] @ [Hkv, 1, D, K]
        # -> [Hkv, G, Qc, K]
        scores = torch.matmul(q_chunk.float(), k_transposed)
        scores.mul_(scale)

        chunk_q_len = q_end - q_start

        if causal or (
            sliding_window_size is not None and sliding_window_size > -1
        ):
            q_pos = torch.arange(
                query_offset + q_start,
                query_offset + q_end,
                device=q.device,
            )

            if causal:
                valid = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            else:
                valid = torch.ones(
                    (chunk_q_len, kv_len),
                    dtype=torch.bool,
                    device=q.device,
                )
        else:
            valid = torch.ones(
                (chunk_q_len, kv_len),
                dtype=torch.bool,
                device=q.device,
            )

        if sliding_window_size is not None and sliding_window_size > -1:
            valid &= k_pos.unsqueeze(0) >= (
                q_pos.unsqueeze(1) - sliding_window_size
            )

        # Apply the model's normal attention mask first.
        scores = scores.masked_fill(
            ~valid.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        if prune_2of4:
            # Pad only for grouping. Padded positions remain invalid.
            pad = (-kv_len) % 4

            if pad:
                grouped_scores = F.pad(
                    scores,
                    (0, pad),
                    value=float("-inf"),
                )
                grouped_valid = F.pad(valid, (0, pad), value=False)
            else:
                grouped_scores = scores
                grouped_valid = valid

            grouped_scores = grouped_scores.reshape(
                hkv,
                gqa_group_size,
                chunk_q_len,
                -1,
                4,
            )
            grouped_valid = grouped_valid.reshape(
                1,
                1,
                chunk_q_len,
                -1,
                4,
            )

            # Top-2 by raw numerical logit value, not absolute magnitude.
            top2_indices = grouped_scores.topk(
                k=2,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices

            keep = torch.zeros_like(grouped_scores, dtype=torch.bool)
            keep.scatter_(-1, top2_indices, True)

            # Prevent padded/causally-invalid locations from being retained.
            keep &= grouped_valid

            grouped_scores = grouped_scores.masked_fill(
                ~keep,
                float("-inf"),
            )

            scores = grouped_scores.reshape(
                hkv,
                gqa_group_size,
                chunk_q_len,
                -1,
            )[..., :kv_len]

        probabilities = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        )

        # [Hkv, G, Qc, K] @ [Hkv, 1, K, Dv]
        # -> [Hkv, G, Qc, Dv]
        chunk_output = torch.matmul(probabilities, v_grouped)
        output_chunks.append(chunk_output.to(q.dtype))

    output = torch.cat(output_chunks, dim=2)
    return output.reshape(hq, q_len, value_dim)
```

这个实现避免了显式复制 GQA 的 K/V heads，但会显式生成 score，因此只能用于 reference/quality 实验。

## 4.2 增加实验开关

在 `__init__()` 中增加：

```python
self.reference_attn = (
    os.getenv("SGLANG_REFERENCE_ATTN", "0") == "1"
)
self.simulate_attn_2of4 = (
    os.getenv("SGLANG_ATTN_2OF4", "0") == "1"
)
self.reference_q_chunk_size = int(
    os.getenv("SGLANG_REFERENCE_ATTN_Q_CHUNK", "128")
)

raw_layers = os.getenv("SGLANG_ATTN_2OF4_LAYERS", "all").strip()

if raw_layers.lower() == "all":
    self.attn_2of4_layers: set[int] | None = None
elif not raw_layers:
    self.attn_2of4_layers = set()
else:
    self.attn_2of4_layers = {
        int(x.strip())
        for x in raw_layers.split(",")
        if x.strip()
    }
```

再增加：

```python
def _enable_2of4_for_layer(self, layer_id: int) -> bool:
    if not self.simulate_attn_2of4:
        return False

    return (
        self.attn_2of4_layers is None
        or layer_id in self.attn_2of4_layers
    )
```

给 `_run_sdpa_forward_extend()` 和 `_run_sdpa_forward_decode()` 增加参数：

```python
prune_2of4: bool = False
```

调用它们时，在 `forward_extend()` 和 `forward_decode()` 中传：

```python
prune_2of4=self._enable_2of4_for_layer(layer.layer_id)
```

## 4.3 Extend/prefill 分支替换

在 `_run_sdpa_forward_extend()` 中，读取并转换完 `per_req_key` 和 `per_req_value` 后，在现有 SDPA 调用之前加入：

```python
if self.reference_attn:
    per_req_out = self._manual_gqa_attention(
        per_req_query,
        per_req_key,
        per_req_value,
        scale=scaling,
        causal=causal,
        # 当前 query 是 full sequence 中 prefix 之后的 token。
        query_offset=int(prefill_seq_len_q.item()),
        sliding_window_size=sliding_window_size,
        prune_2of4=prune_2of4,
        q_chunk_size=self.reference_q_chunk_size,
    )

    # [Hq, Q, Dv] -> [Q, Hq, Dv]
    output[start_q:end_q, :, :] = per_req_out.movedim(1, 0)
    start_q, start_kv = end_q, end_kv
    continue
```

保留原有 SDPA 代码作为：

```python
if not self.reference_attn:
    ...
```

这里不再需要 stock 实现中的 `per_req_query_redudant`。它原本是为了让非方形 SDPA 获得正确的 bottom-right causal offset；显式实现已经用 `query_offset=prefix_len` 解决了这个问题。Stock redundant-query 逻辑可见于源码。

## 4.4 Decode 分支替换

在 `_run_sdpa_forward_decode()` 中加入：

```python
if self.reference_attn:
    per_req_out = self._manual_gqa_attention(
        per_req_query,
        per_req_key,
        per_req_value,
        scale=scaling,
        # Decode KV cache 只包含过去和当前 token，没有未来 token。
        causal=False,
        # 当前 query 的逻辑位置是 K - 1。
        query_offset=int(seq_len_kv.item()) - 1,
        sliding_window_size=sliding_window_size,
        prune_2of4=prune_2of4,
        q_chunk_size=1,
    )

    output[start_q:end_q, :, :] = per_req_out.movedim(1, 0)
    start_q, start_kv = end_q, end_kv
    continue
```

---

# 5. 第一轮受控实验的启动方式

先使用与 NVIDIA 官方相同的 SGLang `v0.5.13`，但关闭所有会引入混杂因素的功能。

Dense reference：

```bash
export SGLANG_REFERENCE_ATTN=1
export SGLANG_ATTN_2OF4=0
export SGLANG_ATTN_2OF4_LAYERS=all
export SGLANG_REFERENCE_ATTN_Q_CHUNK=128

python3 -m sglang.launch_server \
  --model-path /model \
  --tp-size 4 \
  --ep-size 4 \
  --context-length 4096 \
  --mem-fraction-static 0.85 \
  --attention-backend torch_native \
  --kv-cache-dtype bf16 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-backend flashinfer \
  --chunked-prefill-size -1 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --trust-remote-code
```

2:4 reference：

```bash
export SGLANG_REFERENCE_ATTN=1
export SGLANG_ATTN_2OF4=1
export SGLANG_ATTN_2OF4_LAYERS=all
export SGLANG_REFERENCE_ATTN_Q_CHUNK=128
```

然后以完全相同的 server 参数重启。

这里强制 BF16 KV cache，是为了让实验只测 2:4 score pruning。`torch_native` 读取 KV cache 后只做 dtype cast，而生产 `trtllm_mha` 路径还会处理量化 KV scale；如果继续使用自动选择的量化 KV cache，会把 KV 量化误差也混入结果。Torch native 的读取与转换逻辑见源码。

不要在第一轮加入官方命令中的：

```text
--speculative-algorithm EAGLE
--speculative-num-steps ...
```

因为 MTP 会引入额外 attention 层和 acceptance-rate 变化。先测纯 target model。

## Score 内存规模

TP=4 时每个 rank 有 16 个 Q heads。若直接生成完整 FP32 score：

```text
4096 × 4096 × 16 × 4 bytes ≈ 1 GiB / rank / sequence
8192 × 8192 × 16 × 4 bytes ≈ 4 GiB / rank / sequence
```

所以必须进行 Q chunking。`q_chunk_size=128, K=4096` 时，单个主要 score chunk 约为：

```text
16 × 128 × 4096 × 4 bytes ≈ 32 MiB / rank
```

这仍然很慢，但足以做 4K 左右的质量实验。

---

# 6. 质量评估应该怎么做

不要直接比较：

```text
stock trtllm_mha dense
vs
torch_native 2:4
```

因为 backend、KV dtype 和数值累加顺序都变了。

正确的受控比较是：

```text
A: custom torch-native manual dense
B: custom torch-native manual 2:4
```

A 与 B 唯一差别是 `prune_2of4`。

建议按以下实验矩阵执行：

```text
A0: manual dense，所有层
A1: 只对 layer 7 启用 2:4
A2: 只对 layer 14 启用 2:4
...
A12: 只对 layer 98 启用 2:4
Aall: 12 个 attention 层全部启用 2:4
```

环境变量示例：

```bash
export SGLANG_ATTN_2OF4_LAYERS=7
```

或者：

```bash
export SGLANG_ATTN_2OF4_LAYERS=7,14,23,32,39,48,57,64,73,82,89,98
```

最有诊断价值的指标是：

[
\delta_{l,h,t}
==============

\sum_{j\notin S_{l,h,t}}
\operatorname{softmax}(s_{l,h,t})_j
]

即，被 2:4 删除位置原本占据的 dense-softmax 概率质量。建议记录每层的：

```text
mean dropped mass
P50 / P90 / P95 / P99 dropped mass
最大 dropped mass
```

模型级指标至少包括：

```text
平均 token NLL / perplexity 变化
最终 logits KL divergence
top-1 token agreement
首次 greedy token 分叉位置
固定任务集的最终 accuracy / pass rate
```

如果某层的 dropped-mass P99 很高，而单层启用时任务指标也明显下降，这一层就不适合无条件 2:4，可以进一步测试 selective fallback。

---

# 7. 想修改源码 kernel，应改哪些位置

## 7.1 Prefill/extend Triton kernel

文件：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py
```

Prefix 部分的 QK 在：

```python
qk = tl.dot(q.to(k.dtype), k)
...
qk = tl.where(final_mask, qk, float("-inf"))
row_max = tl.max(qk, 1)
```

当前 extend/triangle 部分的 QK 在：

```python
qk = tl.dot(q, k, out_dtype=tl.float32)
...
qk = tl.where(final_mask, qk, float("-inf"))
row_max = tl.max(qk, 1)
```

正确的插入点是：

```python
qk = tl.where(final_mask, qk, float("-inf"))

if ENABLE_2OF4:
    qk = keep_top2_per_logical_quartet(qk, logical_key_positions)

row_max = tl.max(qk, 1)
```

必须在 online-softmax 的 `row_max` 之前。

## 7.2 Decode Triton kernel

Nemotron Ultra 是 GQA，因此 decode 会进入 grouped kernel，而不是普通 MHA kernel。Dispatcher 明确在 `kv_group_num != 1` 时选择 grouped 路径。

实际 QK 与 softmax/PV 位于：

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py
_fwd_grouped_kernel_stage1
```

核心代码是：

```python
qk = tl.dot(q_k, k)
qk *= sm_scale_withk
...
qk = tl.where(valid_mask, qk, float("-inf"))

n_e_max = tl.maximum(tl.max(qk, 1), e_max)
p = tl.exp(qk - n_e_max[:, None])
acc += tl.dot(p.to(v.dtype), v)
```

2:4 mask 同样应插在 normal mask 后、`tl.max()` 前。

对于 Ultra 的 TP=4 配置，本地 `kv_group_num=16`，因此这里的 `qk` 形状大致是：

```text
[BLOCK_H, BLOCK_N]
```

每一行对应一个独立 query head；必须对每行独立做四元组 top-2。

---

# 8. Triton 实现中最容易出错的两个边界

## 8.1 必须按逻辑 key index 分组

Decode kernel 中：

```python
offs_n   # 逻辑序列位置
kv_loc   # paged KV cache 的物理位置
```

应使用：

```python
offs_n % 4
```

而不是：

```python
kv_loc % 4
```

否则 radix cache、prefix sharing、cache eviction 都可能改变结果。

## 8.2 Prefix 与 extend 的 4 元组可能跨界

`extend_attention.py` 将 attention 分为：

```text
stage 1: prefix KV
stage 2: 当前 extend KV
```

假设：

```text
prefix_len = 6
```

那么逻辑 quartet 是：

```text
[4, 5, 6, 7]
```

其中：

```text
4,5 在 prefix stage
6,7 在 extend stage
```

如果两个 stage 分别做 top-2，就可能在同一个逻辑 quartet 中总共保留 4 个，而不是 2 个。

因此首轮 Triton 性能原型应使用：

```text
--disable-radix-cache
--chunked-prefill-size -1
关闭 MTP
完整 prompt 一次 prefill
```

使得 `prefix_len=0`。

要支持正式 online serving，则必须：

```text
保存 prefix 最后一个不完整 quartet 的 logits
与 extend 开头的 logits 合并选择
再更新 online-softmax accumulator
```

或者明确改变算法定义，规定每个 prefill chunk 独立对齐；但后者不再等价于“按绝对 key-position 固定分组”的 2:4。

Decode 的 split-K 边界通常按 32 个 key 对齐，32 可被 4 整除，因此该问题相对简单。

---

# 9. 为什么普通 `-inf` patch 测不出 Rubin 性能

即使在 Triton 中加入：

```python
qk[discarded] = -inf
```

随后：

```python
p = exp(qk - max)
acc += tl.dot(p, v)
```

现有 `tl.dot(p, v)` 仍是 dense PV：

```text
仍加载所有 V
仍执行 dense dot
只是其中一半 p 数值为 0
```

因此这种 patch：

```text
能模拟模型输出
能测 top-2 选择和 mask 的额外开销
不能模拟 Sparse MMA 的算力收益
通常反而会变慢
```

更接近 Rubin 的 kernel 需要：

```text
1. dense QK
2. 每四个 score 选两个
3. 生成两个非零 softmax value
4. 生成两个位置的 metadata
5. 只加载被选中的两个 V rows
6. 执行 sparse P × V
```

在没有 Rubin sparse-MMA 指令的 GPU 上，可以先用 gather/FMA 近似：

```text
每个 quartet:
    output += p0 * V[index0]
    output += p1 * V[index1]
```

这可能减少 V 流量，但不等价于 Rubin 的结构化 Sparse MMA。

而官方 `trtllm_mha` 路径的真正 FMHA 核心来自预编译 CUBIN，所以最终的真实性能实现需要新的 FlashInfer/TRT-LLM kernel，而不是只改 SGLang backend wrapper。

另外，FlashInfer 当前的 `skip_softmax_threshold_scale_factor` 是按 tile 的 local/global max 阈值跳过 softmax/BMM2，不是每四个元素固定保留两个，不能直接用它替代 2:4。

---

# 10. 如何解释最终吞吐结果

Reference PyTorch 版本必然明显变慢，因此它只用于质量分析。

Triton 数值模拟版本可能出现：

```text
dense Triton > 2:4-mask Triton
```

这不表示 Rubin 2:4 无效，只表示你加入了 top-2 开销，却仍然执行 dense PV。

真正的理论 attention 收益可用：

```text
QK 成本 = C
PV 成本 = C
```

若 Rubin 只把 PV 减半：

```text
dense:  C + C     = 2C
2:4:    C + 0.5C  = 1.5C
```

两次主要矩阵运算的理想加速约为：

[
\frac{2C}{1.5C}=1.33\times
]

端到端速度则由 Amdahl 定律决定：

[
S_{\text{e2e}}
==============

\frac{1}{
(1-f_{\text{attn}})
+
f_{\text{attn}}/S_{\text{attn}}
}
]

其中 (f_{\text{attn}}) 是原始运行时间中 full attention 所占比例。

虽然主干只有 12/108 个 full-attention block，但不能简单认为 attention 只占 1/9 时间：

* 短上下文 decode 中，Mamba、MoE、权重与 KV 访问可能占主要时间；
* 长上下文 prefill 中，12 个 (O(L^2)) full-attention block 仍可能成为主要瓶颈。

最终应分别测：

```text
4K / 16K / 64K prefill throughput
batch 1 / 8 / 32 decode throughput
TTFT
inter-token latency
attention kernel GPU time
V-cache read bytes
MTP acceptance rate
```

我没有在本环境加载这个 550B checkpoint 或运行 4×B200，因此这里不给出未经实测的精度或 TPS 数值。可执行路线应是：

```text
第一步：torch-native manual dense vs manual 2:4，确定质量损失
第二步：逐 attention layer 做敏感度扫描
第三步：在 Triton prefill/decode kernel 中实现同一数值语义
第四步：将 dense PV 改成只读取两个 V rows 的 gather/FMA 原型
第五步：最后实现 FlashInfer/TRT-LLM 的 metadata + sparse PV kernel
```

[1]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/blob/main/README.md?utm_source=chatgpt.com "README.md · nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 at main"
[2]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/blob/main/config.json "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/blob/main/config.json"
[3]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/blob/main/README.md "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/blob/main/README.md"
