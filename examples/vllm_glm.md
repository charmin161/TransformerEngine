## 结论

为了在 **GLM-5.2-NVFP4 的 DSA 精确注意力分数上实现 4 选 2，再把另外两个 score 置为 `-inf`**，建议使用：

```text
FLASHMLA_SPARSE
```

不要使用自动选择出来的 `FLASHINFER_MLA_SPARSE`，也不要使用 `TRITON_MLA`。

推荐实验配置是：

```text
vLLM v0.26.0
B200 / SM100
TP=8
KV cache = fp8_ds_mla
attention backend = FLASHMLA_SPARSE
sparse_mla_force_mqa = true
```

原因很明确：

* 在 B200、DSA、FP8 KV cache 配置下，vLLM 自动选择时，会优先选 `FLASHINFER_MLA_SPARSE`；但它把 `QK → softmax → PV` 封装在 FlashInfer/TRT-LLM 的外部融合内核中，vLLM 源码里看不到可直接修改的 score。
* `FLASHMLA_SPARSE` 虽然同样是融合内核，但 vLLM 会编译开源的 `vllm-project/FlashMLA` 源码。它在 CUDA 内核中明确把 QK 结果读入 `float p[]`，先做合法位置 mask，然后求 row max、softmax，正好存在你需要的插入点。
* `TRITON_MLA` 是通用的 dense MLA backend，不是 GLM-5.2 所需的 DSA sparse MLA 路径；强行换过去会改变甚至绕开原本的 DSA top-k 注意力语义，不适合这个实验。

---

## 1. 先明确你实际仿真的 attention 流程

NVIDIA Rubin 博客描述的流程是：

```text
dense QKᵀ
  ↓
将 score 压缩成结构化 2:4
  ↓
sparse softmax
  ↓
稀疏 P × dense V
```

也就是压缩发生在 **QK 分数产生之后、softmax 之前**。博客并没有公开：

* 每 4 个元素具体按照什么物理布局分组；
* 保留 top-2 raw logits，还是 top-2 absolute values；
* 是否存在更复杂的训练后选择规则。

因此，你的仿真应明确命名为：

> contiguous-score top-2-of-4 by raw logit

对于 softmax，保留每组中最大的两个原始 logit 比按绝对值选择更合理。按绝对值可能会保留一个非常大的负值，而它在 softmax 中本来几乎没有贡献。([NVIDIA Developer][1])

GLM-5.2-NVFP4 使用的是 `GlmMoeDsaForCausalLM`，其 DSA IndexShare 先选择最多 2048 个候选 token，然后精确注意力才在这些候选 token 上计算 QK。模型配置中还包含 `index_topk=2048`、`index_topk_freq=4` 和 64 个 attention heads。

所以你的实验实际上是：

```text
完整历史 token
    ↓ IndexShare
每个 query 选择 top-2048 候选 token
    ↓ exact MLA QK score
对候选列表中的 score 每连续 4 个保留 2 个
    ↓
约剩余 1024 个候选 / head
    ↓
softmax → PV
```

这不是对完整上下文 dense QK 的 Rubin 2:4，而是：

> **DSA top-2048 之后的二级 score sparsification**

这个区别在论文或测试报告中必须写清楚。

---

## 2. vLLM 版本应该怎么选

截至 **2026 年 8 月 4 日**，vLLM 最新稳定版是 **v0.26.0，发布于 2026 年 7 月 27 日**；但 NVIDIA 的 GLM-5.2-NVFP4 模型卡仍然使用 `vllm/vllm-openai:v0.23.0` 作为推荐环境。([GitHub][2])

建议建立三组环境或构建：

| 用途             |    vLLM | Backend                          | 是否修改 |
| -------------- | ------: | -------------------------------- | ---- |
| NVIDIA 模型卡复现基线 | v0.23.0 | 自动选择，通常为 `FLASHINFER_MLA_SPARSE` | 否    |
| 后端替换基线         | v0.26.0 | `FLASHMLA_SPARSE`                | 否    |
| 2:4 实验组        | v0.26.0 | `FLASHMLA_SPARSE`                | 是    |

最终用第 2 组和第 3 组对比，才能把差异归因到 2:4，而不是 FlashInfer 和 FlashMLA 之间的数值差异。

---

## 3. 为什么最新 vLLM 还要加 `sparse_mla_force_mqa=true`

vLLM v0.26.0 对 sparse MLA 做了一项调度优化：

* decode 默认使用 sparse `forward_mqa`；
* prefill 如果上下文长度不超过 `index_topk`，可能改走 dense `forward_mha`；
* 只有设置 `sparse_mla_force_mqa=true`，才能保证 prefill 也进入 sparse MLA backend。

如果不设置它，你可能出现：

```text
decode：2:4 已生效
prefill：2:4 没有生效
```

然后下游任务结果会很难解释。

建议启动参数：

```bash
vllm serve /path/to/GLM-5.2-NVFP4 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --trust-remote-code \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --kv-cache-dtype fp8_ds_mla \
    --block-size 64 \
    --attention-config '{"backend":"FLASHMLA_SPARSE","sparse_mla_force_mqa":true}' \
    --enforce-eager
```

这里有三点需要注意：

1. 模型卡使用的是 `fp8_e4m3`，而 `FLASHMLA_SPARSE` 的 FP8 路径使用专门的 `fp8_ds_mla` KV cache 布局。为了避免自动转换和后端选择歧义，实验时建议显式写 `fp8_ds_mla`。
2. `--enforce-eager` 只建议在第一次调试时使用。确认内核确实命中后，正式测速可以移除。
3. 启动日志必须出现类似：

```text
Using FLASHMLA_SPARSE backend
```

否则不要开始跑下游任务。

---

## 4. GLM-5.2 在 vLLM 中的调用路径

在 vLLM v0.26.0 中，模型注册关系仍然是：

```text
GlmMoeDsaForCausalLM
    ↓
vllm/model_executor/models/deepseek_v2.py
    ↓
MLAAttention.forward_impl
    ↓
FlashMLASparseImpl.forward_mqa
    ↓
_forward_fp8_kv_mixed_batch
    ↓
_fp8_flash_mla_kernel
    ↓
flash_mla_with_kvcache
    ↓
FlashMLA CUDA kernel
```

`GlmMoeDsaForCausalLM` 仍映射到 `deepseek_v2` 模型实现。

主要的 vLLM Python 文件是：

```text
vllm/model_executor/models/deepseek_v2.py

vllm/model_executor/layers/attention/mla_attention.py

vllm/v1/attention/backends/mla/flashmla_sparse.py

vllm/v1/attention/ops/flashmla.py
```

其中 `FlashMLASparseImpl.forward_mqa()` 最终调用：

```python
flash_mla_with_kvcache(...)
```

真正的 score 不会作为 PyTorch Tensor 返回，而是在编译后的 FlashMLA CUDA 内核中产生。

---

## 5. 真正应该修改的 CUDA 文件

vLLM v0.26.0 固定使用的 FlashMLA commit 是：

```text
a8f794d1251cbfd88a5011445dd5582289c727e4
```

vLLM 的 CMake 支持通过环境变量 `FLASH_MLA_SRC_DIR` 指向你修改后的本地 FlashMLA 源码。

在 B200、FP8 KV、TP=8 的推荐配置下，主要修改：

```text
FlashMLA/
└── csrc/
    └── sm100/
        └── decode/
            └── head64/
                └── kernel.cuh
```

精确插入点在：

```cpp
// Mask
uint32_t valid_mask = ...;

for (...) {
    if (!(valid_mask >> i & 1))
        p[i] = -CUDART_INF_F;
}

// 在这里插入 2:4

// Get rowwise max of Pi
float cur_pi_max = -CUDART_INF_F;
```

当前代码中，QK 结果已经被加载并归约到：

```cpp
float p[B_TOPK / 2];
```

现有的 `valid_mask` 会先把越界、无效候选位置设为 `-inf`；随后代码立刻对 `p` 求 row max 并进入在线 softmax。因此这里正是：

```text
QK 完成
→ causal/valid mask 完成
→ 你的 2:4 mask
→ row max
→ exp
→ softmax
→ PV
```

对应源码结构可直接在 SM100 decode kernel 中看到。

---

## 6. 建议插入的 4 选 2 代码

先在文件顶部附近增加一个实验开关：

```cpp
#ifndef RUBIN_SCORE_2TO4
#define RUBIN_SCORE_2TO4 1
#endif
```

然后在现有 `// Mask` 之后、`// Get rowwise max of Pi` 之前插入：

```cpp
#if RUBIN_SCORE_2TO4

// p[] represents one half of a 64-candidate QK score tile.
// Keep exactly the two largest raw logits in every contiguous group of 4.
// Ties are resolved deterministically in favor of the lower candidate index.

CUTE_UNROLL
for (int base = 0; base < B_TOPK / 2; base += 4) {
    float x[4] = {
        p[base + 0],
        p[base + 1],
        p[base + 2],
        p[base + 3],
    };

    int rank[4] = {0, 0, 0, 0};

    // rank[i] = number of elements that should be ordered ahead of x[i].
    // The index comparison gives deterministic tie breaking.
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j) {
            const bool j_is_better =
                (x[j] > x[i]) ||
                ((x[j] == x[i]) && (j < i));

            rank[i] += static_cast<int>(j_is_better);
        }
    }

    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        p[base + i] =
            rank[i] < 2 ? x[i] : -CUDART_INF_F;
    }
}

#endif  // RUBIN_SCORE_2TO4
```

这个版本的行为是：

```text
[4.1, 2.3, -1.0, 3.2]
          ↓
[4.1, -inf, -inf, 3.2]
```

若一组中部分位置已经被 causal/valid mask 设为 `-inf`，它们自然不会覆盖有效的较大分数。

因为 attention scale 是正数：

```text
top2(p) == top2(p * positive_scale)
```

所以在乘 softmax scale 之前做选择不会改变 top-2 结果。

### 为什么不是按绝对值

不要写：

```cpp
fabsf(x[j]) > fabsf(x[i])
```

例如：

```text
[8.0, 7.0, -100.0, -90.0]
```

按绝对值会保留 `-100` 和 `-90`，而 softmax 真正重要的是 `8` 和 `7`。因此应该比较原始 logit。

---

## 7. 为什么 TP=8 时通常只改这个 decode kernel 就够了

GLM-5.2 有 64 个 attention heads。TP=8 后每张 GPU 有：

```text
64 / 8 = 8 heads
```

FlashMLA 中：

```python
MIN_HEADS_FOR_BF16_PREFILL = 32
```

每卡 8 heads 小于 32，因此 FP8 sparse MLA 会采用 mixed-batch 路径，将需要走 MQA 的 prefill 和 decode 一起送入 FP8 sparse decode kernel。

在推荐配置：

```text
TP=8
fp8_ds_mla
sparse_mla_force_mqa=true
```

下，主要命中的就是：

```text
csrc/sm100/decode/head64/kernel.cuh
```

### 哪些情况下还要修改 prefill kernel

如果你使用以下配置之一：

* TP=2，此时每卡 32 heads；
* BF16 KV cache；
* 没有设置 `sparse_mla_force_mqa=true`；
* vLLM 调度使 prefill 单独走 BF16 sparse prefill；

那么还需要同步修改：

```text
csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh
```

插入位置为：

```cpp
retrieve_mask_and_reduce_p(..., p);
plan.bar_k_valid_free[...].arrive();

// 在这里插入同样的 2:4

float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
```

这同样是合法 mask 完成之后、row max 与 softmax 之前。

---

## 8. 本地修改和编译方法

```bash
git clone --branch v0.26.0 https://github.com/vllm-project/vllm.git
git clone https://github.com/vllm-project/FlashMLA.git

git -C FlashMLA checkout a8f794d1251cbfd88a5011445dd5582289c727e4

export FLASH_MLA_SRC_DIR="$(realpath FlashMLA)"

cd vllm

python -m pip install -U pip setuptools wheel ninja cmake
python -m pip install -v --no-build-isolation -e .
```

修改 FlashMLA CUDA 文件后，必须重新执行最后一条安装命令，重新生成：

```text
vllm._flashmla_C
```

只修改 Python 文件不会让 CUDA 内核更新。

建议分别建立两个 FlashMLA 分支：

```text
baseline-flashmla
rubin-score-2to4
```

或者分别构建两个环境。不要在正式评测时反复手工修改同一个环境，否则很容易混淆当前载入的是哪一个 `.so`。

---

## 9. 第一次验证时应该检查什么

### 后端确认

在：

```text
vllm/v1/attention/backends/mla/flashmla_sparse.py
```

的 `FlashMLASparseImpl.forward_mqa()` 中临时加入：

```python
logger.warning_once(
    "RUBIN DEBUG: entered FlashMLASparseImpl.forward_mqa, "
    "kv_cache_dtype=%s, num_heads=%d",
    self.kv_cache_dtype,
    self.num_heads,
)
```

预期看到：

```text
kv_cache_dtype=fp8_ds_mla
num_heads=8
```

### 路径确认

可以在 `_forward_fp8_kv_mixed_batch()` 中临时加入：

```python
logger.warning_once(
    "RUBIN DEBUG: entered FlashMLA FP8 mixed-batch path"
)
```

只有确认命中了：

```text
forward_mqa
→ _forward_fp8_kv_mixed_batch
→ _fp8_flash_mla_kernel
```

再跑下游任务。

### 结果确认

第一轮不要直接跑采样任务，先做固定输入、贪心解码：

```text
temperature = 0
seed 固定
相同 TP / EP
相同 KV cache dtype
相同 max_model_len
相同 prompts
```

比较三组：

```text
A. 原始 FLASHINFER_MLA_SPARSE
B. 未修改 FLASHMLA_SPARSE
C. 修改后 FLASHMLA_SPARSE
```

重点看：

```text
A vs B：后端替换误差
B vs C：真正的 2:4 影响
```

---

## 10. 下游测评建议

NVIDIA 模型卡列出了 GPQA Diamond、SciCode、IFBench、AA-LCR 和 τ²-Bench Telecom 等结果，并给出了 `temperature=1.0`、`top_p=0.95`，以及不同任务的输出长度设置。([Hugging Face][3])

建议分两阶段：

### 第一阶段：确定性影响

使用：

```text
temperature=0
固定 seed
固定 prompt
```

测量：

* teacher-forced NLL / perplexity；
* token-level KL divergence；
* baseline 与 2:4 的首个分叉 token 位置；
* 各层 attention output 的 cosine similarity 和相对 MSE；
* 每层、每 head 被设为 `-inf` 的有效 score 比例。

### 第二阶段：真实下游任务

至少覆盖：

* GPQA Diamond：高难推理；
* LiveCodeBench 或 SciCode：代码生成；
* IFBench：指令遵循；
* 长上下文检索任务；
* Agent/tool-use 任务。

每个任务同时报告：

```text
绝对分数
相对下降
多次运行均值
置信区间
平均输出长度
拒答/截断比例
```

采样任务不能只跑一次，因为 2:4 引起的微小 logit 变化可能通过自回归过程被放大。

---

## 最需要避免的两个错误

第一，不要修改：

```text
Indexer 的 topk score
topk_indices_buffer
sparse_attn_indexer
```

那是在改变 DSA 的候选 token 选择，而不是 Rubin 所说的精确 attention score compression。

第二，不要在 softmax 之后把概率简单清零却不重新归一化。正确顺序必须是：

```text
QK
→ valid/causal mask
→ 2:4 设置 -inf
→ softmax
→ PV
```

按照这一配置，最稳妥的第一轮实验就是：

```text
vLLM v0.26.0
+ TP=8
+ fp8_ds_mla
+ FLASHMLA_SPARSE
+ sparse_mla_force_mqa=true
+ 修改 csrc/sm100/decode/head64/kernel.cuh
```

[1]: https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/ "Inside NVIDIA Rubin GPU Architecture: Powering the Era of Agentic AI | NVIDIA Technical Blog"
[2]: https://github.com/vllm-project/vllm/releases "Releases · vllm-project/vllm · GitHub"
[3]: https://huggingface.co/nvidia/GLM-5.2-NVFP4 "nvidia/GLM-5.2-NVFP4 · Hugging Face"
