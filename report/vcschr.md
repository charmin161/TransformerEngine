是的，但**不是你的 `FLASHINFER_MLA_SPARSE` 配错了**。从你这份日志看，真正的问题是：

> **你现在运行的 GLM-5.2 并没有实例化我们前面讨论的 `vllm/models/deepseek_v32/nvidia/DeepseekV32Attention`，而是走了 vLLM 0.26.0 默认的通用 `DeepseekV2MLAAttention → MultiHeadLatentAttentionWrapper → MLAAttention` 路径。**

所以你在：

```text
vllm/models/deepseek_v32/nvidia/attention.py
vllm/models/deepseek_v32/nvidia/kernels.py
```

里加 `print`，**当然不会打印**。

而且你的日志已经把实际调用链完整暴露出来了。

---

## 1. 你的 Attention backend 其实选对了

日志明确有：

```text
Using FLASHINFER_MLA_SPARSE attention backend
```

并且 KV cache 是：

```text
kv_cache_dtype=fp8_e4m3
```

同时 Prefill backend 是：

```text
Using TRTLLM_RAGGED MLA prefill backend.
```

所以你的 attention backend 本身没有问题。

你当前实际上是：

```text
GLM-5.2
   │
   ▼
model_executor/models/deepseek_v2.py
   │
   ▼
DeepseekV2MLAAttention
   │
   ▼
model_executor/layers/mla.py
   │
   ▼
MultiHeadLatentAttentionWrapper
   │
   ▼
MLAAttention
   │
   ├── Prefill → TRTLLM_RAGGED / forward_mha
   │
   └── Decode  → FLASHINFER_MLA_SPARSE / forward_mqa
```

你的 torch.compile traceback 非常明确：

```text
deepseek_v2.py:1174
    return self.mla_attn(...)

mla.py:170
    self.indexer(...)
```

也就是说根本没有经过：

```text
vllm/models/deepseek_v32/nvidia/model.py
```

。

---

# 2. 为什么会这样？

因为 vLLM 0.26.0 里实际上同时存在两套可用于 GLM-5.2 / DSA 的外层模型实现。

### 你现在默认走的

```text
vllm/model_executor/models/deepseek_v2.py
```

里面：

```python
DeepseekV2MLAAttention
    ↓
MultiHeadLatentAttentionWrapper
```

在 v0.26.0 源码里，`DeepseekV2MLAAttention` 最终就是构造：

```python
self.mla_attn = MultiHeadLatentAttentionWrapper(...)
```

。

而 `MultiHeadLatentAttentionWrapper.forward()` 明确是：

```python
qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]

q_c, kv_lora = qkv_lora.split(...)

q_c = self.q_a_layernorm(q_c)
q = self.q_b_proj(q_c)[0]

kv_c, k_pe = kv_lora.split(...)

kv_c_normed = self.kv_a_layernorm(kv_c)

...

q_pe, k_pe = self.rotary_emb(
    positions,
    q_pe,
    k_pe,
)

...

attn_out = self.mla_attn(
    q,
    kv_c_normed,
    k_pe,
    ...
)
```

。

这和你日志中的 traceback 完全吻合。

---

# 3. 我们之前讨论的是另外一套 NVIDIA 专用实现

我们前面一直分析的是：

```text
vllm/models/deepseek_v32/nvidia/model.py
```

它自己构造：

```python
self.self_attn = DeepseekV32Attention(...)
```

源码非常明确。

最终：

```text
DeepseekV32Attention
   ↓
_fused_attention()
   ↓
fused_norm_rope()
   ↓
_fused_norm_rope_kernel()
```

而这个 NVIDIA 专用模型类：

```python
class DeepseekV32ForCausalLM(...)
```

源码注释直接写着：

> serves DeepSeek V3.2 and architectures reusing DSA, e.g. GLM-5.2.



所以前面的分析本身对应的是这套实现，但**你的实际启动配置并没有选择它**。

这一点我需要把前面的结论修正得更准确：

> 我前面默认你已经在跑 NVIDIA `DeepseekV32Attention` 专用路径；从这次实际日志看，你跑的是 vLLM 0.26.0 默认 generic MLA 路径。

---

# 4. 如果你就是想走我们前面分析的 `_fused_norm_rope_kernel()`，需要加一个关键参数

vLLM 0.26.0 官方甚至专门提供了这个参数：

```bash
--model-class-overrides \
'{"GlmMoeDsaForCausalLM":"vllm.models.deepseek_v32.nvidia.model:DeepseekV32ForCausalLM"}'
```

而且非常巧的是，vLLM 0.26.0 官方 CLI 文档给出的 `--model-class-overrides` 示例，**就是 GLM-5.2 这个映射**。([vLLM][1])

所以如果你希望继续按照我们之前说的：

```text
DeepseekV32Attention
→ _fused_attention
→ fused_norm_rope
→ _fused_norm_rope_kernel
```

来改，我建议你的启动命令改成：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /wireless/public/models/GLM-5.2-NVFP4 \
    --tensor-parallel-size 4 \
    --quantization modelopt \
    --trust-remote-code \
    --disable-log-stats \
    --port 8972 \
    --host 127.0.0.1 \
    --enable-expert-parallel \
    --reasoning-parser glm45 \
    --served-model-name GLM5.2-NVFP4 \
    --max-model-len 819200 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend FLASHINFER_MLA_SPARSE \
    --model-class-overrides \
    '{"GlmMoeDsaForCausalLM":"vllm.models.deepseek_v32.nvidia.model:DeepseekV32ForCausalLM"}'
```

其中真正决定你是否进入：

```text
vllm/models/deepseek_v32/nvidia/
```

的不是：

```bash
--attention-backend FLASHINFER_MLA_SPARSE
```

而是：

```bash
--model-class-overrides ...
```

---

# 5. `--attention-backend` 和 `--model-class-overrides` 是两个完全不同的层次

这个特别容易混淆。

你当前日志：

```text
Using FLASHINFER_MLA_SPARSE attention backend
```

**只能证明内层 attention kernel 是 FlashInfer Sparse MLA。**

它不能说明外层模型一定是：

```text
DeepseekV32Attention
```

可以理解成：

```text
模型外层 implementation
│
├── Generic:
│   DeepseekV2MLAAttention
│   MultiHeadLatentAttentionWrapper
│
└── NVIDIA specialized:
    DeepseekV32Attention
          │
          ▼
    内层 attention backend
          │
          ├── FLASHINFER_MLA_SPARSE
          └── FLASHMLA_SPARSE
```

你现在是：

```text
Generic outer MLA
        +
FLASHINFER_MLA_SPARSE backend
```

而我们之前讨论的是：

```text
NVIDIA DeepseekV32 outer MLA
        +
FLASHINFER_MLA_SPARSE backend
```

所以：

```bash
--attention-backend FLASHINFER_MLA_SPARSE
```

单独加上去**也不会让你的 `_fused_norm_rope_kernel()` 开始执行**。

---

# 6. 为了 Debug，我还建议暂时加 `--enforce-eager`

你的日志另外还有：

```text
torch.compile
CUDAGraph
FULL_AND_PIECEWISE
```

以及大量：

```text
Capturing CUDA graphs
```

。

所以即使进入正确模型路径，普通 Python `print()` 在 compile / graph capture 环境下也很容易让判断变得混乱：

* 有的只在 compile 阶段打一次；
* CUDA Graph replay 不会重新执行 Python；
* Triton kernel 内的打印又是另外一种行为。

因此第一轮**纯粹确认调用链**的时候，我强烈建议额外：

```bash
--enforce-eager
```

vLLM 官方说明这个参数会同时关闭 torch.compile 和 CUDA Graph。([vLLM][2])

也就是说 Debug 版启动：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /wireless/public/models/GLM-5.2-NVFP4 \
    --tensor-parallel-size 4 \
    --quantization modelopt \
    --trust-remote-code \
    --enable-expert-parallel \
    --reasoning-parser glm45 \
    --served-model-name GLM5.2-NVFP4 \
    --max-model-len 819200 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend FLASHINFER_MLA_SPARSE \
    --model-class-overrides \
    '{"GlmMoeDsaForCausalLM":"vllm.models.deepseek_v32.nvidia.model:DeepseekV32ForCausalLM"}' \
    --enforce-eager \
    --port 8972 \
    --host 127.0.0.1
```

等你确认：

```text
DeepseekV32Attention.forward
_fused_attention
fused_norm_rope
```

全部走到了，再去掉：

```bash
--enforce-eager
```

做正式性能/精度实验。

---

# 7. 最简单的验证方法

不要第一步就在 Triton kernel 里面 debug。

先在：

```text
vllm/models/deepseek_v32/nvidia/model.py
```

的：

```python
class DeepseekV32DecoderLayer
```

初始化里临时加：

```python
print(
    "[DEBUG] USING NVIDIA DEEPSEEK V32 MODEL",
    type(self.self_attn),
    flush=True,
)
```

或者在：

```text
vllm/models/deepseek_v32/nvidia/attention.py
```

`DeepseekV32Attention.__init__()`：

```python
print(
    "[DEBUG] DeepseekV32Attention CREATED",
    prefix,
    flush=True,
)
```

启动以后应该看到类似：

```text
[DEBUG] DeepseekV32Attention CREATED model.layers.0.self_attn
[DEBUG] DeepseekV32Attention CREATED model.layers.1.self_attn
...
```

只要这一行出来了，就证明：

```text
--model-class-overrides
```

生效了。

然后再在：

```python
def _fused_attention(...):
    print("[DEBUG] ENTER _fused_attention", flush=True)
```

确认 runtime。

---

# 8. 但是还有第二种方案：其实你完全可以不切 NVIDIA model class

这一点对你现在做 NVFP4 fake quant 反而很重要。

如果你的目的只是：

> 对 `kv_c_normed + k_pe_roped` 做 NVFP4 QDQ，并让 Prefill 和 Decode 都使用它。

那么你当前 **generic MLA 路径其实更加容易改**。

因为现在的：

```text
vllm/model_executor/layers/mla.py
```

里面已经非常干净地暴露了：

```python
kv_c, k_pe = kv_lora.split(
    [self.kv_lora_rank, self.qk_rope_head_dim],
    dim=-1,
)

kv_c_normed = self.kv_a_layernorm(kv_c)

...

q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
    positions,
    q[..., self.qk_nope_head_dim:],
    k_pe,
)
```

然后马上：

```python
attn_out = self.mla_attn(
    q,
    kv_c_normed,
    k_pe,
    ...
)
```

。

所以如果你的 fake quant 是普通 PyTorch 函数，你甚至可以直接：

```python
kv_c_normed = self.kv_a_layernorm(kv_c)

...

q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
    positions,
    q[..., self.qk_nope_head_dim:],
    k_pe,
)


# ======================================
# NVFP4 fake quant
# ======================================

kv_c_normed = nvfp4_fake_quant(kv_c_normed)
k_pe = nvfp4_fake_quant(k_pe)

# ======================================


attn_out = self.mla_attn(
    q,
    kv_c_normed,
    k_pe,
    ...
)
```

这里甚至比改 `_fused_norm_rope_kernel()` 更方便。

---

# 9. 而且这个位置确实同时影响 Cache 和 Prefill

现在的通用 `MLAAttention.forward()` 接收到：

```python
q
kv_c_normed
k_pe
```

之后，首先就把：

```python
kv_c_normed
k_pe
```

送入：

```python
self.impl.do_kv_cache_update(...)
```

写入真正 KV cache：

```python
self.impl.do_kv_cache_update(
    kv_for_cache,
    kpe_for_cache,
    self_kv_cache,
    ...
)
```

。

然后同一次 forward 又继续把同一份：

```python
kv_c_normed
k_pe
```

交给 `forward_impl()`：

```python
self.forward_impl(
    q,
    kv_c_normed,
    k_pe,
    self_kv_cache,
    ...
)
```

。

也就是说，如果你在 `mla.py` 这里 fake quant：

```text
kv_c
 ↓
RMSNorm
 ↓
kv_c_normed ──┐
              │
k_pe           │
 ↓ RoPE        │
k_pe_roped ────┤
               │
       【NVFP4 QDQ】
               │
        ┌──────┴──────┐
        ▼             ▼
KV cache write    当前 Prefill
        │
        ▼
      Decode
```

**Prefill + Decode 两边都覆盖。**

而且这种方式还没有 Triton `tl.tensor` 的限制，你现成的 PyTorch fake-quant 代码可以直接用。

---

# 10. 你当前日志还回答了我们上一个问题

你之前问：

> Prefill 是不是也走那个 MQA latent path？

你现在这份真实日志说明，**当前默认 generic implementation 下，并不是简单地全部走同一条 MQA。**

日志明确显示：

```text
Using TRTLLM_RAGGED MLA prefill backend
```

。

而通用 `MLAAttention.forward_impl()` 的源码确实会拆：

```python
num_mqa_tokens = attn_metadata.num_decode_tokens
num_mha_tokens = q.size(0) - num_mqa_tokens
```

Prefill 可以：

```python
self.impl.forward_mha(...)
```

Decode：

```python
self.impl.forward_mqa(...)
```

。

这正好与我们前面讲过的数学结构一致：

```text
Prefill:
kv_c × W_UK
kv_c × W_UV
→ compute-friendly MHA

Decode:
Q × W_UK^T
attention(... V=kv_c)
→ × W_UV
```

所以这里我也纠正前面针对你实际运行环境的判断：

> **你的当前默认 vLLM 0.26.0 GLM-5.2 确实存在 Prefill/Decode 两种 execution form。**

---

## 我现在更推荐你怎么做

如果你的目标只是研究 **`[kv_c_normed, k_pe_roped]` 做 NVFP4 QDQ 对模型精度的影响**，我现在看到真实调用链以后，反而建议你：

**先不要强制切换到 NVIDIA `DeepseekV32Attention`。**

直接改：

```text
vllm/model_executor/layers/mla.py
```

这个位置：

```python
kv_c_normed = self.kv_a_layernorm(kv_c)

...

q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(...)


# ← 在这里 NVFP4 fake quant
kv_c_normed = your_nvfp4_qdq(kv_c_normed)
k_pe = your_nvfp4_qdq(k_pe)


attn_out = self.mla_attn(
    q,
    kv_c_normed,
    k_pe,
    ...
)
```

这是你当前实际配置下**最直接、最不容易漏 Prefill/Decode、也最方便使用现有 PyTorch fake-quant 代码的位置**。

如果你就是希望研究 NVIDIA 那套高度 fused 的 `DeepseekV32Attention` 实现，则加入：

```bash
--model-class-overrides \
'{"GlmMoeDsaForCausalLM":"vllm.models.deepseek_v32.nvidia.model:DeepseekV32ForCausalLM"}'
```

再继续改 `_fused_norm_rope_kernel()`。

**这两条路线不要混着改。**你这次没有看到 print 的原因，现在已经可以确定就是模型 class 没走到我们原来分析的那条路径，而不是 `FLASHINFER_MLA_SPARSE` 没生效。

[1]: https://docs.vllm.ai/en/v0.26.0/cli/serve/?utm_source=chatgpt.com "vllm serve - vLLM"
[2]: https://docs.vllm.ai/en/v0.26.0/api/vllm/config/vllm/?utm_source=chatgpt.com "vllm - vLLM"
