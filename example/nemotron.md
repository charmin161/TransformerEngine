下面是可直接落到代码的梳理。结论先说：**Nemotron-3-Ultra-550B-A55B-NVFP4 不是纯 Transformer MoE，而是 Nemotron-H / Hybrid LatentMoE：Mamba-2 + MoE + 少量 Attention + MTP 的混合结构**。官方模型卡写明总参数约 550B、激活参数约 55B，结构是 interleaved Mamba-2、MoE、select Attention layers，并且使用 LatentMoE 与 MTP；NVFP4 推理推荐 SGLang 或 vLLM 部署。([Hugging Face][1])

## 1. 整体网络结构：不要按纯 Transformer 层理解

在 SGLang / vLLM 里，这个模型的核心结构都是：

```text
NemotronHForCausalLM
 └── NemotronHModel
      ├── embed_tokens
      ├── layers[i]  # 类型由 config.hybrid_override_pattern[i] 决定
      │    ├── Attention layer
      │    ├── MoE layer
      │    ├── Mamba layer
      │    └── Dense MLP layer
      └── final_layernorm
 └── lm_head
```

关键点是：**每一层的类型不是固定 TransformerBlock，而是由 `config.hybrid_override_pattern` 决定**。SGLang 代码中用 layer pattern 选择 decoder layer 类型，模型 forward 里按层循环调用 `layer.forward(...)`；vLLM 代码也用类似机制，通过 `hybrid_override_pattern` 构造各层。([GitHub][2])

SGLang 的层类型映射大致是：

```python
ALL_DECODER_LAYER_TYPES = {
    "ATTENTION": NemotronHAttentionDecoderLayer,
    "MLP": NemotronHMLPDecoderLayer,
    "MAMBA": NemotronHMambaDecoderLayer,
    "MOE": NemotronHMoEDecoderLayer,
}
```

vLLM 侧对应的是：

```python
ALL_DECODER_LAYER_TYPES = {
    "M": NemotronHMambaDecoderLayer,
    "-": NemotronHMLPDecoderLayer,
    "*": NemotronHAttentionDecoderLayer,
    "E": NemotronHMoEDecoderLayer,
}
```

所以你后续抓激活前，第一步不是假设所有层都有 attention / MLP，而是先在本地 `config.json` 里打印：

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    trust_remote_code=True,
)
from collections import Counter
print(len(cfg.hybrid_override_pattern))
print(Counter(cfg.hybrid_override_pattern))
for i, t in enumerate(cfg.hybrid_override_pattern):
    print(i, t)
```

模型卡上的 `config.json` 很大，HF 页面只显示它是一个约 6.94MB 的文件；官方还合并过一个 “minify config.json” 的讨论/PR，所以**精确层数、Attention/MoE/Mamba 分布，建议以本地下载后的 config 为准**。([Hugging Face][3])

---

## 2. Attention：QKV proj 和 output proj 的具体落点

SGLang 里 attention 的核心类是：

```text
python/sglang/srt/models/nemotron_h.py
NemotronHAttention
```

关键 forward 路径是：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
attn_output = self.attn.forward(q, k, v, forward_batch)
output, _ = self.o_proj(attn_output, ...)
```

其中：

```python
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
self.attn = RadixAttention(...)
```

这正是你要抓的 attention QKV 和 output projection。SGLang 的 DP attention 分支也仍然会调用 `qkv_proj`、split Q/K/V、`RadixAttention.forward`、`o_proj`，只是中间有 padding / gather 处理。([GitHub][2])

vLLM 侧对应类也是：

```text
vllm/model_executor/models/nemotron_h.py
NemotronHAttention
```

核心路径同样是：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split(...)
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
```

vLLM 里 `qkv_proj` 是 `QKVParallelLinear`，`o_proj` 是 `RowParallelLinear`。([GitHub][4])

你要抓的 attention 激活建议分成四类：

```text
model.layers.{i}.mixer.qkv_proj input    # attention pre-proj hidden activation
model.layers.{i}.mixer.qkv_proj output   # packed QKV
split 后 q/k/v                            # Q/K/V activation
model.layers.{i}.mixer.o_proj input      # attention output before output projection
model.layers.{i}.mixer.o_proj output     # attention block projection output
```

注意：SGLang 使用 `RadixAttention`，所以如果你只 hook PyTorch module，能稳定抓到 `qkv_proj` 和 `o_proj`，但不一定能优雅抓到 attention kernel 内部的 softmax / score / context 中间值。

---

## 3. Dense MLP：不是 SwiGLU gate/up/down，而是 ReLU² 的 up/down

SGLang 的 dense MLP 是：

```text
NemotronHMLP
 ├── up_proj   = ColumnParallelLinear
 ├── act_fn    = ReLU2()
 └── down_proj = RowParallelLinear
```

forward 是：

```python
x, _ = self.up_proj(x)
x = self.act_fn(x)
x, _ = self.down_proj(x)
return x
```

vLLM 侧结构也类似，`up_proj = ColumnParallelLinear`，`down_proj = RowParallelLinear`，激活函数是 `ReLUSquaredActivation`。([GitHub][2])

所以这里要特别注意：**Nemotron-H 的 dense MLP 不是常见 LLaMA/SwiGLU 的 `gate_proj + up_proj + down_proj`。它只有 `up_proj` 和 `down_proj`，中间是 ReLU² 激活。**

对应抓取点：

```text
model.layers.{i}.mixer.up_proj input
model.layers.{i}.mixer.up_proj output
ReLU2 output
model.layers.{i}.mixer.down_proj input
model.layers.{i}.mixer.down_proj output
```

如果你看到某些 LoRA / loader mapping 里出现 `gate_up_proj`，不要直接等价为实际 forward 里有独立 `gate_proj`。对这个模型的 dense MLP forward 来说，主路径是 `up_proj -> ReLU2 -> down_proj`。

---

## 4. MoE / LatentMoE：这里最容易误判

SGLang 的 `NemotronHMoE` 结构大致是：

```text
NemotronHMoE
 ├── gate                # router, ReplicatedLinear
 ├── topk                # grouped top-k routing
 ├── fc1_latent_proj     # optional, hidden_size -> moe_hidden_size
 ├── experts             # fused MoE backend
 ├── fc2_latent_proj     # optional, moe_hidden_size -> hidden_size
 └── shared_experts      # optional dense shared experts
```

SGLang 代码中，router 是：

```python
self.gate = ReplicatedLinear(
    config.hidden_size,
    config.n_routed_experts,
    bias=False,
    quant_config=None,
    prefix=f"{prefix}.gate",
)
```

普通 forward 核心路径 `_forward_core_normal` 是：

```python
router_logits = torch.mm(hidden_states, self.gate.weight.t(), out_dtype=torch.float32)

shared_output = self.shared_experts(hidden_states)  # if exists

topk_output = self.topk(hidden_states, router_logits)

if self.use_latent_moe:
    hidden_states, _ = self.fc1_latent_proj(hidden_states)

final_hidden_states = self.experts(hidden_states, topk_output)

if self.use_latent_moe:
    final_hidden_states, _ = self.fc2_latent_proj(final_hidden_states)
```

这段是你抓专家 ID 和专家输入激活的核心位置。([GitHub][2])

vLLM 侧也是 `NemotronHMoE`，有 router `GateLinear`，可选 `fc1_latent_proj` / `fc2_latent_proj`，然后用 `FusedMoE(...)` 执行专家计算；并且 vLLM 明确设置了 `is_non_gated_moe=True`。([GitHub][4])

这里最重要的修正是：

### MoE 里的 `gate` 是 router，不是 SwiGLU 的 gate_proj

你原先按 Transformer MoE 理解的：

```text
gate_proj / up_proj / down_proj
```

在这个模型里要拆开看：

```text
router gate:        self.gate(hidden_states) 或 torch.mm(hidden_states, gate.weight.T)
expert up/down:     在 self.experts 这个 fused MoE backend 内部
latent projection:  fc1_latent_proj / fc2_latent_proj
shared expert:      shared_experts.up_proj / shared_experts.down_proj
```

也就是说，**你一定要记录 `router_logits`、`topk_output`、token index、expert id、expert weight、MoE layer id**。否则即使抓到了 `self.experts` 的输入激活，也没法把 token 激活对应到具体 expert weight。

建议你在 SGLang 里优先改这个函数：

```text
python/sglang/srt/models/nemotron_h.py
NemotronHMoE._forward_core_normal
```

直接在这里 dump：

```python
# pseudo patch
router_logits = torch.mm(hidden_states, self.gate.weight.t(), out_dtype=torch.float32)
topk_output = self.topk(hidden_states, router_logits)

dump("moe_hidden_pre_router", hidden_states)
dump("moe_router_logits", router_logits)
dump("moe_topk_output_type", type(topk_output))
dump("moe_topk_output", topk_output)

if self.use_latent_moe:
    hidden_states, _ = self.fc1_latent_proj(hidden_states)
    dump("moe_after_fc1_latent_proj", hidden_states)

final_hidden_states = self.experts(hidden_states, topk_output)
dump("moe_experts_output", final_hidden_states)

if self.use_latent_moe:
    final_hidden_states, _ = self.fc2_latent_proj(final_hidden_states)
    dump("moe_after_fc2_latent_proj", final_hidden_states)
```

第一次不要假设 `topk_output` 的结构，先打印：

```python
print(type(topk_output))
print(topk_output)
if isinstance(topk_output, tuple):
    for j, x in enumerate(topk_output):
        print(j, type(x), getattr(x, "shape", None), getattr(x, "dtype", None))
elif hasattr(topk_output, "__dict__"):
    print(topk_output.__dict__.keys())
```

因为 SGLang 的 TopK 返回对象可能随版本变化；但从调用关系看，**专家 ID / top-k 权重一定是在 `self.topk(hidden_states, router_logits)` 之后传给 `self.experts(...)` 的这个对象里**。([GitHub][2])

---

## 5. SGLang 推理时的主调用链

官方模型卡给的 SGLang 部署命令使用：

```bash
python3 -m sglang.launch_server \
  --model-path /model \
  --tp 4 \
  --mem-fraction-static 0.95 \
  --page-size 64 \
  --chunked-prefill-size 4096 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 50000
```

并且官方示例里设置了：

```bash
SGLANG_DISABLE_DEEP_GEMM=1
```

测试镜像是 `lmsysorg/sglang:v0.5.12.post1`。([Hugging Face][1])

落到模型 forward，核心调用链可以按下面理解：

```text
sglang.launch_server
  -> load model by architecture
    -> NemotronHForCausalLM.__init__
      -> _init_model()
        -> NemotronHModel(...)
          -> make_layers(...)
  -> request decoding / prefill
    -> NemotronHForCausalLM.forward(...)
      -> NemotronHModel.forward(...)
        -> for each layer:
             layer.forward(hidden_states, residual, forward_batch)
               ├── NemotronHAttention.forward
               ├── NemotronHMoE.forward
               ├── NemotronHMLP.forward
               └── NemotronHMamba.forward
```

SGLang 的 `NemotronHForCausalLM` 初始化里会创建 `NemotronHModel`，`NemotronHModel.forward` 会循环调用每一层；不同 layer 的实际类型由 `hybrid_override_pattern` 决定。([GitHub][2])

---

## 6. vLLM 侧对应位置

虽然你主要说 SGLang，但你也提到了 “vLLM 会调用的函数和代码”。vLLM 侧模型文件是：

```text
vllm/model_executor/models/nemotron_h.py
```

关键类对应如下：

| 目标                | vLLM 类 / 函数                                                  | 说明                                           |
| ----------------- | ------------------------------------------------------------ | -------------------------------------------- |
| 模型入口              | `NemotronHForCausalLM`                                       | 创建 `NemotronHModel`、lm_head、logits processor |
| 主干                | `NemotronHModel.forward`                                     | embedding 后逐层循环                              |
| 层选择               | `ALL_DECODER_LAYER_TYPES` + `config.hybrid_override_pattern` | 决定 Mamba / MLP / Attention / MoE             |
| QKV               | `NemotronHAttention.forward -> self.qkv_proj`                | `QKVParallelLinear`                          |
| O proj            | `NemotronHAttention.forward -> self.o_proj`                  | `RowParallelLinear`                          |
| Dense MLP up/down | `NemotronHMLP.forward`                                       | `up_proj -> ReLU² -> down_proj`              |
| MoE router        | `NemotronHMoE.forward -> self.gate`                          | `GateLinear`                                 |
| MoE experts       | `NemotronHMoE.forward -> self.experts`                       | `FusedMoE`                                   |
| LatentMoE         | `fc1_latent_proj` / `fc2_latent_proj`                        | routed expert 前后投影                           |

vLLM 还声明了 q/k/v 权重打包映射：

```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
}
```

SGLang 也有类似的 `packed_modules_mapping`。所以抓权重/激活时，注意 Q/K/V 在运行时是 packed 到 `qkv_proj` 里的。([GitHub][2])

---

## 7. 你真正应该打点的位置表

| 目标                       | SGLang 位置                                                       | 要抓什么                            | 注意事项                         |
| ------------------------ | --------------------------------------------------------------- | ------------------------------- | ---------------------------- |
| Attention QKV 输入         | `NemotronHAttention.forward` 的 `self.qkv_proj(hidden_states)` 前 | `hidden_states`                 | 这是 QKV 矩阵乘输入激活               |
| Attention QKV 输出         | `qkv`                                                           | packed QKV                      | 需要按 `q_size/kv_size` split   |
| Q/K/V                    | `q, k, v = qkv.split(...)` 后                                    | q/k/v                           | 用于分析 attention 输入            |
| Attention output proj 输入 | `self.o_proj(attn_output)` 前                                    | `attn_output`                   | 这是 O proj 矩阵乘输入              |
| Attention output proj 输出 | `output`                                                        | projection output               | attention 子层输出               |
| Dense MLP up_proj 输入     | `NemotronHMLP.forward` 的 `self.up_proj(x)` 前                    | x                               | 只在 MLP layer 有               |
| Dense MLP up_proj 输出     | `self.up_proj(x)` 后                                             | x                               | ReLU² 前                      |
| Dense MLP down_proj 输入   | `self.act_fn(x)` 后                                              | x                               | ReLU² 后                      |
| Dense MLP down_proj 输出   | `self.down_proj(x)` 后                                           | x                               | MLP 输出                       |
| MoE router 输入            | `NemotronHMoE._forward_core_normal`                             | `hidden_states`                 | router 的输入                   |
| MoE router logits        | `router_logits = torch.mm(...)` 后                               | router logits                   | 用于复现 routing                 |
| MoE top-k 专家             | `topk_output = self.topk(...)` 后                                | expert id / weight              | 第一次先打印对象结构                   |
| LatentMoE 输入投影           | `fc1_latent_proj` 前后                                            | hidden / latent hidden          | 专家真正输入可能是 latent hidden      |
| Routed experts 输入        | `self.experts(hidden_states, topk_output)` 前                    | hidden_states + topk_output     | 需要和 expert id 对齐             |
| Routed experts 输出        | `self.experts(...)` 后                                           | final_hidden_states             | fused MoE 聚合输出               |
| LatentMoE 输出投影           | `fc2_latent_proj` 前后                                            | expert output / restored hidden | 恢复到 hidden size              |
| Shared experts           | `self.shared_experts(hidden_states)`                            | shared expert up/down 激活        | shared expert 是 dense MLP 路径 |

---

## 8. 为了抓得稳，建议你先这样跑 SGLang

官方示例已经建议：

```bash
-e SGLANG_DISABLE_DEEP_GEMM=1
```

这对你抓矩阵乘非常关键，因为 fused DeepGEMM / FP4 MoE kernel 会把你想看的单个矩阵乘藏进融合后端里。官方 SGLang 部署示例确实用了这个环境变量。([Hugging Face][1])

你还应该尽量：

```bash
--disable-cuda-graph
```

或者以你安装版本为准，用：

```bash
python3 -m sglang.launch_server --help | grep -E "cuda|graph|compile|attention|moe|gemm|overlap"
```

确认 flag 名称。SGLang 不同版本 flag 会变，模型卡测试的是 `v0.5.12.post1`，你最好固定版本再改代码。([Hugging Face][1])

对 MoE，SGLang 的 `_forward_core` 在 CUDA 且非 torch compile 时会走 shared-routed overlap 路径，这会让 dump 的时序更复杂。为了抓激活，建议你临时强制：

```python
def _forward_core(self, hidden_states):
    return self._forward_core_normal(hidden_states)
```

因为 `_forward_core_normal` 里 router、topk、latent projection、experts 的调用顺序最清晰。SGLang 源码里确实有 normal path 和 overlap path 两套逻辑。([GitHub][2])

---

## 9. 最小可行抓取策略

我建议你分三阶段做，不要一开始就进 fused expert 内部。

### 阶段 A：只抓模块边界激活

先 hook 这些模块名：

```python
target_suffixes = [
    ".mixer.qkv_proj",
    ".mixer.o_proj",
    ".mixer.up_proj",
    ".mixer.down_proj",
    ".mixer.fc1_latent_proj",
    ".mixer.fc2_latent_proj",
    ".mixer.shared_experts.up_proj",
    ".mixer.shared_experts.down_proj",
]
```

记录：

```text
layer_id
module_name
input tensor shape / dtype
output tensor shape / dtype
token range / batch info
prefill or decode
```

### 阶段 B：改 `NemotronHMoE._forward_core_normal`

在 MoE forward 里额外记录：

```text
layer_id
hidden_states before router
router_logits
topk expert ids
topk weights
hidden_states after fc1_latent_proj
experts input
experts output
hidden_states after fc2_latent_proj
```

这是把 token 激活和专家权重对应起来的最小必要信息。

### 阶段 C：深入 `self.experts` fused MoE backend

SGLang 的 routed expert 具体实现由：

```python
self.experts = get_moe_impl_class(quant_config)(...)
```

决定。也就是说，真正的 expert up/down GEMM 不在 `NemotronHMoE` 这一层裸露出来，而是在 `python/sglang/srt/layers/moe/...` 的具体 backend 里。`SGLANG_DISABLE_DEEP_GEMM=1` 可以减少一部分融合，但不保证所有 expert GEMM 都变成普通 PyTorch Linear。([GitHub][2])

所以如果你要“专家内部 up/down matmul 的输入输出”，最终要么：

1. 强制使用更朴素的 MoE backend；
2. 修改 `get_moe_impl_class(quant_config)` 返回的 backend；
3. 在 fused MoE kernel 前后抓 dispatch 后的 expert-local activation；
4. 或先用 `topk_output + experts input + expert weight` 离线重放单个 expert GEMM。

对分析数据来说，第四种通常最稳：**在线只抓 hidden、topk expert id、topk weight、专家权重；离线按 expert id 重放 up/down。**

---

## 10. 最关键的命名修正

你后续写抓取代码时建议统一成下面这些概念：

```text
attention.qkv_proj        # packed QKV projection
attention.o_proj          # attention output projection

dense_mlp.up_proj         # dense MLP first projection
dense_mlp.down_proj       # dense MLP second projection
dense_mlp.act             # ReLU², not SwiGLU

moe.router_gate           # router, not expert gate_proj
moe.topk                  # expert id / weight
moe.fc1_latent_proj       # hidden -> latent expert space
moe.experts               # routed expert fused backend
moe.fc2_latent_proj       # latent expert output -> hidden
moe.shared_experts        # optional shared dense experts
```

最容易踩坑的是这三点：

1. **Nemotron-H 不是每层都有 attention。**
2. **dense MLP 没有标准 SwiGLU `gate_proj`。**
3. **MoE 的 `gate` 是 router；expert up/down 在 fused backend 里，不一定是裸露的 `up_proj/down_proj` module。**

[1]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 · Hugging Face"
[2]: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/nemotron_h.py "sglang/python/sglang/srt/models/nemotron_h.py at main · sgl-project/sglang · GitHub"
[3]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/tree/main "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 at main"
[4]: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nemotron_h.py "vllm/vllm/model_executor/models/nemotron_h.py at main · vllm-project/vllm · GitHub"
