下面按 **模型结构 → vLLM 入口 → 每层前向路径 → 你要抓的矩阵乘插桩点** 梳理。这里以 vLLM 当前 MiniMax-M3 专用实现为准；官方 vLLM 文档说明 MiniMax-M3 支持尚未进入稳定版，需要使用专门镜像或 main/nightly 代码路径。([vLLM Recipes][1])

## 1. MiniMax-M3-NVFP4 的整体结构

NVIDIA 发布的 `MiniMax-M3-NVFP4` 是 MiniMax-M3 的 **ModelOpt NVFP4 量化版本**，模型卡写明它是 MoE 多模态模型，支持 1M token 级上下文，运行时目标是 vLLM，硬件目标是 Blackwell，量化工具为 `nvidia-modelopt v0.44.0`。([Hugging Face][2])

从结构上看，它不是单纯 LLaMA/Qwen 风格的 dense decoder，而是：

```text
MiniMaxM3SparseForConditionalGeneration
├── vision tower / ViT encoder                # 多模态图像/视频输入
├── multimodal projector                      # 视觉特征投影到 text hidden size
└── language_model / MiniMaxM3SparseForCausalLM
    └── MiniMaxM3Model
        ├── embed_tokens
        ├── 60 × MiniMaxM3DecoderLayer
        │   ├── input_layernorm
        │   ├── self_attn
        │   │   ├── dense attention 或 sparse attention / MSA
        │   │   ├── qkv_proj
        │   │   └── o_proj
        │   ├── post_attention_layernorm
        │   └── mlp 或 block_sparse_moe
        │       ├── dense MLP: gate_up_proj + down_proj
        │       └── MoE: router gate + FusedMoE experts
        └── norm
```

Transformers 的 MiniMax-M3 配置文档给出的默认 text backbone 参数包括：`hidden_size=6144`、`num_hidden_layers=60`、`num_attention_heads=64`、`num_key_value_heads=4`、`head_dim=128`、`dense_intermediate_size=12288`、`intermediate_size=3072`、`top_k=4`、`num_local_experts=128`、`routed_scaling_factor=2.0`、`index_topk_blocks=16`、`index_block_size=128` 等。([Hugging Face][3])

MiniMax 官方说明 M3 使用 **MiniMax Sparse Attention / MSA**，目标是长上下文下减少 attention 计算和 KV 访问；vLLM 也有专门的 NVIDIA MSA sparse attention backend。([GitHub][4])

---

## 2. vLLM 中会进入哪个模型类

vLLM registry 里 MiniMax-M3 的映射是：

```python
"MiniMaxM3SparseForCausalLM"
    -> ("vllm.models.minimax_m3", "MiniMaxM3SparseForCausalLM")

"MiniMaxM3SparseForConditionalGeneration"
    -> ("vllm.models.minimax_m3", "MiniMaxM3SparseForConditionalGeneration")
```

也就是说，HF config 里的 `architectures` 会决定进入 causal-only 还是 multimodal conditional generation 类。([GitHub][5])

`vllm.models.minimax_m3` 本身是一个硬件隔离入口，会根据当前平台分发到 `nvidia/` 或 `amd/` 实现；NVIDIA 侧实现主要在：

```text
vllm/models/minimax_m3/__init__.py
vllm/models/minimax_m3/nvidia/model.py
vllm/models/minimax_m3/nvidia/sparse_attention_msa.py
vllm/models/minimax_m3/nvidia/indexer_msa.py
vllm/models/minimax_m3/common/sparse_attention.py
vllm/models/minimax_m3/common/indexer.py
```

vLLM 文档明确列出 NVIDIA MiniMax-M3 目录中包含 `indexer_msa`、`model`、`mtp`、`sparse_attention_msa` 等模块。([vLLM][6])

---

## 3. vLLM 前向推理主调用链

推理时，主体调用链可以抽象成：

```text
LLMEngine / worker / model_runner
  -> MiniMaxM3SparseForCausalLM.forward(...)
      -> MiniMaxM3Model.forward(...)
          -> for layer in self.layers:
                MiniMaxM3DecoderLayer.forward(...)
          -> final norm
      -> compute_logits(...)
```

每个 decoder layer 的逻辑大致是：

```text
MiniMaxM3DecoderLayer.forward
├── residual = hidden_states
├── hidden_states = input_layernorm(hidden_states)
├── hidden_states = self_attn(hidden_states, positions, ...)
│   ├── MiniMaxM3Attention.forward
│   └── 或 MiniMaxM3SparseAttention.forward
├── hidden_states = residual + attention_output
├── hidden_states = post_attention_layernorm(hidden_states)
├── hidden_states = mlp 或 block_sparse_moe(hidden_states)
└── hidden_states = residual + ffn_output
```

vLLM 的实现里，decoder layer 会根据 `sparse_attention_config` 决定某层使用 dense attention 还是 sparse attention，同时根据 MoE 层配置选择 `block_sparse_moe` 或 dense `mlp`。([GitHub][7])

---

## 4. Attention：QKV proj 与 output proj 定位

### 4.1 Dense attention 路径

Dense attention 类是：

```python
MiniMaxM3Attention
```

核心模块：

```python
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
self.attn = Attention(...)
```

forward 中先调用：

```python
qkv, _ = self.qkv_proj(hidden_states)
```

然后经过 fused Q/K norm、RoPE、KV cache insert，再 split 出 query/key/value，最后：

```python
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
```

vLLM 文档源码片段显示 `MiniMaxM3Attention` 使用 `QKVParallelLinear` 做 QKV 融合投影，并用 `RowParallelLinear` 做 `o_proj`。([vLLM][8])

你要抓 dense attention 激活，最直接的位置是：

```text
model.layers.{layer_id}.self_attn.qkv_proj
model.layers.{layer_id}.self_attn.o_proj
```

对应关系：

| 目标                    | vLLM 模块                  | 该抓什么                              |
| --------------------- | ------------------------ | --------------------------------- |
| Q/K/V projection 输入激活 | `self_attn.qkv_proj` 的输入 | attention norm 后的 `hidden_states` |
| Q/K/V projection 输出   | `self_attn.qkv_proj` 的输出 | fused qkv，后续再 split               |
| output projection 输入  | `self_attn.o_proj` 的输入   | `attn_output`                     |
| output projection 输出  | `self_attn.o_proj` 的输出   | attention branch 输出               |

注意：如果你要抓 **RoPE/QKNorm 之后的 Q/K**，仅 hook `qkv_proj` 不够，因为 vLLM 后面会走 `ops.fused_minimax_m3_qknorm_rope_kv_insert` 这类 fused op。要么在该 op 之后加 patch，要么抓 KV cache / query buffer。

---

### 4.2 Sparse attention / MSA 路径

MiniMax-M3 的特殊性主要在这里。Sparse attention 类是：

```python
MiniMaxM3SparseAttention
```

它的 qkv projection 不是普通 QKV，而是融合了：

```text
Q / K / V / index_Q / index_K
```

vLLM 代码里使用：

```python
MinimaxM3QKVParallelLinearWithIndexer
```

也就是说 sparse attention 的 qkv 投影输出中额外包含 indexer 分支需要的 query/key。vLLM 文档说明 sparse attention 类会拥有 qkv + index q/k projections，index value/output projection 被禁用或不创建。([vLLM][8])

Sparse attention forward 的关键路径是：

```text
MiniMaxM3SparseAttention.forward
├── qkv, _ = self.qkv_proj(hidden_states)
│   └── 输出 Q/K/V/index_Q/index_K
├── fused qk norm + rope + kv cache insert
├── self._run_attention(...)
│   ├── self.indexer(index_query)
│   │   └── 计算 sparse attention 需要访问哪些 KV blocks
│   └── self.impl.forward(...)
│       └── sparse/block attention kernel
└── output, _ = self.o_proj(attn_output)
```

vLLM common sparse attention 文档说明：indexer 会把选中的 block 写入共享的 `layer.topk_indices_buffer`，主 attention kernel 再读取这个 buffer。([vLLM][9])

所以 sparse attention 下你要抓：

```text
model.layers.{layer_id}.self_attn.qkv_proj
model.layers.{layer_id}.self_attn.o_proj
model.layers.{layer_id}.self_attn.indexer
model.topk_indices_buffer
```

但注意这里的 `topk_indices_buffer` 是 **sparse attention 的 KV block top-k**，不是 MoE expert id。不要把它和 MoE router 的 expert id 混淆。

---

## 5. Dense MLP：gate/up/down proj 定位

Dense MLP 类是：

```python
MiniMaxM3MLP
```

它不是三个独立 Linear，而是：

```python
self.gate_up_proj = MergedColumnParallelLinear(...)
self.down_proj = RowParallelLinear(...)
self.act_fn = SiluAndMulWithClamp(...)
```

forward 是：

```python
gate_up, _ = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x, _ = self.down_proj(x)
```

vLLM 文档源码显示 `MiniMaxM3MLP` 将 gate 和 up 合并为 `gate_up_proj`，再经 SwiGLU/OAI 风格激活，最后进入 `down_proj`。([vLLM][8])

对应插桩点：

```text
model.layers.{layer_id}.mlp.gate_up_proj
model.layers.{layer_id}.mlp.down_proj
```

对应关系：

| 目标                 | vLLM 模块               | 说明                                |
| ------------------ | --------------------- | --------------------------------- |
| dense gate_proj 输入 | `mlp.gate_up_proj` 输入 | FFN norm 后 hidden                 |
| dense up_proj 输入   | `mlp.gate_up_proj` 输入 | 与 gate_proj 输入相同                  |
| dense gate/up 输出   | `mlp.gate_up_proj` 输出 | fused tensor，需要 split 成 gate / up |
| dense down_proj 输入 | `mlp.down_proj` 输入    | SwiGLU/OAI 激活后的中间值                |
| dense down_proj 输出 | `mlp.down_proj` 输出    | FFN 输出                            |

所以 MiniMax-M3 dense MLP 中，不要找单独的：

```text
gate_proj
up_proj
```

vLLM 里它们合并成：

```text
gate_up_proj
```

---

## 6. MoE：router、expert gate/up/down 的真实位置

MoE 类是：

```python
MiniMaxM3MoE
```

核心结构是：

```python
self.gate = GateLinear(...)
self.experts = FusedMoE(...)
```

forward 是：

```python
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
)
```

vLLM 源码文档显示 `MiniMaxM3MoE` 使用 `GateLinear` 得到 router logits，然后把 `hidden_states` 和 `router_logits` 交给 `FusedMoE`。([vLLM][8])

这里最容易混淆：

```text
block_sparse_moe.gate
```

不是 FFN 的 `gate_proj`，而是 **MoE router projection**：

```text
hidden_states -> num_experts router_logits
```

真正的 expert FFN 矩阵在：

```text
block_sparse_moe.experts
```

并且通常被 vLLM FusedMoE 打包，不表现为普通 PyTorch Linear module。

vLLM 的 MiniMax-M3 weight mapping 明确把 checkpoint 中的 expert 权重映射为：

```text
w1 = gate_proj
w2 = down_proj
w3 = up_proj
```

加载时会把 gate/up 合并到 fused expert 参数里。([GitHub][7])

对于 NVFP4 / ModelOpt 量化，vLLM 的 ModelOpt MoE 参数会以打包形式存储，例如 `w13_weight` 表示 gate/up fused 权重，`w2_weight` 表示 down 权重；文档中还说明 NVFP4 权重按 uint8 打包存储，每个 byte 中有两个 FP4 值。([vLLM][10])

所以 MoE 的插桩点应该分两层：

### 6.1 Python 层可直接 hook 的位置

```text
model.layers.{layer_id}.block_sparse_moe.gate
model.layers.{layer_id}.block_sparse_moe.experts
```

其中：

| 目标            | 位置                            | 能直接拿到什么                       |
| ------------- | ----------------------------- | ----------------------------- |
| router 输入     | `block_sparse_moe.gate` 输入    | MoE 前 hidden_states           |
| router logits | `block_sparse_moe.gate` 输出    | 每 token 对每 expert 的 logits    |
| MoE 输入        | `block_sparse_moe.experts` 输入 | hidden_states + router_logits |
| MoE 输出        | `block_sparse_moe.experts` 输出 | combine 后的 FFN 输出             |

### 6.2 真正 expert GEMM 级别需要 patch 的位置

如果你要抓：

```text
某 token 被分到哪个 expert
该 token 对应 expert 的 gate_proj / up_proj 输入
gate/up GEMM 输出
SwiGLU 后 down_proj 输入
down_proj 输出
```

仅 hook `block_sparse_moe.experts` 不够，因为 expert routing、token permutation、grouped GEMM、dequant、activation、combine 都在 FusedMoE 内部完成。vLLM 的 fused MoE 接口中，底层 `fused_experts(...)` 明确接收：

```python
hidden_states
w1
w2
topk_weights
topk_ids
...
```

也就是说，**expert id 在 FusedMoE 内部 router/topk 之后才成为实际 kernel 输入**。([vLLM][11])

为了让激活和权重精确对应，建议 patch FusedMoE 内部，保存：

```text
layer_id
token_id / batch position
global expert id
local expert id / physical expert id
topk weight
expert input hidden_states
w13 GEMM output
split 后的 expert gate / up 输出
SwiGLU/OAI 激活后 down 输入
w2 GEMM output
combine 前后输出
```

如果开启 expert parallel，还必须记录 global expert id 到 local/physical expert id 的映射；vLLM 的 fused MoE 代码中存在 `expert_local_to_global`、`expert_physical_to_global` 以及 global/local expert id 映射函数。([vLLM][11])

---

## 7. 你最终需要定位的矩阵乘清单

### 7.1 Attention

| 你关心的矩阵乘          | vLLM 模块                           | 代码层级                                                              | 备注                                           |
| ---------------- | --------------------------------- | ----------------------------------------------------------------- | -------------------------------------------- |
| QKV proj         | `self_attn.qkv_proj`              | `MiniMaxM3Attention.forward` / `MiniMaxM3SparseAttention.forward` | dense 是 Q/K/V；sparse 是 Q/K/V/index_Q/index_K |
| output proj      | `self_attn.o_proj`                | attention forward 末尾                                              | 输入是 attention kernel 输出                      |
| sparse index q/k | `self_attn.qkv_proj` fused 输出的一部分 | sparse attention only                                             | 用于 MSA block selection，不是 MoE expert id      |

### 7.2 Dense MLP

| 你关心的矩阵乘   | vLLM 模块                  | 备注                 |
| --------- | ------------------------ | ------------------ |
| gate_proj | `mlp.gate_up_proj` 的前半部分 | fused              |
| up_proj   | `mlp.gate_up_proj` 的后半部分 | fused              |
| down_proj | `mlp.down_proj`          | 输入是 SwiGLU/OAI 后激活 |

### 7.3 MoE

| 你关心的矩阵乘               | vLLM 中真实位置                                    | 备注                    |
| --------------------- | --------------------------------------------- | --------------------- |
| router                | `block_sparse_moe.gate`                       | 不是 FFN gate_proj      |
| expert gate_proj / w1 | `block_sparse_moe.experts` 内部 fused MoE `w13` | 与 up_proj fused       |
| expert up_proj / w3   | `block_sparse_moe.experts` 内部 fused MoE `w13` | 与 gate_proj fused     |
| expert down_proj / w2 | `block_sparse_moe.experts` 内部 fused MoE `w2`  | grouped GEMM          |
| expert id             | FusedMoE router/topk 内部                       | 推荐 patch 内部，不建议只从外部重算 |

---

## 8. 推荐抓激活方案

你的 `disable-cudagraph` / eager 推理思路是对的，但要区分两类目标：

### A. 普通 Linear 级别激活

这些可以先用 PyTorch forward hook：

```text
self_attn.qkv_proj
self_attn.o_proj
mlp.gate_up_proj
mlp.down_proj
block_sparse_moe.gate
```

这能拿到大部分输入输出 tensor。关闭 cudagraph 后，hook 的行为更直观，不会被 graph replay 干扰。

### B. MoE expert 内部激活

这部分不建议只靠 module hook，因为 `FusedMoE` 内部可能一次性完成：

```text
routing topk
token dispatch / permutation
w13 grouped GEMM
SwiGLU/OAI activation
w2 grouped GEMM
weighted combine
```

你应该直接 patch：

```text
vllm/model_executor/layers/fused_moe/
```

重点找：

```text
FusedMoE.forward
RoutedExperts.forward
fused_experts(...)
router / topk 计算逻辑
```

保存 `topk_ids`、`topk_weights`、expert mapping、permuted hidden states、`w13` 输出、activation 后中间值和 `w2` 输出。vLLM 的 fused MoE 文档显示 `fused_experts` 这一层已经显式接收 `topk_ids` 和 `topk_weights`，因此这里是最适合打点的位置。([vLLM][11])

---

## 9. backend / fusion 对抓数的影响

几个关键判断：

1. **attention backend 主要影响 attention kernel，不影响 qkv_proj/o_proj 的模块存在。**
   也就是说，你抓 QKV projection 和 output projection，通常不必先换 attention backend；hook `qkv_proj` 和 `o_proj` 就能抓到 projection 输入输出。

2. **Sparse attention 的 indexer 是 MiniMax-M3 特有重点。**
   如果你想分析 MSA 选择了哪些 KV block，要额外记录 `indexer` 输出或 `topk_indices_buffer`；但这和 MoE expert id 是两套 top-k。vLLM 文档说明 sparse attention indexer 会把 block 选择结果写入共享 buffer，然后主 sparse attention kernel 读取。([vLLM][12])

3. **MoE 是最需要改源码的部分。**
   对于 expert gate/up/down，vLLM 不是普通 `nn.Linear` 调用，而是 FusedMoE grouped GEMM。尤其 NVFP4 下还会进入 ModelOpt 量化 Linear/MoE backend，权重可能是 packed FP4，普通 hook 只能拿模块边界输入输出，拿不到每个 expert 内部 GEMM 中间激活。([vLLM][10])

4. **调试时先不要开 expert parallel。**
   第一轮建议：TP 固定、batch 小、`max_num_seqs=1` 或很小、关闭 cudagraph、不开 EP。等你确认 token → expert → weight 的对应关系后，再处理 EP 下 global/local/physical expert id 映射。

---

## 10. 最小插桩优先级

建议你按这个顺序做：

```text
第一层：确认调用路径
1. 打印 model.named_modules()
2. 找 self_attn.qkv_proj / o_proj
3. 找 mlp.gate_up_proj / down_proj
4. 找 block_sparse_moe.gate / experts

第二层：抓普通 projection
1. hook qkv_proj 输入输出
2. hook o_proj 输入输出
3. hook dense mlp gate_up/down 输入输出
4. hook block_sparse_moe.gate router logits

第三层：MoE 内部 patch
1. 在 FusedMoE router/topk 后保存 topk_ids/topk_weights
2. 保存 global/local expert id 映射
3. 在 grouped GEMM 前保存 per-expert 输入
4. 在 w13 后保存 gate/up fused 输出
5. 在 activation 后保存 down 输入
6. 在 w2 后保存 expert 输出
7. 在 combine 后保存最终 MoE 输出

第四层：Sparse attention 附加分析
1. 保存 index_q/index_k
2. 保存 sparse attention topk block indices
3. 保存 attention backend 输入输出
```

结论上，MiniMax-M3 对你原先 transformer 抓点理解的主要修正是：

```text
普通 QKV / o_proj：仍然存在，可以按 attention module 抓。

dense MLP gate/up/down：gate/up 被 gate_up_proj 融合。

MoE gate/up/down：不在普通 Linear module 里，而在 FusedMoE experts 内部；
checkpoint 语义是 w1=gate, w2=down, w3=up；
vLLM 运行时通常 w1/w3 fused 成 w13。

MiniMax 特有 sparse attention：qkv_proj 额外融合 index_q/index_k，
并有 MSA indexer/topk block 选择；这不是 MoE expert id。
```

[1]: https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3 "MiniMaxAI/MiniMax-M3 | vLLM Recipes"
[2]: https://huggingface.co/nvidia/MiniMax-M3-NVFP4 "nvidia/MiniMax-M3-NVFP4 · Hugging Face"
[3]: https://huggingface.co/docs/transformers/model_doc/minimax_m3_vl "MiniMax-M3-VL · Hugging Face"
[4]: https://github.com/MiniMax-AI/MiniMax-M3 "GitHub - MiniMax-AI/MiniMax-M3 · GitHub"
[5]: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py "vllm/vllm/model_executor/models/registry.py at main · vllm-project/vllm · GitHub"
[6]: https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/ "minimax_m3 - vLLM"
[7]: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/minimax_m3/nvidia/model.py "raw.githubusercontent.com"
[8]: https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/amd/model/ "model - vLLM"
[9]: https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/common/sparse_attention/ "sparse_attention - vLLM"
[10]: https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/modelopt/ "modelopt - vLLM"
[11]: https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/ "fused_moe - vLLM"
[12]: https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/common/indexer/ "indexer - vLLM"
