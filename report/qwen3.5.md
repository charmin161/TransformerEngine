结论：**attention 的投影线性层不走 NVFP4 fused MoE 路径；MoE routed experts 才是这个 checkpoint 的 NVFP4 重点。**
你要抓激活值，建议分三层插桩：`qwen3_5.py` 抓模型语义位置，`linear.py` 抓普通 Linear/GEMM，`fused_moe_triton/layer.py + modelopt_quant.py` 抓 routed expert 的 fused GEMM。

## 1. 先判断这个模型哪些层被 NVFP4 量化

HF 上 NVIDIA 的 `Qwen3.5-397B-A17B-NVFP4` 配置是 `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`，并且是 60 层，`layer_types` 是 **3 个 `linear_attention` + 1 个 `full_attention` 周期重复**。配置里还有 `hidden_size=8192`、`num_attention_heads=32`、`num_key_value_heads=2`、`head_dim=256`、`num_experts=512`、`num_experts_per_tok=10`、`moe_intermediate_size=1024` 等关键维度。([Hugging Face][1])

关键是它的 `quantization_config`：`quant_algo=NVFP4`、`quant_method=modelopt`、weight/input activations 都是 4 bit、`group_size=16`，但 `ignore` 里排除了大量 `linear_attn*` 和 `self_attn*`。模型卡也明确写了：**只有 transformer block 内 MoE 的 linear operators 的 weights 和 activations 被量化到 NVFP4**。([Hugging Face][2])

所以：

| 模块                                                                      |                               是否 NVFP4 | 你该在哪里抓                                                            |
| ----------------------------------------------------------------------- | -------------------------------------: | ----------------------------------------------------------------- |
| linear attention 的 q/k/v/z、ba、out projection                            |        否，SGLang 强制 `quant_config=None` | `qwen3_5.py` + `linear.py`                                        |
| full attention 的 qkv_proj、o_proj                                        |        否，SGLang 强制 `quant_config=None` | `qwen3_5.py` + `linear.py`                                        |
| full attention core，即 QK/softmax/V 或 TRTLLM/FlashInfer attention kernel |                            不是 Linear 层 | `RadixAttention` backend                                          |
| MoE router gate                                                         |                  否，`quant_config=None` | `qwen2_moe.py` + `linear.py`                                      |
| MoE routed experts 的 w13 / w2                                           |                      是，NVFP4 fused MoE | `FusedMoE.run_moe_core()` / `ModelOptNvFp4FusedMoEMethod.apply()` |
| shared expert                                                           | 取决于是否被 fused 到 MoE；非 fused 时走普通 Linear | `qwen2_moe.py` / `linear.py` / `FusedMoE`                         |

---

## 2. Attention 线性层在哪里调用

### 2.1 Linear Attention 层

SGLang 在 `Qwen3_5LinearDecoderLayer.__init__` 里专门写了：

```python
linear_attn_quant_config = None if quant_config and quant_config.get_name() == "modelopt_fp4" else quant_config
```

也就是说，使用 `modelopt_fp4` 时，linear attention 的线性层会被强制设为非量化路径。([GitHub][3])

Linear attention 的主要线性层在 `Qwen3_5GatedDeltaNet` 里：

| 语义                       | SGLang 模块                               | 调用位置                                                         |
| ------------------------ | --------------------------------------- | ------------------------------------------------------------ |
| q/k/v/z fused projection | `self.in_proj_qkvz`                     | `_forward_input_proj()` 里 `self.in_proj_qkvz(hidden_states)` |
| beta/alpha projection    | `self.in_proj_ba`                       | `_forward_input_proj()` 里 `self.in_proj_ba(hidden_states)`   |
| core linear attention    | `self.attn = RadixLinearAttention(...)` | `self.attn(...)`                                             |
| output projection        | `self.out_proj`                         | `self.out_proj(core_attn_out)`                               |

源码里 `in_proj_qkvz`、`in_proj_ba`、`out_proj` 都是在 `Qwen3_5GatedDeltaNet` 初始化；forward 里先做 input projection，再调用 `self.attn(...)`，最后调用 `self.out_proj(core_attn_out)`。([GitHub][3])

**抓数建议：**

```text
python/sglang/srt/models/qwen3_5.py

Qwen3_5GatedDeltaNet._forward_input_proj()
	- 抓 hidden_states
	- 抓 projected_states_qkvz
	- 抓 projected_states_ba

Qwen3_5GatedDeltaNet.forward()
	- 抓 q, k, v, z, beta, alpha
	- 抓 self.attn(...) 的输入/输出
	- 抓 self.out_proj(core_attn_out) 的输入/输出
```

### 2.2 Full Attention 层

Full attention 也一样，在 SGLang 里有：

```python
attn_quant_config = None if quant_config and quant_config.get_name() == "modelopt_fp4" else quant_config
```

因此 full attention 的 `qkv_proj` 和 `o_proj` 也不走 ModelOpt FP4 quantized linear。([GitHub][3])

Full attention 的关键路径是：

```python
qkv, _ = self.qkv_proj(hidden_states)
...
attn_output = self.attn(q, k, v, forward_batch)
output, _ = self.o_proj(attn_output)
```

其中 `qkv_proj` 是 `QKVParallelLinear`，`o_proj` 是 `RowParallelLinear`，核心 attention 是 `RadixAttention`。([GitHub][3])

**抓数建议：**

```text
python/sglang/srt/models/qwen3_5.py

Qwen3_5AttentionDecoderLayer 的 full attention forward block
	- 抓 hidden_states
	- 抓 qkv_proj 输出 qkv
	- 抓 split 后的 q / k / v
	- 抓 self.attn(q, k, v, forward_batch) 输入/输出
	- 抓 o_proj 输入/输出
```

注意：如果你想看真正 attention kernel 里的 QK、softmax、PV，中间值不会在 `Linear` 层出现，需要去 `RadixAttention` 对应 backend 里插，而不是只改 `linear.py`。

---

## 3. 普通 Linear 的底层 GEMM 入口在哪里

SGLang 的普通线性层都统一走：

```text
python/sglang/srt/layers/linear.py
```

关键逻辑是：

```python
output_parallel = self.quant_method.apply(self, input_, bias)
```

`ColumnParallelLinear.forward()` 里注释就是 “Matrix multiply”，然后调用 `self.quant_method.apply(...)`；`ReplicatedLinear.forward()` 也一样调用 `self.quant_method.apply(...)`。如果 `quant_config is None`，`LinearBase` 会使用 `UnquantizedLinearMethod`。([GitHub][4])

所以：

```text
attention qkv_proj / in_proj_qkvz / in_proj_ba / o_proj
router gate
非 fused shared expert
普通 dense linear
```

这些都可以在：

```text
python/sglang/srt/layers/linear.py

ColumnParallelLinear.forward()
RowParallelLinear.forward()
ReplicatedLinear.forward()
```

里的 `self.quant_method.apply(...)` 前后抓输入输出。

但要注意：**MoE routed experts 的 w13/w2 不会经过这些普通 Linear forward。** 它们被 `FusedMoE` 接管。

---

## 4. MoE routed experts 在哪里调用

### 4.1 Qwen2MoeSparseMoeBlock 是上层入口

在 `qwen2_moe.py` 中，MoE block 里：

```python
self.experts = get_moe_impl_class(quant_config)(...)
```

这里会根据 `quant_config` 创建 MoE 实现，通常是 `FusedMoE`。([GitHub][5])

router gate 是：

```python
self.gate = ReplicatedLinear(
	config.hidden_size,
	config.num_experts,
	bias=False,
	quant_config=None,
	...
)
```

也就是说 router gate 不量化。([GitHub][5])

普通 MoE forward 路径里：

```python
router_logits, _ = self.gate(hidden_states)
topk_output = self.topk(hidden_states, router_logits)
return self.experts(hidden_states, topk_output)
```

因此你可以在这里抓：

```text
hidden_states
router_logits
topk_ids / topk_weights
experts 输入输出
```

对应位置：

```text
python/sglang/srt/models/qwen2_moe.py

Qwen2MoeSparseMoeBlock._forward_router_experts()
Qwen2MoeSparseMoeBlock._forward_deepep()
Qwen2MoeSparseMoeBlock.forward()
```

源码里的 `_forward_router_experts()` 和 forward 都能看到 router、topk、experts 调用链。([GitHub][5])

### 4.2 FusedMoE 是 routed expert 的真正入口

`FusedMoE` 的注释已经说明：它包含 fused 的 `gate_up_proj / w13` 和 `down_proj / w2` 权重。也就是说，MoE experts 不是普通的 `Linear` 模块，而是 fused weight tensor。([GitHub][6])

初始化里会做：

```python
self.quant_method = quant_config.get_quant_method(self, prefix)
...
self.quant_method.create_weights(...)
self.quant_method.create_moe_runner(...)
```

如果是 ModelOpt NVFP4，会走 `ModelOptNvFp4FusedMoEMethod`。源码里还专门判断了 `isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)`。([GitHub][6])

真正 forward 调用链是：

```text
FusedMoE.forward()
	-> FusedMoE.forward_impl()
		-> dispatch_output = self.dispatcher.dispatch(...)
		-> combine_input = self.run_moe_core(dispatch_output)
			-> self.quant_method.apply(layer=self, dispatch_output=dispatch_output)
		-> self.dispatcher.combine(...)
```

源码里 `run_moe_core()` 直接返回 `self.quant_method.apply(layer=self, dispatch_output=dispatch_output)`。([GitHub][7])

**所以 routed MoE 的第一推荐插桩点是：**

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py

FusedMoE.forward_impl()
FusedMoE.run_moe_core()
```

这里能抓到：

```text
dispatch 前的 hidden_states
dispatch 后按 expert/token 重排后的 dispatch_output
run_moe_core 输出 combine_input
combine 后的 final_hidden_states
topk_ids / topk_weights
```

如果你只想抓“进入专家前”和“专家输出后”的激活，插在 `FusedMoE.run_moe_core()` 前后最稳。

---

## 5. NVFP4 MoE 的底层矩阵乘在哪里

ModelOpt 的 quant config 选择逻辑在：

```text
python/sglang/srt/layers/quantization/modelopt_quant.py
```

对于 `FusedMoE`，如果解析到 `quant_algo == "NVFP4"`，会返回：

```python
ModelOptNvFp4FusedMoEMethod(self.nvfp4_config)
```

对于普通 `LinearBase`，如果是 `NVFP4`，才返回 `ModelOptFp4LinearMethod`；但这个模型的 attention 被 ignore / 强制 `quant_config=None`，所以 attention 不走这个路径。([GitHub][8])

普通 FP4 GEMM primitive 是：

```text
modelopt_quant.py::fp4_gemm(...)
```

它内部按 backend 分发：

```python
if fp4_backend.is_cutlass():
	return cutlass_fp4_gemm(...)
elif enable_flashinfer_fp4_gemm:
	return flashinfer_fp4_gemm(...)
else:
	return cutlass_fp4_gemm(...)
```

也就是说，普通 FP4 GEMM 会落到 CUTLASS 或 FlashInfer 的 `mm_fp4`。([GitHub][8])

但对于 **fused MoE**，你不能假设每次 expert GEMM 都会经过 `fp4_gemm()` 这个 Python 函数。SGLang 会通过：

```text
FusedMoE.run_moe_core()
	-> ModelOptNvFp4FusedMoEMethod.apply()
		-> MoE runner backend
```

下沉到 fused MoE runner。这个路径可能直接调用 FlashInfer / TRTLLM / CUTLASS fused MoE kernel。

所以你要分两种目标：

| 目标                              | 插桩位置                                                                    |
| ------------------------------- | ----------------------------------------------------------------------- |
| 抓 routed expert 输入 / 输出         | `FusedMoE.run_moe_core()` 前后                                            |
| 抓 dispatch 后每个 expert 收到的 token | `FusedMoE.forward_impl()` dispatch 后                                    |
| 抓 w13 GEMM 之后、activation 之前的中间值 | 需要改 `ModelOptNvFp4FusedMoEMethod.apply()` 或 runner/kernel；Python 层可能拿不到 |
| 抓 FP4 packed activation / scale | `ModelOptNvFp4FusedMoEMethod.apply()` 或更底层 runner                       |
| 抓普通 FP4 Linear GEMM             | `modelopt_quant.py::fp4_gemm()`                                         |
| 抓 attention projection 激活       | `qwen3_5.py` 或 `linear.py`，不是 `modelopt_quant.py::fp4_gemm()`           |

---

## 6. 最推荐的插桩方案

### A. 抓 attention projection 激活

改：

```text
python/sglang/srt/models/qwen3_5.py
```

插在：

```text
Qwen3_5GatedDeltaNet._forward_input_proj()
Qwen3_5GatedDeltaNet.forward()
Qwen3_5AttentionDecoderLayer full attention forward block
```

同时可以在：

```text
python/sglang/srt/layers/linear.py
```

的这些方法里统一抓：

```text
ColumnParallelLinear.forward()
RowParallelLinear.forward()
ReplicatedLinear.forward()
```

这样能覆盖：

```text
linear attention: in_proj_qkvz / in_proj_ba / out_proj
full attention: qkv_proj / o_proj
router gate
非 fused shared expert
```

### B. 抓 MoE routed expert 激活

改：

```text
python/sglang/srt/models/qwen2_moe.py
```

插在：

```text
Qwen2MoeSparseMoeBlock._forward_router_experts()
```

抓：

```text
hidden_states
router_logits
topk_ids
topk_weights
```

然后改：

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

插在：

```text
FusedMoE.forward_impl()
FusedMoE.run_moe_core()
```

抓：

```text
dispatch_output
combine_input
final_hidden_states
```

### C. 抓 NVFP4 packed activation / scale / GEMM 输入

继续下沉到：

```text
python/sglang/srt/layers/quantization/modelopt_quant.py

ModelOptNvFp4FusedMoEMethod.apply()
```

这里才是 NVFP4 routed MoE 最关键的位置。你可以在这里抓：

```text
fused MoE 输入 hidden states
topk 信息
FP4 quantized activation
activation scale
w13_weight / w2_weight
w13_scale / w2_scale
runner 输出
```

如果你发现 `apply()` 内部直接调用 runner，且 w13 中间值没有返回，那么要抓 `w13 -> activation -> w2` 中间激活，只能：

```text
1. 临时切换到非 fused / debug runner；
2. 修改 fused MoE runner Python wrapper；
3. 或改 CUDA / CUTLASS / FlashInfer kernel，把中间 tensor 暴露出来。
```

---

## 7. 一句话定位

你可以按这个顺序找：

```text
Attention projection:
python/sglang/srt/models/qwen3_5.py
	linear attention: Qwen3_5GatedDeltaNet._forward_input_proj / forward
	full attention: qkv_proj -> RadixAttention -> o_proj

普通 Linear GEMM:
python/sglang/srt/layers/linear.py
	ColumnParallelLinear.forward / RowParallelLinear.forward / ReplicatedLinear.forward
		-> self.quant_method.apply(...)

MoE routing:
python/sglang/srt/models/qwen2_moe.py
	Qwen2MoeSparseMoeBlock._forward_router_experts()
		-> self.gate(...)
		-> self.topk(...)
		-> self.experts(hidden_states, topk_output)

MoE routed expert fused GEMM:
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
	FusedMoE.forward_impl()
		-> FusedMoE.run_moe_core()
			-> self.quant_method.apply(layer=self, dispatch_output=dispatch_output)

NVFP4 method/backend:
python/sglang/srt/layers/quantization/modelopt_quant.py
	ModelOptNvFp4FusedMoEMethod.apply()
	fp4_gemm()  # 普通 FP4 GEMM primitive；fused MoE 不一定逐次经过这里
```

最实用的抓数策略是：**先在 `qwen2_moe.py` 抓 router/topk，再在 `FusedMoE.run_moe_core()` 抓 expert 前后激活；attention projection 则在 `qwen3_5.py` 或 `linear.py` 抓。**

[1]: https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4/blob/1ea7852ec974926be5705dd281eebacbc99bb4bd/config.json "config.json · nvidia/Qwen3.5-397B-A17B-NVFP4 at 1ea7852ec974926be5705dd281eebacbc99bb4bd"
[2]: https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4/raw/1ea7852ec974926be5705dd281eebacbc99bb4bd/config.json "huggingface.co"
[3]: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5.py "sglang/python/sglang/srt/models/qwen3_5.py at main · sgl-project/sglang · GitHub"
[4]: https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/layers/linear.py "raw.githubusercontent.com"
[5]: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_moe.py "sglang/python/sglang/srt/models/qwen2_moe.py at main · sgl-project/sglang · GitHub"
[6]: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py "sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py at main · sgl-project/sglang · GitHub"
[7]: https://github.com/sgl-project/sglang/raw/refs/heads/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py "raw.githubusercontent.com"
[8]: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/modelopt_quant.py "sglang/python/sglang/srt/layers/quantization/modelopt_quant.py at main · sgl-project/sglang · GitHub"
