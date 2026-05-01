Qwen3_5AttentionDecoderLayer / Qwen3_5LinearDecoderLayer
	-> Qwen2MoeSparseMoeBlock.forward()
		-> _forward_router_experts()
			-> router_logits = self.gate(hidden_states)
			-> topk_output = self.topk(hidden_states, router_logits)
			-> self.experts(hidden_states, topk_output)
				-> FusedMoE.forward()
					-> FusedMoE.run_moe_core()
						-> self.quant_method.apply(...)
							-> Fp8MoEMethod.apply(...)
								-> fused_experts(...)
									-> invoke_fused_moe_kernel(... w13 ...)
									-> silu_and_mul(...)
									-> invoke_fused_moe_kernel(... w2 ...)
可以按 **4 层打点** 来确认：

```text
qwen2_moe.py
	确认 get_moe_impl_class 返回了哪个 MoE 实现类

fused_moe_triton/layer.py
	确认 self.experts 实例化成了哪个 FusedMoE，quant_method 是谁，runner 是谁

quantization/fp8.py / unquant.py / modelopt_quant.py
	确认 runtime 进入了 Fp8MoEMethod / UnquantizedFusedMoEMethod / ModelOptNvFp4FusedMoEMethod 的哪个 apply

fused_moe_triton/fused_moe.py
	确认最终是否进入 Triton fused_experts，并定位 w13 -> silu -> w2 的位置
```

---

## 1. 先加一个通用 trace 函数

建议不要直接到处 `print()`，否则多卡多进程会刷屏。可以先在 `qwen2_moe.py` 顶部临时加：

```python
def debug_trace(message: str) -> None:
	import os
	rank = os.environ.get("RANK", "?")
	local_rank = os.environ.get("LOCAL_RANK", "?")
	print(f"[MOE_TRACE][rank={rank}][local_rank={local_rank}] {message}", flush=True)
```

如果日志太多，改成只打印 rank0：

```python
def debug_trace(message: str) -> None:
	import os
	if os.environ.get("RANK", "0") != "0":
		return
	local_rank = os.environ.get("LOCAL_RANK", "?")
	print(f"[MOE_TRACE][rank=0][local_rank={local_rank}] {message}", flush=True)
```

如果你担心 stdout 被吞，改成写文件：

```python
def debug_trace(message: str) -> None:
	import os
	rank = os.environ.get("RANK", "0")
	local_rank = os.environ.get("LOCAL_RANK", "?")
	with open(f"/tmp/sglang_moe_trace_rank{rank}.log", "a", encoding="utf-8") as trace_file:
		trace_file.write(f"[rank={rank}][local_rank={local_rank}] {message}\n")
		trace_file.flush()
```

---

## 2. 在 `qwen2_moe.py` 确认 get_moe_impl_class 返回什么

你现在定位到：

```python
self.experts = get_moe_impl_class(quant_config)(...)
```

把它拆开：

```python
moe_impl_class = get_moe_impl_class(quant_config)
debug_trace(
	f"Qwen2MoeSparseMoeBlock.__init__: layer_id={layer_id}, "
	f"quant_config={None if quant_config is None else quant_config.get_name()}, "
	f"moe_impl_class={moe_impl_class}"
)

self.experts = moe_impl_class(
	layer_id=self.layer_id,
	top_k=(
		config.num_experts_per_tok
		if not self.enable_shared_expert_fusion
		else config.num_experts_per_tok + self.num_fused_shared_experts
	),
	num_experts=(
		config.num_experts + get_global_server_args().ep_num_redundant_experts
		if not self.enable_shared_expert_fusion
		else config.num_experts
		+ get_global_server_args().ep_num_redundant_experts
		+ self.num_fused_shared_experts
	),
	hidden_size=config.hidden_size,
	intermediate_size=config.moe_intermediate_size,
	quant_config=quant_config,
	prefix=add_prefix("experts", prefix),
	routing_method_type=RoutingMethodType.RenormalizeNaive,
	num_fused_shared_experts=self.num_fused_shared_experts,
)

debug_trace(
	f"Qwen2MoeSparseMoeBlock.__init__: self.experts={type(self.experts)}, "
	f"quant_method={type(getattr(self.experts, 'quant_method', None))}, "
	f"runner={getattr(getattr(self.experts, 'runner', None), 'runner_backend', None)}, "
	f"dispatcher={type(getattr(self.experts, 'dispatcher', None))}"
)
```

你期望看到类似：

```text
quant_config=fp8
moe_impl_class=<class '...FusedMoE'>
self.experts=<class '...FusedMoE'>
quant_method=<class '...Fp8MoEMethod'>
runner=MoeRunnerBackend.TRITON
dispatcher=<class '...StandardDispatcher'>
```

如果你看到：

```text
quant_method=UnquantizedFusedMoEMethod
```

说明这个 expert 被 ignored 了或者 quant_config 没传进去。

如果看到：

```text
runner=FLASHINFER_TRTLLM
runner=FLASHINFER_CUTLASS
runner=DEEP_GEMM
```

说明没有走默认 Triton 路径。

---

## 3. 在 `qwen2_moe.py` 的 forward 确认运行时真的进入 MoE

在 `_forward_router_experts` 里加：

```python
def _forward_router_experts(self, hidden_states: torch.Tensor):
	debug_trace(
		f"Qwen2MoeSparseMoeBlock._forward_router_experts: "
		f"layer_id={self.layer_id}, hidden_states={tuple(hidden_states.shape)}, "
		f"dtype={hidden_states.dtype}, device={hidden_states.device}"
	)

	router_logits, _ = self.gate(hidden_states)
	debug_trace(
		f"Qwen2MoeSparseMoeBlock._forward_router_experts: "
		f"router_logits={tuple(router_logits.shape)}, dtype={router_logits.dtype}"
	)

	topk_output = self.topk(hidden_states, router_logits)
	debug_trace(
		f"Qwen2MoeSparseMoeBlock._forward_router_experts: "
		f"topk_output_type={type(topk_output)}"
	)

	if hasattr(topk_output, "topk_ids"):
		debug_trace(
			f"Qwen2MoeSparseMoeBlock._forward_router_experts: "
			f"topk_ids={tuple(topk_output.topk_ids.shape)}, "
			f"topk_weights={tuple(topk_output.topk_weights.shape)}"
		)

	if self.enable_shared_expert_fusion and TopKOutputChecker.format_is_standard(topk_output):
		topk_output = self._append_shared_to_topk_output(topk_output, hidden_states)

	output = self.experts(hidden_states, topk_output)
	debug_trace(
		f"Qwen2MoeSparseMoeBlock._forward_router_experts: "
		f"experts_output={tuple(output.shape)}, dtype={output.dtype}"
	)
	return output
```

这一步确认：

```text
router gate 是否执行
topk 是否执行
self.experts(...) 是否执行
```

---

## 4. 在 `fused_moe_triton/layer.py` 确认 FusedMoE 运行路径

找到：

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

在 `FusedMoE.__init__` 里，`self.quant_method` 创建之后加：

```python
debug_trace(
	f"FusedMoE.__init__: layer_id={layer_id}, prefix={prefix}, "
	f"quant_config={None if quant_config is None else quant_config.get_name()}, "
	f"quant_method={type(self.quant_method)}, "
	f"use_triton_kernels={getattr(self, 'use_triton_kernels', None)}, "
	f"use_flashinfer_trtllm_moe={getattr(self, 'use_flashinfer_trtllm_moe', None)}, "
	f"enable_flashinfer_cutlass_moe={getattr(self, 'enable_flashinfer_cutlass_moe', None)}"
)
```

在 `self.quant_method.create_moe_runner(...)` 之后加：

```python
debug_trace(
	f"FusedMoE.__init__: layer_id={layer_id}, "
	f"runner={getattr(getattr(self, 'runner', None), 'runner_backend', None)}, "
	f"dispatcher={type(getattr(self, 'dispatcher', None))}, "
	f"num_local_experts={getattr(self, 'num_local_experts', None)}, "
	f"intermediate_size_per_partition={getattr(self, 'intermediate_size_per_partition', None)}"
)
```

然后在 forward 相关函数里加。你本地可能叫 `forward` / `forward_impl` / `run_moe_core`，都可以加：

```python
debug_trace(
	f"FusedMoE.forward: layer_id={self.layer_id}, "
	f"hidden_states={tuple(hidden_states.shape)}, "
	f"topk_output={type(topk_output)}, "
	f"quant_method={type(self.quant_method)}, "
	f"runner={getattr(getattr(self, 'runner', None), 'runner_backend', None)}"
)
```

在 `run_moe_core` 里重点加：

```python
debug_trace(
	f"FusedMoE.run_moe_core: layer_id={self.layer_id}, "
	f"dispatch_output={type(dispatch_output)}, "
	f"hidden_states={tuple(dispatch_output.hidden_states.shape)}, "
	f"quant_method={type(self.quant_method)}"
)

combine_input = self.quant_method.apply(layer=self, dispatch_output=dispatch_output)

debug_trace(
	f"FusedMoE.run_moe_core: layer_id={self.layer_id}, "
	f"combine_input={type(combine_input)}, "
	f"output={tuple(combine_input.hidden_states.shape)}"
)

return combine_input
```

如果你本地 `run_moe_core` 是一行：

```python
return self.quant_method.apply(layer=self, dispatch_output=dispatch_output)
```

就临时拆开。

---

## 5. 在 `quantization/fp8.py` 确认是否进入 Fp8MoEMethod

如果是 Qwen FP8 版，最关键是确认进入：

```text
python/sglang/srt/layers/quantization/fp8.py
	class Fp8MoEMethod
```

在 `Fp8MoEMethod.create_moe_runner` 加：

```python
debug_trace(
	f"Fp8MoEMethod.create_moe_runner: "
	f"moe_runner_backend={get_moe_runner_backend()}, "
	f"block_quant={self.block_quant}, "
	f"use_mxfp8={self.use_mxfp8}, "
	f"weight_block_size={getattr(self.quant_config, 'weight_block_size', None)}"
)
```

在 `Fp8MoEMethod.apply` 或 `forward_cuda` 加：

```python
debug_trace(
	f"Fp8MoEMethod.apply: "
	f"layer_id={getattr(layer, 'layer_id', None)}, "
	f"runner={getattr(getattr(self, 'runner', None), 'runner_backend', None)}, "
	f"x={tuple(dispatch_output.hidden_states.shape)}, "
	f"dtype={dispatch_output.hidden_states.dtype}, "
	f"w13={tuple(layer.w13_weight.shape)}, dtype={layer.w13_weight.dtype}, "
	f"w2={tuple(layer.w2_weight.shape)}, dtype={layer.w2_weight.dtype}"
)
```

然后在每个分支前加明确标记，例如：

```python
debug_trace("Fp8MoEMethod.apply: entering TRITON branch")
```

```python
debug_trace("Fp8MoEMethod.apply: entering FLASHINFER_TRTLLM branch")
```

```python
debug_trace("Fp8MoEMethod.apply: entering DEEP_GEMM branch")
```

```python
debug_trace("Fp8MoEMethod.apply: entering CUTLASS branch")
```

这样你一眼就知道实际分支。

---

## 6. 在 `fused_moe_triton/fused_moe.py` 确认最终 Triton fused MoE

你本地版本应在：

```text
python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
```

或者同目录下类似文件。找：

```python
def fused_experts(...)
def fused_experts_impl(...)
invoke_fused_moe_kernel(...)
silu_and_mul(...)
```

在 `fused_experts` 开头加：

```python
debug_trace(
	f"fused_moe_triton.fused_experts: "
	f"hidden_states={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}, "
	f"w1={tuple(w1.shape)}, dtype={w1.dtype}, "
	f"w2={tuple(w2.shape)}, dtype={w2.dtype}, "
	f"use_fp8_w8a8={use_fp8_w8a8}, "
	f"use_int8_w8a8={use_int8_w8a8}, "
	f"use_int4_w4a16={use_int4_w4a16}, "
	f"topk={topk_ids.shape[1] if hasattr(topk_ids, 'shape') else 'unknown'}"
)
```

在第一次 GEMM 前后加：

```python
debug_trace("fused_moe_triton: before GEMM1 w13")
invoke_fused_moe_kernel(...)
debug_trace(
	f"fused_moe_triton: after GEMM1, "
	f"intermediate_cache1={tuple(intermediate_cache1.shape)}, "
	f"dtype={intermediate_cache1.dtype}"
)
```

在 activation 后加：

```python
debug_trace("fused_moe_triton: before silu_and_mul")
silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
debug_trace(
	f"fused_moe_triton: after silu_and_mul, down_input={tuple(intermediate_cache2.shape)}, "
	f"dtype={intermediate_cache2.dtype}"
)
```

在第二次 GEMM 前加：

```python
debug_trace("fused_moe_triton: before GEMM2 w2")
invoke_fused_moe_kernel(intermediate_cache2, w2, ...)
debug_trace("fused_moe_triton: after GEMM2 w2")
```

如果这些 print 出现，说明你确实走的是 Triton fused MoE；`intermediate_cache2` 就是你要抓的 `down_proj` 输入。

---

## 7. 最小确认方案：只打 6 个点

如果你不想加太多 print，最小打这些：

### 点 1：`qwen2_moe.py`

```python
debug_trace(f"moe_impl_class={get_moe_impl_class(quant_config)}")
```

### 点 2：`qwen2_moe.py` after `self.experts = ...`

```python
debug_trace(
	f"experts={type(self.experts)}, "
	f"quant_method={type(getattr(self.experts, 'quant_method', None))}, "
	f"runner={getattr(getattr(self.experts, 'runner', None), 'runner_backend', None)}"
)
```

### 点 3：`Qwen2MoeSparseMoeBlock._forward_router_experts`

```python
debug_trace("entered Qwen2MoeSparseMoeBlock._forward_router_experts")
```

### 点 4：`FusedMoE.run_moe_core`

```python
debug_trace(f"entered FusedMoE.run_moe_core, quant_method={type(self.quant_method)}")
```

### 点 5：`Fp8MoEMethod.apply`

```python
debug_trace(f"entered Fp8MoEMethod.apply, runner={getattr(self.runner, 'runner_backend', None)}")
```

### 点 6：`fused_moe_triton/fused_moe.py`

```python
debug_trace("entered fused_moe_triton.fused_experts")
```

如果输出顺序是：

```text
moe_impl_class=<class FusedMoE>
experts=<class FusedMoE>, quant_method=<class Fp8MoEMethod>, runner=TRITON
entered Qwen2MoeSparseMoeBlock._forward_router_experts
entered FusedMoE.run_moe_core
entered Fp8MoEMethod.apply
entered fused_moe_triton.fused_experts
```

那你的路径就是：

```text
Qwen2MoeSparseMoeBlock
	-> FusedMoE
	-> Fp8MoEMethod
	-> Triton fused_moe
```

---

## 8. 注意事项

`print` 只能证明 Python wrapper 走到了这里，不能证明 CUDA kernel 已执行完成。
如果你要确认某个 tensor 已经写完，再取数前加：

```python
torch.cuda.synchronize()
```

但不要长期保留，会严重拖慢推理。

另外不要这样写：

```python
tensor = tensor.cpu()
tensor = tensor.float()
```

然后继续把 `tensor` 传回原路径。安全写法是：

```python
tensor_to_save = tensor.detach().float().cpu()
```

不要覆盖原变量，不要 in-place 改 `hidden_states / topk_ids / topk_weights / intermediate_cache2`。

