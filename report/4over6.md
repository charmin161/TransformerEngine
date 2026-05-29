我看下来的结论是：**TE 里的 4over6 不是一个新的 FP4 dtype，也不是 Megatron 层的 feature；它是挂在 `NVFP4BlockScaling` recipe 下的一个“per-block 自适应量化策略”。** 底层核心逻辑是：每个 NVFP4 block 同时尝试 **map-to-6** 和 **map-to-4** 两种 scale，分别量化、反量化并计算误差，然后选择误差更小的那个候选。这个功能是 Transformer Engine PR `#2972` 合入的，PR 名为 “Implement 4over6 NVFP4 recipe”，已在 2026-05-22 合入 main。([GitHub][1])

## 1. TE 中 4over6 放在哪个结构里

TE 把它放在这个层次结构里：

```text
transformer_engine.common.recipe.NVFP4BlockScaling
    └── QParams for input / weight / grad
        └── NVFP4Quantizer
            └── NVFP4Tensor metadata
                └── C++/CUDA quantize path
                    └── quantize_4over6_nvfp4.cuh
```

关键入口在 Python recipe：

```python
transformer_engine/common/recipe/__init__.py
```

`NVFP4BlockScaling` 里新增了这些字段/环境变量：

```python
nvfp4_4over6
nvfp4_4over6_e4m3_use_256
nvfp4_4over6_err_mode
```

对应环境变量是：

```bash
NVTE_NVFP4_4OVER6=none|weights|activations|all
NVTE_NVFP4_4OVER6_E4M3_USE_256=none|weights|activations|all
NVTE_NVFP4_4OVER6_ERR_MODE=MAE|MSE
```

TE 文档里也明确写了：`nvfp4_4over6` 可以按 scope 作用到 weights、activations 或 all；启用后会在 selected FP4 blocks 上比较 map-to-4 和 map-to-6，并选择配置误差更低的结果。文档还特别说明，当前 4over6 目标是 RL 和 post-training 场景，pre-training 中 4over6 + RHT 的组合路径尚未实现。([NVIDIA GitHub][2])

PR 中给出的实现链路更完整，大致是：

```text
NVFP4BlockScaling(nvfp4_4over6="all")
    ↓
quantization.py 根据 tensor_type 决定 input / weight 是否启用 4over6
    ↓
NVFP4Quantizer(
    nvfp4_4over6_mode=MinMAE or MinMSE,
    nvfp4_e4m3_max=256 or 448
)
    ↓
pytorch/csrc/quantizer.cpp
    ↓
common/cast/nvfp4/quantize.cuh
    ↓
common/cast/nvfp4/quantize_4over6_nvfp4.cuh
```

相关文件包括 `quantize_4over6_nvfp4.cuh`、`quantization.py`、`cast.cpp`、`dequantize_nvfp4.cuh`、`nvfp4.cu`、`nvfp4_tensor.py`、`quantizer.cpp`、`quantization_ref_nvfp4.py` 和 C API header。([GitHub][1])

## 2. 4over6 kernel 底层怎么做

核心 CUDA 文件是：

```text
transformer_engine/common/cast/nvfp4/quantize_4over6_nvfp4.cuh
```

TE 的实现不是把 FP4 E2M1 codebook 换掉，而是对每个 1×16 NVFP4 group 同时构造两个候选：

```text
candidate A: map-to-6，也就是标准 NVFP4 scale
candidate B: map-to-4，也就是把 block scale 放大 1.5 倍
```

为什么是 1.5？因为 E2M1 FP4 的最大有效值是 6。标准 NVFP4 会让 block amax 约映射到 FP4 的 6；而 map-to-4 则把 E4M3 block scale 扩大 1.5 倍，使同样的动态范围落到 FP4 value 4 附近。这样做可以避开 FP4 高端区间较粗的量化间隔，对某些 block 的误差更小。TE kernel 注释里直接说明：map-to-6 使用 normal scale，map-to-4 使用 expanded E4M3 scale；两个候选都在寄存器中量化、反量化并计算误差，tie 时选择 map-to-6。([GitHub][3])

kernel 中的 scale 逻辑可以概括为：

```text
fp4_max = 6
map6_scale = base_scale
map4_scale = min(base_scale * 1.5, e4m3_max)
```

其中 `e4m3_max` 可以是 448，也可以是 256。TE 通过 `NVTE_NVFP4_4OVER6_E4M3_USE_256` 控制某些 scope 是否用 256；默认在 4over6 中偏向使用 256。kernel 会针对 MAE 或 MSE 两种误差模式计算候选误差。([GitHub][3])

简化后的 per-block 伪代码是：

```python
for each 1x16 block:
    amax = max(abs(x))

    scale6 = base_scale(amax)          # standard NVFP4, map amax -> 6
    scale4 = min(scale6 * 1.5, fp8max) # 4over6 candidate, map amax -> 4

    q6 = quantize_e2m1(x, scale6)
    q4 = quantize_e2m1(x, scale4)

    x6 = dequantize_e2m1(q6, scale6)
    x4 = dequantize_e2m1(q4, scale4)

    err6 = MAE_or_MSE(x, x6)
    err4 = MAE_or_MSE(x, x4)

    if err4 < err6:
        write(q4, scale4)
    else:
        write(q6, scale6)
```

实际 CUDA kernel 更复杂，因为它同时处理 rowwise / columnwise 输出、2D 16×16 weight block scaling、shared memory staging、warp-level reduction 和 Blackwell FP4 PTX 指令。TE 的 4over6 path 使用 Blackwell FP4 convert 指令，例如 `cvt.rn.satfinite.e2m1x2.f32`，并且在非 SM100+ 架构上会报错。([GitHub][3])

几个重要限制也在代码和 PR 中体现出来：

```text
1. 需要 Blackwell / SM100+ FP4 指令。
2. 4over6 path 当前不支持 stochastic rounding。
3. fprop RHT + 4over6 当前不支持。
4. grouped tensors 组合会被拒绝。
5. 2D quantization / columnwise path 对 rows、cols 有 16 对齐要求。
6. row-scaled activation 不支持 2D 或 columnwise 组合。
```

PR 明确写了当前 4over6 kernel 是单独 kernel，没有直接融合进已有 quantization kernel；原因是它要做 FP32 dequantization error 计算，因此当前偏 compute-bound。([GitHub][1])

## 3. TE 里有没有 post-training / RL example

**公开 TE 仓库里我没有看到独立的 post-training 或 RL training example。** 公开能确认的 TE 内部用法主要在测试里，而不是 `examples/` 目录。

最接近的公开用法在：

```text
tests/pytorch/test_numerics.py
```

里面有一个 `nvfp4_4over6()` recipe fixture，形态大致是：

```python
from transformer_engine.common import recipe

def nvfp4_4over6():
    nvfp4_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        nvfp4_4over6="all",
    )
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(fp4_2d_quantization=True)
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
    return nvfp4_recipe
```

这个测试 recipe 会被放进 TE numerics test 的 recipe 列表里，然后配合 TE module/autocast 路径测试。也就是说，**TE 公开 repo 里展示的是“怎么构造 recipe”，不是完整的 RLHF / GRPO / SFT / PTQ 示例脚本。**([GitHub][4])

PR 里提到过 “Functionality has been verified by internal RL experiments”，但这指的是 NVIDIA 内部 RL 实验验证，不是公开 example。([GitHub][1])

## 4. post-training 公开 example 放在哪里

公开的 post-training quantization 示例主要不在 TE，而是在 MIT Han Lab 的 FourOverSix 仓库中。这个仓库说明它包含 4/6 和 IF 数据类型的 block-scaled quantization / matmul kernels，以及 post-training quantization experiments。([GitHub][5])

它的结构大致是：

```text
fouroversix/
    scripts/
        ptq.py        # 新版 vLLM-based PTQ setup
        ptq_hf.py     # 旧版 HuggingFace PTQ setup
    src/fouroversix/
        quantize_model(...)
        quantize(...)
    kernels / backends
        CUDA / Triton / PyTorch reference
```

README 里给出的 PTQ 命令示例包括：

```bash
# vLLM-based PTQ
python -m scripts.ptq \
  --model Qwen/Qwen3.5-35B-A3B \
  --quantization-scheme if4 \
  --tasks arc_easy
```

以及旧版 HuggingFace PTQ：

```bash
# RTN + 4/6
python -m scripts.ptq_hf \
  --model-name meta-llama/Llama-3.2-1B \
  --ptq-method rtn \
  --task wikitext

# AWQ + 4/6
python -m scripts.ptq_hf \
  --model-name meta-llama/Llama-3.2-1B \
  --ptq-method awq \
  --task wikitext
```

它还提供 Python API：

```python
from fouroversix import quantize_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
quantize_model(model)  # NVFP4 using 4/6 with MSE block selection
```

所以如果你问的是**后训练量化 / PTQ**，目前公开 example 更像是放在 `mit-han-lab/fouroversix` 的 `scripts.ptq` / `scripts.ptq_hf` / `quantize_model` 结构中，而不是 TE 的 `examples/` 里。([GitHub][5])

## 5. RL 相关公开集成在哪里

RL 方向公开能看到的更明确集成不在 TE 本体，而是在 FlashInfer 的 fused MoE / quantizer 路径里。FlashInfer PR `#3264` 名为 “Support 4over6 nvfp4 for quantizer and fused MoE”，在 2026-05-21 合入 main。PR 说明里提到它实现了 4over6 NVFP4 quantizer 和 fused MoE 支持，并引用了 TE 的 4over6 PR。([GitHub][6])

FlashInfer 里的结构大致是：

```text
csrc/nv_internal/cpp/common/envUtils.cpp
csrc/nv_internal/tensorrt_llm/common/envUtils.h
csrc/nv_internal/include/tensorrt_llm/common/quantization.h
csrc/nv_internal/cpp/kernels/quantization.cu
csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh
csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh
csrc/trtllm_fused_moe_runner.cu
```

它支持的路径包括：

```text
cutlass_fused_moe
trtllm_fp4_block_scale_routed_moe
trtllm_fp4_block_scale_moe
```

FlashInfer 的 4over6 env 不是 `NVTE_...`，而是自己的：

```bash
FLASHINFER_NVFP4_4OVER6
FLASHINFER_NVFP4_4OVER6_E4M3_USE_256
FLASHINFER_NVFP4_4OVER6_ERR_MODE
TRTLLM_DISABLE_FP4_QUANT_FAST_MATH
```

PR 还强调它希望 FlashInfer quantization backend 与 TE-style implementation 在 RL use cases 中 bitwise 对齐。([GitHub][6])

所以如果你的 RL workload 是 **Megatron/TE 训练主干**，4over6 入口是 TE recipe/env；如果你的 RL workload 涉及 **rollout / inference / fused MoE / TRT-LLM style backend**，公开实现更可能落在 FlashInfer 的 fused MoE 和 quantization kernel 结构里。

## 6. 对你当前 Megatron + TE 用法的实际含义

你之前的入口：

```bash
--fp4-format e2m1 \
--fp4-recipe nvfp4 \
--transformer-impl transformer_engine
```

仍然是进入 TE NVFP4 的正确入口。4over6 不需要新的 `fp4-format`，因为底层还是 E2M1 NVFP4；变化发生在 block scale 的选择策略上。

如果只是想最小风险启用 weights 的 4over6：

```bash
export NVTE_NVFP4_4OVER6=weights
export NVTE_NVFP4_4OVER6_E4M3_USE_256=weights
export NVTE_NVFP4_4OVER6_ERR_MODE=MAE

torchrun ... \
  --transformer-impl transformer_engine \
  --fp4-format e2m1 \
  --fp4-recipe nvfp4 \
  ...
```

如果你想仿照 TE test 那种 “all” 的方式，需要同时规避 RHT / stochastic rounding 组合：

```bash
export NVTE_NVFP4_4OVER6=all
export NVTE_NVFP4_4OVER6_E4M3_USE_256=all
export NVTE_NVFP4_4OVER6_ERR_MODE=MAE
export NVTE_NVFP4_DISABLE_RHT=1
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1
```

但我不建议你一开始直接用 `all` 跑大规模训练。TE 文档已经说明 4over6 + RHT 的 pre-training path 尚未实现；PR 也说当前 unsupported combos 会被拒绝。更稳的顺序是：

```text
第一步：NVTE_NVFP4_4OVER6=weights
第二步：验证 loss / numerics / throughput
第三步：再尝试 activations 或 all，并显式关闭 RHT
```

一句话总结：**TE 的 4over6 底层是 `NVFP4BlockScaling -> NVFP4Quantizer -> quantize_4over6_nvfp4.cuh` 的 per-block candidate-selection kernel；TE 本体公开的是 recipe/test fixture，没有完整 post-training 或 RL example；PTQ example 在 MIT Han Lab 的 `fouroversix` repo，RL/fused-MoE 相关公开集成主要在 FlashInfer。**

[1]: https://github.com/NVIDIA/TransformerEngine/pull/2972 "Implement 4over6 NVFP4 recipe by zianglih · Pull Request #2972 · NVIDIA/TransformerEngine · GitHub"
[2]: https://nvidia.github.io/TransformerEngine/api/common.html "Common API — Transformer Engine 2.17.0.dev0 documentation"
[3]: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/cast/nvfp4/quantize_4over6_nvfp4.cuh "TransformerEngine/transformer_engine/common/cast/nvfp4/quantize_4over6_nvfp4.cuh at main · NVIDIA/TransformerEngine · GitHub"
[4]: https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/test_numerics.py?utm_source=chatgpt.com "TransformerEngine/tests/pytorch/test_numerics.py at main"
[5]: https://github.com/mit-han-lab/fouroversix "GitHub - mit-han-lab/fouroversix: Code for the papers: “Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling” and “Adaptive Block-Scaled Data Types” · GitHub"
[6]: https://github.com/flashinfer-ai/flashinfer/pull/3264 "Support 4over6 nvfp4 for quantizer and fused MoE by zianglih · Pull Request #3264 · flashinfer-ai/flashinfer · GitHub"
