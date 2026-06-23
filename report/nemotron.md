## 核心判断

HF 的 NVFP4 checkpoint 在 vLLM、SGLang 原生实现中，模型入口都是 `NemotronHForCausalLM`。不要优先修改 Hugging Face Transformers 的 `modeling_nemotron_h.py`：

* **QKV、普通 MLP、shared expert 的 Linear**：模型层 → parallel linear → `quant_method.apply()` → NVFP4 activation quant → CUTLASS FP4 GEMM。
* **routed experts**：模型层 → `FusedMoE`/可选 `SharedFusedMoE` → NVFP4 MoE quant method/runner → token permutation → W13 grouped GEMM → activation → W2 grouped GEMM → reduce。
* `shared_fused_moe.py` 只是可能的一层包装，不是所有配置下最终执行矩阵乘的位置。真正的 expert GEMM 通常继续下沉到 CUTLASS、FlashInfer 或 TensorRT-LLM MoE runner。

我把本次检查使用的 vLLM、SGLang commit、源码行号、直接调用函数和 GitHub 定位链接抽取成了两份报告：

* [精简调用符号索引](sandbox:/mnt/data/nemotron_trace/resolved_map.md)
* [完整源码调用链与代码片段](sandbox:/mnt/data/nemotron_trace/nemotron_ultra_nvfp4_callgraph.md)

下面是最值得关注的稳定调用关系。

---

# 1. vLLM 调用链

## 1.1 QKV projection

模型实现位置：

```text
vllm/model_executor/models/nemotron_h.py
```

主要调用链：

```text
NemotronHForCausalLM.forward
  └─ NemotronHModel.forward
      └─ NemotronHDecoderLayer.forward
          └─ NemotronHAttention.forward
              └─ self.qkv_proj(hidden_states)
                  └─ QKVParallelLinear.__call__
                      └─ ColumnParallelLinear.forward
                          └─ self.quant_method.apply(
                                 layer=self,
                                 x=input_parallel,
                                 bias=bias
                             )
                              └─ ModelOpt NVFP4 LinearMethod.apply
                                  ├─ scaled_fp4_quant(x)
                                  └─ cutlass_scaled_fp4_mm(...)
                                      └─ torch.ops._C.<FP4 CUTLASS op>
                                          └─ CUTLASS Blackwell FP4 GEMM
```

通用 Linear 包装层位置：

```text
vllm/model_executor/layers/linear.py
```

关键点是：

* `QKVParallelLinear` 通常不实现自己的矩阵乘；
* 它继承 `ColumnParallelLinear.forward`；
* 真正选择 BF16、FP8、NVFP4 等实现的是：

```python
self.quant_method.apply(self, input_parallel, bias)
```

因此有三个不同粒度的插桩位置。

### 抓融合 QKV 的输入、输出

在：

```text
NemotronHAttention.forward
```

中的：

```python
qkv, _ = self.qkv_proj(hidden_states)
```

前后插入。

此处拿到的是：

```text
输入：hidden_states，通常已经 flatten 为 [num_tokens, hidden_size]
输出：融合后的 local QKV shard
```

### 分别抓 Q、K、V

需要放在 `qkv_proj` 之后、QKV split 之后：

```python
q, k, v = qkv.split(...)
```

只在 `QKVParallelLinear.forward` 插桩，拿到的仍然是拼接后的 QKV，不能直接区分三者。

### 抓 NVFP4 GEMM 的原始参数

在 ModelOpt/NVFP4 的：

```text
*LinearMethod.apply
```

插入。这里通常可以拿到：

```text
x                  原始 BF16/FP16 activation
x_fp4              动态量化后的 packed FP4 activation
x_block_scale      activation block scale
weight             packed NVFP4 weight
weight_scale       weight block scale
weight_global_scale
output             GEMM 输出
```

具体类名在不同 vLLM revision 中可能是：

```text
ModelOptNvFp4LinearMethod
ModelOptFp4LinearMethod
```

不要只依赖类名，最可靠的定位方法是搜索：

```bash
rg -n "class .*Fp4.*LinearMethod|class .*NvFp4.*LinearMethod" \
  vllm/model_executor/layers/quantization

rg -n "scaled_fp4_quant|cutlass_scaled.*fp4.*mm|cutlass_scaled_mm" \
  vllm vllm/_custom_ops.py csrc
```

精确到本次源码 revision 的类名和行号已在报告中列出。

---

## 1.2 普通 MLP / shared expert

模型级调用通常是：

```text
NemotronHMLP.forward
  ├─ self.gate_up_proj(hidden_states)
  │   └─ MergedColumnParallelLinear
  │       └─ ColumnParallelLinear.forward
  │           └─ quant_method.apply
  │               └─ NVFP4 W13/Gate-Up GEMM
  │
  ├─ activation/gating
  │
  └─ self.down_proj(intermediate_states)
      └─ RowParallelLinear.forward
          └─ quant_method.apply
              └─ NVFP4 W2/Down GEMM
```

对应位置：

```text
vllm/model_executor/models/nemotron_h.py
vllm/model_executor/layers/linear.py
vllm/model_executor/layers/quantization/
```

抓数建议：

| 目标                   | 插入点                         |
| -------------------- | --------------------------- |
| MLP 原始输入             | `NemotronHMLP.forward` 入口   |
| Gate、Up 融合输出         | `gate_up_proj(...)` 之后      |
| 激活函数前后               | `act_fn(...)` 前后            |
| Down projection 输入   | `down_proj(...)` 之前         |
| packed FP4 权重和 scale | NVFP4 `LinearMethod.apply`  |
| CUTLASS 实际参数         | `cutlass_scaled_fp4_mm` 调用前 |

需要注意：普通 MLP 的 gate/up 通常融合为一个 `MergedColumnParallelLinear`，因此权重一般也是融合、分片并打包的。

---

## 1.3 Routed MoE

模型级入口仍在：

```text
vllm/model_executor/models/nemotron_h.py
```

总体调用链：

```text
NemotronHMoE.forward
  ├─ router/gate(hidden_states)
  │   └─ router_logits
  │
  └─ self.experts(hidden_states, router_logits)
      └─ FusedMoE.forward
          └─ FusedMoE.forward_impl / dispatch
              └─ self.quant_method.apply(...)
                  └─ NVFP4 FusedMoE method
                      ├─ top-k / expert routing
                      ├─ token permutation / expert grouping
                      ├─ grouped GEMM W13
                      ├─ SiLU × gate
                      ├─ grouped GEMM W2
                      ├─ top-k weight multiplication
                      └─ unpermute / reduce
```

相关目录主要是：

```text
vllm/model_executor/layers/fused_moe/layer.py
vllm/model_executor/layers/fused_moe/shared_fused_moe.py
vllm/model_executor/layers/fused_moe/
vllm/model_executor/layers/quantization/
```

### `shared_fused_moe.py` 应该怎么理解

它可能负责：

```text
routed experts 输出
    +
shared expert 输出
    × shared gate
```

或者把两条路径交给更底层 runner 做一定程度的融合。

但是它通常不是最终的矩阵乘实现。继续往下追会进入：

```text
FusedMoE quant_method.apply
  → MoE backend/runner
    → CUTLASS grouped GEMM
       或 FlashInfer fused MoE
       或 TensorRT-LLM MoE kernel
```

所以只在 `SharedFusedMoE.forward` 抓数据，适合抓：

```text
MoE 总输入
shared/routed 两支输出
合并前后的结果
```

不适合抓：

```text
每个 expert 的实际输入 token
W13/W2 的 packed FP4 operands
expert-local block scales
排序后的 token offset
```

这些数据应该在 fused-MoE runner、quant method 或最后的 grouped-GEMM wrapper 前抓。

---

# 2. SGLang 调用链

## 2.1 QKV projection

模型文件：

```text
python/sglang/srt/models/nemotron_h.py
```

整体关系与 vLLM 类似：

```text
NemotronHForCausalLM.forward
  └─ NemotronHModel.forward
      └─ NemotronHDecoderLayer.forward
          └─ NemotronHAttention.forward
              └─ self.qkv_proj(hidden_states)
                  └─ QKVParallelLinear
                      └─ ColumnParallelLinear.forward
                          └─ quant_method.apply
                              └─ ModelOpt/NVFP4 LinearMethod.apply
                                  ├─ scaled_fp4_quant
                                  └─ CUTLASS scaled FP4 GEMM
                                      └─ sgl_kernel / torch.ops.sgl_kernel
                                          └─ CUDA CUTLASS kernel
```

通用 Linear 位置：

```text
python/sglang/srt/layers/linear.py
```

量化实现位于：

```text
python/sglang/srt/layers/quantization/
```

不同 SGLang revision 可能把 ModelOpt NVFP4 实现放在：

```text
modelopt_quant.py
modelopt.py
nvfp4.py
fp4.py
```

最可靠的搜索方式是：

```bash
rg -n "class .*Fp4.*LinearMethod|class .*NvFp4.*LinearMethod" \
  python/sglang/srt/layers/quantization

rg -n "scaled_fp4_quant|cutlass_scaled.*fp4.*mm|cutlass_scaled_mm" \
  python/sglang sgl-kernel
```

SGLang 的 Python 可见底层一般是：

```text
sgl_kernel.scaled_fp4_quant
sgl_kernel.cutlass_scaled_fp4_mm
```

或者版本中命名更通用的：

```text
sgl_kernel.cutlass_scaled_mm
```

再向下才是：

```text
torch.ops.sgl_kernel.<op>
  → sgl-kernel C++ binding
    → CUDA/CUTLASS kernel
```

---

## 2.2 MLP

```text
NemotronHMLP.forward
  ├─ gate_up_proj
  │   └─ MergedColumnParallelLinear.forward
  │       └─ quant_method.apply
  │           └─ FP4 GEMM
  ├─ gated activation
  └─ down_proj
      └─ RowParallelLinear.forward
          └─ quant_method.apply
              └─ FP4 GEMM
```

最实用的 SGLang 插桩点和 vLLM 相同：

```text
模型 forward：用于区分语义层，例如 QKV、gate/up、down
quant method：用于统一获取 packed weight、scale、activation
sgl_kernel wrapper：用于获取真正传入 CUDA op 的参数
```

---

## 2.3 SGLang MoE

主要代码会在：

```text
python/sglang/srt/layers/moe/
```

内部再根据版本和启动参数分流到类似：

```text
fused_moe_triton/
moe_runner/
flashinfer/
cutlass/
trtllm/
```

调用关系可以概括为：

```text
NemotronHMoE.forward
  ├─ router logits
  └─ FusedMoE.forward / SharedFusedMoE.forward
      └─ forward_impl / runner.run / quant_method.apply
          ├─ select experts
          ├─ sort tokens by expert
          ├─ W13 grouped GEMM
          ├─ activation
          ├─ W2 grouped GEMM
          └─ weighted reduce
```

对 NVFP4 来说，不应该默认最终一定进入 Triton 文件。packed FP4 expert GEMM 可能切换到 CUTLASS、FlashInfer 或 TensorRT-LLM runner。应以实际构造出的 runner 类型为准。

---

# 3. 最推荐的插桩层级

## 3.1 只关心 QKV 数值

直接在模型文件中：

```python
# NemotronHAttention.forward

qkv_input = hidden_states

qkv, _ = self.qkv_proj(hidden_states)

# 如需分别抓 Q/K/V，在 split 后抓
q, k, v = qkv.split(...)
```

这是最容易理解的数据，但看不到 FP4 packed operand 和 scale。

---

## 3.2 关心所有 NVFP4 Linear 的真实输入

在 NVFP4 `LinearMethod.apply` 中：

```python
def apply(self, layer, x, bias=None):
    dump_tensor("input", x)
    dump_tensor("packed_weight", layer.weight)

    for attr in (
        "weight_scale",
        "weight_scale_2",
        "weight_block_scale",
        "weight_global_scale",
        "input_scale",
    ):
        value = getattr(layer, attr, None)
        if value is not None:
            dump_tensor(attr, value)

    output = original_nvfp4_path(...)

    dump_tensor("output", output)
    return output
```

这一层会同时覆盖：

```text
QKV projection
attention output projection
gate/up projection
down projection
router linear
shared expert linear
LM head（取决于其量化配置）
```

因此应通过 module prefix、weight shape 或调用者信息过滤，否则抓数规模会非常大。

---

## 3.3 关心每个 expert 的矩阵乘

至少抓两层。

第一层放在 router/top-k 之后：

```text
router_logits
topk_ids
topk_weights
```

第二层放在 grouped GEMM 前：

```text
sorted_token_ids
expert_ids / expert_offsets
permuted_input
W13 packed weight
W13 block scale
W2 packed weight
W2 block scale
```

仅在 `NemotronHMoE.forward` 抓 `hidden_states`，无法知道每个 expert 实际拿到了哪些 token。

---

## 3.4 关心最终 CUDA op 的参数

Dense Linear 的最后 Python 插桩点是：

```text
cutlass_scaled_fp4_mm(...)
```

或：

```text
cutlass_scaled_mm(... FP4 arguments ...)
```

MoE 的最后 Python 插桩点则是所选 backend 的：

```text
cutlass_fused_moe(...)
flashinfer_*fused_moe(...)
trtllm_*moe(...)
```

这里拿到的数据已经经过：

```text
tensor parallel 分片
expert parallel 本地 expert 映射
token permutation
FP4 packing
block-scale layout 转换
```

它最接近真实 kernel，但最不方便和原始 token、原始 expert ID 对齐。

---

# 4. 运行时确认实际类和源码位置

框架版本变化较快，最稳妥的方法是在 worker 完成模型加载后运行：

```python
import inspect

def print_relevant_modules(model):
    keywords = (
        "qkv_proj",
        "gate_up_proj",
        "down_proj",
        "experts",
        "shared_expert",
    )

    for name, module in model.named_modules():
        if not any(key in name for key in keywords):
            continue

        module_cls = type(module)
        quant_method = getattr(module, "quant_method", None)

        try:
            module_file = inspect.getsourcefile(module_cls)
        except (TypeError, OSError):
            module_file = None

        print(
            f"\nmodule={name}\n"
            f"  class={module_cls.__module__}.{module_cls.__qualname__}\n"
            f"  source={module_file}"
        )

        if quant_method is not None:
            quant_cls = type(quant_method)
            try:
                quant_file = inspect.getsourcefile(quant_cls)
                _, quant_line = inspect.getsourcelines(quant_cls.apply)
            except (TypeError, OSError, AttributeError):
                quant_file = None
                quant_line = None

            print(
                f"  quant_method="
                f"{quant_cls.__module__}.{quant_cls.__qualname__}\n"
                f"  quant_source={quant_file}:{quant_line}"
            )
```

这段代码能直接告诉你当前安装版本究竟使用：

```text
哪个 QKVParallelLinear
哪个 FP4 LinearMethod
哪个 FusedMoE
哪个 quant method
源码文件和 apply 行号
```

它比根据仓库 `main` 分支猜测更加可靠。

---

# 5. 抓数时容易踩的四个问题

### CUDA Graph

vLLM 建议抓数时开启：

```bash
--enforce-eager
```

SGLang 建议使用：

```bash
--disable-cuda-graph
```

否则 Python 代码可能主要在 graph capture 阶段执行，后续 replay 不会再次经过你的 Python dump 逻辑。

### TP/EP 分片

`QKVParallelLinear` 是 column-parallel。每个 TP rank 拿到的是部分 heads，不是完整 QKV。

MoE 开启 EP 后，每个 rank 只持有部分 experts。保存数据时至少记录：

```text
global rank
TP rank
EP rank
layer id
local expert id
global expert id
request/token id
```

### NVFP4 权重不能直接当普通 Tensor 解读

`layer.weight` 通常是 packed FP4，一个字节包含多个 FP4 元素，并配套 block scale/global scale。直接：

```python
layer.weight.float()
```

不会得到正确的反量化权重。

### Hybrid Mamba 层没有 QKV

NemotronH 是 hybrid Mamba-Transformer。只有 attention block 会进入 `qkv_proj`；Mamba block 使用的是 `in_proj`、SSM/conv 和 `out_proj` 路径。抓所有层时发现部分 layer 从未命中 QKV hook，属于正常情况。
