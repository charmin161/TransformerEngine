有，而且 TE 里能找到非常明确的 **dequantize → quantize** 代码。但需要强调：**有这类通用/特殊路径，不等于 Megatron 默认 MXFP8 Linear 的 A/W 转置表示就是这样生成的。默认 Linear 路径不是。**

### 1. MXFP8Tensor 自己就有明确的“反量化再量化”

位置：

```text
transformer_engine/pytorch/tensor/mxfp8_tensor.py
```

`MXFP8Tensor.quantize_()` 中直接写了：

```python
if isinstance(tensor, QuantizedTensor):
    return self.quantize_(tensor.dequantize())

return super().quantize_(tensor, noop_flag=noop_flag)
```

也就是：

```text
QuantizedTensor
      ↓
 dequantize()
      ↓
BF16/FP32 等高精度 Tensor
      ↓
 quantize_
      ↓
MXFP8Tensor
```

这是字面意义上的反量化再量化。

NVFP4、Float8Blockwise 也有几乎一样的逻辑。

不过这个更像是 **通用 Tensor conversion API**：

```text
“我拿一个已经量化的 tensor，
现在要求把它写入另一个 MXFP8 tensor”
```

TE 为了保证语义正确，会走：

$$
Q_1 \rightarrow FP \rightarrow Q_2
$$

而不是假设两个 quantized representation 可以直接转换。

---

## 2. 更有意思：TE 甚至专门支持“rowwise 反量化后，再做 columnwise 量化”

这个跟我们前面讨论的问题非常接近。

位置：

```text
transformer_engine/pytorch/tensor/hybrid_tensor.py
```

现在 TE 有一个：

```python
HybridQuantizer
```

其中参数：

```python
columnwise_source
```

可以选择：

```text
"original"
"rowwise_dequantized"
```

源码文档明确写：

```text
columnwise_source="original"

原始 tensor
 ├─> rowwise quantization
 └─> columnwise quantization
```

而：

```text
columnwise_source="rowwise_dequantized"
```

则是：

```text
original
   ↓
rowwise quantize
   ↓
rowwise quantized
   ↓
dequantize
   ↓
high precision reconstructed value
   ↓
columnwise quantize
```

源码甚至明确描述：

> `"rowwise_dequantized"` quantizes rowwise first, dequantizes the rowwise result, then uses that value as the columnwise source.

实际实现也很直接：

```python
def _columnwise_src_from_rowwise(
    self,
    tensor,
    rowwise_result,
):
    if rowwise_result is None:
        rowwise_result = self.rowwise_quantizer.quantize(tensor)

    return rowwise_result.dequantize(dtype=tensor.dtype)
```

然后：

```python
columnwise_src = self._columnwise_src_from_rowwise(
    tensor,
    rowwise_result
)

columnwise_result = (
    self.columnwise_quantizer.quantize(columnwise_src)
)
```

所以这里就是完整的：

$$
X
\xrightarrow{Q_{row}}
X_{row}^{Q}
\xrightarrow{DQ}
\hat X
\xrightarrow{Q_{col}}
X_{col}^{Q}
$$

而且 TE 自己把这种东西称为类似 **double-quantization** 的用途。

---

# 3. 为什么 TE 要提供这种 `rowwise_dequantized`？

这是为了支持 **Hybrid / Custom Recipe**。

例如你可能想规定：

```text
Forward:
    weight rowwise = MXFP8

Backward:
    weight columnwise = NVFP4
```

或者：

```text
Forward:
    rowwise = MXFP8

Backward:
    columnwise = high precision
```

TE 给出的例子就是：

```python
HybridQuantizer(
    rowwise_quantizer=mxfp8_quantizer,
    columnwise_quantizer=IdentityQuantizer(),
    columnwise_source="rowwise_dequantized",
)
```

也就是说 backward 看到的高精度数值，并不是原始 BF16：

$$
W
$$

而是：

$$
DQ(Q_{\rm MXFP8}(W))
$$

因此 forward 的量化误差也会体现在 backward representation 中。

TE 的源码文档明确给出了这个例子。

---

# 4. distributed.py 里也有非常明确的 dequantize → quantizer

还有一个你做 Megatron 时值得关注的地方：

```text
transformer_engine/pytorch/distributed.py
```

里面直接出现：

```python
inp = quantizer(
    inp.dequantize(dtype=dtype)
)
```

即：

```text
quantized inp
      ↓
dequantize
      ↓
high precision
      ↓
quantizer(...)
      ↓
新的 quantized representation
```

这是分布式/communication 相关路径里用于重新构建适当 quantized representation 的 fallback。代码本身非常明确。

因此如果你后面研究：

```text
TP
Sequence Parallel
AllGather
FSDP
UserBuffer
```

不要简单认为 TE 永远不 dequant/requant。

**确实存在这种 fallback。**

---

# 5. 但是，回到我们最关心的 Megatron 默认 MXFP8 Linear

这里一定要区分清楚。

普通：

```text
Megatron
+
TE MXFP8BlockScaling
+
Linear
```

对于 A：

```text
A BF16
  │
  ├─────────────┐
  ↓             ↓
Qrow(A)       Qcol(A)
```

而不是：

```text
A BF16
  ↓
Qrow(A)
  ↓
DQ
  ↓
Qcol(A)
```

这两种的数值含义完全不同。

标准 MXFP8 Linear 的设计就是尽量使用前一种。

---

## 一个非常强的代码证据

看：

```text
transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py
```

它的：

```python
update_usage()
```

如果你已经有一个只包含：

```text
rowwise_data
```

的 MXFP8Tensor，然后突然要求：

```python
update_usage(columnwise_usage=True)
```

TE **不会**帮你：

```text
rowwise
 → dequantize
 → columnwise quantize
```

而是直接报错：

```text
Requested column-wise usage,
but MXFP8Tensor is missing column-scaled FP8 data
```

这非常重要。

说明对于标准 `MXFP8TensorStorage`：

> **columnwise representation 必须在量化原始输入时就产生，不能事后由 rowwise representation lazy reconstruct。**

---

# 6. 所以你可以把 TE 中的行为分成三类

| 场景                            | 数据流                     | 默认 Megatron MXFP8 Linear？                              |
| ----------------------------- | ----------------------- | ------------------------------------------------------ |
| 正常 MXFP8 row+col              | `BF16 → Qrow + Qcol`    | **是**                                                  |
| 保存 BF16 后 backward 重量化        | `BF16(saved) → Qcol`    | 某些配置会                                                  |
| Quantized → dequant → requant | `Qrow → BF16 → Qcol/Q2` | **正常 Linear 不这么做**，但 Hybrid/distributed/API fallback 有 |

所以我们前面说：

> “TE 中没有反量化再量化”

如果理解为整个代码库，这是**不准确的**。

准确说法应该是：

> **TE 代码库明确存在 dequantize→requantize 路径，甚至 HybridQuantizer 专门支持 `rowwise_dequantized`。但 Megatron 默认 `MXFP8BlockScaling` 的普通 Linear Forward/Dgrad/Wgrad 不依赖 `Qrow → dequantize → Qcol` 来生成转置方向数据；A/W 在需要两个方向时，会从原始高精度输入直接生成 rowwise 和 columnwise 两套 MXFP8 representation。**

---

## 7. 如果你现在就是想验证“我的 Megatron 实际跑起来有没有走 dequant→requant”

我建议你直接盯三个地方：

```text
① transformer_engine/pytorch/tensor/mxfp8_tensor.py

MXFP8Tensor.quantize_()
```

在：

```python
if isinstance(tensor, QuantizedTensor):
```

里面打 log。

然后：

```text
② transformer_engine/pytorch/tensor/hybrid_tensor.py

_columnwise_src_from_rowwise()
```

这里打 log。

最后：

```text
③ transformer_engine/pytorch/distributed.py
```

在：

```python
inp = quantizer(inp.dequantize(...))
```

这里打 log。

而正常 MXFP8 cast：

```text
④ transformer_engine/pytorch/tensor/mxfp8_tensor.py

MXFP8Quantizer.quantize_impl()
    -> tex.quantize()
```

也打一个 log。

这样你跑一个 Megatron iteration 后，就能非常明确地区分：

```text
BF16 → MXFP8(row+col)
```

和：

```text
MXFP8 → BF16 → MXFP8
```

到底各发生了几次。

**如果你现在研究的目标就是标准 Megatron MXFP8 训练，我认为下一步最值得追的是 `tex.quantize()` 到 C++/CUDA 后，`rowwise=True,columnwise=True` 时是否真的是一个 fused kernel 一次读取 BF16 同时产出两套数据，还是内部实际 launch 两个 quantization kernel。** 这会进一步回答“两套表示的额外带宽和时间开销到底多大”。
