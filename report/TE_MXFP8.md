可以把 Transformer Engine 里的 MXFP8 训练先抓住一个核心：

> **MXFP8 不是“量化一份 FP8，然后反向需要转置时再把这份 FP8 转置”。**
> 对一个高精度矩阵，TE 可以从同一份 BF16/FP16 输入一次性生成两套 MXFP8 表示：`rowwise` 和 `columnwise`。这两套表示的数据字节本身通常不同，因为它们沿不同方向每 32 个元素计算 E8M0 block scale。TE 官方 recipe 代码甚至明确说明：当原矩阵和 transpose 都需要时，要从 high-precision input 分别生成两种表示，以避免 “double quantization error”。

以一个 Linear 为例，定义：

$$
A\in R^{M\times K},\quad W\in R^{N\times K},\quad Y=AW^T
$$

反向梯度记为：

$$
G=\frac{\partial L}{\partial Y}\in R^{M\times N}
$$

那么三个 GEMM 是：

$$
\text{Forward}:Y=AW^T
$$

$$
\text{Dgrad}:\frac{\partial L}{\partial A}=GW
$$

$$
\text{Wgrad}:\frac{\partial L}{\partial W}=G^TA
$$

在当前 TE 的标准 MXFP8 路径中，它们实际上对应：

| GEMM    | 数学形式     | TE `general_gemm`                 | 实际使用的 MXFP8 表示                  | 输出                            |
| ------- | -------- | --------------------------------- | ------------------------------- | ----------------------------- |
| Forward | \(AW^T\) | `general_gemm(W, A)`, `TN`        | `W.rowwise` + `A.rowwise`       | 通常 BF16/FP16                  |
| Dgrad   | \(GW\)   | `general_gemm(W, G, layout="NN")` | `W.columnwise` + `G.rowwise`    | 通常 BF16/FP16                  |
| Wgrad   | \(G^TA\) | `general_gemm(A, G, layout="NT")` | `A.columnwise` + `G.columnwise` | BF16/FP16 或 `main_grad` dtype |

这个映射直接来自 `transformer_engine/pytorch/cpp_extensions/gemm.py`：当 GEMM operand 被 transpose 使用时和不 transpose 使用时，`general_gemm()` 会选择不同的 rowwise/columnwise representation。

---

# 1. 先搞清楚：MXFP8 的“转置数据”到底是什么

这是最容易误解的地方。

假设：

```text
A.shape = [M, K]
```

一个 `MXFP8TensorStorage` 内部可以同时有：

```python
_rowwise_data
_rowwise_scale_inv

_columnwise_data
_columnwise_scale_inv
```

源码：

```text
transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py
```

这四个成员是 MXFP8 tensor 的核心存储。

但注意：

```text
_rowwise_data.shape    = [M, K]
_columnwise_data.shape = [M, K]
```

**`columnwise_data` 并不是 `[K,M]` 的物理 transpose tensor。**

`MXFP8Quantizer.get_columnwise_shape()` 直接返回原始 data shape，而 scale shape 才按照另外一个方向组织。

可以把二者理解为：

```text
BF16 A[M,K]
         │
         ├──── 沿 K 方向，每连续 32 个数求一个 scale
         │
         ▼
   A.rowwise_data
   A.rowwise_scale
         
         │
         └──── 沿 M 方向，每连续 32 个数求一个 scale
              ▼
        A.columnwise_data
        A.columnwise_scale
```

也就是说：

### rowwise MXFP8

对于：

```text
A[m, 0:32]
A[m, 32:64]
...
```

每 32 个元素共用一个 E8M0 scale。

### columnwise MXFP8

对于：

```text
A[0:32, k]
A[32:64, k]
...
```

每 32 个元素共用一个 E8M0 scale。

因此：

```text
Qrow(A)^T  !=  Qcol(A^T)
```

数值上并不等价。

这就是为什么 TE **不能把 `rowwise_data` 简单 transpose 一下作为反向数据**。

MXFP8 recipe 的注释直接说明了这一点：

> scaling direction matters，因此 quantized tensor 和它的 transpose 并不数值等价；如果两个方向都需要，就从原始高精度 input 同时计算。

---

# 2. 一个 MXFP8 tensor 在显存中到底存了什么

Python 层：

```text
transformer_engine/pytorch/tensor/mxfp8_tensor.py
```

`MXFP8Quantizer.inner_tensor_specs()` 会根据：

```python
rowwise_usage
columnwise_usage
```

决定到底分配哪些 buffer。

物理存储的数据都是：

```text
_rowwise_data           : uint8
_columnwise_data        : uint8

_rowwise_scale_inv      : uint8
_columnwise_scale_inv   : uint8
```

其中 data 的 `uint8` bit pattern 被解释成 FP8，而 scale 的 `uint8` bit pattern 被解释成 E8M0。MXFP8 block size 固定为 32。

对于 `[M,K]`，忽略 padding：

$$
\text{rowwise data}=MK\text{ Bytes}
$$

$$
\text{rowwise scale}\approx\frac{MK}{32}\text{ Bytes}
$$

所以一套方向大约是：

$$
1+\frac1{32}=1.03125\ \text{Bytes/value}
$$

两套：

$$
2.0625\ \text{Bytes/value}
$$

这也是后面分析显存时非常重要的数字。

实际 scale buffer 还有 alignment。源码中：

```python
rowwise scale:
[round_up(M,128), round_up(K/32,4)]

columnwise scale:
[round_up(M/32,4), round_up(K,128)]
```

所以小矩阵时 padding 占比可能更明显。

---

# 3. 关键问题：两套量化数据是一次生成，还是先量化再反量化？

对于标准 MXFP8 Linear：

**一次从 BF16/FP16 输入生成。**

不是：

```text
BF16
 ↓
MXFP8-row
 ↓ dequant
BF16
 ↓ quant
MXFP8-col
```

而是：

```text
                 ┌─> MXFP8 rowwise
BF16 / FP16 ─────┤
                 └─> MXFP8 columnwise
```

甚至 CUDA kernel 有专门的 bidirectional/bidimensional quantize 路径。

核心位置：

```text
transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh
```

quantizer 会检查：

```cpp
output->has_data()
output->has_columnwise_data()
```

然后选择：

```text
ROWWISE
COLWISE
BIDIMENSIONAL
```

如果两种 usage 都开启，就直接启动同时输出 rowwise + columnwise 的 kernel。

Python：

```python
input_quantizer(input)
```

最终进入：

```python
tex.quantize(...)
```

然后：

```text
transformer_engine/pytorch/csrc/quantizer.cpp
```

进入：

```cpp
MXFP8Quantizer::quantize(...)
    -> nvte_quantize_v2(...)
```



这就是整个 MXFP8 数据转换最底层的主链路。

---

# 4. Forward：A 和 W 到底发生了什么

先看普通 Linear，没有特殊 FSDP/TP communication case。

源码主入口：

```text
transformer_engine/pytorch/module/linear.py
```

## A 的量化

Forward 一开始输入还是：

```text
A : BF16 / FP16
```

TE 知道：

```text
Forward 需要 A-row
Wgrad   需要 A-col
```

所以正常训练、并且 Wgrad 需要 A 时，会设置：

```python
input_quantizer.set_usage(
    rowwise=True,
    columnwise=True,
)
```

然后：

```python
inputmat = input_quantizer(inputmat)
```

这一次 quantize 就得到：

```text
A_row_data
A_row_scale

A_col_data
A_col_scale
```

对应 `linear.py` Forward 的 input prepare 部分。

---

## W 的量化

W 原来通常是：

```text
W : BF16 / FP16 parameter
```

Forward 需要：

```text
W-row
```

Dgrad 需要：

```text
W-col
```

所以正常情况下：

```python
weight_quantizer.set_usage(
    rowwise=True,
    columnwise=True,
)
```

然后进入：

```python
quantize_weight(...)
```

得到：

```text
W_row
W_col
```



---

# 5. Forward GEMM 实际吃的是哪两份

源码：

```text
transformer_engine/pytorch/module/linear.py
```

代码旁边已经直接写了：

```python
# y = x * w^T
general_gemm(
    weightmat,
    inputmat_total,
    ...
)
```

默认：

```text
layout = "TN"
```



然后来到：

```text
transformer_engine/pytorch/cpp_extensions/gemm.py
```

这里：

```python
transa = layout[0] == "T"
transb = layout[1] == "T"

A = _unwrap_tensor(
    A,
    "rowwise" if transa else "columnwise"
)

B = _unwrap_tensor(
    B,
    "columnwise" if transb else "rowwise"
)
```



Forward 是：

```text
general_gemm(W, A, "TN")
```

于是：

```text
W -> transa=True  -> W.rowwise
A -> transb=False -> A.rowwise
```

因此：

```text
Forward:

W_row ──────┐
            ├── MXFP8 GEMM ──> Y BF16
A_row ──────┘
```

这里 `A_col` 和 `W_col` 并没有参与 Forward GEMM。

它们是给 backward 留的。

---

# 6. Forward 结束后 A_row 会不会一直留到 backward？

**不会。这个地方 TE 专门做了显存优化。**

这是回答你“会不会显存爆炸”的一个关键细节。

Forward GEMM 做完以后：

```python
inputmat.update_usage(
    rowwise_usage=False,
    columnwise_usage=True,
)
```

也就是：

```text
A_row          -> 释放
A_row_scale    -> 释放

A_col          -> 保留
A_col_scale    -> 保留
```



所以 Forward 量化的瞬间：

```text
A_row + A_col
```

的确同时存在。

但跨越整个：

```text
Forward
     ↓
...
     ↓
Backward
```

保存的是：

```text
A_col
```

而不是：

```text
A_row + A_col
```

因此激活缓存不是持续 2.06 B/value，而大致是：

$$
\boxed{1.03\ {\rm Bytes/value}}
$$

外加 scale padding。

---

# 7. 那 A_col 到底保存在哪里？

它最终通过 PyTorch autograd context 保存。

流程是：

```text
_Linear.forward()
   ↓
prepare_for_saving(...)
   ↓
ctx.save_for_backward(...)
```

`linear.py` 在 Forward 末尾将需要 backward 的 tensor 放进 autograd ctx。

对于 `MXFP8TensorStorage`，专门实现了：

```python
prepare_for_saving()
```

它把：

```python
[
    self._rowwise_data,
    self._columnwise_data,
    self._rowwise_scale_inv,
    self._columnwise_scale_inv,
]
```

拆成普通 PyTorch tensors 给：

```python
ctx.save_for_backward()
```

保存。

这里有一个很重要的显存细节：

**它不是 clone。**

相当于：

```text
MXFP8 object
      │
      └── underlying CUDA buffers
                   ↑
              autograd ctx
```

是把底层 tensor reference 交给 autograd 保存，而不是再复制一遍。

所以不会因为：

```python
ctx.save_for_backward()
```

再多出来一套 `A_col`。

---

# 8. W 为什么和 A 有一点不同

W 是 parameter，而不是 activation。

普通情况下：

```text
Forward 需要 W_row
Dgrad   需要 W_col
```

所以 W 很适合：

```text
Forward 一次量化
      ↓
{W_row, W_col}
      ↓
后面的 Dgrad 继续复用
```

而且 TE 支持 **FP8/MXFP8 weight workspace cache**。

`Linear.forward()` 中的：

```python
is_first_microbatch
```

就是和这个缓存相关。

TE 文档注释说明，设置 `is_first_microbatch` 可以缓存 FP8 versions of weights，避免 gradient accumulation 多个 microbatch 每次都重新 quantize。

Megatron-Core 又有：

```text
disable_parameter_transpose_cache
```

这个配置最终影响这一类 weight workspace caching。

所以一个比较常见的生命周期是：

```text
BF16 W parameter
       │
       ├────> W_row MXFP8 ── Forward
       │
       └────> W_col MXFP8 ── Dgrad
                 
      两者缓存为 FP8 weight workspace
```

这部分确实是比较实在的显存开销。

如果 BF16 W 本身还存在：

$$
W_{\mathrm{BF16}}\approx2P
$$

同时 MXFP8 dual workspace：

$$
W_{\mathrm{MXFP8,row+col}}\approx2.0625P
$$

单看这一块就大约：

$$
4.06P\ {\rm Bytes}
$$

当然这还没有算 optimizer states。

---

# 9. 进入 Backward：G 是什么时候量化的？

这里和 A/W 完全不同。

G 是：

$$
G=\frac{\partial L}{\partial Y}
$$

Forward 时它根本不存在。

所以：

> **G 的 MXFP8 表示一定是在 backward 才生成。**

进入 `Linear.backward` 后，TE 先执行：

```text
grad_output_preprocess
```

源码：

```text
transformer_engine/pytorch/module/base.py
```

以及调用处：

```text
transformer_engine/pytorch/module/linear.py
```



因为：

```text
Dgrad 需要 G-row
Wgrad 需要 G-col
```

标准路径会：

```python
grad_output_quantizer.set_usage(
    rowwise=True,
    columnwise=True,
)
```

然后对高精度：

```text
G BF16
```

执行一次：

```text
quantizer(G)
```

得到：

```text
G_row
G_col
```

所以：

```text
              ┌──> G_row ── Dgrad
G BF16 ──Q────┤
              └──> G_col ── Wgrad
```



这不是从 Forward 保存出来的。

---

# 10. Dgrad 的完整数据流

数学上：

$$
dA=GW
$$

TE 代码：

```python
general_gemm(
    weight_for_dgrad,
    grad_output,
    layout="NN",
    ...
)
```

源码：

```text
transformer_engine/pytorch/module/linear.py
```



再根据 `general_gemm()` 的规则：

```text
NN:

first operand  N -> columnwise
second operand N -> rowwise
```

所以：

```text
W -> W_col
G -> G_row
```

完整路径：

```text
                 W_col
                   │
                   │
G BF16 -> MXFP8 -> G_row
                   │
                   ▼
             MXFP8 GEMM
                   │
                   ▼
             dA BF16/FP16
```

也就是：

$$
\boxed{Dgrad=G_{\rm row}\times W_{\rm col}}
$$

这里所谓的：

```text
W "转置数据"
```

实际上就是前面 Forward 时从 BF16 W 同时生成的：

```text
W.columnwise
```

正常情况下根本不需要：

```text
W_row
 ↓ dequantize
BF16
 ↓ requantize
W_col
```

---

# 11. Wgrad 的完整数据流

数学：

$$
dW=G^TA
$$

源码中：

```python
general_gemm(
    x,
    dy,
    layout="NT",
)
```

而代码注释直接写：

```text
dw = dy^T * x
```



根据 `general_gemm()`：

```text
NT:

A operand N -> columnwise
B operand T -> columnwise
```

所以：

```text
A -> A_col
G -> G_col
```

最终：

$$
\boxed{Wgrad=G_{\rm col}^{T}\times A_{\rm col}}
$$

可以画成：

```text
Forward 已保存
A_col ─────────────────────┐
                           │
                           │
Backward                   │
G BF16 ── MXFP8 ── G_col ──┤
                           ▼
                        Wgrad GEMM
                           │
                           ▼
                     dW / main_grad
```

所以这里正好回答你的问题：

**A_col 是 Forward 就准备好的；G_col 是 Backward 才准备的。**

---

# 12. 因此 A / W / G 的生命周期可以浓缩成这张图

```text
====================== FORWARD ======================

A BF16
   │
   ├─ quantize ──> A_row ───────┐
   │                             │
   └─────────────> A_col         │
                     │           │
                     │      Forward GEMM
                     │           ▲
W BF16               │           │
   │                  │           │
   ├─ quantize ──> W_row ────────┘
   │
   └─────────────> W_col
                     │
                     │
Forward GEMM done    │
                     │
A_row ── FREE        │
                     │
A_col ── SAVE ───────┼──────────────────────┐
W_col ── SAVE/CACHE ─┼──────────┐           │
W_row ── CACHE       │          │           │
                     │          │           │


====================== BACKWARD =====================

G BF16
   │
   ├─ quantize ──> G_row ───────┼─ + W_col ──> Dgrad
   │                             │
   └─────────────> G_col ────────┼─ + A_col ──> Wgrad
                                 │
                                 ▼
                          backward finished

A_col / G_row / G_col
     ── released when no longer needed
```

这基本就是你要找的核心机制。

---

# 13. 有没有“反向重新量化”的情况？有，但不是从 FP8 反量化再量化

这里需要加几个很重要的例外。

### `save_original_input=True`

某些路径不会 Forward 保存：

```text
A_col MXFP8
```

而是保存原始高精度：

```text
A BF16
```

那么 Wgrad 前：

```python
input_quantizer.set_usage(
    rowwise=False,
    columnwise=True,
)
inputmat = input_quantizer(inputmat)
```

也就是说：

```text
saved BF16 A
      ↓ backward quantize
A_col
```



仍然不是：

```text
A_row MXFP8
 → dequant
 → requant
```

---

### FSDP2 / distributed weight

有些分布式 weight 路径 TE 刻意不跨 Forward→Backward 保存整个 MXFP8 weight workspace。

例如 FSDP2 的源码注释明确说：

```text
不保存 FP8 weight workspace，
backward 时从 gathered high-precision weight 重新 quantize。
```



所以：

```text
Forward:
BF16 W -> MXFP8 W

workspace 丢弃

Backward:
gather BF16 W
       ↓
MXFP8 quantize
       ↓
W_col
```

这是典型的：

> **用算力换显存。**

---

### UserBuffer / communication overlap

某些 TP + communication overlap 路径也可能不能直接复用已经生成的 G representation。

`linear.py` 里甚至有一段针对 MXFP8 的明确注释：

> 不能把 row-scaled MXFP8 转成 column-scaled MXFP8，因此不能直接重用 dgrad 所使用的 grad output。

于是它会对原始：

```text
grad_output_arg
```

重新做一次：

```text
columnwise MXFP8 quantization
```

供 Wgrad 使用。

注意这里依然是：

```text
BF16 G -> G_col
```

而不是：

```text
G_row -> dequant -> G_col
```

---

# 14. `update_usage()` 并不会偷偷帮你“转换”

这个源码细节也非常能说明问题。

`MXFP8TensorStorage.update_usage()` 的注释明确说：

> 对 MXFP8，columnwise output 只能由 x2 scaling kernel 等量化过程产生，因此这个函数只能 disable usages。

也就是说：

```python
tensor.update_usage(columnwise_usage=True)
```

如果 `columnwise_data` 本来没有生成：

**它不会现场把 rowwise 变成 columnwise。**

甚至会直接报：

```text
Requested column-wise usage,
but MXFP8Tensor is missing column-scaled FP8 data
```



这几乎可以从代码层面完全排除一种错误理解：

```text
“TE 保存 rowwise FP8，
反向的时候看到要 transpose，
update_usage() 自动 transpose/requant”
```

不是这样。

---

# 15. Forward、Dgrad、Wgrad 的输入输出 dtype 再总结一下

这里区分：

```text
GEMM operand precision
```

和：

```text
GEMM result precision
```

标准 Megatron + TE MXFP8 Linear 大致是：

| 阶段      | Operand 1 | Operand 2 | Tensor Core 输入 | GEMM 输出                                |
| ------- | --------- | --------- | -------------- | -------------------------------------- |
| Forward | W-row     | A-row     | MXFP8          | `activation_dtype`                     |
| Dgrad   | W-col     | G-row     | MXFP8          | `activation_dtype`                     |
| Wgrad   | A-col     | G-col     | MXFP8          | `activation_dtype` 或 `main_grad.dtype` |

Megatron 里最常见的：

```text
activation_dtype = BF16
```

因此你看到的是：

```text
MXFP8 × MXFP8
      ↓
accumulate
      ↓
BF16 output
```

Wgrad 如果使用 Megatron 的：

```text
gradient_accumulation_fusion
```

则可能直接 accumulate 到：

```text
weight.main_grad
```

具体 dtype 根据你的 optimizer/main-grad 配置，常见是 FP32。`linear.py` 对 Wgrad output dtype 就是根据 `main_grad.dtype` 或 `activation_dtype` 选择。

标准 `Linear` 里 `output_quantizer` 和 `grad_input_quantizer` 默认没有开启，所以 Y 和 dA 并不是自动继续以 MXFP8 tensor 形式返回；只有显式要求 `fp8_output` / `fp8_grad` 等路径才会量化 GEMM output。

---

# 16. Forward 与 Backward 的 FP8 format 还可能不同

这里还有一个容易忽略的细节。

Megatron 中：

```text
Fp8Recipe.mxfp8
```

最终创建：

```python
MXFP8BlockScaling(...)
```



TE 支持：

```text
E4M3
HYBRID
```

recipe。

如果：

```text
fp8_format = E4M3
```

那么：

```text
A : E4M3
W : E4M3
G : E4M3
```

如果是：

```text
HYBRID
```

那么通常：

```text
Forward operands:
A, W -> E4M3

Backward operand:
G -> E5M2
```

这是通过：

```text
MXFP8BlockScalingRecipeState
```

调用：

```python
get_fp8_te_dtype(recipe, fprop_tensor)
```

实现的。

scale 仍然是：

```text
E8M0
```

block size 仍是 32。

---

# 17. 为什么不直接存一份 A_row，然后 backward 时 transpose？

这不是简单的软件设计选择，而是 MXFP8 数学定义导致的。

比如：

```text
A =
row0: [100, 1, 1, ... 32 values]
row1: [1,   1, 1, ...]
...
```

如果 rowwise：

```text
row0 的 32 个值
```

会因为其中有一个 `100`，整组 scale 被 100 决定。

而 columnwise 时：

```text
col0 的 32 个值
```

这一组又可能包含完全不同的数据分布。

所以：

```text
scale_row(m, k//32)
```

和：

```text
scale_col(m//32, k)
```

根本不是同一组 scale。

因此：

$$
Q_{\rm row}(A)^T
\neq
Q_{\rm row}(A^T)
$$

而所谓 `A_col` 本质上就是：

$$
Q_{\rm row}(A^T)
$$

在 TE 内部对应的一种未真正 materialize `[K,M]` transpose 的存储形式。

这也是 MXFP8 相比普通 per-tensor FP8，transpose cache 更值得关注的原因。

---

# 18. 那显存到底会不会“爆炸”？

不会出现你可能直觉想象的：

```text
每层长期保存

A_row
A_col
W_row
W_col
G_row
G_col
```

六份全部横跨整个 iteration。

真正的生命周期明显错开。

对普通 Linear 来说：

| 数据    | Forward GEMM 时 | Forward 后等待 backward | 当前层 Backward |
| ----- | -------------: | -------------------: | -----------: |
| A-row |              有 |                   释放 |            无 |
| A-col |              有 |               **保存** |     Wgrad 使用 |
| W-row |              有 |             可能 cache |       一般不再需要 |
| W-col |              有 |         **保存/cache** |     Dgrad 使用 |
| G-row |              无 |                    无 |     **临时生成** |
| G-col |              无 |                    无 |     **临时生成** |

所以 activation 侧真正跨越 Forward→Backward 的关键增量，大约是：

$$
A_{\rm col}
\approx1.03\ {\rm Bytes/value}
$$

而不是 2.06 B/value。

G 是 backward 当前层的临时 workspace：

$$
G_{\rm row+col}
\approx2.06\ {\rm Bytes/value}
$$

等这层 backward 完成后即可释放。`linear.py` 也有显式清理 input/grad-output temporary storage 的逻辑。

真正更容易形成**持续性显存开销**的其实是：

$$
W_{\rm row}+W_{\rm col}
$$

因为 weight workspace 很适合跨 microbatch cache。

所以如果你实际 profile MXFP8 训练显存，我会重点把显存拆成：

```text
BF16/FP32 parameter / master weight
+
optimizer states
+
MXFP8 W row+col workspace
+
saved A_col activation
+
当前 backward 层 G row+col workspace
+
TP/FSDP communication buffers
```

而不是简单理解成“FP8 后所有 tensor 都只有原来的 1/2”。

---

# 19. Megatron-LM → TE → CUDA 的调用链

如果你准备直接下代码打 print/NVTX，我建议按这条链追。

| 层级                     | 源码位置                                                                | 重点看什么                                                |
| ---------------------- | ------------------------------------------------------------------- | ---------------------------------------------------- |
| Megatron recipe        | `megatron/core/extensions/transformer_engine.py`                    | `Fp8Recipe.mxfp8 -> MXFP8BlockScaling`               |
| TE recipe 定义           | `transformer_engine/common/recipe/__init__.py`                      | MXFP8=32-value block；transpose 两套独立 quantization     |
| recipe → quantizer     | `transformer_engine/pytorch/quantization.py`                        | `MXFP8BlockScalingRecipeState`，Forward/Bwd dtype     |
| Linear 主路径             | `transformer_engine/pytorch/module/linear.py`                       | A/W quantize、Forward/Dgrad/Wgrad、save/free           |
| MXFP8 Python quantizer | `transformer_engine/pytorch/tensor/mxfp8_tensor.py`                 | `MXFP8Quantizer`、row/col scale shape                 |
| MXFP8 storage          | `transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py` | 四个真实 CUDA buffer、`prepare_for_saving`、`update_usage` |
| GEMM layout 选择         | `transformer_engine/pytorch/cpp_extensions/gemm.py`                 | `TN/NN/NT → rowwise/columnwise`                      |
| C++ quantizer          | `transformer_engine/pytorch/csrc/quantizer.cpp`                     | buffer allocation、`nvte_quantize_v2`                 |
| CUDA MXFP8 cast        | `transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh`           | row/col/bidirectional block quantization kernel      |

其中最值得你直接下断点/加日志的是这几个位置：

```text
linear.py
  Forward input quantization        ~ input_quantizer.set_usage(...)
  Forward weight quantization       ~ weight_quantizer.set_usage(...)
  Forward GEMM                      ~ general_gemm(weightmat, inputmat_total)

  Backward grad_output preprocess   ~ grad_output_quantizer
  Dgrad                             ~ layout="NN"
  Wgrad                             ~ layout="NT"
```

然后在：

```text
cpp_extensions/gemm.py
```

的：

```python
A = _unwrap_tensor(...)
B = _unwrap_tensor(...)
```

前后打印：

```python
layout
type(A)
A.get_usages()
type(B)
B.get_usages()
```

你会非常直观地看到：

```text
FWD:
TN -> W.rowwise + A.rowwise

DGRAD:
NN -> W.columnwise + G.rowwise

WGRAD:
NT -> A.columnwise + G.columnwise
```

这实际上是理解 TE MXFP8 GEMM 最干净的切入点。

---

## 最后浓缩成一句话

**标准 TE MXFP8 训练不是“Forward 量化一份，Backward 反量化后再量化 transpose”。而是 A、W 在 Forward 时根据未来 GEMM 的需求，直接从 BF16/FP16 一次生成 rowwise 和 columnwise 两套 MXFP8；Forward 用 rowwise，A 的 rowwise 随后释放、columnwise 存入 autograd context 给 Wgrad，W 的 columnwise 留给 Dgrad。G 因为只有 Backward 才产生，所以 Backward 时再从高精度 G 一次生成 rowwise + columnwise，分别给 Dgrad 和 Wgrad。特殊 FSDP/通信/`save_original_input` 路径会选择在 Backward 从保存的高精度原始数据重新量化，以计算换显存，但正常不会做 MXFP8 → dequant → MXFP8 的二次量化。**
