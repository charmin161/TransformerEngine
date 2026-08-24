截至 **2026 年 8 月 21 日**，我认为可以先给出一个比较明确的结论：

**你的判断基本是对的：Rubin 已经出现了一个与 TurboQuant 的 codebook 反量化高度匹配的原生 Tensor Core 硬件机制。** NVIDIA 官方把它称为 **3-bit lookup-table format for matrix B**；更关键的是 CUDA PTX 9.4 已经加入 `tcgen05.mma` 的 **`.decompress::lut::b`**，也就是说“3-bit index → 查 LUT → 得到数值 → 直接参与 MMA”可以发生在 Tensor Core 数据通路内部，而不需要先在 CUDA Core/Triton 中显式反量化。([NVIDIA Developer][1])

但目前还不能进一步说成“**Rubin 官方已经原生支持 TurboQuant KV Cache**”。NVIDIA 目前把这个功能描述成 Matrix B/weight compression；我没有找到 NVIDIA 官方把它和 TurboQuant 绑定起来的材料。更准确的说法是：

> **Rubin 提供了一个几乎正好对应 TurboQuant MSE/codebook 部分的硬件原语。**

---

## 1. 先把 TurboQuant 的量化过程简化清楚

TurboQuant 最核心的东西其实并不复杂。先忽略论文里用于消除 inner-product bias 的 QJL，只看主体 `TurboQuant_mse`。

假设一个 Key 向量：

[
K\in R^d
]

首先计算它的 L2 norm：

[
s=|K|_2,\qquad x=K/s
]

论文理论推导默认单位向量；非单位向量就额外存这个 norm，反量化以后再乘回来。([arXiv][2])

然后做随机正交旋转：

[
y=\Pi x
]

这里真正的目的不是“旋转本身”，而是**把原来 K 中很难预测的 outlier、不同 channel 的统计差异打散**。随机旋转以后，每个维度的分布变成已知的、接近：

[
y_i\sim\mathcal N(0,1/d)
]

因此不需要针对模型做 calibration，也不需要为不同 channel 学习 scale。TurboQuant 可以事先针对这个分布，用 Lloyd-Max 求一个固定的最优 scalar quantization codebook。([arXiv][2])

例如 3-bit：

[
C={c_0,c_1,\ldots,c_7}
]

只有 **8 个 centroid**。

每个 (y_i) 不保存数值，而只保存：

[
idx_i=\arg\min_j |y_i-c_j|
]

所以真正写进 KV Cache 的主要东西就是：

```text
K 原始值
   ↓ L2 normalize
K / ||K||
   ↓ random rotation / WHT
rotated K
   ↓ nearest centroid
3-bit index  3-bit index  3-bit index ...
```

论文的反量化是：

[
\tilde y_i=C[idx_i]
]

然后：

[
\tilde K=s\Pi^T\tilde y
]

也就是：

```text
3-bit index
     ↓
codebook lookup
     ↓
centroid
     ↓
inverse rotation
     ↓
× L2 norm
     ↓
K approximately
```

论文 Algorithm 1 就是这个过程：**rotation → nearest-centroid index；dequant 时 index → centroid → inverse rotation**。([arXiv][2])

---

## 2. TurboQuant 为什么特别适合 attention 硬件融合

这里有个非常关键的数学变换。

实际上做 attention 时根本没必要真的恢复：

[
\tilde K=s\Pi^T C[idx]
]

因为：

[
q^T\tilde K
===========

q^T(s\Pi^TC[idx])
]

利用正交矩阵：

# [

s(\Pi q)^TC[idx]
]

也就是说可以变成：

```text
Q ──rotation──> Q'
                  │
                  ×
                  │
K cache: 3-bit index ──LUT──> centroid
                  │
                  ↓
                 QK
                  ↓
             × per-token norm
```

**这一步非常重要。**

你不再需要：

```text
index
 → LUT
 → 完整 BF16 K
 → inverse rotation
 → 写 register/shared memory
 → Tensor Core MMA
```

理论上完全可以：

```text
3-bit index
       ↓
Tensor Core 内部 LUT
       ↓
centroid
       ↓
直接乘 rotated Q
```

所以从硬件设计角度看，TurboQuant 真正希望拥有的是：

> **一个能够把 low-bit index 当作 MMA operand，然后在 MMA 内部进行 codebook lookup 的 Tensor Core。**

而这恰好就是 Rubin 新出现的东西。

---

# 3. Rubin：证据非常强，确实出现了“codebook Tensor Core”

NVIDIA 在 **2026 年 7 月 21 日**发布的 Rubin 架构官方文章明确说：

Rubin Transformer Engine 支持 **Matrix B 的 3-bit lookup-table format**。矩阵里不直接存 weight value，而是存一个 **3-bit index**；Tensor Core 在计算时 **inline resolve** 这些 index，从一个小的 representative-value table 中取真正的数值。([NVIDIA Developer][1])

这已经不是“可能支持”了，而是非常明确的硬件功能。

更重要的是 **PTX ISA 9.4**。你之前关注的 CUDA 13.4 developer preview 正好把它暴露出来了：

```text
tcgen05.mma
    .decompress::lut::b
```

PTX 9.4 release note 明确列出：

```text
Adds support for .decompress::lut::b qualifier
for tcgen05.mma instruction.

Adds support for .collector::b::* qualifier
for tcgen05.mma instruction.
```

([NVIDIA Docs][3])

PTX 文档的 figure 目录甚至已经直接出现：

```text
Layout of Lookup table in Tensor Memory

B matrix of NxK compressed for 8B LUT

Matrix B usage in computation

GMEM and TMEM layout of look-up-table
```

([NVIDIA Docs][3])

所以硬件数据流大致已经不是：

```text
HBM
│
├─ packed int3
│
↓
SMEM / registers
│
CUDA Core unpack
│
LUT lookup
│
FP8/BF16
│
↓
Tensor Core MMA
```

而是更接近：

```text
HBM
│
├── 3-bit indices
│
└── LUT / codebook
       ↓
Tensor Memory / collector
       ↓
┌─────────────────────────┐
│      Rubin Tensor Core   │
│                         │
│ index ── LUT ── value   │
│               ↓         │
│             MMA         │
└─────────────────────────┘
```

这正是它最有意义的地方：**dequant 不再成为一个独立的数据搬运阶段。**

---

## 4. 它与 TurboQuant 到底有多像？

非常像。

| 特征             | TurboQuant 3-bit     | Rubin LUT-B                     |
| -------------- | -------------------- | ------------------------------- |
| 存储内容           | 3-bit codebook index | 3-bit LUT index                 |
| codebook 大小    | (2^3=8)              | 8-entry LUT                     |
| codebook 是否非均匀 | 是，Lloyd-Max centroid | 是，可编程 LUT                       |
| 解码             | `C[index]`           | `LUT[index]`                    |
| 解码位置           | 软件/kernel            | **Tensor Core MMA 内部**          |
| 是否训练得到         | 不需要                  | 与硬件无关                           |
| 目标             | KV/vector            | NVIDIA 当前主要描述为 Matrix-B weights |

NVIDIA 官方没有公开全部 LUT block 细节，但基于 PTX 9.4 的公开内容，SemiAnalysis 对该机制进行了进一步拆解：每个值是 3-bit index，指向 **8 个 E4M3 value**，一个 LUT 服务于一个 **8×64=512 element block**；因此包含 LUT 开销后的实际存储约为：

[
3+\frac{8\times8}{512}
=3.125\ bit/element
]

它还指出 lookup 是在 MMA 内完成的，而不是事先 materialize 一个解压后的 B 矩阵。这个 block/LUT datatype 的细节目前我会把它归为**强技术分析，而不是 NVIDIA 官方明文规格**。([InferenceX][4])

---

# 5. 为什么我认为它尤其可能用于 TurboQuant 的 K Cache

PTX 这里明确说的是 **Matrix B 的 (N\times K) compressed representation**。([NVIDIA Docs][3])

attention：

[
QK^T
]

如果：

[
Q:[M,d]
]

而 K Cache 按：

[
K:[N,d]
]

存储，那么从 MMA 的视角正好可以写成：

[
A=Q:[M,K]
]

[
B=K:[N,K]
]

计算：

[
AB^T
]

所以 **K Cache 天然就是 Rubin LUT-B 的 B operand**。

再结合前面的 TurboQuant：

[
QK^T
\approx
(\Pi Q)C[idx]^T \cdot s_K
]

就出现了非常漂亮的数据流：

```text
                Q
                │
          Hadamard / rotation
                │
                ▼
                A
                │
                │
                ▼
        ┌───────────────────┐
        │ Rubin Tensor Core │
        │                   │
K idx ─►│ 3-bit LUT decode  │
        │       ↓           │
LUT ───►│ centroid × A      │
        │       ↓           │
        │       MMA         │
        └───────────────────┘
                │
                ▼
             QK score
                │
           × K L2 norm
```

**从计算图上几乎就是 TurboQuant MSE-only K quantization 所需要的硬件。**

---

# 6. 但现在还不能说“Rubin 原生支持完整 TurboQuant”

这里至少有四个边界需要注意。

第一，**NVIDIA 目前明确说的是 weight/Matrix B LUT format，而没有说 TurboQuant KV Cache**。从数学上 K 可以作为 Matrix B，但是否 CUDA/CUTLASS/FlashAttention 会把 LUT-B 暴露给 KV attention kernel，是软件层面的下一步。

第二，Rubin 当前公开的是 **3-bit / 8-entry LUT**。所以它最自然对应：

[
TQ\text{-}3bit
]

而 AMD 和当前 vLLM 的实际经验反而表明 **4-bit TQ 往往是更稳妥的 production sweet spot**。vLLM 的大规模评测里，`turboquant_4bit_nc` 比 3-bit variant 稳定得多，而部分长上下文和 reasoning workload 上 3-bit 会明显掉精度。([GitHub][5])

第三，Rubin LUT 的 entry 据目前技术分析是 E4M3，所以原始 Lloyd-Max centroid 需要：

[
c_i\rightarrow FP8\ E4M3
]

会再多一层 centroid quantization。这可能需要重新求“hardware-aware codebook”，而不能机械地把论文中的 FP32 centroid 填进去。([InferenceX][4])

第四，Rubin LUT-B 只解决 TurboQuant 的：

[
idx\rightarrow centroid
]

而**不自动解决 rotation、per-token norm，以及论文完整版 QJL residual**。

尤其论文真正用于 unbiased inner product 的完整版是：

[
Q_{\rm prod}
============

Q_{\rm mse}^{b-1}
+
1\text{-bit QJL residual}
]

也就是先用 (b-1) bit codebook，然后对 residual 再存一组 sign bit；反量化还要加入 QJL reconstruction。([arXiv][2])

Rubin LUT-B 没有解决第二项。

不过有趣的是，**生产实现正在主动把 QJL 去掉**。

---

# 7. AMD：已经把 TurboQuant 做得非常深，但目前是“软件 LUT + 硬件 MFMA”

AMD 在 2026 年 6 月专门发布了一篇官方文章：

**Productionizing TurboQuant on AMD GPUs for KV-Cache-Bound LLM Inference**

而且不是实验性质，是针对 vLLM + MI355X 做的 production optimization。([Rocm Blog][6])

AMD 实际上对原论文做了相当大的工程化修改：

| 原始 TurboQuant              | AMD production TQ            |
| -------------------------- | ---------------------------- |
| random orthogonal rotation | signed Walsh-Hadamard        |
| K/V 都可以 codebook           | 主要在 **K 使用 rotation+LUT**    |
| V 也可 TQ                    | V 更倾向普通低 bit quant           |
| QJL residual               | **4-bit production 中去掉**     |
| scalar LUT                 | **Pair LUT**                 |
| 普通 layout                  | SoA KV layout                |
| generic Triton             | Triton → HIP → MFMA → FlyDSL |

AMD 明确说，当前 vLLM OSS TurboQuant 是：

> scalar LUT lookup → dequantization → attention

而 AMD 优化成 **Pair LUT**：

```text
两个 4-bit index
       ↓
组成 1 Byte
       ↓
pair LUT
       ↓
一次 gather
       ↓
得到两个 centroid
```

这相当于预计算：

[
LUT[i,j]=(C_i,C_j)
]

从而把两次 scalar lookup 合成一次。AMD 官方明确写了“一次 byte load + 一次 gather 恢复两个 dequantized values”。([Rocm Blog][7])

---

## 8. AMD 哪些部分用了真正的硬件加速？

AMD 做得很激进。

他们的 native HIP kernel 直接使用 **GCN ISA MFMA intrinsic** 做 QK：

```text
compressed KV
   ↓
Pair LUT / gather       ← 软件
   ↓
BF16 values
   ↓
MFMA                     ← Matrix Core
   ↓
QK
   ↓
softmax
   ↓
MFMA                     ← Matrix Core
   ↓
PV
```

同时还用了：

* GQA grouping，让多个 query heads 共享一次 KV load；
* 128-bit coalesced HBM load；
* LDS staging；
* native MFMA dispatch；
* 4-wave parallel QK；
* QK → softmax → PV 的 register handoff；
* 更宽的 MFMA。([Rocm Blog][7])

最终优化 kernel 相比开源基础版本最高达到约 **12.7× kernel-level speedup**；其 FlyDSL TQ4/4 在测试配置下能达到 BF16 throughput 的约 95%。([Rocm Blog][7])

但是注意数据流中的分界线：

```text
index → centroid
```

仍然是 **LUT + gather 软件操作**，

而：

```text
centroid × Q
```

才进入 MFMA。

AMD CDNA4/MI355X 确实有原生 FP8/FP6/FP4 MFMA，例如 `V_MFMA_F32_16X16X128_F8F6F4`，但目前公开 ISA/ROCm 材料中，我没有找到类似 NVIDIA：

```text
.decompress::lut::b
```

这样能够在 **MFMA 内部做 programmable codebook lookup** 的 operand format。AMD 自己的 TurboQuant 博客仍然明确把 Pair-LUT/gather 和 MFMA dispatch 分成两个优化阶段，这也是一个相当强的旁证。([ROCm Documentation][8])

因此截至现在我的判断是：

> **AMD MI355X 对 TurboQuant 是“为算法写高度优化的硬件感知 kernel”，Rubin 则第一次出现了“硬件本身提供 TurboQuant-like codebook MMA operand”的迹象。**

---

# 9. Hopper / Blackwell 又是什么状态？

Google 自己报告过，在 **H100** 上 4-bit TurboQuant 的 attention-logit computation 相对 FP32 key 可以做到最高约 **8×**。([Google Research][9])

但这个数字不要理解成：

> H100 有 TurboQuant 硬件。

不是。

H100 上是通过：

```text
更少 HBM traffic
+
packed low-bit load
+
LUT/dequant fusion
+
optimized attention kernel
```

获得速度优势。

vLLM 2026 年对 H100 的系统评测甚至非常明确地指出：FP8 KV Cache 可以直接使用 hardware-native FP8 attention，而 TurboQuant **只是以 3–4 bit 存储 KV，attention computation 前仍要恢复成 BF16**。因此在 KV 内存不紧张时，TQ 的吞吐反而经常比 FP8/BF16 差。([GitHub][5])

所以可以把 NVIDIA 演进简单理解成：

| GPU             | TurboQuant/codebook 支持性质                                        |
| --------------- | --------------------------------------------------------------- |
| A100            | 软件 LUT/dequant                                                  |
| H100/H200       | 软件 LUT/dequant；HBM 节省可带来明显收益                                    |
| B100/B200/GB200 | 更强 Tensor Core，但目前没找到原生 programmable LUT-B 的官方证据                |
| **Rubin**       | **3-bit index + programmable LUT + Tensor Core inline resolve** |

这最后一步是性质上的变化。

---

# 10. 我目前对 Rubin × TurboQuant 的判断

我会把证据强度划成这样：

**可以基本确认：**

[
\boxed{\text{Rubin 有 programmable 3-bit LUT Matrix-B Tensor Core}}
]

NVIDIA 官方博客 + PTX 9.4 两条独立官方证据都已经成立。([NVIDIA Developer][1])

**可以从数学和数据布局上较强推断：**

[
\boxed{
\text{TurboQuant K-cache}
\rightarrow
\text{Rubin LUT-B}
}
]

因为 TQ 3-bit 本身恰好就是 8 centroid codebook，而 QK 的 K 又天然可以作为 MMA 的 Matrix B。

**现在还没有公开证据证明：**

[
\boxed{
\text{NVIDIA 已经实现 TurboQuant KV attention kernel on Rubin}
}
]

我目前没有找到 NVIDIA/CUTLASS/FlashInfer/TRT-LLM 明确发布这种 kernel。因此这一层暂时不能从“硬件 capability”跳到“已经支持 TurboQuant”。

---

我觉得下一步最值得做的反而不是继续泛泛搜 TurboQuant，而是**直接沿 PTX 9.4 的 `.decompress::lut::b` 往下拆**：把 Rubin LUT-B 的 **TMEM LUT layout、8×64 block、collector B、支持的 MMA shape、E4M3 codebook 限制**全部梳理出来，然后拿 **TurboQuant 3-bit K-cache 的 `[token, head_dim]` layout** 一项项对齐。这样基本可以判断：如果以后你要在 Rubin 上自己实现 TurboQuant Attention，哪些部分 Tensor Core 能直接吃，哪些部分还需要在 kernel 中做 WHT/norm。我可以下一步直接按这个方向继续帮你拆。

主要一手资料：[TurboQuant 原论文](https://arxiv.org/abs/2504.19874?utm_source=chatgpt.com) · [NVIDIA Rubin 架构官方文章](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/?utm_source=chatgpt.com) · [AMD TurboQuant 官方实现分析](https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html?utm_source=chatgpt.com)

如果你希望长期关注，我也可以持续跟踪 PTX/CUTLASS/FlashInfer 是否开始正式暴露 Rubin LUT-B。

[1]: https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/?utm_source=chatgpt.com "Inside NVIDIA Rubin GPU Architecture: Powering the Era of Agentic AI | NVIDIA Technical Blog"
[2]: https://arxiv.org/html/2504.19874 "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
[3]: https://docs.nvidia.com/cuda/developer-preview/13.4/pdf/ptx_isa_9.4.pdf?utm_source=chatgpt.com "PTX ISA"
[4]: https://inferencex.semianalysis.com/blog/vera-rubin-nvl72-vs-gb200-nvl72-inference?utm_source=chatgpt.com "Vera Rubin NVL72 vs GB200 NVL72? Inference TCO & Architecture ..."
[5]: https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-05-11-turboquant.md "vllm-project.github.io/_posts/2026-05-11-turboquant.md at main · vllm-project/vllm-project.github.io · GitHub"
[6]: https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html?utm_source=chatgpt.com "Productionizing TurboQuant on AMD GPUs for KV-Cache-Bound LLM Inference — ROCm Blogs"
[7]: https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html "Productionizing TurboQuant on AMD GPUs for KV-Cache-Bound LLM Inference — ROCm Blogs"
[8]: https://rocm.docs.amd.com/projects/ai-developer-hub/en/v14.0/notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4.html?utm_source=chatgpt.com "FP8 GEMM optimization on AMD CDNA4-based GPUs — Tutorials for AI developers 14.0"
[9]: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/?utm_source=chatgpt.com "TurboQuant: Redefining AI efficiency with extreme compression"
