有**实质性新进展**，而且这次比上一次 TransformerEngine 的 UE5M3 POC 更有价值：**Rubin 的 E5M3/NVFP4 block-scale 路线已经从“量化原型”进一步落到了 NVIDIA 官方 cuDNN Frontend 的实际 GEMM / MoE GEMM pipeline；与此同时，Rubin 专用 SDPA 代码开始暴露更多 Tensor Memory + Softmax 的具体实现方式。** 但仍然没有看到公开代码直接揭示 attention 2:4 adaptive compression 的“压缩指令/metadata 生成器”。

### 发生了什么变化

第一项，也是本轮最重要的新证据，是 NVIDIA `cudnn-frontend` 在 **2026 年 8 月 14 日**创建并当天合并 PR #593 `frost(gemm): Add Rubin kernel pipelines`。它不是概念验证，而是已经合并到 `develop` 的 Rubin SM107 FROST 路径。PR 明确加入了 **FP8_E5M3 datatype、E5M3 block-scale/quantize、SM107 block-scaled GEMM、MoE grouped block-scaled GEMM**，并明确写出这些 Rubin kernel 使用 **64-byte-K MMA**；同时还有 Rubin 专属的 B-operand collector reuse、`num_mma_m > 1` 和超大 SMEM 配置。 PR 于 8 月 14 日 03:38 UTC 创建，17:17 UTC 即被合并，涉及 50 个文件、约 8K 行新增。

这对我们上次看到的 TransformerEngine #3325 是一个很重要的补充。TE #3325 目前仍然是 **Draft / Proof-of-Concept**，描述仍写着“UE5M3 NVFP4 quantization 已支持，但 GEMM 尚不可用”。 到 8 月 15 日它虽然已经扩展到 11 个 commit、57 个文件，但依然没有 merge。 因此现在不能再把“TE 中 GEMM unavailable”理解成 **Rubin 平台没有 E5M3/NVFP4 GEMM**；更准确的说法是：**TE 的公开 recipe/API 接口还在原型阶段，但更底层的 cuDNN FROST Rubin kernel pipeline 已经在落地。**

第二项很值得关注的是 **Rubin attention 内部数据流**。cuDNN Frontend 在 **8 月 13 日**合并了 PR #580，Rubin SM107 的 FP8 SDPA kernel 现在有两个非常具体的优化：其一，在无 mask Softmax 中直接用 `tcgen05.ld.red.f32.max`，把 **从 Tensor Memory 加载 score + row-max reduction 融合成一次操作**；其二，不再把 probability `P` 拉回寄存器做 row-sum，而是额外做一个 `P × ones` 的 N=16 MMA，把 Softmax denominator 直接累加到 O 邻接的 Tensor Memory 列中。该实现还明确使用 Rubin 的 **576-column exclusive Tensor Memory allocation**。 该 PR 于 8 月 13 日创建并当天合并。

这不是 adaptive compression 的 sparse kernel——它处理的是 dense FP8 SDPA——但它第一次给我们一个相当具体的 Rubin attention 实现侧写：**Rubin 明显倾向于让 score / Softmax 中间状态长期留在 Tensor Memory 中，并把 reduction 甚至 denominator computation 都尽可能变成 TMEM-native / Tensor-Core-native 操作。** 这与 NVIDIA 7 月 21 日官方描述“dense QKᵀ 结果从 Tensor Memory load 时转换为 2:4 compressed representation，然后 sparse Softmax 和 sparse MMA 继续消费”的数据流非常一致。([NVIDIA Developer][1])

### 证据可靠性与来源等级

这次两项核心证据我都给 **A+ 级来源**：都是 NVIDIA 官方 `cudnn-frontend` 仓库，而且已经 merge，不是第三方逆向，也不是未落地的设计文档。尤其 #593 比上一次 TE #3325 的可信度高一个层级——后者虽然也是 NVIDIA 官方代码，但仍然是 Draft POC。

不过要严格区分：**#593 证明的是 Rubin block-scale GEMM / E5M3 scale 路线；#580 证明的是 Rubin dense attention 的 TMEM/Softmax 优化。二者都没有直接证明 2:4 adaptive compression 的 metadata generator 是怎样实现的。** 后面这一部分仍然只能做架构推断。

### 对可能技术路线的含义

我现在会把 Rubin 的实现模型进一步收敛为两个相对独立、最后在 Tensor Core/TMEM 数据流上汇合的“平面”：

**第一条是低精度数值表示平面：**

`NVFP4 value + E5M3 block scale → SM107 block-scaled 64B-K MMA`

这条路线现在已经不仅是 TE recipe，而是进入了 cuDNN FROST 的 GEMM 和 MoE Grouped GEMM。#593 的 **64-byte-K MMA** 也与 NVIDIA 官方说 Rubin Tensor Core 通过加倍每条指令处理的 K 维来提高吞吐的描述吻合。 ([NVIDIA Developer][1])

**第二条是 attention activation compression 平面：**

`dense QKᵀ in TMEM → 2:4 selection/compression + metadata → sparse Softmax → sparse MMA × V`

目前官方架构文章仍然明确描述这个流程：压缩发生在 **从 Tensor Memory 加载 intermediate score 时**，同时生成 nonzero 和 metadata，后续 Softmax 只处理 nonzero，第二次 attention GEMM 使用 sparse MMA。([NVIDIA Developer][1])

而 #580 新出现的 TMEM row-max / row-sum-in-MMA 代码让我更倾向于一个具体实现：

`QK MMA`
→ `score 留在 TMEM`
→ **TMEM load/reduction/compression 单元**
→ `compressed score + metadata`
→ `Softmax / reduction 尽量不回 RF`
→ `sparse MMA(P,V)`

也就是说，我现在更看好 **“compression 是 Tensor Memory 出口附近的数据变换能力，而不是 Tensor Core MMA 本身动态改变 K_phy”**。这是推断，但证据比以前更强了。NVIDIA 自己的描述就是“load intermediate data from Tensor Memory into structured 2:4 sparse compressed form”。([NVIDIA Developer][1])

这也进一步降低了此前我们讨论的另一种可能性：**Tensor Core 内部根据当前 nonzero 数量动态改变实际 K issue 数量。** 当前公开代码反而越来越呈现一个固定、规则的硬件流水：Rubin dense/block-scale MMA 明确走更宽的 64B-K；而 sparsity 是在数据进入 sparse MMA 之前，通过 compressed operand + metadata 来表达。换句话说，**更像“固定宽 Tensor Core + operand-side compression/metadata”，而不是“动态 K 宽度 Tensor Core”。**

### 与此前判断相比是否需要修正

需要修正一处表述，但核心判断不变。

上次我们说：

> TE 的 UE5M3 POC 只有 quantize/dequantize，没有 GEMM，因此还不知道它最终是否进入 Tensor Core GEMM datapath。

现在这句话需要改成：

> **TE 的公开 UE5M3 recipe 仍然没有 GEMM，但 NVIDIA 的 cuDNN FROST 已经有了 Rubin SM107 E5M3/block-scale GEMM 和 MoE GEMM pipeline。**

所以 **UE5M3 是 Rubin 原生 block-scale 数据路径一部分**的置信度，我会从此前大约 **70–80% 提高到 95% 左右**。

但与此同时，对“UE5M3/4-over-6 与 adaptive compression 是同一件事”的置信度应继续下降。现在证据越来越支持：

**E5M3 / NVFP4 / 64B-K = precision + block-scaling 路线**

**2:4 + metadata + sparse Softmax/MMA = activation adaptive-compression 路线**

两者最终共用 Rubin Tensor Core / Tensor Memory 基础设施，但不是同一个机制。

另外，本轮我没有发现新的公开 PTX 指令或 descriptor 把 adaptive compression 直接暴露出来。当前官方 PTX 仍是 **PTX ISA 9.3**；公开文档已经有 `tcgen05.mma.sp`、`tcgen05.mma.ws.sp` 以及完整的 sparsity metadata matrix layout，但 9.3 的新增特性列表里没有新的“adaptive compression”或 TMEM→2:4 compression 指令。([NVIDIA Docs][2]) 同样，当前公开的 cuBLAS 13.6.1 patch note 只涉及 Grouped GEMM heuristic、workspace/stream capture 等修复，没有 Rubin sparse/adaptive-compression API 新线索。([NVIDIA Docs][3])

因此，本轮最值得记住的结论是：

**Rubin 的 E5M3/NVFP4 路线已经从 TE 原型跨入实际 cuDNN GEMM pipeline；与此同时，Rubin SDPA 开始公开展示“score/Softmax 尽量留在 Tensor Memory 内完成 load-reduction/MMA-reduction”的实现。两者共同增强了一个判断：adaptive compression 很可能位于 TMEM→operand 的转换边界，通过生成 compressed nonzero + metadata 喂给固定宽 sparse MMA，而不是依赖 Tensor Core 内部动态改变 K 执行宽度。**

[1]: https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/?utm_source=chatgpt.com "Inside NVIDIA Rubin GPU Architecture: Powering the Era of Agentic AI | NVIDIA Technical Blog"
[2]: https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html?utm_source=chatgpt.com "Contents — PTX ISA 9.3 documentation"
[3]: https://docs.nvidia.com/cuda/cublas-patch-release-notes/?utm_source=chatgpt.com "cuBLAS Patches - Release Notes — CUBLAS Release Notes 13.2 documentation"


先把型号纠正一下：我没有查到 AMD 有 **MI335** 这个正式型号。结合你前面说的 PR #47896、`gfx950`，这里应该是 **MI355X**。MI355X 属于 **CDNA4 / gfx950**；如果你实际想问的是 **MI325X / gfx942**，我在后面单独给出区别。AMD 官方也明确把 MI325X 归为 CDNA3、MI355X 归为 CDNA4。([ROCm 文档][1])

先给核心结论：

> **PR #47896 的 TurboQuant 4-bit KV，并没有利用 MI355X 的原生 MXFP4 Tensor Core 来直接做 4-bit QK/PV。**
>
> 它做的是：
>
> **4-bit HBM 存储 → tile 内即时解码成 BF16 → BF16 MFMA → FP32 accumulator**
>
> 所以在这条 FlyDSL kernel 里：
>
> [
> \boxed{QK:\ BF16\times BF16\rightarrow FP32}
> ]
>
> [
> \boxed{Softmax:\ FP32}
> ]
>
> [
> \boxed{PV:\ BF16\times BF16\rightarrow FP32}
> ]
>
> TurboQuant 的“4 bit”是 **KV cache 的压缩表示**，不是 QK/PV 的 Tensor Core 计算精度。

而且这一点可以直接从现在 vLLM main 的 FlyDSL 源码里确认。

---

# 一、PR #47896 现在到底合进了什么

这个 PR 已经在 **2026-08-11** 合并进 vLLM，正是你说的这周。

最终 main 分支的自动选择条件比“AMD 有 4bit 就用”窄得多。`vllm/v1/attention/backends/turboquant_attn.py` 里实际判断：

```python
_gqa = self.num_kv_groups

flydsl_gqa_ok = (_gqa in (8, 16)) or (
    _gqa == 6 and is_flydsl_gqa6_available()
)

flydsl_eligible = (
    not self.tq_config.key_fp8
    and self.tq_config.key_mse_bits == 4
    and self.tq_config.effective_value_quant_bits == 4
    and self.head_size == 128
    and flydsl_gqa_ok
    and self.sinks is None
    and not (self.sliding_window and self.sliding_window > 0)
)
```

再加上 `is_flydsl_available()` 本身要求：

```text
ROCm
gfx950
FlyDSL 可用
```

否则退回 SoA Triton TurboQuant decode。

所以当前真正的 FlyDSL TQ4 profile 是：

| 条件                     | FlyDSL              |
| ---------------------- | ------------------- |
| MI355X / gfx950        | ✅                   |
| K = 4-bit MSE centroid | ✅                   |
| V = 4-bit uniform      | ✅                   |
| head_dim = 128         | ✅                   |
| GQA = 8                | ✅                   |
| GQA = 16               | ✅                   |
| GQA = 6                | ✅ 独立 MiniMax kernel |
| GQA 其他值                | Triton fallback     |
| K=FP8 / K=3bit         | Triton fallback     |
| sliding window         | Triton fallback     |
| attention sink         | Triton fallback     |

MiniMax-M2.5 的 GQA=6 确实是另一份 `tq_decode_gqa6.py`，内部计算方案与 8/16 路径相同。

---

# 二、MI355X 原生到底支持哪些“4 bit”？

这里要把 **硬件数值格式** 和 **软件压缩格式** 分开。

MI355X 是 CDNA4，AMD 产品页明确给出 **MXFP4 Matrix 峰值约 10.1 PFLOPS**，并支持 MXFP4/MXFP6。([AMD][2])

ROCm 数据类型文档进一步明确：

```text
__hip_fp4_e2m1
```

只在 CDNA4 上获得支持；CDNA3 没有。Matrix Core 的 `float4` 也只有 CDNA4 支持。([ROCm 文档][1])

因此 MI355X 上值得区分的几种“4 bit”是：

| 名称                 | 4-bit payload      | Scale                 | MI355X 原生 Matrix Core       | 说明                   |
| ------------------ | ------------------ | --------------------- | --------------------------- | -------------------- |
| **FP4 E2M1**       | E2M1               | 无/外部                  | ✅                           | 原始 4-bit float       |
| **OCP MXFP4**      | E2M1               | E8M0 / 32 elements    | **✅ 重点支持**                  | CDNA4 主力 FP4 GEMM 格式 |
| INT4 packed        | 4-bit integer/code | 软件定义                  | 不是公开的一等 4-bit MFMA datatype | 常用于压缩/权重             |
| **TurboQuant TQ4** | 4-bit index/code   | K norm / V scale+zero | ❌ 不能直接喂 MXFP4 MFMA          | 本次 PR                |
| **NVFP4**          | E2M1               | E4M3 / 16 + global    | ❌ 无原生 NVFP4 path            | NVIDIA 格式            |

AMD 当前文档里 MXFP4 的定义非常标准：

[
\boxed{
\text{32 个 E2M1}
+
\text{1 个 E8M0 scale}
}
]

也就是每 32 个值共用一个 8-bit exponent scale。([ROCm 文档][3])

gfx950 有专门的：

```text
v_mfma_scale_f32_16x16x128_f8f6f4
```

直接消费低比特 payload 和 scale，并累加到 FP32。AMD 自己也明确说明，这条指令可以直接处理 FP4/FP6 operand，而不是先恢复 BF16。([ROCm博客][4])

所以真正的 MXFP4 硬件路径是：

```text
E2M1 payload ─────┐
E8M0 scale ───────┤
                  ↓
     V_MFMA_SCALE_F32_...F8F6F4
                  ↓
               FP32 acc
```

这个和 NVIDIA Blackwell 的 block-scaled MMA 在思想上非常接近。

---

# 三、但 TurboQuant 4bit 根本不是 E2M1

这是理解 PR #47896 最关键的地方。

`turboquant_4bit_nc` 在 vLLM 中定义为：

```python
"turboquant_4bit_nc": {
    "key_quant_bits": 4,
    "value_quant_bits": 4,
    "norm_correction": True,
}
```

但这里的 4-bit K 和 V 完全不是一种格式。

## K：4 bit 是 centroid index

对于 K，TurboQuant 大致先做：

[
K \rightarrow \text{normalize}
\rightarrow Hadamard
\rightarrow Lloyd-Max
]

然后每一个元素不存真实数值，而是存：

[
i\in[0,15]
]

这个 4 bit 表示：

> “选第几个 centroid”。

有一个长度 16 的 centroid 表：

[
C=[c_0,c_1,\ldots,c_{15}]
]

运行时恢复：

[
\boxed{
K_i\approx C[\text{index}_i]\times norm_K
}
]

单元测试中 reference 就直接这么写：

```python
k_ref = centroids[k_idx.long()].float() * knorm.float().unsqueeze(-1)
```

并且两个 4-bit index 打包成一个 byte。

因此：

```text
0xA
```

不是：

```text
E2M1 的某个 FP4 数
```

而是：

```text
centroid[10]
```

---

# 四、V 的 4 bit 又是另一回事

V 使用普通的 per-vector affine uniform quantization：

[
v_{\min}=\min(V)
]

[
v_{\max}=\max(V)
]

[
s_V=\frac{v_{\max}-v_{\min}}{15}
]

然后：

[
q_i=
\operatorname{round}
\left(
\frac{V_i-v_{\min}}{s_V}
\right)
\in[0,15]
]

存的是：

```text
4-bit q_i
FP16 v_scale
FP16 v_zero
```

恢复：

[
\boxed{
V_i=q_i\times s_V+v_{\rm zero}
}
]

vLLM 当前 SoA store kernel 就直接这么实现：

```python
v_scale = (val_max - val_min) / 15.0

q_all = ...
    .clamp(0, 15)

packed_val = ...
```

然后 scale、zero 都存 FP16。

所以从物理意义看：

```text
TurboQuant K 4-bit = codebook index
TurboQuant V 4-bit = affine integer code
MXFP4 4-bit        = E2M1 floating-point
```

三个完全不是一个东西。

因此 **MI355X 的 MXFP4 MFMA 没办法直接吃 TurboQuant cache**。

---

# 五、所以 FlyDSL 真正在做什么？

这次 PR 的关键优化不是：

> “MI355X 有 FP4 Tensor Core，所以拿 4bit KV 直接做 attention”。

而是：

> **不要把整个 KV cache 从 4bit 解压成 BF16 写回 HBM，而是在 FlashAttention tile 被读到时，在 kernel 内部即时恢复这一小块 K/V，再立刻进行 BF16 MFMA。**

也就是：

```text
              HBM
               │
      TurboQuant 4-bit cache
               │
         ┌─────┴─────┐
         ↓           ↓
      K 4-bit      V 4-bit
      indices      affine code
         │           │
         ↓           ↓
   centroid LUT   scale + zero
         │           │
         ↓           ↓
        BF16        BF16
         │           │
         │           │
Q BF16 ──┘           │
   ↓                 │
 BF16 MFMA QK         │
   ↓                  │
 FP32 score           │
   ↓                  │
 FP32 softmax         │
   ↓                  │
 P BF16 ──────────────┘
            ↓
        BF16 MFMA PV
            ↓
         FP32 acc
```

这就是这条 kernel 的本质。

AMD 自己的 TurboQuant 博客也描述了：cache 仍然是 4-bit codebook representation，但 decode 时 on-the-fly dequant，并通过 GQA grouping、SoA layout、MFMA、LDS V staging 等方式消化这个额外恢复开销。([ROCm博客][5])

---

# 六、Q 在进入 QK 之前是什么精度？

这里源码非常漂亮。

FlyDSL launcher 先做 Hadamard/rotation：

```python
q_rot = (query.float() @ PiT).bfloat16()
```

主分支当前实际实现是：

```python
_q_float.copy_(query)       # BF16 -> FP32

torch.mm(
    _q_float,
    PiT_f32,
    out=_q_rot_f32,
)

_q_rot_out.copy_(_q_rot_f32)  # FP32 -> BF16
```

所以 rotation 本身：

[
\boxed{FP32\ GEMM}
]

rotation 后送入 attention kernel 的 Q：

[
\boxed{BF16}
]

注意这里之所以 Q 也要做 rotation，是因为 K 在存 cache 之前已经做了相同正交变换：

[
QK^T=(QP)(KP)^T
]

只要：

[
PP^T=I
]

QK 数学结果保持不变。

---

# 七、K 的 4bit → BF16 到底怎么恢复？

这是 `tq_decode.py` 最有价值的一段代码。

首先 centroid table 本身：

```python
cent_lds = ... T.f32
```

也就是说 centroid 放 LDS 后是：

[
FP32
]

K norm 是 cache 中的：

[
FP16
]

读取后：

```python
knorm_f16 = ...
knorm_f32 = arith.extf(T.f32, knorm_f16)
```

然后每个 4-bit nibble：

```python
nibble_idx = ...
cent_f32 = cent_lds.load([nibble_idx])

elem_bf16 =
    arith.trunc_f(
        T.bf16,
        cent_f32 * knorm_f32
    )
```

所以完整过程是：

[
index_{4bit}
\rightarrow C[index]_{FP32}
]

[
norm_{FP16}\rightarrow FP32
]

[
K_{tmp,FP32}
============

C[index]\times norm
]

然后：

[
\boxed{
K_{BF16}
========

cast_{BF16}(K_{tmp,FP32})
}
]

再写进 LDS。

因此甚至可以更精确地说：

> **TurboQuant K 的解码运算用 FP32 做，真正提供给 Matrix Core 的 K operand 是 BF16。**

---

# 八、QK 到底是什么精度？

源码直接调用：

```python
k_op = vector.bitcast(
    T.vec(8, T.bf16),
    kv_load
)

qk_acc =
    rocdl.mfma_f32_16x16x32_bf16(
        T.f32x4,
        [
            k_op,
            q_chunks[chk],
            qk_acc,
            ...
        ],
    )
```

对应 CDNA4 指令：

```text
v_mfma_f32_16x16x32_bf16
```

因此非常明确：

[
\boxed{
K: BF16
}
]

[
\boxed{
Q: BF16
}
]

[
\boxed{
QK accumulator: FP32
}
]

也就是：

[
\boxed{
BF16\times BF16\rightarrow FP32
}
]

而且 CDNA4 相比 CDNA3 新增了更宽的 BF16：

```text
16 × 16 × 32
```

MFMA，这也是为什么源码注释叫：

```python
# CDNA4 wide-K
```

AMD 的 CDNA4 优化文档也明确列出 `v_mfma_f32_16x16x32` 作为 BF16/FP16 的宽 K MFMA。([ROCm博客][6])

---

# 九、Score 和 softmax 又是什么精度？

QK 得到：

```text
qk_acc : FP32
```

然后直接：

```python
qk_acc = _vsplat_mul(qk_acc, QK_SCALE)
```

所以：

[
\frac{QK^T}{\sqrt d}
]

仍然在：

[
\boxed{FP32}
]

之后 online softmax：

```python
new_max
running_max
running_sum
exp2(...)
```

全部是 FP32。

所以：

[
\boxed{
Score=FP32
}
]

[
\boxed{
Softmax max/sum=FP32
}
]

[
\boxed{
P_{\rm softmax}=FP32
}
]

到这里和一个正常的高质量 BF16 FlashAttention 路径非常接近。

---

# 十、但是进入 PV 之前，P 被降成 BF16

这个细节很关键。

代码写得非常直接：

```python
# P operand B layout matches qk_acc
p_bf16 = arith.trunc_f(
    T.vec(4, T.bf16),
    qk_acc
)

p_op = vector.bitcast(
    T.vec(4, T.i16),
    p_bf16
)
```

也就是说：

[
P_{FP32}
\rightarrow
\boxed{P_{BF16}}
]

然后才进入第二个 MFMA。

---

# 十一、V 也是 4bit → FP32 → BF16

V cache 中实际存：

```text
4 bit code
FP16 v_scale
FP16 v_zero
```

kernel 里：

```python
vscale_f32 =
    extf(FP32, vscale_FP16)

vzero_f32 =
    extf(FP32, vzero_FP16)

nibble_f32 =
    int4_code -> FP32

elem_f32 =
    nibble_f32 * vscale_f32
    + vzero_f32

elem_bf16 =
    trunc(BF16, elem_f32)
```

因此：

[
q_{4bit}
\rightarrow FP32
]

[
V_{tmp}
=q_{4bit}s_V+z_V
]

然后：

[
\boxed{
V_{BF16}=cast_{BF16}(V_{tmp})
}
]

同样没有把完整 BF16 V cache 写回 HBM，而只放当前 tile 到 LDS。

---

# 十二、PV 到底是什么精度？

代码：

```python
acc_pv[h] =
    rocdl.mfma_f32_16x16x16bf16_1k(
        T.f32x4,
        [
            v_op_raw,
            p_op,
            acc_pv[h],
            ...
        ],
    )
```

因此：

[
P=BF16
]

[
V=BF16
]

[
Accumulator=FP32
]

也就是：

[
\boxed{
PV:\ BF16\times BF16\rightarrow FP32
}
]

最终每个 partition 的 output 再存成 BF16，而 online-softmax 的 max/sum 保持 FP32。FlyDSL launcher 对 buffer 类型也写得很清楚：

```text
segm_out : BF16
segm_max : FP32
segm_sum : FP32
```

---

# 十三、因此整条 FlyDSL TQ4 precision graph 可以定死了

对当前 PR #47896 的 gfx950 路径：

| 阶段                      | 存储/计算格式                                 |
| ----------------------- | --------------------------------------- |
| 原始 Q                    | BF16                                    |
| Q rotation 输入           | FP32                                    |
| Q rotation GEMM         | FP32                                    |
| rotation 后 Q            | **BF16**                                |
| K cache                 | **4-bit centroid index + FP16 norm**    |
| K centroid LUT          | FP32                                    |
| K 恢复运算                  | FP32                                    |
| QK 输入 Q                 | **BF16**                                |
| QK 输入 K                 | **BF16**                                |
| QK MFMA                 | **BF16 × BF16**                         |
| QK accumulator          | **FP32**                                |
| score scaling           | FP32                                    |
| online softmax          | **FP32**                                |
| P → PV                  | **BF16**                                |
| V cache                 | **4-bit affine code + FP16 scale/zero** |
| V 恢复运算                  | FP32                                    |
| PV 输入 V                 | **BF16**                                |
| PV MFMA                 | **BF16 × BF16**                         |
| PV accumulator          | **FP32**                                |
| partition output        | BF16                                    |
| partition softmax stats | FP32                                    |

所以一句话概括：

[
\boxed{
\text{TQ4 Storage}
\rightarrow
\text{BF16 Attention Compute}
}
]

---

# 十四、这和我们前面看的 NVIDIA NVFP4 KV 有一个很有意思的对照

两家公司当前实现实际上都在做：

> **低 bit 是 HBM/cache format，不一定是 Attention compute format。**

但目标 compute dtype 不一样。

NVIDIA 当前 production NVFP4 KV：

```text
NVFP4 cache
   ↓
FP8 E4M3
   ↓
FP8 QK / PV
```

而 AMD 这条 TurboQuant：

```text
TurboQuant 4bit cache
   ↓
BF16
   ↓
BF16 QK / PV
```

区别的根本原因不是：

> MI355X 没有 FP4 Tensor Core。

恰恰相反，MI355X 的 **原生 MXFP4 能力非常强**，峰值约 10 PFLOPS。([AMD][2])

真正原因是：

[
\boxed{
TurboQuant\ 4bit \neq MXFP4
}
]

TurboQuant 的码字不能作为：

```text
E2M1
```

直接送进：

```text
v_mfma_scale_f32_16x16x128_f8f6f4
```

---

# 十五、为什么 AMD 不把恢复后的 KV 再转成 MXFP4？

理论上可以设计类似：

```text
TQ4
 ↓ centroid decode
BF16
 ↓
MXFP4
 ↓
FP4 MFMA
```

但这很可能得不偿失。

因为 TQ4 的主要目标本来就是：

[
\text{减少 HBM KV read}
]

如果每个 decode tile 又要：

1. TQ codebook decode；
2. 重新计算每 32 values MXFP4 scale；
3. BF16→E2M1 quant；
4. scale 重排；
5. 再送 scaled MFMA；

那么你为了省一点 MFMA arithmetic，增加了一堆在线量化工作。

而 long-context decode 本身更主要受制于：

[
\boxed{\text{HBM bandwidth}}
]

不是 BF16 MFMA 峰值。

所以 AMD 现在选择：

```text
HBM：4bit
片上：快速恢复 BF16
Tensor Core：成熟 BF16 MFMA
```

是相当合理的平衡。

AMD 自己的性能结果也表明，优化后的 FlyDSL TQ 在测试中已经接近 BF16 baseline，4-bit cache 的核心收益来自显著降低 HBM 流量。([ROCm博客][5])

---

# 十六、如果你真正想研究 MI355X 的“原生 4-bit Attention”，应该看另一条路线

如果目标不是 TurboQuant，而是：

> **能不能让 KV cache 直接保存硬件兼容的 FP4，然后 QK/PV 真正执行 FP4 MFMA？**

那么 MI355X 应该重点研究的是：

[
\boxed{\text{MXFP4}}
]

因为 CDNA4 有：

```text
V_MFMA_SCALE_F32_16X16X128_F8F6F4
```

并且 scale 是 instruction 的直接输入，而不是先恢复 BF16。AMD 已公开展示 MXFP4 GEMM 就是：

```text
FP4 E2M1 payload
        +
E8M0 / 32 scale
        ↓
scaled MFMA
        ↓
FP32 accumulator
```

([ROCm博客][7])

如果把这种思路搬到 Attention，理论目标会是：

```text
Q MXFP4 + K MXFP4
        ↓
scaled MFMA
        ↓
FP32 Score
        ↓
FP32 softmax
        ↓
P MXFP4/BF16 + V MXFP4
        ↓
scaled MFMA
```

但 **PR #47896 完全没有走这条路线**。

---

## 最后补充 MI325X

如果你原来“MI335”其实想说的是 **MI325X**，那么差别非常大：

[
MI325X = CDNA3 = gfx942
]

ROCm 官方精度矩阵显示：

```text
__hip_fp4_e2m1
CDNA3: ❌
CDNA4: ✅
```

并且 CDNA3 Matrix Core 也没有 FP4/FP6，只有 FP8；CDNA4 才增加 FP4/FP6 和 block-scaled MFMA。([ROCm 文档][1])

因此 PR #47896 的这个 FlyDSL kernel：

[
\boxed{\text{不会在 MI325X 上启用}}
]

因为源码直接 gate：

```python
on_gfx950()
```

它就是为 MI350X/MI355X 这一代 CDNA4 写的。

---

我认为这条工作最值得进一步挖的地方，已经不是 TurboQuant 算法本身，而是**为什么 AMD 在有 native MXFP4 MFMA 的情况下，依然选择 TQ4→BF16 MFMA**。下一步可以把 `tq_decode.py` 的实际生成 ISA 拆出来，与真正 MXFP4 kernel 的 `v_mfma_scale_f32_16x16x128_f8f6f4` 做逐条对照：HBM bytes、LDS bytes、dequant 指令数、MFMA 数量和 decode roofline 会非常直观。

[1]: https://rocm.docs.amd.com/en/docs-7.2.4/reference/precision-support.html "Data types and precision support — ROCm Documentation"
[2]: https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html "AMD Instinct™ MI355X GPUs"
[3]: https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/diffusion-models/mxfp4_training.html "MXFP4 training for Flux models — AMD Primus 26.4"
[4]: https://rocm.blogs.amd.com/artificial-intelligence/w4a6-quant-mm/README.html?utm_source=chatgpt.com "MXFP6 and MXFP4 Mixed Precision for Accelerating Dense LLMs on AMD Instinct MI355X — ROCm Blogs"
[5]: https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html "Productionizing TurboQuant on AMD GPUs for KV-Cache-Bound LLM Inference — ROCm Blogs"
[6]: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html?utm_source=chatgpt.com "FP8 GEMM Optimization on AMD CDNA™4 Architecture — ROCm Blogs"
[7]: https://rocm.blogs.amd.com/software-tools-optimization/kimi-k25-mxfp4-atom/README.html?utm_source=chatgpt.com "Serve Kimi-K2.5-MXFP4 on MI355X with ATOM — ROCm Blogs"

有，而且这里需要区分 **“UltraQuant 论文里的完整算法”** 和 **“已经公开 upstream 的底层实现”**。截至 **2026-08-17**，我的判断是：

> **UltraQuant 的核心硬件路线已经在 AMD 的 AITER / vLLM 代码中明显落地；但我没有找到一个与论文 1:1 对应、直接叫 `UltraQuant`、可以通过 `--kv-cache-dtype ultraquant` 打开的通用 vLLM backend。**
>
> 更准确地说：
>
> * **Ultra-TQ**：已经非常明确地进入 vLLM，就是我们上一轮看的 PR #47896。
> * **UltraQuant proper**：`FP4 E2M1 + UE8M0 + scaled-MFMA` 这些关键硬件组件已经进入 AITER，并出现在 vLLM 的 DeepSeek-V4 专用路径里；但**当前公开代码的量化 recipe 与论文 UltraQuant 并不完全相同**。

### 1. 先明确 UltraQuant 论文到底创新在哪

UltraQuant 不是把 TurboQuant 的 centroid kernel 再优化一下。论文明确把两条路线分开：

* **Ultra-TQ**：仍然保存 TurboQuant 的 4-bit centroid/codebook representation，通过 layout、lookup、MFMA scheduling 优化软件 dequant。
* **UltraQuant**：直接扔掉 centroid lookup，用硬件原生 **FP4 E2M1 grid** 近似它，从而把 dequant 融入 CDNA4 的 scaled-MFMA。([arXiv][1])

论文里的 UltraQuant cache 是：

[
\boxed{
32\times E2M1 + 1\times UE8M0
}
]

即每 32 个元素：

* 32 个 E2M1，4 bit/value；
* 一个 8-bit UE8M0 scale；

总共：

[
16+1=17\text{ bytes}
]

所以有效 bit 数：

[
\frac{17\times8}{32}=4.25\text{ bit/value}
]

论文明确说这是为了让 CDNA4 Matrix Core 直接消费，不需要把 KV materialize 成 BF16。([arXiv][1])

计算目标是：

[
\boxed{
Q_{FP8}\times K_{FP4}\rightarrow FP32
}
]

以及相应的 scaled-FP4 attention 路径。Q 会先 Hadamard rotate，再转成 FP8 E4M3；KV 则保持 FP4+UE8M0，使用 `MFMA_SCALE_F32_*_F8F6F4`。([arXiv][1])

这与前面 TurboQuant FlyDSL 的：

[
BF16\times BF16\rightarrow FP32
]

是完全不同的路线。

---

## 2. Ultra-TQ 已经明确进了 vLLM

这一部分就是我们前面看的 **vLLM PR #47896**。

它 upstream 的：

```text
FlyDSL TurboQuant 4-bit KV decode
gfx950 / MI355X
```

本质仍然是：

```text
4-bit centroid/index cache
          ↓
tile 内 lookup/dequant
          ↓
BF16 K/V
          ↓
BF16 MFMA
```

所以它基本可以直接对应论文所说的：

[
\boxed{\text{Ultra-TQ}}
]

而不是 UltraQuant proper。

论文自己也把 Ultra-TQ 描述成“保持 TurboQuant representation，优化 lookup/layout/MFMA scheduling”。([arXiv][1])

---

# 3. 真正有意思的是：AITER 已经出现 UltraQuant 风格的 FP4 KV cache 写入代码

最直接的一笔是：

**AITER PR #4029：**

> `DeepSeek-V4 FP4: fused_compress FP4 scatter + rmsnorm_rope_rotate FP4 KV-cache kernel`

它已经在 **2026-07-28 合并**。

这个 PR 加入：

```text
csrc/kernels/dsv4_rotate_quant.cu
```

核心功能就是：

```text
RMSNorm
   ↓
RoPE
   ↓
optional Hadamard rotation
   ↓
FP4 E2M1 quantization
   ↓
UE8M0 scale
   ↓
paged KV-cache write
```

PR 自己明确写的是：

> fused RMSNorm + RoPE + optional Hadamard rotate + FP4(E2M1) quant + paged KV-cache write.

这已经和 UltraQuant 的 encoder 非常接近了。

---

# 4. 我把这段 kernel 往下追了，确实是真 FP4，不是伪量化

现在 AITER：

```text
csrc/kernels/dsv4_rotate_quant.cu
```

中有：

```cpp
hadamard_rotate_activation_fp4quant_kernel
```

它先做真正的 Hadamard butterfly，最后归一化：

```cpp
af[i] = af[i] * dim_rsqrt;
```

然后：

```cpp
if constexpr(std::is_same_v<DTYPE_O, opus::fp4_t>)
```

进入 FP4 quantization。

真正的数据类型是：

```cpp
opus::fp4_t
```

而不是 uint4/int4 做软件模拟。

之后求 group absmax：

```cpp
absMax = fmaxf(absMax, fabsf(af[i]));
```

然后生成 E8M0：

```cpp
uint8_t scale_e8m0 = ceil_pow2(
    absMax * (1.0f / fp4_max)
);
```

再：

```cpp
scale_f32 =
    bitcast<float>(scale_e8m0 << 23);
```

最后：

```cpp
store_vector<..., opus::fp4_t>(
    ...,
    af,
    ...,
    scale_f32
);
```

所以公开代码里已经存在真正的：

[
\boxed{
BF16
\rightarrow Hadamard
\rightarrow E2M1
+
UE8M0
}
]

这绝对不是普通的“INT4 KV cache”。

---

# 5. paged KV cache layout 也已经按 FP4 hardware format 做了

更具体地，代码专门定义了：

```cpp
kv_fp4_preshuffle_offset(...)
kv_scale_preshuffle_offset(...)
store_kv_fp4_preshuffle(...)
```

并注明：

```text
FlyDSL / pa_mqa_logits_fp4 KV preshuffle
```

也就是说 cache layout 根本不是：

```text
[token][head][packed int4]
```

这么简单。

它专门排成适合后面的 FP4 Matrix Core reader 的 layout：

```text
[num_blocks,
 k_tiles,
 4,
 kv_block_size,
 16]
```

scale 同样单独 preshuffle。

这已经是典型的：

> **storage format 与 MFMA operand layout 联合设计**

而不是单纯为了省 HBM。

---

# 6. 更关键：AITER 已经有直接读取 FP4 KV 的 scaled-MFMA kernel

另一个关键 PR 是：

**AITER #4230**

```text
[FLYDSL] Support paged mqa logits fp4 varqlen kernel
```

同样在 **2026-07-28** 合并。

文件：

```text
aiter/ops/flydsl/kernels/pa_mqa_logits_fp4.py
```

这里已经出现我们上一轮讨论过、但 TurboQuant kernel **没有使用**的 CDNA4 指令：

```python
rocdl.mfma_scale_f32_16x16x128_f8f6f4(...)
```

注意这里不是：

```text
mfma_f32_..._bf16
```

而是：

```text
mfma_scale_f32_16x16x128_f8f6f4
                    ^^^^^^^
```

这才是 UltraQuant 论文真正想利用的硬件能力。

它同时给 MFMA 输入：

```text
q operand
q_scale (UE8M0)

kv operand
kv_scale (UE8M0)
```

核心调用：

```python
accs[mi_idx] =
    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.f32x4,
        [
            q_a_ops[k_tile][mi_idx],
            kv_b,
            accs[mi_idx],
            ...
            q_scale_ops[k_tile][mi_idx],
            ...
            kv_scale_val,
        ],
    )
```

所以这里是真正：

[
\boxed{
lowbit\ operand
+
UE8M0\ scale
\rightarrow
scaled\ MFMA
\rightarrow
FP32
}
]

而不是：

```text
FP4 cache
 ↓
BF16 tensor
 ↓
BF16 MFMA
```

---

# 7. 但是：这一条还不能直接等号成“论文 UltraQuant”

这里有两个重要差异。

### 差异一：当前 AITER 这个 DeepSeek-V4 kernel 的 Q 也是 FP4

它的公开 API 明确是：

```python
def flydsl_pa_mqa_logits_fp4(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    ...
)
```

所以这个具体 kernel 更接近：

[
\boxed{
Q_{FP4}\times K_{FP4}
}
]

而论文 UltraQuant 明确规定：

[
\boxed{
Q_{FP8}\times K_{FP4}
}
]

论文写得非常清楚：query Hadamard rotation 之后被 round 到 **FP8 E4M3**，再与 FP4 K 走 scaled-MFMA。([arXiv][1])

因此：

> AITER 已经把 **FP4+UE8M0+scaled-MFMA 的底层机制**实现了，但这个 DeepSeek-V4 `pa_mqa_logits_fp4` 不是论文 MiniMax/Qwen UltraQuant kernel 的原样公开版本。

---

# 8. 第二个差异更加关键：scale recipe 不一样

UltraQuant 论文不是标准 MXFP4 的 absmax recipe。

论文给的是：

[
s=c\cdot m
]

其中：

[
m=\max_i |x_i|
]

并通过：

[
E=
\operatorname{round}
\left(
\log_2(c,m)
\right)
]

生成 UE8M0 exponent。

它专门离线搜索：

[
\boxed{c=0.156}
]

并称这个为 **constant-optimized scaling**。([arXiv][1])

这是 UltraQuant 的一个重要数值创新。

---

## 9. 但现在公开 vLLM / AITER 用的是标准 MXFP4 scale

例如当前 vLLM DeepSeek-V4 文件：

```text
vllm/models/deepseek_v4/common/ops/
fused_compress_quant_cache.py
```

已经明确有：

```python
_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
```

而 cache 定义是：

```text
32 × E2M1
+ UE8M0
```

这一部分和 UltraQuant 一样。([GitHub][2])

但它的 scale 是：

[
\boxed{
s=
2^{\lceil\log_2(amax/6)\rceil}
}
]

源码注释原文就是：

```text
Per-32-element block scale =
    2^ceil(log2(amax / 6.0))
```

([GitHub][3])

也就是经典 MXFP4 的：

[
\frac{amax}{E2M1_{\max}}
]

再 round-up 到 power-of-two。

我也专门在当前 vLLM/AITER 中查了：

```text
0.156
UltraQuant
```

没有找到相应实现。

所以：

[
\boxed{
\text{当前公开 DeepSeek-V4 FP4 KV}
\neq
\text{论文 exact UltraQuant quantizer}
}
]

---

# 10. 因此现在可以把 AMD 的公开路线画成三层

| 路径                   | Cache             | Attention compute         | 公开状态                    |
| -------------------- | ----------------- | ------------------------- | ----------------------- |
| vLLM OSS TurboQuant  | centroid/int code | 软件恢复                      | 已公开                     |
| **Ultra-TQ**         | centroid 4bit     | 恢复 BF16 → BF16 MFMA       | **vLLM #47896 已合入**     |
| **UltraQuant paper** | E2M1 + UE8M0      | **FP8 × FP4 scaled MFMA** | 完整通用版本未发现公开 upstream    |
| DeepSeek-V4 FP4 path | E2M1 + UE8M0      | native scaled MFMA        | **AITER/vLLM 已公开，模型特化** |

所以我会把目前的状态概括成：

> **UltraQuant 的“硬件架构路线”基本已经公开实现；UltraQuant 的“论文完整数值 recipe + 通用 MHA serving backend”尚未完整公开。**

---

# 11. 这也解释了为什么之前我们看 TurboQuant 时没有发现 native FP4 MFMA

这两条开发线实际上是并行存在的：

### vLLM PR #47896

```text
TQ4 code
  ↓
centroid lookup
  ↓
BF16 K
  ↓
BF16 MFMA
```

核心：

```text
mfma_f32_16x16x32_bf16
```

### AITER DeepSeek-V4 / UltraQuant-style

```text
E2M1 payload
      +
UE8M0 scale
      ↓
MFMA_SCALE_F32_*_F8F6F4
      ↓
FP32
```

第二种才真正释放 MI355X/CDNA4 的 FP4 Matrix Core。

而论文 UltraQuant 的核心论点恰恰就是：

> TurboQuant centroid 的 MSE 更好，但软件 lookup 太贵；**牺牲一点 codebook 最优性，把表示改成硬件原生 E2M1，反而获得更好的端到端 serving performance。** ([arXiv][1])

---

## 我现在最关注的代码位置

如果继续追 UltraQuant，我建议下一步就盯这四处：

```text
ROCm/aiter
├── csrc/kernels/dsv4_rotate_quant.cu
│   └── Hadamard → FP4 → E8M0
│
├── aiter/ops/flydsl/kernels/
│   ├── pa_mqa_logits_fp4.py
│   └── pa_mqa_logits_fp4_prefill.py
│       └── MFMA_SCALE_F32_...F8F6F4
│
└── aiter/ops/flydsl/kernels/fused_compress_attn.py

vllm
└── vllm/models/deepseek_v4/common/ops/
    └── fused_compress_quant_cache.py
        └── MXFP4 E2M1 + UE8M0 paged cache
```

其中真正最值得继续拆的是：

```python
rocdl.mfma_scale_f32_16x16x128_f8f6f4
```

我们可以下一步**直接把这个函数的 9 个 operand 各自对应到 CDNA4 ISA 的 A/B format、A/B scale format，弄清楚 `4,4,0,...` 这些立即数到底编码什么**；然后再对照 UltraQuant 论文的 **FP8 Q × FP4 K**，判断 AITER 现有 kernel 只差一个 Q operand format 改动，还是需要重新设计 operand layout。这个会比只看论文更能判断“UltraQuant 离正式 vLLM 通用落地还有多远”。

[1]: https://arxiv.org/abs/2606.20474 "UltraQuant: 4-bit KV Caching for Context-Heavy Agents"
[2]: https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py?utm_source=chatgpt.com "vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py at main · vllm-project/vllm · GitHub"
[3]: https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py "vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py at main · vllm-project/vllm · GitHub"





