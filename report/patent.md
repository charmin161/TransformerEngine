我按 **Google Patents** 页面里的 **Publication date（页面公布/公告时间）** 和 **Current Assignee / Original Assignee（当前/原始受让单位）** 来整理。先提醒一句：Google Patents 页面自己也注明了 assignee 可能并非最终法律结论，下面的“单位”是按页面字段记录的。([谷歌专利][1])

先给一个总判断：这 10 篇专利基本围绕一条主线——**如何让神经网络里的稀疏性（sparsity，很多元素是 0 或近似 0）被硬件直接利用**。更具体地说，可以分成四组：
一组是**在取数阶段就判断“这个值值不值得算”**；一组是**稀疏卷积 / 稀疏矩阵乘的专用加速器**；一组是**压缩、解压、转置友好的矩阵格式**；最后一组是**面向结构化稀疏矩阵乘加（MMA）的软件-硬件接口**。这 10 篇里，除了 **US11586417B2** 属于 **Qualcomm Inc**，其余页面上的 Current Assignee 都是 **Nvidia Corp**。([谷歌专利][1])

下面先把高频硬件名词统一解释掉，后面就不重复展开太长了：

* **Load/Store Unit（LSU，加载/存储单元）**：负责执行“从内存读数据 / 把数据写回内存”的指令单元。
* **Predicate signal / predicate register（谓词信号 / 谓词寄存器）**：本质上是一个布尔标记，后续指令可以根据它决定“执行”还是“跳过”。
* **Decoder（译码器）**：把机器指令里的比特字段解释成“要做什么、比较什么、结果写到哪”。
* **PE, Processing Element（处理单元）**：加速器里重复排布的基础计算块。
* **Multiplier array（乘法阵列）**：很多乘法器并行工作的一组硬件。
* **Accumulator / accumulator array / accumulator tree（累加器 / 累加阵列 / 累加树）**：把大量乘法结果合并相加的硬件。
* **MAC, Multiply-Accumulate（乘加）**：先乘再加，是神经网络里最基础的计算。
* **MMA, Matrix Multiply Accumulate（矩阵乘加）**：把 MAC 扩展到矩阵 / 子矩阵级别的块运算。
* **HMMA / IMMA**：分别是 **Half-precision Matrix Multiply Accumulate（半精度矩阵乘加）** 和 **Integer Matrix Multiply Accumulate（整数矩阵乘加）**。
* **Metadata（元数据）**：压缩后留下来的“位置信息 / 索引信息”，告诉硬件非零值原来在什么位置。
* **Structured sparsity（结构化稀疏）**：零值不是随机分布，而是按固定规则分布。
* **2:4 structured sparsity（2:4 结构化稀疏）**：每连续 4 个元素里，有 2 个是 0、2 个是非 0。
* **Crossbar（交叉开关）**：把多个输入灵活路由到多个输出的互连电路。
  这些术语在这组专利里是高频核心概念。([谷歌专利][2])

---

## 1. US11586417B2

**Exploiting activation sparsity in deep neural networks**
**中文**：在深度神经网络中利用激活稀疏性

**基本信息**：页面的 Publication date 是 **2023-02-21**，Current Assignee / Original Assignee 都是 **Qualcomm Inc**，Filing date 是 **2018-09-28**。([谷歌专利][1])

**主要内容**：这篇专利的核心不是“把整个网络重新设计成稀疏格式”，而是抓住 **activation tensor（激活张量：某一层输出的多维数据块）** 里天然会出现很多 0 这一点，只把非零激活压成一个 **compressed activation tensor（压缩激活张量）**，然后再和 **weight tensor（权重张量）** 继续计算。专利还进一步写到，非零激活会被打包到 **MAC hardware（乘加硬件）** 的 **vector lanes（向量通道：并行数据通道）** 上，并把对应的原始通道号之类的 **metadata（元数据）** 存到 **on-chip memory（片上存储）**，必要时还会经过 **FIFO buffer（先进先出缓冲）** 暂存。它的本质目标是：当输入激活很稀疏时，不要让乘加阵列空转，而是尽量把“有效非零值”重新装配后送给 MAC。([谷歌专利][1])

**一句话看法**：这是比较典型的“**利用激活稀疏性来提高 MAC 利用率**”路线，偏 **数据打包 + lane 重映射 + 片上元数据管理**。([谷歌专利][1])

---

## 2. US10503507B2

**Inline data inspection for workload simplification**
**中文**：用于简化工作负载的内联数据检查

**基本信息**：Publication date 是 **2019-12-10**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2017-08-31**。([谷歌专利][2])

**主要内容**：这篇专利的想法非常直接：数据刚被 **LSU（加载/存储单元）** 取出来时，就由一个 **inspection circuit（检查电路）** 顺手判断“它是不是 0”，然后把结果作为 **predicate signal（谓词信号）** 一起送回去。这样后面的计算单元就能根据这个布尔标志跳过无意义的运算。说明书里进一步扩展到“是否小于某个 **threshold value（阈值）**”的判断，并且可以通过替换成“带 inline inspection 的 load 指令”来实现；如果谓词成立，后面的一些乘法甚至整个 tile（分块）内的部分乘法都可以不做。它本质是在**取数阶段提前做零值 / 阈值判定**，把“数据依赖的跳过”前移。([谷歌专利][2])

**一句话看法**：这是“**把稀疏判断嵌进 load 指令路径**”的做法，偏 **指令级优化 + 谓词控制 + 跳过无效乘法**。([谷歌专利][2])

---

## 3. US11609761B2

**Inline data inspection for workload simplification**
**中文**：用于简化工作负载的内联数据检查

**基本信息**：Publication date 是 **2023-03-21**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2019-12-09**。页面说明它是对 **US15/693,345** 的 **Continuation（继续申请 / 延续申请）**。([谷歌专利][3])

**主要内容**：如果说上面那篇更强调“加载时顺手检查”，这篇则更明确地把逻辑写成：由 **decoder（译码器）** 对 load instruction（加载指令）译码，驱动处理器里的电路判断“加载的数据是否**超过**阈值”，并把这个 indication（指示结果）存下来。专利权利要求甚至写到：只有当数据超过阈值时，load 才执行；后面的指令序列也可以基于这个 indication 决定是否继续执行。也就是说，这一版更像是把 **threshold-based gating（基于阈值的门控）** 机制正式固化到指令和处理器语义里。([谷歌专利][3])

**和 US10503507B2 的关系**：两篇标题相同，但 **US11609761B2** 明确是前案的 continuation。你可以把它理解成同一条技术路线从“零值判定 / 谓词跳过”进一步细化到“阈值判断 / 结果存储 / 后续指令门控”。([谷歌专利][3])

---

## 4. US10096134B2

**Data compaction and memory bandwidth reduction for sparse neural networks**
**中文**：面向稀疏神经网络的数据压紧与内存带宽降低

**基本信息**：Publication date 是 **2018-10-09**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2017-02-01**。([谷歌专利][4])

**主要内容**：这篇专利处理的是更底层的“搬数据”问题。摘要里写得很清楚：当送往 **PE（Processing Element，处理单元）** 的多比特数据等于 0 时，**memory interface（内存接口）** 不必把完整多比特值都搬过去，而是可以只送一个 1-bit signal（1 位信号）说明“这里是 0”；反过来，如果拿到的是 **compacted data sequence（压紧数据序列）**，就由 **expansion engine（展开引擎）** 把非零值插回原来的位置、把 0 补回来，再交给 PE。说明书里还提到可以对 activations（激活）和 weights（权重）做 thresholding（阈值化）和 pruning（剪枝）后再压紧。核心价值是：**减少内存带宽消耗**，把“0 的存在”从“显式搬运很多比特”变成“更便宜的表示”。([谷歌专利][4])

**一句话看法**：这篇是典型的 **memory bandwidth optimization（内存带宽优化）** 专利，偏 **压紧 / 展开引擎 + 内存接口到计算阵列之间的传输优化**。([谷歌专利][4])

---

## 5. US20200285618A1

**Decompression techniques for processing compressed data suitable for artificial neural networks**
**中文**：适用于人工神经网络压缩数据处理的解压技术

**基本信息**：Publication date 是 **2020-09-10**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2019-03-20**。同一页面还列出该家族后来授权为 **US11379420B2**，授权公告时间是 **2022-07-05**。([谷歌专利][5])

**主要内容**：它解决的是“稀疏数据压缩以后，怎么以硬件友好的方式解压”。专利定义了 **N:M compression（N:M 压缩：原本 M 个位置里只保留 N 个有效值，并配元数据指明原位置）**，输入是 N 个有效值和相应 metadata，输出是恢复后的 M 元数据结构。更有意思的是，它不只讲“完全解压”，还讲 **partial decompression（部分解压）**：比如从更密的压缩格式，先变成“较少压缩”的中间格式，例如页面里就列出 **2:16 to 2:4 partial decompression（从 2:16 到 2:4 的部分解压）**。专利明确说它的目标之一是避免传统解压对复杂 arithmetic units（算术单元）的依赖，降低硬件成本。([谷歌专利][5])

**一句话看法**：这篇很关键，因为它把“压缩表示”和“实际可执行硬件数据流”连起来了，偏 **instruction-assisted decompression（指令辅助解压）** 和 **多级压缩格式转换**。([谷歌专利][5])

---

## 6. US11127167B2

**Efficient matrix format suitable for neural networks**
**中文**：适用于神经网络的高效矩阵格式

**基本信息**：Publication date 是 **2021-09-21**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2019-04-29**。([谷歌专利][6])

**主要内容**：这篇不是在讲某个单独算子，而是在讲一种“**格式**”：矩阵被存成压缩形式之后，**transpose operation（转置操作：把行列互换）** 可以在 decompression（解压）过程中顺手完成，而不必先还原成 dense matrix（稠密矩阵）再做转置。页面还写到，甚至可以在不真正生成原始矩阵、转置矩阵、dense 矩阵的情况下，先根据 metadata 推出 non-zero values（非零值）的位置。这个思路非常适合神经网络里常见的“矩阵乘之前经常要换布局 / 转置 / 重排”的情况。([谷歌专利][6])

**一句话看法**：这篇偏 **data layout（数据布局）设计**，目标是让“稀疏 + 转置 + 解压”三件事尽量在同一条流水里完成。([谷歌专利][6])

---

## 7. US11392829B1

**Managing data sparsity for neural networks**
**中文**：神经网络中的数据稀疏性管理

**基本信息**：Publication date 是 **2022-07-19**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2019-04-02**。([谷歌专利][7])

**主要内容**：这篇专利的重点是 **structured sparsity（结构化稀疏）**。摘要里说得很明确：它不是只利用“碰巧有零”，而是给 sparse matrix（稀疏矩阵）施加一个 **sparsity constraint（稀疏约束）**，并且这个约束要在所有 submatrices（子矩阵）上都满足，从而形成“细粒度但分布均匀”的结构化稀疏。说明书给的流程是：先确定全局稀疏约束，再把矩阵切成等大小子矩阵，然后把数据分配到这些子矩阵里，使每块都满足约束；之后矩阵就可以从例如 **16×16 压成 16×8**。最后这些矩阵用于 **MMA（矩阵乘加）**。这其实是在为硬件创造“可预测、可均衡、易并行”的稀疏分布。([谷歌专利][7])

**一句话看法**：这篇的价值不在“压得多狠”，而在**把稀疏分布规范化**，让硬件更容易稳定吃到收益。([谷歌专利][7])

---

## 8. US20220366007A1

**Performing matrix value indication**
**中文**：执行矩阵值指示 / 标示矩阵中的有效值

**基本信息**：Publication date 是 **2022-11-17**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2022-05-12**，页面状态是 **Pending（待审 / 未授权）**。页面还显示这是一组并列申请中的一篇，同族里还有：

* **US17/743,340**：Application programming interface to decompress data（用于解压数据的应用程序接口）
* **US17/743,334**：Matrix multiplication and accumulation operations on compressed matrices（在压缩矩阵上做矩阵乘和累加）
* **US17/743,330**：Application programming interface to compress data（用于压缩数据的应用程序接口）
  这说明它不是孤立的一篇，而是一个“压缩 / 解压 / MMA / 值指示”组合方案的一部分。([谷歌专利][8])

**主要内容**：摘要写到，这篇覆盖四件事：

1. 指示矩阵里哪些值是 non-zero values（非零值）；
2. 通过 API（Application Programming Interface，应用程序接口：软件调用底层能力的标准入口）压缩矩阵；
3. 在至少一个输入矩阵已压缩的前提下执行 MMA；
4. 通过 API 解压矩阵。
   页面里还明确给了 **2:4 structured sparsity（2:4 结构化稀疏）** 的例子，并写到 MMA 既可以是 **HMMA（半精度矩阵乘加）**，也可以是 **IMMA（整数矩阵乘加）**。所以这篇更像是“**把压缩矩阵计算正式做成编程接口和执行语义**”。([谷歌专利][8])

**一句话看法**：这篇偏 **software-hardware contract（软硬件契约）**——不是单讲某块电路，而是在定义“软件如何告诉硬件哪里是非零、何时压缩、何时解压、如何在压缩矩阵上做 MMA”。([谷歌专利][8])

---

## 9. US10891538B2

**Sparse convolutional neural network accelerator**
**中文**：稀疏卷积神经网络加速器

**基本信息**：Publication date 是 **2021-01-12**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2017-07-25**。([谷歌专利][9])

**主要内容**：这篇是非常硬核的 **sparse CNN accelerator（稀疏卷积神经网络加速器）** 设计。摘要里说它会解码两个 index vector（索引向量）来得到非零元素的坐标集合；而说明部分进一步展示了硬件结构：每个 **PE（处理单元）** 里有一个 **F×I multiplier array（F×I 乘法阵列）**，输入是一组非零 weights（权重）和一组非零 activations（激活），阵列直接算它们的 **Cartesian product（笛卡尔积：两组元素两两相乘）**，只产生有效乘积，不做多余零乘法。随后由 **destination calculation unit（目标地址计算单元）** 计算这些乘积应该累加到输出张量的哪个位置，再通过 **crossbar（交叉开关）** 散射到 **accumulator array（累加阵列）** 中。页面还写到：只传输 non-zero weights 和 non-zero activations 到 multiplier array。([谷歌专利][9])

**一句话看法**：这篇是“**压缩稀疏卷积硬件**”的代表作，核心是 **只算非零 × 非零，再把结果散射累加回去**。([谷歌专利][9])

---

## 10. US11966835B2

**Deep neural network accelerator with fine-grained parallelism discovery**
**中文**：带细粒度并行性发现机制的深度神经网络加速器

**基本信息**：Publication date 是 **2024-04-23**，Current Assignee / Original Assignee 都是 **Nvidia Corp**，Filing date 是 **2019-01-23**。([谷歌专利][10])

**主要内容**：这篇和上一篇很像，但更强调“**先找对，再计算**”。摘要里说，它会从 compacted input activation array（压紧后的输入激活数组）和 compacted weight array（压紧后的权重数组）里，动态找出 matching pairs（可配对的非零激活和非零权重），这些配对由 **parallelism discovery unit, PDU（并行性发现单元）** 完成。PDU 会做 **parallel search（并行搜索）**，利用 channel index（通道索引）去找“可以相乘并归约”的配对；后面再由 sequence decoder（序列译码器）把编码序列拆成行/列选择，并输出 valid bit（有效位），最后送给 MAC。和上一篇那种“把一整组非零值做全笛卡尔积”相比，这篇更像是通过显式匹配来发现 **fine-grained parallelism（细粒度并行性）**。([谷歌专利][10])

**一句话看法**：它比 US10891538B2 更强调 **动态匹配和并行性发现**，而不是单纯依靠 PE 内的全交叉乘法。([谷歌专利][10])

---

## 最后给你一个“关系图式”的总结

如果把这 10 篇看成一条技术演进链，可以大致这么理解：

* **US10503507B2 / US11609761B2**：先在 **指令 / load 路径** 上判断哪些数据值得算。
* **US10096134B2 / US20200285618A1 / US11127167B2**：再解决 **数据怎么压、怎么解、怎么换布局 / 转置**。
* **US10891538B2 / US11966835B2**：然后把压缩后的非零数据真正送进 **稀疏卷积 / 稀疏矩阵乘加速器**。
* **US11392829B1 / US20220366007A1**：进一步把稀疏从“自然稀疏”推进到“**结构化稀疏 + API / MMA 语义**”，让软件和硬件都更好对齐。([谷歌专利][2])

如果你愿意，我下一步可以继续帮你做两件更有价值的事：

1. 把这 10 篇按**技术脉络画成时间线**；
2. 重点对比它们和 **NVIDIA Ampere / Hopper / Blackwell / Rubin** 上常见的 **2:4 稀疏、MMA、压缩矩阵执行路径** 之间的对应关系。

[1]: https://patents.google.com/patent/US11586417B2/en "US11586417B2 - Exploiting activation sparsity in deep neural networks 
        \- Google Patents"
[2]: https://patents.google.com/patent/US10503507B2/en "US10503507B2 - Inline data inspection for workload simplification 
        \- Google Patents"
[3]: https://patents.google.com/patent/US11609761B2/en "US11609761B2 - Inline data inspection for workload simplification 
        \- Google Patents"
[4]: https://patents.google.com/patent/US10096134B2/en "US10096134B2 - Data compaction and memory bandwidth reduction for sparse neural networks 
        \- Google Patents"
[5]: https://patents.google.com/patent/US20200285618A1/en "US20200285618A1 - Decompression techniques for processing compressed data suitable for artificial neural networks 
      \- Google Patents"
[6]: https://patents.google.com/patent/US11127167B2/en "US11127167B2 - Efficient matrix format suitable for neural networks 
        \- Google Patents"
[7]: https://patents.google.com/patent/US11392829B1/en "US11392829B1 - Managing data sparsity for neural networks 
        \- Google Patents"
[8]: https://patents.google.com/patent/US20220366007A1/en "US20220366007A1 - Performing matrix value indication 
        \- Google Patents"
[9]: https://patents.google.com/patent/US10891538B2/en "US10891538B2 - Sparse convolutional neural network accelerator 
        \- Google Patents"
[10]: https://patents.google.com/patent/US11966835B2/en "US11966835B2 - Deep neural network accelerator with fine-grained parallelism discovery 
        \- Google Patents"

可以，下面我直接把这 10 篇专利和 **NVIDIA Ampere / Hopper / Blackwell / Rubin** 上你关心的三条主线对齐起来：

1. **2:4 structured sparsity（2:4 结构化稀疏）**：每连续 4 个元素里，固定有 2 个为 0。
2. **MMA（Matrix Multiply-Accumulate，矩阵乘加）**：张量核心里的核心块运算。
3. **compressed-matrix execution path（压缩矩阵执行路径）**：先压缩非零值和位置，再让硬件只算非零项。 ([NVIDIA Developer][1])

## 先给结论

**最强的主线关系**是这样的：

* **Ampere**：把“固定规则的结构化稀疏 + 稀疏 MMA”第一次做成公开、可商用、可编程的主路径。它最像你那批专利里的
  **US11392829B1**（管理结构化稀疏）

  * **US20220366007A1**（矩阵值指示 / 压缩矩阵上的 MMA / API）
  * **US20200285618A1**（解压与格式转换）
  * **US10096134B2**（压紧数据、减带宽）。 ([NVIDIA Developer][1])

* **Hopper**：没有推翻这条路，而是在保留 **2:4 sparse Tensor Core（2:4 稀疏张量核心）** 思路的基础上，把 MMA 前端做得更大、更异步，并引入 **Transformer Engine（Transformer 引擎，面向大模型混合精度的张量核心调度/数值机制）** 和 **FP8（8 位浮点）**。所以它本质上还是 Ampere 那套稀疏主线的增强版。 ([NVIDIA Developer][2])

* **Blackwell**：公开资料里，宣传重点从“2:4 本身”明显转向了 **FP4（4 位浮点）**、**micro-tensor scaling（微张量缩放：更细粒度的缩放因子管理）** 和新的张量核心指令族；但从 NVIDIA 自己的 CUTLASS 文档看，**稀疏 MMA 没消失**，而是升级成了 **tcgen05.mma(.sp)** 这一代指令，所以它更像“**保留压缩稀疏 MMA 主线，同时把低比特和 block-scaled（块缩放）做大做强**”。 ([NVIDIA][3])

* **Rubin**：目前公开官方材料的重点已经进一步从“固定 2:4”转向 **hardware-accelerated adaptive compression（硬件加速自适应压缩）**，并明确说它 **Fully compatible with NVIDIA Blackwell（与 Blackwell 完全兼容）**。所以 Rubin 和你这批专利里最强的对应关系，不再是固定 2:4，而更像
  **US10503507B2 / US11609761B2**（inline data inspection，内联数据检查 / 阈值判定）

  * **US11586417B2**（activation sparsity，激活稀疏压缩）
  * **US10096134B2**（压紧和减带宽）。也就是：从“静态规则稀疏”进一步走向“运行时自适应压缩”。 ([NVIDIA][4])

---

## 一、先把这条公开硬件路径讲清楚

### 1. Ampere 的公开路径，本质就是“2:4 + metadata + sparse MMA”

NVIDIA 对 Ampere 的公开描述非常清楚：
**A100** 支持 **2:4 fine-grained structured sparsity（2:4 细粒度结构化稀疏）**；稀疏矩阵压缩后只存 **non-zero values（非零值）** 和 **metadata（元数据，记录这些非零值原来在哪）**；**Sparse Tensor Core（稀疏张量核心）** 在做矩阵乘时只处理这些非零值，并利用 metadata 从另一个未压缩操作数里取出对应元素，因此理论上可得到 2 倍等效吞吐。NVIDIA 还明确说，这条路径主要适用于已经提供 **2:4 sparse weights（2:4 稀疏权重）** 的全连接层和卷积层。 ([NVIDIA Developer][1])

这和专利的对应关系非常直接：

* **US11392829B1**：本质上是在讲“如何把稀疏做成规则、均匀、可硬件消费的形式”；这和 Ampere 的固定 2:4 约束最贴。 ([谷歌专利][5])
* **US20220366007A1**：它讲“矩阵值指示、压缩、在压缩矩阵上做 MMA、再解压”，这和 Ampere 的公开软件/硬件契约几乎同方向。 ([谷歌专利][6])
* **US20200285618A1**：它偏“解压和部分解压”，而 Ampere 公开描述里已经明确说 metadata 会用来从另一个操作数中“挑出必要值”，这就是典型的 metadata 驱动选择/解压逻辑。 ([NVIDIA Developer][1])
* **US10096134B2**：它偏“压紧 + 降带宽”，这也是 Ampere 2:4 之所以有价值的基础，因为压缩后不只是少算，也少搬。 ([谷歌专利][7])

所以你可以把 **Ampere** 理解成：
**专利层面的“结构化稀疏 + 压紧 + 元数据 + 稀疏矩阵乘加”第一次被 NVIDIA 公开收敛成统一产品路径。** ([NVIDIA Developer][1])

---

## 二、从指令层看：Ampere/Hopper/Blackwell 的 MMA 主线其实是连续演进

### 1. Ampere：`mma.sp`

**PTX（Parallel Thread Execution，并行线程执行中间汇编）** 文档从早期就公开了 **`mma.sp`**，也就是 **sparse matrix multiply-accumulate（稀疏矩阵乘加）** 指令。文档里还直接写了 sparse `mma` 的矩阵片段布局，以及 metadata 如何跟随稀疏操作数 A 一起传入。 ([NVIDIA Docs][8])

这个层面对应的专利最强的是：

* **US20220366007A1**：压缩矩阵上的 MMA / API。
* **US20200285618A1**：metadata 驱动的恢复 / 选择。
* **US11127167B2**：高效矩阵格式，尤其是和 transpose（转置）/ layout（布局）相关的处理。
* **US10096134B2**：压缩带来的带宽收益。 ([谷歌专利][6])

### 2. Hopper：`mma.sp` 继续存在，但又加了 `wgmma.mma_async` / `wgmma.mma_async.sp`

到了 Hopper，公开 PTX 内容里同时出现了：

* **`mma.sp`**：传统 warp-level（线程束级）稀疏 MMA
* **`wgmma.mma_async`**：warpgroup-level asynchronous MMA（线程束组级异步矩阵乘加）
* **`wgmma.mma_async.sp`**：线程束组级异步稀疏 MMA。 ([NVIDIA Docs][9])

这里的 **warp（线程束）** 可以简单理解成 GPU 里固定协同执行的一小组线程；**warpgroup（线程束组）** 则是更大的协同计算单元。
这说明 Hopper 不是把稀疏路线改了，而是把它从“warp 级稀疏 MMA”扩展到了“更大粒度、更异步的稀疏 MMA”。同时，Hopper 官方架构页强调了 **FP8** 和 **Transformer Engine**，说明它在“数值格式”和“调度/执行粒度”上都往前迈了一步。 ([NVIDIA][10])

所以 **Hopper 对这些专利的对应关系** 基本还是 Ampere 那套，但更偏向：

* **US20220366007A1**：因为它更像“压缩矩阵 MMA 的编程接口/执行语义”；
* **US11127167B2**：因为更大粒度的 MMA 对布局和转置更敏感；
* **US11392829B1**：因为 2:4 规则仍在。 ([谷歌专利][5])

---

## 三、Blackwell 的变化：主卖点从“2:4”转向“更宽的数据格式 + 新一代 sparse MMA 指令”

Blackwell 官方架构页重点讲的是：

* **Second-Generation Transformer Engine（第二代 Transformer 引擎）**
* **FP4**
* **micro-tensor scaling（微张量缩放）**
* 新的更强张量核心。 ([NVIDIA][3])

如果只看官网宣传，会觉得它不像 Ampere 那样把“2:4”放在最中心位置；但 NVIDIA 的 CUTLASS 文档把底层路线说得更清楚：
**Blackwell SM100** 引入了新的 **`tcgen05.mma`** 指令族，并且明确列出了 **`tcgen05.mma(.sp)`** 稀疏版本，覆盖 **tf32 / f16 / i8** 以及多种低比特浮点、块缩放格式；文档还说这些新指令相对 Hopper 的 **WGMMA（warpgroup MMA，线程束组级矩阵乘加）** 可达 **2x 到 4x** 的提升。 ([NVIDIA Docs][11])

这很关键，因为它说明：

**Blackwell 不是放弃稀疏 MMA，而是把稀疏 MMA 升级到了新 Tensor Core 代际里。**
只是公开叙事的重点，从 Ampere 的“固定 2:4 稀疏”转到了“低比特 + block scale（块缩放）+ FP4/FP6/FP8”。 ([NVIDIA Docs][11])

因此，**Blackwell 和专利的对应关系**是：

* **继续强对应**

  * **US20220366007A1**：压缩矩阵上的 MMA，仍然是核心。
  * **US20200285618A1**：更多低比特/块缩放格式下，metadata 和解压/选择仍然重要。
  * **US10096134B2**：块缩放和低比特让“少搬数据”的价值更大。 ([NVIDIA Docs][11])
* **相对弱化**

  * **US11392829B1**：仍然 relevant（相关），但公开卖点不再只是固定 2:4。 ([NVIDIA Docs][11])

再补一层软件侧证据：
`cuSPARSELt` 官方文档明确把自己定义成“至少一个操作数是 50% 结构化稀疏矩阵”的高性能矩阵乘库，并列出了它支持的架构范围，已经覆盖 Ampere、Hopper 以及更新的后续 SM 架构。这说明 **从软件栈角度，NVIDIA 仍在维持“稀疏 MMA + 压缩矩阵”这条主线**。 ([NVIDIA Docs][12])

---

## 四、Rubin 的变化最大：从“固定 2:4”更明显地走向“自适应压缩”

Rubin 官方页面目前最关键的一句话是：
**The Rubin GPU features a new Transformer Engine with hardware-accelerated adaptive compression … Fully compatible with NVIDIA Blackwell.**
也就是：Rubin 的新 Transformer Engine 带有**硬件加速的自适应压缩**，并且与 Blackwell 完全兼容。 ([NVIDIA][4])

这句话的信息量很大：

1. **兼容 Blackwell**：说明 Rubin 不是推倒重来，而是在 Blackwell 路线上继续演进。 ([NVIDIA][4])
2. **adaptive compression（自适应压缩）**：公开叙事重点已经从“固定 N:M 模式”切换到“运行时根据数据特征压缩”。 ([NVIDIA][4])
3. 官方公开页没有像 Ampere 那样把 **2:4 structured sparsity** 作为头号概念来展开，所以至少从目前公开材料看，Rubin 的主卖点不再是“静态固定 2:4”，而是“动态、自适应、硬件内建的压缩执行”。这个判断来自官方公开内容的侧重点，而不是我臆测它完全没有 2:4 路径。 ([NVIDIA][4])

因此，**Rubin 和专利的最强对应**已经变了：

### 最强对应的专利

* **US10503507B2 / US11609761B2**
  这两篇的核心是 **inline data inspection（内联数据检查）**、阈值判断、谓词门控，本质上是在“数据刚进来时就决定后面要不要算”。这和 Rubin 的“自适应压缩”思路更接近，而不像 Ampere 那样主要依赖离线剪枝得到固定 2:4。 ([谷歌专利][13])

* **US11586417B2**
  这篇讲的是 **activation sparsity（激活稀疏）**：把非零激活压成更紧凑的形式，再重分配到 **vector lanes（向量通道）** / **MAC hardware（乘加硬件）** 上，提高利用率。Rubin 的自适应压缩如果落在推理前端或 Transformer Engine 附近，这种“动态压激活、减少空槽”的思路就比固定 2:4 更像。 ([谷歌专利][14])

* **US10096134B2**
  自适应压缩最终还是要兑现成“少搬数据、少写回、少走无效带宽”，所以这篇的带宽优化价值在 Rubin 反而更像“底座型”专利。 ([谷歌专利][7])

### 相对弱一点，但仍有关联

* **US11392829B1 / US20220366007A1**
  它们更贴固定 2:4 + 压缩矩阵 MMA 的时代，也就是 Ampere/Hopper/Blackwell 的公开路径；Rubin 并不是和它们无关，而是更像把这条路再往“动态压缩”推进了一步。 ([谷歌专利][5])

---

## 五、哪几篇专利和哪一代，关系最强

### 最像 Ampere / Hopper / Blackwell 公开 2:4 sparse-MMA 路径的

* **US11392829B1**：定义“规则化、可硬件消费的结构化稀疏”
* **US20220366007A1**：矩阵值指示、压缩矩阵上的 MMA、压缩/解压 API
* **US20200285618A1**：metadata 驱动的解压 / 部分解压
* **US10096134B2**：压紧与带宽节省。 ([谷歌专利][5])

### 更像 Rubin “adaptive compression” 方向的

* **US10503507B2 / US11609761B2**：取数时检查、阈值判断、条件执行
* **US11586417B2**：激活稀疏的动态压缩和 lane 重映射
* **US10096134B2**：压紧与带宽节省。 ([谷歌专利][13])

### 更像“概念相关，但不是 Tensor Core 公开主路径”的

* **US10891538B2**
* **US11966835B2**
  这两篇更像 **SCNN-style accelerator（压缩稀疏卷积加速器风格）**：显式发现非零对、显式做 products（乘积）和 scatter / reduction（散射累加 / 归约）。它们和 NVIDIA 稀疏 Tensor Core 的精神是同向的，但不太像 Ampere/Hopper/Blackwell 公开的那条“统一 Tensor Core MMA 指令”路径的字面实现。 ([谷歌专利][15])

---

## 六、最后给你一句最实用的判断

如果你是站在 **NVIDIA 产品演进** 的角度看这批专利，我会这样概括：

* **Ampere**：把“静态、固定规则的结构化稀疏”正式工程化。
* **Hopper**：保留这条稀疏主线，同时把 MMA 做大做异步，并把 FP8 / Transformer Engine 引进来。
* **Blackwell**：稀疏 MMA 继续存在，但公开重心转向低比特、块缩放、FP4。
* **Rubin**：开始把“固定稀疏模式”进一步抽象成“硬件加速自适应压缩”，也就是从**静态结构化稀疏**走向**动态数据感知压缩**。 ([NVIDIA Developer][1])

你要是继续深挖，我下一步最值得做的是：**把这 10 篇专利按“静态 2:4 路线”和“动态自适应压缩路线”分成两棵树，再映射到 Ampere/Hopper/Blackwell/Rubin 的具体执行阶段**。

[1]: https://developer.nvidia.com/blog/?p=34218&utm_source=chatgpt.com "Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT | NVIDIA Technical Blog"
[2]: https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/?utm_source=chatgpt.com "Structured Sparsity in the NVIDIA Ampere Architecture and Applications in Search Engines | NVIDIA Technical Blog"
[3]: https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/?utm_source=chatgpt.com "The Engine Behind AI Factories | NVIDIA Blackwell Architecture"
[4]: https://www.nvidia.com/en-us/data-center/technologies/rubin/ "Infrastructure for Scalable AI Reasoning | NVIDIA Vera Rubin Platform"
[5]: https://patents.google.com/patent/US11392829B1/en?utm_source=chatgpt.com "US11392829B1 - Managing data sparsity for neural networks - Google Patents"
[6]: https://patents.google.com/patent/US20220366007A1/en?utm_source=chatgpt.com "US20220366007A1 - Performing matrix value indication - Google Patents"
[7]: https://patents.google.com/patent/US10096134B2/en?utm_source=chatgpt.com "US10096134B2 - Data compaction and memory bandwidth reduction for sparse neural networks - Google Patents"
[8]: https://docs.nvidia.com/cuda/archive/11.1.0/parallel-thread-execution/index.html?utm_source=chatgpt.com "PTX ISA :: CUDA Toolkit Documentation"
[9]: https://docs.nvidia.com/cuda/hopper-tuning-guide/parallel-thread-execution/contents.html "Contents — PTX ISA 8.8 documentation"
[10]: https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/?utm_source=chatgpt.com "Hopper GPU Architecture | NVIDIA"
[11]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html "Blackwell SM100 GEMMs — NVIDIA CUTLASS Documentation"
[12]: https://docs.nvidia.com/cuda/cusparselt/ "cuSPARSELt: A High-Performance CUDA Library for Sparse Matrix-Matrix Multiplication — NVIDIA cuSPARSELt"
[13]: https://patents.google.com/patent/US11977888B2/en?utm_source=chatgpt.com "US11977888B2 - Inline data inspection for workload simplification - Google Patents"
[14]: https://patents.google.com/patent/US11586417B2/en?utm_source=chatgpt.com "US11586417B2 - Exploiting activation sparsity in deep neural networks - Google Patents"
[15]: https://patents.google.com/patent/US20180046900A1/en?utm_source=chatgpt.com "US20180046900A1 - Sparse convolutional neural network accelerator - Google Patents"







