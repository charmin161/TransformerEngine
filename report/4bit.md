下面按**严格“最近几个月”**和**近一年关键基线**分开看。结论先说：真正值得关注的近期方向不是“全链路都 4bit/8bit”，而是**把低比特集中放在线性层 GEMM 的 Fprop / Dgrad / Wgrad**，再用高精度保住 optimizer、梯度累加、部分通信、norm/softmax/attention 关键路径。FP4 训练的核心难点集中在 **Wgrad、outlier、scale 选择、rounding bias**。

## 1. 最近几个月最相关论文

| 论文                                                                                                         |      时间 | 低比特类型 | 低比特用在哪里                                                                     | 稳定收敛策略                                                                                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------------------------------- | ------: | ----- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pretraining Large Language Models with MXFP4 on Native FP4 Hardware**                                    | 2026-05 | MXFP4 | Transformer linear layer 的 Fprop / Dgrad / Wgrad，逐步从 FP8 baseline 替换为 MXFP4 | 发现 **Wgrad 是主要不稳定来源**；普通 stochastic rounding 和 randomized Hadamard 对 full MXFP4 不够，**deterministic Hadamard** 才恢复稳定训练。Fprop-only 约 8–9% token overhead，Fprop+Dgrad 约 10–11%，全 Fprop+Dgrad+Wgrad 若 naive 会到 26–27%；加 deterministic Hadamard 后回到约 8–9%。([arXiv][1]) |
| **Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation**                | 2026-01 | NVFP4 | 目标是 linear layers 的 major matmuls，包括前向和反向                                   | 提出 **MS-EDEN**，替代普通 stochastic rounding，目标是对 micro-scaled formats 做更低误差的无偏梯度估计；公开摘要称量化误差低于 SR，并在 1.9B、38B tokens 规模验证。([arXiv][2])                                                                                                                                |
| **FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning**                      | 2026-01 | FP8   | 主要用于 RL rollout：FP8 W8A8 linear layer rollout，并扩展到 **FP8 KV cache**         | 重点不是预训练，而是 RL 中 rollout 与 trainer 的精度不一致问题；用 per-step QKV scale recalibration 处理 KV cache，并用 token-level importance sampling 缓解 FP8 rollout 与高精度 trainer 的 policy mismatch。([arXiv][3])                                                                           |
| **Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow** | 2026-01 | FP8   | 训练和 rollout 统一 FP8 precision flow                                           | 指出 BF16 training + FP8 rollout 在长 horizon 或复杂任务中会出现 off-policy / numerical mismatch，甚至 collapse；核心策略是让训练和 rollout 的精度流统一，减少精度切换导致的不一致。([arXiv][4])                                                                                                                |

## 2. 近一年必须一起看的基线论文

| 论文                                                                                        |                     时间 | 低比特类型              | 主要贡献                                                                                                                                                                                |
| ----------------------------------------------------------------------------------------- | ---------------------: | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pretraining Large Language Models with NVFP4**                                          |       2025-09，2026 有修订 | NVFP4              | NVIDIA 的系统性 NVFP4 预训练方案：FP4 用在线性层 GEMM；用 Random Hadamard、2D scaling、gradient stochastic rounding、selective high precision layers 保持收敛；12B、10T tokens 上接近 FP8 baseline。([arXiv][5])  |
| **MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling** |                2025-11 | FP8 / microscaling | 面向 FP8 训练的两级 microscaling：activation 用 global FP32 scale + local E8M0 microscale，weight 用自动 scaling；7B 级别训练接近 BF16，并报告吞吐提升。([arXiv][6])                                             |
| **Recipes for Pre-training LLMs with MXFP8**                                              |                2025-06 | MXFP8              | 重点解决 MXFP8 的 scale factor rounding。论文指出 OCP 默认 round-down 会导致训练问题，推荐 scale factor round-up / round-to-infinity 来避免 saturation 和 divergence。([arXiv][7])                             |
| **Quartet: Native FP4 Training Can Be Optimal for LLMs**                                  |                2025-05 | MXFP4 / FP4        | 更理论化地讨论 end-to-end FP4 training：前向关注最小化 MSE，反向关注无偏梯度估计；把 major linear-layer matmuls 放到 FP4。([arXiv][8])                                                                             |
| **FP4 All the Way: Fully Quantized Training of LLMs**                                     |                2025-05 | NVFP4-style FP4    | 主张权重、激活、梯度大部分都用 FP4；使用 NVFP4 格式、合适的 block scale 和 rounding 策略，在较大 token 规模上接近 BF16。([arXiv][9])                                                                                     |
| **Training LLMs with MXFP4**                                                              | 2025-02 / AISTATS 2025 | MXFP4              | 主要把 MXFP4 放在 **backward pass 的 decoder linear layers**，前向默认 BF16，也测试过 FP8 forward + MXFP4 backward；核心是 stochastic rounding + randomized Hadamard，降低反向量化误差和 outlier 影响。([arXiv][10]) |

## 3. 收敛与稳定性的共同策略

### 3.1 Wgrad 是 FP4 训练最脆弱的位置

最近的 MXFP4 原生硬件论文把 Fprop、Dgrad、Wgrad 分开替换，结果很清楚：**Fprop 和 Dgrad 相对容易，Wgrad 一上 MXFP4，训练损失和 token overhead 明显恶化**。更重要的是，它发现 stochastic rounding 和 randomized Hadamard 并不一定能救 full MXFP4，最后是 deterministic Hadamard 才把全链路 MXFP4 拉回稳定区间。([arXiv][1])

这说明 FP4 训练不能只说“用 SR 就稳定”。SR 是否有效，取决于：

* 量化对象是 activation、weight 还是 gradient；
* 是 Fprop、Dgrad 还是 Wgrad；
* micro-scale 的 block 方向；
* outlier 是否被 rotation / Hadamard 处理；
* scale factor 是否因为 power-of-two 约束损失动态范围。

### 3.2 Hadamard / rotation 基本成为 FP4 训练标配，但放置位置不同

MXFP4 早期方案用 **randomized Hadamard transform** 降低 outlier 带来的 stochastic rounding 方差；NVFP4 方案则更精细，主要把 RHT 用在 Wgrad GEMM 的输入上，因为论文发现 Fprop / Dgrad 从 RHT 获益不明显。([arXiv][10])

2026-05 的原生 MXFP4 论文进一步说明，随机 Hadamard 也不总是够，**deterministic Hadamard** 在 full MXFP4 下更有效。这个差异很关键：它暗示 FP4 的误差不是纯随机噪声，还可能有和 micro-scale/block layout 绑定的结构性误差。([arXiv][1])

### 3.3 Rounding 策略：前向和反向不能一刀切

比较一致的经验是：

* **weights / activations 的前向量化**：通常用 round-to-nearest / round-to-nearest-even / MSE-optimal quantization，而不是随便 SR。
* **gradients / backward / update 相关量化**：更倾向用 stochastic rounding 或其他无偏估计方法，避免梯度系统性偏移。
* NVIDIA NVFP4 论文明确提到，gradient 用 stochastic rounding 有帮助，但 activation 或 weight 上用 SR 可能导致训练发散；12B 规模需要对进入 Dgrad 和 Wgrad 的 gradients 使用 SR。([ar5iv][11])

Quartet II 的 MS-EDEN 也是沿着这个方向：不是简单用普通 SR，而是为 micro-scaled FP4 设计更好的无偏量化估计。([arXiv][2])

### 3.4 Scale 设计比格式本身还重要

FP4/MXFP4/NVFP4 的核心不是“4bit 值本身”，而是 **block scale 怎么选、scale 是什么格式、block 方向如何对齐 GEMM**。

NVFP4 的优势之一是两级 scaling：global FP32 tensor scale + local FP8 E4M3 scale；相比 MXFP4 的 power-of-two scale，NVFP4 的 E4M3 local scale 更能保留动态范围。NVIDIA 的 NVFP4 训练方案还对 weights 使用 2D scaling，对 activations / gradients 使用 1D scaling，以解决前向权重和反向转置权重在 chain rule 中的表示一致性问题。([ar5iv][11])

FP8/MXFP8 里也类似。MXFP8 paper 的重点就是 scale factor rounding：默认 round-down 会导致 saturation 和不稳定，推荐 round-up / round-to-infinity。([arXiv][12])

### 3.5 Selective high precision 仍然很常见

NVIDIA NVFP4 不是全模型所有地方都 FP4。它保留了一批 sensitive linear layers 为高精度，论文建议约 15% 敏感线性层保持高精度，通常集中在模型后部；embedding、output projection、norm、nonlinearity、softmax、QK/AV attention batched GEMM 等也保留 BF16/FP32。([ar5iv][11])

这点非常重要：很多论文标题写“NVFP4 training”或“FP4 training”，实际低比特主要覆盖的是**线性层 GEMM 主计算量**，不是所有张量、所有通信、所有状态都 FP4。

## 4. 低比特具体用在哪些位置？

### 4.1 FP4 / MXFP4 / NVFP4

**最常见位置：Transformer linear layers 的 GEMM。**

包括：

* Fprop：`X @ W^T`
* Dgrad：`dY @ W`
* Wgrad：`dY^T @ X` 或等价形式

不同论文覆盖程度不同：

* **Training LLMs with MXFP4**：主要是 backward pass 的 decoder linear layers；前向默认 BF16，附录里测试过 FP8 forward + MXFP4 backward。它明确使用 FP32 master weights 和 BF16 parameter copies。([ar5iv][13])
* **Pretraining LLMs with NVFP4**：Fprop / Dgrad / Wgrad 的输入都量化到 NVFP4，输出为 BF16 或 FP32；但 attention softmax、QK/AV batched GEMM、norm、nonlinear 等保持 BF16/FP32。([ar5iv][11])
* **Pretraining LLMs with MXFP4 on Native FP4 Hardware**：系统地把 transformer linear layers 的 FP8 GEMM 替换为 MXFP4 GEMM，并逐步测试 Fprop、Dgrad、Wgrad。([arXiv][1])

### 4.2 FP8 / MXFP8

FP8 训练通常更成熟，低比特覆盖范围比 FP4 更容易扩大。常见用法：

* linear layer GEMM 的 activation/weight FP8；
* 某些方案把 gradients 或一阶动量也做 FP8；
* activation checkpoint / activation memory 可能用 FP8 降低内存；
* FSDP / ZeRO 场景下，低精度 activation 或 gradient 表示会降低通信量。

MOSS 用两级 microscaling：activation 用 FP32 global scale + E8M0 local microscale，weight 用 per-tensor FP32 scale；它报告 7B 训练接近 BF16，并降低 activation memory 和 AllReduce 体积。([arXiv][14])

## 5. 其他位置的精度：通信、梯度累加、optimizer、KV cache

| 位置                                  | 近期论文里的常见做法                                                                                                                                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **梯度累加 / optimizer states**         | FP4 训练里通常不降到 FP4。NVIDIA NVFP4 明确保留 main weights、用于 microbatch / data-parallel replica 累加的 weight gradients、optimizer states 为 FP32。([ar5iv][11])                                                       |
| **Tensor-parallel reductions / 通信** | NVIDIA NVFP4 明确写 tensor-parallel reductions 用 BF16，而不是 FP4。([ar5iv][11]) MOSS 报告 AllReduce 体积和延迟下降，但其重点是 FP8 training framework 和 activation/weight microscaling，不应简单理解为所有通信都 FP8。([arXiv][14])        |
| **Attention QK / AV GEMM**          | NVIDIA NVFP4 明确保留 QK/AV batched GEMM、softmax 等 attention components 为 BF16/FP32。([ar5iv][11]) 所以很多 FP4 论文的“major GEMMs”主要指 linear projection / MLP / output linear，不一定包括 attention score/value matmul。 |
| **Norm / nonlinear / softmax**      | 通常保持 BF16/FP32。NVFP4 论文明确保留 embeddings、output projection、norms、nonlinearities、attention components 为原精度。([ar5iv][11])                                                                                  |
| **KV cache**                        | 预训练论文一般不讨论 KV cache，因为训练时不是 inference-style KV cache。KV cache 低精度主要出现在 RL rollout / inference 相关论文。FP8-RL 把 FP8 扩展到 KV cache，并用 per-step QKV scale recalibration 稳定 rollout。([arXiv][3])               |
| **模型权重存储**                          | 训练时通常仍有 FP32 master weights 或高精度主权重。MXFP4 backward 论文明确使用 FP32 master weights 和 BF16 parameter copies。([ar5iv][13])                                                                                    |
| **GEMM accumulation 输出**            | NVFP4 论文里，FP4 GEMM 输入量化为 NVFP4，输出为 BF16 或 FP32。([ar5iv][11]) MOSS 的 FP8 GEMM 描述中，partial sum / dequantization 到 FP32。([arXiv][14])                                                                     |

## 6. 总体判断

当前低比特训练大致形成了三条线：

**第一条：FP8 训练已经相对工程化。**
FP8/MXFP8 的主要问题是 scale granularity、scale update overhead、activation outlier、通信和内存效率。MOSS、MXFP8 recipes 这类工作主要是在 FP8 scale 机制上做工程优化。

**第二条：FP4/NVFP4 预训练开始可行，但还不是“所有地方都 FP4”。**
最近的 NVFP4/MXFP4 论文基本都把 FP4 放在线性层 GEMM，尤其是 MLP 和 projection 相关矩阵乘。optimizer、梯度累加、norm、softmax、attention QK/AV、部分敏感层仍然保留 BF16/FP32。

**第三条：Wgrad 是 FP4 训练能否成功的核心瓶颈。**
Fprop 和 Dgrad 相对容易；Wgrad 需要 rotation/Hadamard、careful scaling、unbiased quantization，甚至 selective high precision。2026-05 的 MXFP4 原生硬件论文把这一点验证得最直接。([arXiv][1])

如果你后续要自己判断一篇低比特训练论文是否“真有价值”，建议直接看四个表述：

1. 它是否真的覆盖 **Wgrad**，还是只做 forward / backward activation？
2. optimizer states、master weights、gradient accumulation 是否仍是 FP32？
3. attention QK/AV、softmax、norm 有没有低比特，还是只低比特 linear layers？
4. 它的稳定策略是普通 SR，还是有 rotation、2D scaling、selective high precision、precision switching、scale rounding 等系统设计？

[1]: https://arxiv.org/html/2605.09825v2 "Pretraining Large Language Models with MXFP4 on Native FP4 Hardware"
[2]: https://arxiv.org/abs/2601.22813 "[2601.22813] Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation"
[3]: https://arxiv.org/abs/2601.18150?utm_source=chatgpt.com "FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning"
[4]: https://arxiv.org/abs/2601.14243?utm_source=chatgpt.com "Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow"
[5]: https://arxiv.org/abs/2509.25149 "[2509.25149] Pretraining Large Language Models with NVFP4"
[6]: https://arxiv.org/abs/2511.05811?utm_source=chatgpt.com "MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling"
[7]: https://arxiv.org/abs/2506.08027 "[2506.08027] Recipes for Pre-training LLMs with MXFP8"
[8]: https://arxiv.org/html/2505.14669v1 "Quartet: Native FP4 Training Can Be Optimal for Large Language Models"
[9]: https://arxiv.org/abs/2505.19115?utm_source=chatgpt.com "FP4 All the Way: Fully Quantized Training of LLMs"
[10]: https://arxiv.org/abs/2502.20586 "[2502.20586] Training LLMs with MXFP4"
[11]: https://ar5iv.org/abs/2509.25149v1 "[2509.25149] Pretraining Large Language Models with NVFP4"
[12]: https://arxiv.org/html/2506.08027v1?utm_source=chatgpt.com "Recipes for Pre-training LLMs with MXFP8"
[13]: https://ar5iv.org/html/2502.20586v3 "[2502.20586] 1 Introduction"
[14]: https://arxiv.org/html/2511.05811v2 "MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling"

下面把两篇分开拆，然后做交叉比较。为方便表述，我把 **Pretraining Large Language Models with MXFP4 on Native FP4 Hardware** 简称为 **MXFP4-native**，把 **Normalized Architectures are Natively 4-Bit** 简称为 **nGPT-NVFP4**。

## 总体结论

这两篇论文其实给出了两种完全不同的 FP4 训练路线：

**MXFP4-native 的核心问题是：在标准 Transformer/Llama 架构上，哪些 GEMM 路径一旦换成 MXFP4 会破坏收敛？** 它的结论非常直接：Fprop 和 Dgrad 相对安全，真正脆弱的是 **Wgrad**。普通 stochastic rounding 和 randomized Hadamard 在 full MXFP4 下救不回来，**deterministic Hadamard** 才能把 full Fprop + Dgrad + Wgrad 的 MXFP4 训练拉回接近 FP8 baseline。该论文在 AMD Instinct MI355X 上用 native MXFP4 做实验，不是软件模拟。([arXiv][1])

**nGPT-NVFP4 的核心问题是：能不能从模型架构上让 4bit 训练天然稳定，而不是靠外部量化补丁？** 它的答案是：nGPT 这种把 hidden states 和 weights 约束到 unit hypersphere 的 normalized architecture，使 dot product 的信号在高维求和时更“coherent”，从而在 NVFP4 下获得更高 SNR。它不是说所有东西都 4bit，而是说主要 GEMM 可以 NVFP4，同时可以去掉标准 NVFP4 recipe 里的 RHT 和动态 per-tensor scaling 开销；但 **SR 仍然保留**，embedding、nonlinear、batched-GEMM 等仍有 BF16 路径，大模型实验还保留最后 15% 层为高精度。([arXiv][2])

---

# 1. MXFP4-native：这篇到底证明了什么？

## 1.1 实验设计：非常“诊断型”

这篇不是提出一个复杂完整训练框架，而是做了一个控制变量实验：在 Llama 3.1–8B 的 MLPerf C4 pretraining 设置下，把原来 FP8 baseline 里的 transformer linear GEMM 路径逐步换成 MXFP4，分别看：

* **Fprop**：前向线性层 GEMM；
* **Dgrad**：反向传播里对 activation/input gradient 的 GEMM；
* **Wgrad**：反向传播里对 weight gradient 的 GEMM。

收敛目标定义为 validation perplexity ≤ 3.3，然后看达到该目标所需 token 数相对 FP8 baseline 的额外开销。论文明确说它是在 AMD Instinct MI355X 上使用 native FP4 tensor support 做 MXFP4 计算，避免软件模拟带来的性能和稳定性混淆。([arXiv][1])

这点很关键：它不是“证明 MXFP4 比 NVFP4 好”，也不是“证明 FP4 全链路成熟”。它是在回答一个更底层的问题：**FP4 训练到底坏在哪个 GEMM path 上？**

## 1.2 关键实验结果：Wgrad 是主要收敛瓶颈

论文的 stage-wise enablement 结果如下：

| 配置                         | MXFP4 放在哪里            | token overhead vs FP8 |
| -------------------------- | --------------------- | --------------------: |
| FP8 baseline               | 全 GEMM FP8            |                    0% |
| 无 stabilizer               | Fprop                 |                  8–9% |
| 无 stabilizer               | Fprop + Dgrad         |                10–11% |
| 无 stabilizer               | Fprop + Dgrad + Wgrad |                26–27% |
| Stochastic rounding        | Fprop + Dgrad + Wgrad |                   不收敛 |
| Randomized Hadamard H16    | Fprop + Dgrad + Wgrad |                   不收敛 |
| Deterministic Hadamard H32 | Fprop + Dgrad + Wgrad |                  8–9% |
| Deterministic Hadamard H16 | Fprop + Dgrad + Wgrad |                  8–9% |

同一张表还给出 kernel 侧结果：H16 比 H32 快 8%，所以工程上更倾向 H16。([arXiv][1])

这个结果的含义很强：

**第一，Fprop-only 的 MXFP4 不是主要问题。** 8–9% token overhead 说明前向量化会影响训练效率，但不是灾难性。

**第二，Dgrad 也不是主要问题。** Fprop + Dgrad 只从 8–9% 增加到 10–11%，说明 activation gradient path 的 MXFP4 误差还能被训练动态吸收。

**第三，Wgrad 一加进来，问题明显变大。** Full Fprop + Dgrad + Wgrad 的 naive MXFP4 直接到 26–27% token overhead。Wgrad 的误差不只是某一层 activation 的临时误差，而是进入参数更新路径，会跨 step 累积，因此对优化轨迹影响更持久。

**第四，随机性不是万能解。** Stochastic rounding 和 randomized Hadamard 在 partial setting 下看起来不坏，但 full pipeline 一到 Wgrad 就不收敛。论文据此认为，问题不是“随机性不足”，而是 sensitive gradient path 上存在结构化 micro-scaling error。([arXiv][3])

## 1.3 为什么 Wgrad 比 Fprop/Dgrad 更难？

线性层里三个 GEMM 可以写成：

* Fprop: `Y = X W?`
* Dgrad: `dX = dY W`
* Wgrad: `dW = dY? X`

从低比特训练角度看，Wgrad 最难，原因有三层。

第一，**Wgrad 同时乘 activation 和 upstream gradient**。这两个张量都可能有 outlier、heavy-tail 或局部 block 内动态范围不均衡。MXFP4 是 micro-scaling，小 block 共用 scale；一个 outlier 会抬高 block scale，让其他较小值有效精度下降。

第二，**Wgrad 的量化误差进入 optimizer update**。Fprop/Dgrad 的误差更多是当前 step 的计算噪声，Wgrad 的误差会改变参数更新方向，并影响后续所有 step。

第三，**micro-scaling error 是结构化的，不一定能靠 SR 平均掉**。如果 block layout、scale 选择、Hadamard basis、tensor layout 之间产生系统性偏差，SR 增加的随机噪声可能只是把 Wgrad path 搞得更嘈杂，而不是消除偏差。论文正是用 stochastic rounding 和 randomized Hadamard 失败、deterministic Hadamard 成功这一对照来支持这个解释。([arXiv][3])

## 1.4 Deterministic Hadamard 为什么有效？

论文在 Appendix 里说，Hadamard transform 是一个正交旋转，作用是把集中在少数维度上的能量/outlier 分散到更多维度；由于 Hadamard 是正交矩阵，成对插入后理论上不会改变线性层原本计算，只改变进入量化器之前的数据分布。它们在实现中把 Hadamard rotation 插到 GEMM 前面，并用 AMD ROCm Transformer Engine 把 transformer linear module 里的 FP8 GEMM 替换为 selected pass 的 MXFP4 GEMM。([arXiv][3])

这篇最有意思的地方是：它不是说“Hadamard 一定要 random”。相反，实验发现 random signs 在这个设置下可能伤害 full pipeline 收敛；所以它选择 deterministic transform。作者给出的解释是，full MXFP4 的不稳定更像是结构化 micro-scaling error，而不是单纯 outlier 随机噪声问题。([arXiv][3])

这和 NVIDIA 之前 NVFP4 recipe 的结论不同：NVIDIA 的标准 NVFP4 recipe 里，Random Hadamard Transform 用在 Wgrad 输入上，并且认为大模型长 token horizon 下 random sign 有帮助。这个差异不能简单理解为谁对谁错，更合理的解释是：**MXFP4 vs NVFP4 格式不同、block scale 不同、硬件不同、模型/数据设置不同、Hadamard 放置方式不同，导致最优 stabilizer 不同。** NVIDIA NVFP4 论文明确推荐 Wgrad 输入做 RHT、weights 用 2D scaling、activations/gradients 用 1D scaling、gradients 用 SR，而 weights/activations 用 round-to-nearest-even。([arXiv][4])

## 1.5 MXFP4-native 里的低比特到底放在哪？

这篇论文的低比特范围很集中：**transformer linear layers 的 GEMM path**。表格里写的是“MXFP4 GEMMs，others FP8”，说明它是在 FP8 baseline 上逐步把 Fprop、Dgrad、Wgrad GEMM 替换为 MXFP4。([arXiv][3])

它没有系统报告 optimizer states、gradient accumulation、communication、KV cache、attention QK/AV、softmax、norm 等其他路径的精度。严格说，不能从这篇论文推出“全训练链路都是 MXFP4”。它证明的是：**在被替换的 transformer linear GEMM 路径里，full Fprop + Dgrad + Wgrad 的 MXFP4 可以在 deterministic Hadamard 下稳定，并获得 end-to-end speedup。**

效率数据也要正确解读。稳定的 MXFP4 + H16 配置相对 FP8 有 +20% train-step throughput，但由于仍有 +8–9% token overhead，最终 end-to-end speedup 是 +9–10%。也就是说，FP4 的 raw throughput 优势必须先被收敛稳定性“兑现”，否则 token-to-target 增加会吃掉硬件收益。([arXiv][3])

---

# 2. nGPT-NVFP4：NVIDIA 这篇“normalized architectures are natively 4-bit”到底新在哪里？

## 2.1 它不是普通量化论文，而是架构论文

这篇的核心主张是：标准 GPT/Transformer 需要 RHT、dynamic per-tensor scaling、mixed-precision exceptions 这些补丁才能稳定 FP4；但 nGPT 这种 normalized architecture 本身就对 4bit 算术更鲁棒。nGPT 把 hidden representations 和 weights 约束到 unit hypersphere，论文认为这种几何结构让 NVFP4 训练天然更稳。([arXiv][2])

它的目标不是“给现有 Llama checkpoint 套一个量化方法”，而是：**从预训练一开始就使用一个更适合 4bit 的架构。**

## 2.2 nGPT 做了哪些架构改动？

Appendix 里列了主要改动：

| nGPT 改动                                     | 低比特稳定性的意义                                              |
| ------------------------------------------- | ------------------------------------------------------ |
| 去掉 RMSNorm / LayerNorm                      | 不再依赖传统 norm layer，而是把整个表示学习放到 hypersphere 上            |
| 每步后沿 embedding dim normalize 所有 weights     | 控制 weight 范数，限制单个坐标无限放大                                |
| residual update 改成 normalized interpolation | hidden state 保持在单位球附近                                  |
| q/k normalize 并 rescale                     | attention dot product 的幅度被结构性控制                        |
| MLP / logits 做 rescaling                    | 保留表达能力，同时维持尺度可控                                        |
| 去掉 weight decay 和 LR warmup                 | 因为 weight normalization / hypersphere dynamics 改变了优化行为 |

论文特别说明，它们比官方 nGPT implementation 更激进：**normalize all inputs**，而官方 nGPT 对 attention out projection GEMM 和 MLP 的 FFN2 GEMM 输入没有做 normalization。([t.co][5])

这点对 FP4 很重要。标准 Transformer 可以靠少数特别大的 hidden coordinates 或 weight coordinates 来贡献 dot product；nGPT 把这条路堵住了，迫使模型用许多维度上的分布式 alignment 来形成信号。

## 2.3 关键机制：不是噪声变小，而是 signal sum 变强

论文的分析很漂亮。它把一个 linear layer 输出看成 dot product：

[
y = \sum_i w_i x_i
]

然后比较 GPT 和 nGPT 在 NVFP4 量化后的 SNR。结果是：

* 在单个 weight、activation、element-wise product 层面，GPT 和 nGPT 的 SNR 几乎一样；
* 真正差异出现在 **dot product 求和之后**；
* nGPT full dot product SNR 约 26 dB，GPT 约 18.6 dB；
* nGPT 从 product 到 dot product 的 averaging gain 是 +9.4 dB，GPT 只有 +2.2 dB；
* 这个约 7 dB 差距在所有层上基本一致。([arXiv][2])

这说明 nGPT 的优势不是“每个元素量化得更准”，而是**高维求和时，信号更有组织地累加，而量化噪声仍然近似不相关地平均掉**。

论文进一步测量 dot-product 元素之间的相关性，发现 nGPT 的 signal terms 有弱但稳定的正相关，而 noise correlations 在两种架构里都接近零。因此，nGPT 的 SNR 增益来自 signal coherence，而不是 noise suppression。([arXiv][2])

用更工程化的话说：标准 Transformer 做低比特训练时，很多算法在努力“压 outlier、修 scale、降噪声”；nGPT 则是让模型从训练动态上学出一种对低比特更友好的 dot-product 结构。

## 2.4 为什么这个机制对大模型可能更有利？

论文的理论解释是：如果 signal terms 之间有平均正相关 (\rho_s)，那么 signal sum 的方差里会出现类似 (1 + (D-1)\rho_s) 的放大项；而量化噪声如果近似不相关，则 noise sum 仍按 (D) 线性增长。因此 hidden dimension (D) 越大，nGPT 的 signal coherence 越能放大 SNR。论文把宽度分成三个 regime，并称当前 D=4096 的模型位于优势仍在增长的 Regime II；在 D=16384 这种 405B-scale 宽度下，理论预测 nGPT/GPT 的 SNR ratio 会接近 10×。([t.co][5])

这是一条很强但仍需谨慎的 scaling claim。论文自己也承认，SNR 机制主要在较小模型上分析，3B/30B MoE 只训了约 500B tokens，相比 20T+ token 的生产级训练仍是短 horizon。([arXiv][2])

## 2.5 nGPT-NVFP4 里的低比特到底放在哪？

这篇比标题听起来更保守。它的低比特覆盖情况可以概括为：

| 位置                                                    | 精度                                    |
| ----------------------------------------------------- | ------------------------------------- |
| 1.2B dense 的所有 layer GEMM                             | NVFP4                                 |
| 400M/600M hybrid Mamba-Transformer MoE 的所有 layer GEMM | NVFP4                                 |
| 3B/30B hybrid MoE                                     | NVFP4，但最后 15% layers 保持高精度            |
| embedding                                             | BF16                                  |
| nonlinear operations                                  | BF16                                  |
| batched GEMMs                                         | BF16                                  |
| RHT                                                   | nGPT-NVFP4 去掉                         |
| dynamic per-tensor scaling / amax scaling             | nGPT-NVFP4 去掉或固定化                     |
| stochastic rounding                                   | 仍保留                                   |
| KV cache                                              | 预训练论文未讨论，不适用 inference-style KV cache |

Appendix 明确说，在 1.2B dense NVFP4 实验里，GEMMs of all layers 被量化，但 embeddings、nonlinear operations、batched-GEMMs 保持 BF16；400M/600M hybrid MoE 也是同样处理。3B/30B hybrid MoE 实验中，GPT 和 nGPT 都保留最后 15% layers 为高精度。([arXiv][2])

另外要注意一个容易误解的点：论文里的 “No Scale” 主要指去掉 **dynamic per-tensor amax / scaling computation**，不是说 NVFP4 format 完全没有 scale 概念。NVFP4 作为格式仍是 FP4 value + block scale / tensor scale 这一类 microscaling 体系；nGPT 的贡献是因为 hidden states 和 weights 被结构性约束，所以 activation scale 可以固定，RHT path 也可以移除。论文的 overhead 表显示 nGPT-NVFP4 去掉 per-tensor scaling 和 RHT，但 **SR 仍然打勾**。([arXiv][2])

## 2.6 实验结果：质量、LR 鲁棒性、速度

**1.2B dense，1T tokens。** Table 1 里，nGPT 在 BF16 和 NVFP4 下都优于标准 GPT。比如：

| 配置                          | Loss | HellaSwag |  PIQA | Winogrande |  MMLU | MMLU-Pro | GSM8K |
| --------------------------- | ---: | --------: | ----: | ---------: | ----: | -------: | ----: |
| GPT BF16                    | 1.47 |     60.54 | 73.72 |      58.17 | 45.23 |    13.98 | 33.66 |
| nGPT BF16                   | 1.46 |     61.57 | 74.27 |      59.51 | 45.50 |    14.34 | 36.24 |
| GPT NVFP4                   | 1.52 |     58.86 | 73.78 |      58.64 | 42.56 |    13.06 | 29.49 |
| nGPT NVFP4, No RHT/No Scale | 1.50 |     59.93 | 74.37 |      58.64 | 43.57 |    14.59 | 35.56 |

这说明它不是只在 quantization error 指标上好看，而是在 downstream 上也显示出优势。([t.co][5])

**3B/30B hybrid MoE。** 论文在约 500B tokens 的 1T token horizon 中比较 relative error，称 nGPT 在没有标准 NVFP4 interventions，即 RHT 和 per-tensor scaling 的情况下达到约 0% relative error；但这组实验保留了最后 15% layers 高精度。([t.co][5])

**学习率鲁棒性。** nGPT 的 BF16 optimal LR 可以直接迁移到 NVFP4；标准 GPT 则不行，论文 PDF 里写标准 GPT 的 NVFP4 optimal LR 比 BF16 optimum 大 16×，需要重新 sweep。这个结论很实用，因为低比特训练的大规模 LR sweep 成本很高。([t.co][5])

**速度。** 它的速度实验是 single transformer layer benchmark，不是完整大模型 end-to-end pretraining。GB200 上，nGPT NVFP4 “ours” 配置移除了 dynamic per-tensor amax scaling 和 RHT；在最大 hidden size 下，优化后的 nGPT NVFP4 path 相对 BF16 GPT layer baseline 达到约 3.3–3.6× speedup。([t.co][5])

---

# 3. 两篇论文的核心差异

| 维度           | MXFP4-native                                | nGPT-NVFP4                                                                     |
| ------------ | ------------------------------------------- | ------------------------------------------------------------------------------ |
| 目标           | 诊断标准 Transformer/Llama 上 MXFP4 训练哪里坏        | 设计一种天然适合 NVFP4 的 architecture                                                  |
| 硬件           | AMD Instinct MI355X native FP4              | NVIDIA Blackwell / GB200 native NVFP4                                          |
| 格式           | MXFP4                                       | NVFP4                                                                          |
| 模型           | Llama 3.1–8B，MLPerf C4                      | 1.2B dense；400M/600M 和 3B/30B hybrid Mamba-Transformer MoE                     |
| 主要低比特位置      | Transformer linear GEMM 的 Fprop/Dgrad/Wgrad | layer GEMM；但 embedding/nonlinear/batched-GEMM 等仍 BF16                          |
| 主要瓶颈         | Wgrad quantization                          | 标准 GPT dot product signal SNR 不够鲁棒                                             |
| 稳定策略         | Deterministic Hadamard H16/H32，尤其救 Wgrad    | 架构归一化：weights/hidden states on hypersphere；去掉 RHT 和 dynamic per-tensor scaling |
| 是否保留 SR      | SR 单独救不了 full MXFP4，full pipeline 下不收敛      | SR 仍保留；nGPT 只是去掉 RHT 和 per-tensor scaling                                      |
| 是否“全模型 4bit” | 不是；主要是 selected GEMM path                   | 不是；GEMM 是 NVFP4，其他关键路径 BF16，高阶实验还保留最后 15% layers 高精度                           |
| 主要风险         | recipe 可能不泛化；论文自己强调 setting-dependent       | 架构变更大，不是现有 GPT 的 drop-in 量化；超长 token horizon 未充分验证                             |

---

# 4. 和 NVIDIA 之前 NVFP4 recipe 的关系

理解 nGPT-NVFP4，最好把 NVIDIA 之前的 **Pretraining Large Language Models with NVFP4** 当作背景。那篇标准 NVFP4 recipe 的核心组件是：

1. 少量敏感 linear layers 保持高精度，约 15%，多数在网络后部；
2. Wgrad 输入做 Random Hadamard Transform；
3. weights 用 16×16 的 2D scaling，activations/gradients 用 1×16 的 1D scaling；
4. gradients 用 stochastic rounding，weights/activations 用 round-to-nearest-even。([arXiv][4])

同时，那篇也明确说：linear layer 的 Fprop/Dgrad/Wgrad GEMM 输入量化到 FP4，输出 BF16 或 FP32；但 embeddings、output projection head、normalization、nonlinearities、attention softmax、QK/AV batched GEMMs 保留 BF16/FP32。main weights、用于 microbatch/DP 累加的 weight gradients、optimizer states 保持 FP32，tensor-parallel reductions 用 BF16。([arXiv][4])

nGPT-NVFP4 的贡献可以看作是：**把标准 NVFP4 recipe 里最重的两个补丁——RHT 和动态 per-tensor scaling——用 architecture geometry 替代掉。** 但它没有替代 SR，也没有把 embedding/nonlinear/batched-GEMM/所有 final layers 都推到 FP4。

---

# 5. 格式层面：MXFP4 vs NVFP4 对训练稳定性的影响

这两篇分别用 MXFP4 和 NVFP4，不能只看“都是 FP4”。

**MXFP4** 通常是 E2M1 FP4 value + block scale。NVIDIA 的 NVFP4 背景论文说明，MXFP4 每个 block 通常 32 个连续元素共享一个 8-bit UE8M0 scale，这个 scale 是 power-of-two，因此 scale rounding 可能带来额外动态范围损失。([arXiv][4])

**NVFP4** 则把 block size 从 32 降到 16，并用 E4M3 FP8 block scale，加上 FP32 tensor-level scale。这样 local dynamic range 更细，scale 本身还有 mantissa，数值上比 MXFP4 更灵活。NVIDIA 论文认为这使 NVFP4 在训练行为上通常优于 MXFP4。([arXiv][4])

这解释了一个现象：标准 NVIDIA NVFP4 recipe 可以让 12B model 10T tokens 接近 FP8，但 AMD/Penn State 的 MXFP4-native 需要 deterministic Hadamard 才让 full Wgrad 稳定。两者不是同一格式，也不是同一硬件实现。

---

# 6. 对工程实现最有用的 takeaways

## 6.1 现有 GPT/Llama 架构上做 FP4：先盯住 Wgrad

MXFP4-native 最有工程价值的结论是：不要只测 forward quantization，也不要只看 short-run loss。真正要测的是：

* Wgrad 是否低比特；
* Wgrad 低比特以后 token-to-target 是否明显增加；
* 是否需要 rotation/Hadamard；
* stochastic rounding 是否只是让训练噪声更大；
* final target 是不是只看 step throughput，而忽略了 token overhead。

如果一个 FP4 训练方案没有覆盖 Wgrad，或者 Wgrad 仍然高精度，那它的难度和 full FP4 training 不是一回事。

## 6.2 “随机”不一定比“确定性”好

NVIDIA 标准 NVFP4 recipe 认为 RHT + SR 对大模型训练有效；MXFP4-native 却发现 full MXFP4 下 SR 和 randomized Hadamard 都失败，deterministic Hadamard 成功。这个冲突非常值得重视。合理结论不是“以后都用 deterministic”，而是：

**FP4 稳定性强依赖 format、block size、scale 类型、GEMM path、模型架构、硬件 kernel layout。**

尤其是 Wgrad，量化误差直接进入优化器更新，任何额外随机性都可能从“去偏”变成“扰乱优化轨迹”。

## 6.3 nGPT 是架构路线，不是量化补丁

nGPT-NVFP4 的价值在于指出：与其不断给标准 Transformer 加 RHT、scale search、selective BF16 fallback，不如从架构上控制 dot product 的 signal accumulation。它的问题也很明显：这不是 drop-in 到现有 GPT checkpoint 的方案，而是需要从头训练 normalized architecture。

## 6.4 两篇都没有真正证明“所有东西都 4bit”

这点尤其重要。按你关心的位置总结：

| 位置                                        | MXFP4-native                                  | nGPT-NVFP4 / NVIDIA NVFP4 相关结论                                       |
| ----------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------- |
| Linear Fprop                              | MXFP4 tested                                  | NVFP4 GEMM                                                           |
| Linear Dgrad                              | MXFP4 tested                                  | NVFP4 GEMM                                                           |
| Linear Wgrad                              | MXFP4 tested，最脆弱                              | NVFP4 GEMM，但标准 recipe 重点给 Wgrad 做 RHT/SR                             |
| Embedding                                 | MXFP4-native 未详细说明                            | BF16                                                                 |
| Nonlinear / activation function           | MXFP4-native 未详细说明                            | BF16                                                                 |
| Attention QK / AV batched GEMM            | MXFP4-native 未详细说明                            | 标准 NVFP4 recipe 保留 BF16/FP32；nGPT Appendix 也说 batched-GEMMs BF16     |
| Softmax                                   | MXFP4-native 未详细说明                            | BF16/FP32                                                            |
| Norm                                      | MXFP4-native 未详细说明                            | 标准 NVFP4 保留；nGPT 架构移除 RMSNorm/LayerNorm                              |
| Gradient accumulation                     | MXFP4-native 未详细说明                            | 标准 NVFP4 中 main weights、累加用 weight gradients、optimizer states 是 FP32 |
| Tensor-parallel reduction / communication | MXFP4-native 未详细说明                            | 标准 NVFP4 中 TP reductions 是 BF16                                      |
| KV cache                                  | 两篇都是 pretraining，不讨论 inference-style KV cache | 不适用；不能从这些论文推出 KV cache FP4/FP8                                       |

---

# 7. 我的判断

**MXFP4-native 的最大价值是“定位病灶”。** 它用很干净的 stage-wise 实验说明：FP4 训练不稳定的核心不是所有路径都一样难，而是 Wgrad path 最敏感。它还提醒我们，SR 和 randomized Hadamard 不是万能稳定器；至少在 native MXFP4 + Llama 3.1–8B + MLPerf C4 这个 setting 下，deterministic Hadamard 才是关键。

**nGPT-NVFP4 的最大价值是“改变问题定义”。** 它不是继续给标准 GPT 打补丁，而是让模型几何结构本身更适合低比特 dot product。它给出的 SNR 分析很有启发：低比特鲁棒性不一定来自降低局部量化噪声，也可以来自让信号在高维求和时更 coherent。

**短期工程上，MXFP4-native 更像可直接参考的调参/诊断指南；中长期，nGPT-NVFP4 更像架构方向。** 如果目标是现有 Llama/GPT 训练栈降精度，优先围绕 Wgrad、Hadamard、scale/block layout 做系统 ablation。如果目标是下一代从头预训练架构，nGPT 这种 normalized architecture 很值得跟进，但还需要更长 token horizon、更大 dense model、更多下游任务和端到端系统吞吐验证。

[1]: https://arxiv.org/pdf/2605.09825 "Pretraining large language models with MXFP4 on Native FP4 Hardware"
[2]: https://arxiv.org/html/2605.06067v1 "Normalized Architectures are Natively 4-Bit"
[3]: https://arxiv.org/html/2605.09825v2 "Pretraining Large Language Models with MXFP4 on Native FP4 Hardware"
[4]: https://arxiv.org/html/2509.25149v1 "Pretraining Large Language Models with NVFP4"
[5]: https://t.co/FkxC8Tp9s2 "Normalized Architectures are Natively 4-Bit"
是的，你的理解基本对：**NVIDIA NVFP4 recipe 推荐的是“固定 random signs 的 Random Hadamard Transform”，并且主要只用于 Wgrad GEMM 的输入**。这和 MXFP4-native 论文里说的 “deterministic Hadamard 成功、randomized Hadamard 失败”表面上冲突，但其实要拆成几个层次看。

## 1. 三种 Hadamard 其实不能混在一起说

可以把它们写成：

[
H_{\text{det}} = H_d / \sqrt{d}
]

[
H_{\text{rand-fixed}} = S H_d / \sqrt{d}
]

[
H_{\text{rand-dynamic}} = S_t H_d / \sqrt{d}
]

其中 (H_d) 是标准 Hadamard 矩阵，(S) 是对角线为 (\pm 1) 的 sign matrix。

| 版本                                       | 数学形式      | 训练过程中是否变化                        | 含义                            |
| ---------------------------------------- | --------- | -------------------------------- | ----------------------------- |
| **Deterministic Hadamard**               | (H_d)     | 不变                               | 标准固定 Hadamard，没有 random signs |
| **Fixed random-sign Hadamard**           | (S H_d)   | (S) 初始化一次后不变                     | NVIDIA NVFP4 recipe 用的主要是这个   |
| **Dynamic/random per-instance Hadamard** | (S_t H_d) | 每个 layer / step / transform 重新采样 | 真的会持续引入随机扰动                   |

最容易误解的是第二种。**固定 random signs 虽然名字里有 random，但从训练动力学看，它不是 stochastic perturbation；它只是初始化时选了一个固定正交基。** 之后每一步、每一层都用同一个 sign vector，训练过程是确定的。

NVIDIA paper 明确说，Random Hadamard 的随机性来自一个 diagonal random matrix，sign entries 会翻转 Hadamard 的行；它们在 setup 中使用 **single random sign vector shared across all linear layers throughout training**，并且实验发现增加 random sign vectors 没有可测收益。([arXiv][1]) Transformer Engine 文档也写得更工程化：这个 vector 是 fixed 的，RHT matrix 初始化时算一次并缓存。([NVIDIA Docs][2])

---

## 2. NVIDIA NVFP4 recipe 里为什么只在 Wgrad 前用 RHT？

NVIDIA 的结论是：**Wgrad 是最值得加 RHT 的地方，Fprop / Dgrad 加 RHT 反而可能不划算甚至伤质量。**

它们的推荐 recipe 是：

1. 少量敏感 linear layers 保持高精度；
2. 对 **weight-gradient GEMM 的 inputs** 使用 (16 \times 16) Random Hadamard；
3. weights 用 (16 \times 16) 2D scaling，activations / gradients 用 (1 \times 16) 1D scaling；
4. gradients 用 stochastic rounding，weights / activations 用 round-to-nearest-even。([arXiv][1])

论文在 Appendix 里进一步说明：对 1.2B 模型，RHT 用在 **Wgrad inputs** 会改善 validation loss；但用在 Fprop 或 Dgrad inputs 会降低 model quality，因此它们把 RHT 限定在 Wgrad。([arXiv][1])

Transformer Engine 文档也一致：RHT applied to columnwise quantization of inputs and gradients, which are operands for the Wgrad GEMM；Wgrad GEMM 对量化误差特别敏感，所以额外做 outlier smoothing。([NVIDIA Docs][2])

这里的 Wgrad GEMM 大致是：

[
dW = dY^T X
]

RHT 的作用是在 quantize (X) 和 (dY) 前，把 reduction dimension 上的 outlier 分散掉。高精度下如果两边都乘匹配的正交矩阵，dot product 本身不变：

[
(AH)(H^T B) = AB
]

但量化是在 transform 之后发生的，所以分布变平滑后，FP4 block quantization 的误差会变小。

---

## 3. NVIDIA 的 fixed random signs 到底有什么用？

标准 Hadamard (H_d) 是固定结构。如果某些 outlier pattern 恰好和 Hadamard basis 或 tile/block layout 有结构性对齐，那么标准 Hadamard 可能没有把能量充分打散。加一个固定 random sign vector：

[
x \mapsto x S H_d
]

等价于先随机翻转输入维度的符号，再做 Hadamard mixing。这样会改变 Hadamard 后每个输出坐标的加减组合，降低“结构化 outlier 刚好穿过 transform”的概率。

NVIDIA 做了一个很关键的 ablation：比较

1. 每个 transform instance 都用新的 random sign；
2. 整个训练用 single fixed seed；
3. 不用 random sign vector。

结果是：**没有 random sign vector 时 12B 模型质量更低；每个 instance 重新随机也没有比 single fixed seed 更好。因此 single fixed seed 就够了。** 1.2B 小模型上三者差异不明显，说明这个技巧在更大模型、更长 token horizon 下更关键。([arXiv][1])

所以 NVIDIA 的结论不是“训练时持续加随机噪声有帮助”，而是：

> 选一个固定随机正交基，比裸 Hadamard basis 更稳；但没必要每次重采样。

---

## 4. MXFP4-native 的 deterministic Hadamard 和 NVIDIA fixed-RHT 的真正区别

MXFP4-native 里所谓 **deterministic Hadamard**，按论文描述，是不用 random signs 的固定 Hadamard transform。作者明确说，虽然 random sign flips 被广泛使用，但他们的实验发现 randomized signs 在该 setting 下会伤害 convergence stability，所以使用 deterministic transform。([arXiv][3])

它们的实验结果是：

| Stabilizer             | MXFP4 GEMMs           | 结果                    |
| ---------------------- | --------------------- | --------------------- |
| None                   | Fprop + Dgrad + Wgrad | 26–27% token overhead |
| Stochastic rounding    | Fprop + Dgrad + Wgrad | does not converge     |
| Randomized Hadamard    | Fprop + Dgrad + Wgrad | does not converge     |
| Deterministic Hadamard | Fprop + Dgrad + Wgrad | 8–9% token overhead   |

论文还指出，Wgrad quantization 是主要 convergence degradation 来源，而 deterministic Hadamard 能让 full-pipeline MXFP4 接近 FP8 baseline。([arXiv][3])

所以二者区别可以概括为：

| 维度                 | MXFP4-native                                                                                                      | NVIDIA NVFP4 recipe                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 格式                 | MXFP4                                                                                                             | NVFP4                                                      |
| Hadamard           | deterministic (H_d)，不加 random signs                                                                               | fixed random-sign RHT，即 (S H_d)，(S) 初始化后固定                 |
| RHT 位置             | 论文表里对 Fprop / Dgrad / Wgrad staged enablement 都做了比较；full pipeline 成功时是 deterministic Hadamard + Fprop/Dgrad/Wgrad | 推荐只放在 Wgrad inputs                                         |
| 对 random signs 的结论 | randomized signs 在该 setting 下不稳定                                                                                  | fixed random signs 对 12B 长训练有益；per-instance random 没额外收益   |
| 主要解释               | 控制 MXFP4 micro-scaling error，不是增加 stochasticity                                                                   | 打散 Wgrad inputs 的 outliers，避免结构化 outlier 对齐 Hadamard basis |

---

## 5. 为什么两篇结论不矛盾？

我认为至少有五个原因。

### 第一，MXFP4 和 NVFP4 的 scale 机制不同

NVIDIA paper 专门比较过 MXFP4 和 NVFP4 scale factor。MXFP4 的 scale factor 受限于 power-of-two，因此 block amax 不能总是精确映射到 FP4 可表示范围，可能浪费 FP4 bins 或损失动态范围；NVFP4 用 E4M3 block scale，可以把 block amax 更精确地贴近 FP4 最大可表示值，从而保留更多动态范围。([arXiv][1])

这意味着，同样一个 random sign Hadamard 后的分布，在 MXFP4 和 NVFP4 里会被完全不同地 scale/round。**MXFP4 的 power-of-two microscale 更容易产生结构化 scale error**，所以 random signs 改变 block 内 cancellation pattern 后，可能反而让 Wgrad path 的 scale/quantization error 更不稳定。

### 第二，NVIDIA 的 random signs 是固定的，不是持续随机噪声

NVIDIA 明确比较过 per-instance random signs 和 single fixed seed，发现 per-instance 没有改进，single fixed seed 足够。([arXiv][1])

MXFP4-native 论文只写 randomized Hadamard / randomized signs，不够详细说明它到底是 per-instance 采样、per-layer 固定、还是 single global fixed signs。严格地说，不能把 MXFP4-native 的“randomized Hadamard failed”直接等同于“NVIDIA fixed random signs would fail”。

如果 MXFP4-native 的 randomized variant 是动态重采样，那它和 NVIDIA recipe 根本不是同一个东西。即使它也是固定 random signs，格式、block size、scaling、应用位置也都不同。

### 第三，NVIDIA 只在 Wgrad 用 RHT；MXFP4-native full success 不是同一个 placement

NVIDIA ablation 显示，RHT 放到 Fprop/Dgrad 反而可能 degrade quality，所以最后只放 Wgrad。([arXiv][1])

MXFP4-native 的表格是 staged enablement：Fprop、Fprop+Dgrad、Fprop+Dgrad+Wgrad。它显示 deterministic Hadamard 在 full Fprop+Dgrad+Wgrad 下成功，但没有清楚隔离出“只对 Wgrad inputs 用 deterministic Hadamard，Fprop/Dgrad 不用 Hadamard”的结果。也就是说，它证明的是 deterministic Hadamard 在它的 full MXFP4 setup 下有效；不能直接推出最佳 placement 一定和 NVIDIA 不同。

### 第四，NVIDIA 避免在 weights 上使用 RHT

NVFP4 paper 提到，Hadamard / scaling 沿 dot-product dimension 做，会导致 forward 和 backward 中同一 tensor 的量化表示不一致，可能破坏 chain rule；因此 Random Hadamard 不施加到 weight tensors 上。([arXiv][1])

这点很重要。Wgrad 的两个输入是 activation 和 output gradient，不是 weight，所以它可以在 Wgrad 前做 RHT 而不引入 weight representation consistency 问题。Fprop/Dgrad 如果要把 transform cancel 掉，通常会牵涉 weight operand，代价和一致性问题更复杂。

### 第五，两篇 paper 的目标和系统条件不同

MXFP4-native 是 AMD MI355X native MXFP4、Llama 3.1–8B、MLPerf C4 target perplexity 的诊断实验。它的核心结论是：在这个 setting 下，Wgrad 是病灶，deterministic Hadamard 是唯一成功的 tested stabilizer。([arXiv][3])

NVIDIA NVFP4 是 Blackwell NVFP4 recipe，目标是 12B、10T-token horizon 的稳定训练。它不是单靠 RHT，而是 RHT + 2D weight scaling + stochastic rounding on gradients + selective high precision layers 的组合。论文明确说这些组件共同构成 12B 长训练稳定性的 recipe。([arXiv][1])

---

## 6. 实现层面的判断

我会把实践建议归纳成这样：

**如果你在复现 NVIDIA NVFP4 recipe：**
优先照它的 recipe 做：RHT 只放 Wgrad inputs，Hadamard size 16，random signs 用 single fixed seed，全训练共享；不要每 step / 每 instance 重采样。NVIDIA 已经明确看到 per-instance random 没收益，而 no random signs 在 12B 长训练上质量更差。([arXiv][1])

**如果你在做 MXFP4，尤其是 E8M0/power-of-two scale 的 MXFP4：**
不要直接照搬 NVIDIA 的 fixed-random-sign RHT。MXFP4-native 的证据显示，在它们的 setting 下 deterministic Hadamard 更稳，randomized signs 可能伤 convergence。([arXiv][3])

**如果你要做 ablation，最有信息量的不是“Hadamard vs no Hadamard”，而是这四组：**

| Ablation                                 | 目的                    |
| ---------------------------------------- | --------------------- |
| no H                                     | 看 Wgrad FP4 原始损伤      |
| deterministic H only on Wgrad            | 验证 bare Hadamard 是否足够 |
| fixed random-sign H only on Wgrad        | 对齐 NVIDIA recipe      |
| per-instance random-sign H only on Wgrad | 检查动态随机性是否伤害优化         |

然后再加两组 placement：

| Placement             | 目的                                  |
| --------------------- | ----------------------------------- |
| Wgrad only            | 对齐 NVIDIA，最可能收益/成本比高                |
| Fprop + Dgrad + Wgrad | 对齐 MXFP4-native full-pipeline table |

我个人会优先怀疑：**Wgrad-only + fixed random signs** 在 NVFP4 上更合理；**Wgrad-only 或 full-pipeline deterministic H** 在 MXFP4 上更值得先试。核心原因是 NVFP4 的 E4M3 block scale 能更好消化 random-sign 后的分布变化，而 MXFP4 的 power-of-two scale 对 block distribution 更敏感。

[1]: https://arxiv.org/html/2509.25149v2 "Pretraining Large Language Models with NVFP4"
[2]: https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html "NVFP4 — Transformer Engine 2.13.0 documentation"
[3]: https://arxiv.org/html/2605.09825v3 "Pretraining Large Language Models with MXFP4 on Native FP4 Hardware"

