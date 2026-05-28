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
