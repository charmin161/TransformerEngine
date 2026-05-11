你的判断是对的：**不要再用额外脚本硬拉 SM Active**。你已经实测 MoE 训练耗时翻倍，说明辅助 kernel 在真实抢资源。

对于 Megatron-LM/Megatron-Core 的 MoE 训练，**SM Active 只有 35% 很常见**，尤其是 EP、All-to-All、动态 routing、小 expert GEMM、pipeline/host overhead 混在一起时。更合理的目标不是“人为把 SM Active 拉高”，而是让训练本身在通信等待、CPU gap、小 kernel、负载不均时少空转。

## 为什么 MoE 的 SM Active 会低

MoE 不是一个连续的大 GEMM 工作负载，而是：

```text
Router
Token permutation / dispatch
EP All-to-All
Grouped expert GEMM
Token combine / unpermutation
Backward 里再来一遍类似过程
```

其中很多阶段不是强计算阶段。

Megatron 的 MoE 优化文档把瓶颈分成三类：memory、communication、compute efficiency，并明确指出 MoE 的 compute efficiency 问题常来自 **small expert GEMMs + host overhead**，指标上就表现为 GPU SM utilization 偏低。([NVIDIA Docs][1])

具体到你的 MoE 训练，低 SM Active 常见原因有这些：

### 1. EP All-to-All 暴露出来，GPU 在等通信

如果 expert parallel 跨 GPU 分发 token，MoE 层会发生 token dispatch/combine。Megatron 文档也明确说 EP/TP 是 communication-intensive，最好把 EP×TP 限制在 NVLink 域内；跨节点扩 EP/TP 通常会暴露更多通信。([NVIDIA Docs][2])

这种情况下，`SM Active=35%` 不代表 expert GEMM 很慢，而可能是：

```text
一部分时间 GPU 在跑 GEMM；
一部分时间在等 NCCL / All-to-All / dispatch；
一部分时间在等 CPU launch / synchronization。
```

### 2. expert GEMM 太小，单个 GEMM 喂不满 GPU

MoE 每个 expert 只处理被 router 分到的 token。若 micro-batch 小、sequence length 小、top-k 小、expert 数多、EP 切得太碎，则每个 local expert 的 token 数很少。

结果是：

```text
每个 expert GEMM 的 M 维很小
Grouped GEMM 不够大
kernel launch 多
单 kernel SM 覆盖率不高
```

Megatron 文档中也提到，EP 相比 TP 对 expert 层通常有更好的 GEMM efficiency，因为本地矩阵尺寸更大，GPU utilization 更好；例如 Mixtral 8x7B 上 EP8×TP1 优于 EP4×TP2。([NVIDIA Docs][2])

### 3. token permutation / unpermutation / router 是一堆小 kernel

MoE 层除了 GEMM，还有 routing、topk、scatter/gather、permute/unpermute、combine 等操作。这些 kernel 往往：

```text
计算量小
访存/索引开销高
kernel 数量多
单次执行时间短
SM Active 不稳定
```

所以 Megatron 推荐打开 router fusion、permute fusion、grouped GEMM。官方性能建议里直接列了：

```bash
--moe-grouped-gemm
--moe-router-fusion
--moe-permute-fusion
```

([NVIDIA Docs][2])

### 4. expert 负载不均，部分 rank/SM 在等慢 expert

MoE router 如果把 token 分配得不均，某些 expert/rank token 多，某些 expert/rank token 少。少的 rank 很快结束，多的 rank 成为 straggler，整体同步点等待。

Megatron 支持多种 load balancing，例如 `aux_loss`、`seq_aux_loss`、`global_aux_loss`、`sinkhorn`、aux-loss-free bias balancing 等。([NVIDIA Docs][2])

### 5. CPU launch gap / Python 抖动

如果 Nsight Systems timeline 里 kernel 与 kernel 之间有明显空白，SM Active 会被拉低。Megatron 文档把这个归为 CPU overhead bottleneck，并建议 manual GC、CUDA Graph、减少 kernel launch、降低 TP 或增大 micro-batch。([NVIDIA Docs][2])

---

## 能不能“速度几乎不变，但 SM Active 提升”？

可以，但前提是：**提升的 SM Active 必须来自训练本身的有效 overlap/fusion，而不是额外无意义计算。**

也就是说，有两类方案：

```text
A. 有意义提升：
	隐藏通信等待
	融合小 kernel
	减少 CPU launch gap
	让 expert GEMM 变大
	减少负载不均

B. 无意义提升：
	额外跑 busy kernel
	额外做空转计算
```

你前面的脚本属于 B，所以训练耗时翻倍是正常结果。

你现在要做的是 A。

---

## 最小改动优先级

### 第一优先级：确认这些通用性能开关是否已打开

先检查你的 Megatron-LM 启动参数里有没有这些：

```bash
--moe-grouped-gemm
--moe-router-fusion
--moe-permute-fusion
--use-distributed-optimizer
--overlap-param-gather
--overlap-grad-reduce
--tp-comm-overlap
--manual-gc
--manual-gc-interval 100
```

这些属于相对低风险配置项。Megatron 官方文档也把 grouped GEMM、router fusion、permute fusion、通信 overlap、manual GC 列为 MoE training 的通用性能建议。([NVIDIA Docs][2])

如果你只想先试最小集合，我建议先加：

```bash
--moe-grouped-gemm \
--moe-router-fusion \
--moe-permute-fusion
```

这三个最直接针对 MoE 的低 SM utilization。

---

### 第二优先级：如果 EP > 1，优先优化 token dispatcher

如果你现在是 EP 训练，确认 dispatcher。

Megatron 基础示例里 MoE token dispatcher 用的是：

```bash
--moe-token-dispatcher-type alltoall
```

([NVIDIA Docs][2])

如果是跨节点 EP 或 fine-grained MoE，可以试 DeepEP：

```bash
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend deepep
```

Megatron 文档说 DeepEP 针对大规模 MoE token dispatch/combine 优化，尤其推荐给 DeepSeek-V3 这类 fine-grained MoE；文档也给出了这组启用参数。([NVIDIA Docs][3])

如果是 GB200 / B200 / H100，且场景是 intra-node 或 multi-node NVLink，可试 HybridEP：

```bash
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep
```

官方文档描述 HybridEP 是 NVIDIA 的优化 dispatcher，目标是降低 SM resource usage，并支持 intra-node 和 multi-node NVLink 场景。([NVIDIA Docs][3])

注意：DeepEP/HybridEP 不是“保证提升 SM Active”。它们更可能减少通信开销、减少等待和资源浪费，最终 step time 下降或持平。SM Active 可能升，也可能因为通信更高效而不明显升，但 tokens/sec 应该是更重要指标。

---

### 第三优先级：打开 MoE EP All-to-All overlap

如果 Nsight 里看到大量 EP All-to-All 暴露在关键路径上，试：

```bash
--overlap-moe-expert-parallel-comm
```

如果 TransformerEngine 版本支持，再加：

```bash
--delay-wgrad-compute
```

Megatron 文档中 EP All-to-All 的 overlap 配置就是这两个参数；新版本还提到 batch-level overlap 用 `--overlap-moe-expert-parallel-comm`，可选 `--delay-wgrad-compute`。([NVIDIA Docs][2])

这类改动最符合你的目标：**让原本通信等待期间有计算可以跑**。如果配置适配得好，SM Active 会升，step time 可能不变甚至变快。

---

### 第四优先级：增大 micro-batch，让 expert GEMM 变大

如果每个 expert token 数太少，`--moe-grouped-gemm` 也救不了太多。最直接的办法是增大每 GPU micro-batch：

```bash
--micro-batch-size N
```

或者在显存允许的情况下减少过度切分，例如降低 TP/CP，让每 GPU 本地矩阵更大。

Megatron 性能指南明确提醒：过大的 TP 或 CP 会让每 GPU 并行度不足、通信过高，甚至 host-performance bound；文档举例说 TP=8 可能导致低 GPU utilization。([NVIDIA Docs][4])

如果你要保持 global batch 不变，可以：

```text
增大 micro-batch-size
减少 gradient-accumulation-steps
```

这样 step 数语义不变，可能提升单 step 的 SM Active 和 tokens/sec。

---

### 第五优先级：调整 EP / TP / ETP 映射

如果你现在 MoE 层用了较大的 TP，可能会把 expert GEMM 切得太碎。Megatron 文档建议 MoE expert 层优先 EP 而不是 TP，因为 EP 通常有更好的 GEMM efficiency、更低通信，并且 `EP = num_experts` 时可以消除 local token permutation。([NVIDIA Docs][2])

优先检查：

```bash
--expert-model-parallel-size
--expert-tensor-parallel-size
--tensor-model-parallel-size
```

经验方向：

```text
Attention 层可能需要 TP
MoE expert 层尽量 ETP 小一些，EP 合理一些
EP × TP 尽量留在 NVLink 域内
```

Megatron 文档也提到 Parallel Folding 可以把 attention 和 MoE 的并行策略解耦：attention 用 TP×CP×DP×PP，MoE 用 ETP×EP×EDP×PP。([NVIDIA Docs][1])

这不是最小改动，但如果你的当前 parallel mapping 不合理，它可能比小开关更有效。

---

## 我建议你按这个实验顺序来

### Step 1：先不要追 SM Active，先记录基线

记录这些：

```text
每 step 时间
tokens/s
GPU SM Active
GPU Tensor pipe active
DRAM active / HBM bandwidth
NCCL time
MoE layer time
expert GEMM time
token dispatch/combine time
```

如果只能先看简单指标，至少看：

```bash
nvidia-smi dmon -s pucvmet -d 1
```

以及 Megatron 自带 timer log。

---

### Step 2：只加 MoE fusion / grouped GEMM

```bash
--moe-grouped-gemm \
--moe-router-fusion \
--moe-permute-fusion
```

预期：

```text
step time 下降或基本不变
SM Active 可能上升
kernel 数量减少
小 kernel gap 减少
```

---

### Step 3：优化 dispatcher

如果是普通 EP：

```bash
--moe-token-dispatcher-type alltoall
```

如果是跨节点或 DeepSeek/Qwen 这类 fine-grained MoE：

```bash
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend deepep
```

如果是合适的 NVIDIA NVLink/GB/Hopper 场景：

```bash
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep
```

---

### Step 4：打开通信 overlap

```bash
--overlap-moe-expert-parallel-comm
```

可选：

```bash
--delay-wgrad-compute
```

这一步最有可能做到你要的“速度几乎不变甚至更快，同时 SM Active 提升”。

---

### Step 5：如果还有 CPU gap，再试 CUDA Graph

对于 dropless MoE，完整 MoE layer 因为动态 shape 不一定容易 capture。Megatron 文档说，dropless MoE 的 full CUDA Graph 受动态 expert computation 限制，partial CUDA Graph 更安全，通常 capture attention、router、moe_preprocess 等静态部分。([NVIDIA Docs][1])

可以试：

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-scope attn moe_router moe_preprocess
```

如果你的版本不支持这些 scope，就先用 attention-only：

```bash
--cuda-graph-scope attn
```

---

## 你要特别避免的方向

### 1. 不要再用外部 busy kernel

你已经验证了它会让训练耗时翻倍。它提升的是监控数字，不是训练效率。

### 2. 不要为了 SM Active 盲目增大无效计算

比如增大 padding、强行 pad expert input 到 capacity，可能让 shape 更静态、更容易 graph capture，但也可能增加大量无效 expert GEMM。只有当它换来了 CUDA Graph / grouped kernel / overlap 的收益时才值得。

### 3. 不要只看 SM Active

MoE 训练里更重要的是：

```text
tokens/s
step time
MFU
NCCL exposed time
expert GEMM efficiency
load balance
```

SM Active 从 35% 提到 60%，但 step time 不变，可能只是把以前等待的时间填上了；这有价值。
SM Active 从 35% 提到 80%，但 step time 翻倍，那就是反优化。

---

## 一个比较现实的判断

如果你的 MoE 训练当前 `SM Active≈35%`，最可能的真实瓶颈排序是：

```text
1. EP All-to-All / token dispatch 暴露
2. expert GEMM 太小，Grouped GEMM 没开或效果不足
3. router/permute/unpermute 小 kernel 太多
4. CPU launch gap / Python GC 抖动
5. pipeline bubble 或 TP/CP/PP 映射过度切分
6. expert load imbalance
```

最小改动建议就是：

```bash
--moe-grouped-gemm \
--moe-router-fusion \
--moe-permute-fusion \
--overlap-moe-expert-parallel-comm \
--use-distributed-optimizer \
--overlap-param-gather \
--overlap-grad-reduce \
--tp-comm-overlap \
--manual-gc \
--manual-gc-interval 100
```

如果 EP dispatcher 仍是瓶颈，再试：

```bash
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend deepep
```

或硬件合适时：

```bash
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep
```

一句话：**可以在速度几乎不变甚至更快的情况下提高 SM Active，但要靠 Megatron MoE 本身的 grouped GEMM、fusion、dispatcher、communication overlap、CUDA Graph 和更合理 parallel mapping；不能靠外部脚本。**

[1]: https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/moe-optimization.html "MoE Training Optimization — Megatron Bridge"
[2]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html "Mixture of Experts — Megatron Core"
[3]: https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html "Mixture of Experts package — Megatron-LM"
[4]: https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html "Performance Tuning Guide — Megatron Bridge"
