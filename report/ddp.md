下面按你 profiler 里这组函数来解释。**这基本对应 `--use-distributed-optimizer` 路径**：
Backward 阶段先把 DP 梯度做 **reduce-scatter**，每个 DP rank 只保留一段 gradient shard；optimizer 只更新本 rank 的 main-param shard；更新完再把参数 shard 写入 param buffer，并通过 **all-gather** 还原完整模型参数供下一轮 forward 使用。

你的总体配置是 `--bf16`，在默认 Megatron 设置下，如果没有显式加：

```bash
--grad-reduce-in-bf16
```

那么通常是：

```text
模型参数：BF16
DDP grad buffer / main_grad：FP32
DP 梯度 reduce-scatter：FP32
optimizer main params：FP32
optimizer state：通常 FP32
参数 all-gather：BF16
```

---

# 0. 总体调用链

一次 optimizer step 近似是：

```text
backward
  -> param.grad 累加到 param.main_grad
  -> DDP grad buffer 做 reduce-scatter / all-reduce
  -> finish_grad_sync 等通信完成

optimizer.step()
  -> prepare_grads()
       -> _copy_model_grads_to_main_grads()
            -> _copy_group_grads()
  -> get_grad_norm()
       -> get_main_grads_for_grad_norm()
       -> get_grad_stats_parallel_group()
       -> get_grad_norm_fp32()
  -> clip_grad_by_total_norm_fp32()
  -> step_with_ready_grads()
       -> inner optimizer.step()
       -> copy_main_params_to_model_params()
            -> copy_group_params()
       -> params all-gather
  -> logical_and_across_model_parallel_group()
  -> reduce_max_stat_across_model_parallel_group()
```

其中真正的大规模 DP 片间通信主要是两类：

```text
1. backward 后的 grad reduce-scatter
2. optimizer 更新后的 param all-gather
```

`get_grad_norm_fp32`、`logical_and_across_model_parallel_group`、`reduce_max_stat_across_model_parallel_group` 也有通信，但只是 scalar 或很小 tensor 的 all-reduce，不是大梯度/大参数传输。

---

# 1. Backward 后：梯度先进 DDP grad buffer

Megatron DDP 的 backward post hook 会把 `param.grad` 加到 `param.main_grad`，然后把 `param.grad` 置空；如果启用了 overlap grad reduce，还会在 bucket ready 后注册通信。代码里能看到：

```python
param.main_grad.add_(param.grad.data)
param.grad = None
...
register_grad_ready(...)
```

也就是说，optimizer step 看到的主要不是 PyTorch 原生 `param.grad`，而是 Megatron DDP buffer 里的 `param.main_grad`。([GitHub][1])

在 `_ParamAndGradBuffer` 初始化时，Megatron 会创建 contiguous `grad_data`，并把每个参数的 `param.main_grad` 映射成这个 `grad_data` 的 view。代码里：

```python
self.grad_data = torch.zeros(..., dtype=self.grad_dtype, ...)
...
param.main_grad = self._get(..., buffer_type=BufferType.GRAD)
```

所以 `param.main_grad` 本质上是 DDP grad buffer 的切片。([GitHub][2])

---

# 2. DDP grad reduce-scatter：真正的梯度 DP 片间传输

在 `start_grad_sync()` 里，如果使用 distributed optimizer，会对 `bucket.grad_data` 做 reduce-scatter；否则做 all-reduce。核心逻辑是：

```python
if self.ddp_config.use_distributed_optimizer:
    dist_reduce_scatter_func(local_data_view, bucket.grad_data, ...)
else:
    torch.distributed.all_reduce(bucket.grad_data, ...)
```

这一步是 **片间传输**：同一个 DP group 内的多个 GPU 通过 NCCL 交换梯度。单机内一般走 NVLink / PCIe，跨机走 IB / 以太网；Megatron 代码层只看到 `torch.distributed` collective，不区分物理链路。([GitHub][2])

默认 `--bf16` 下，Megatron 的 `grad_dtype` 通常是 FP32，因为 `group_params_for_buffers()` 里写的是：

```python
grad_dtype = torch.float if grad_reduce_in_fp32 else param.dtype
```

而 bf16 默认会打开 `grad_reduce_in_fp32`。因此 `bucket.grad_data` 是 FP32，reduce-scatter 传输的 tensor 也是 FP32。([GitHub][2])

这一步的累加精度可以按这个规则理解：

```text
默认 --bf16:
  bucket.grad_data dtype = FP32
  reduce-scatter 输入/输出 dtype = FP32
  reduction op = SUM 或 AVG
  通信和 reduction 的外部可见 dtype = FP32

如果加 --grad-reduce-in-bf16:
  bucket.grad_data dtype = BF16
  reduce-scatter 输入/输出 dtype = BF16
  外部可见通信 dtype = BF16

如果加 --ddp-reduce-scatter-with-fp32-accumulation:
  先用 all-to-all 收集低精度 shard
  本地 torch.sum(..., dtype=torch.float32)
  再 copy 回输出 tensor，输出 tensor dtype 仍按原 buffer
```

`reduce_scatter_with_fp32_accumulation.py` 里明确写了：先 `all_to_all_single`，再 `torch.sum(..., dtype=torch.float32)`，最后 `output_tensor.copy_(output_tensor_in_fp32)`。([GitHub][3])

---

# 3. `prepare_grads()`

`prepare_grads()` 是 optimizer step 的第一步。对于 mixed precision optimizer，它做两件事：

```text
1. 把 model grad / DDP buffer 里的 grad 交给 main optimizer param 的 grad 字段
2. 如果有 grad scaler，则 unscale 并检查 inf/nan
```

代码中 `prepare_grads()` 先调用：

```python
self._copy_model_grads_to_main_grads()
```

然后如果 `self.grad_scaler` 存在，会调用 `_unscale_main_grads_and_check_for_nan()`。bf16 默认通常没有 fp16 那种动态 loss scaler，所以这一段大多不会成为主要路径。([GitHub][4])

传输性质：

```text
片内传输：有，主要是 GPU 本地 copy / view / cast
片间传输：通常没有
累加：没有新的大规模累加
```

如果是 fp16 动态 loss scaling，`_unscale_main_grads_and_check_for_nan()` 会对 `found_inf` 做一次 all-reduce MAX，用于同步是否出现 inf/nan；这是 scalar 级别通信，不是梯度大 tensor 通信。([GitHub][4])

---

# 4. `_copy_model_grads_to_main_grads()`

这个函数是 distributed optimizer 的关键。注释说得很清楚：它发生在 DDP grad buffer 的 reduce-scatter 之后，负责把已经 reduce-scatter 好的 grad 从 grad buffer 拷到 main shard 的 `.grad` 字段。([GitHub][5])

核心代码是：

```python
model_grad = model_param.main_grad
shard_model_grad = model_grad.view(-1)[param_range.start:param_range.end]

if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
    shard_main_param.decoupled_grad = shard_model_grad
else:
    shard_main_param.grad = shard_model_grad.float()
```

所以默认非 precision-aware optimizer 下，会把 `shard_model_grad` 转成 FP32 后放到 `shard_main_param.grad`。如果前面的 grad buffer 已经是 FP32，这个 `.float()` 基本保持 FP32；如果你强制 BF16 grad reduce，这里会 BF16 -> FP32。([GitHub][5])

传输性质：

```text
片内传输：有
  从 DDP grad buffer 的 shard slice -> optimizer main shard 的 grad 字段

片间传输：无
  reduce-scatter 已经在前面完成

默认 dtype：
  输入 shard_model_grad：FP32
  输出 shard_main_param.grad：FP32

如果 --grad-reduce-in-bf16：
  输入 shard_model_grad：BF16
  shard_main_param.grad = shard_model_grad.float() 后：FP32
```

---

# 5. `_copy_group_grads()`

你在 profiler 里看到的 `_copy_group_grads` 实际是 `_copy_model_grads_to_main_grads()` 内部的 nested helper。它按参数组遍历：

```text
model_float16_groups -> shard_fp32_from_float16_groups
model_fp32_groups    -> shard_fp32_groups
```

也就是说，对 bf16 模型参数，它会把模型侧梯度 shard 对应到 **FP32 main param shard** 的 `.grad` 上。Megatron distributed optimizer 初始化时会为 bf16/fp16 model param 创建 `shard_fp32_from_float16_groups`，注释里也写了这是 “fp32 copy of float16 parameters”。([GitHub][5])

传输性质：

```text
片内传输：有
  本 GPU 内 HBM copy / cast

片间传输：无

累加：无
  它不是 reduce，只是切片 + 赋值 / 转 FP32
```

---

# 6. `get_grad_norm()`

`get_grad_norm()` 是计算 global grad norm，用于日志和 clipping。普通 optimizer 里它做：

```python
grads_for_norm = self.get_main_grads_for_grad_norm()
total_norm = get_grad_norm_fp32(
    grads_for_norm,
    grad_stats_parallel_group=self.get_grad_stats_parallel_group()
)
```

所以它本身只是入口，真正计算在 `get_grad_norm_fp32()`。([GitHub][4])

如果你当前是 `ChainedOptimizer`，它会先 `prepare_grads()`，然后调用 `get_grad_norm()`，再按 `clip_grad > 0` 决定是否真正缩放梯度。代码里 `ChainedOptimizer.step()` 是先 `grad_norm = self.get_grad_norm()`，再循环调用 `clip_grad_by_total_norm_fp32()`。([GitHub][4])

传输性质：

```text
片内传输：有，读取各个 grad tensor 计算 norm
片间传输：有，但只是 scalar all-reduce
大 tensor DP 传输：无
```

---

# 7. `get_main_grads_for_grad_norm()`

这个函数负责收集参与 norm 计算的梯度。它会过滤：

```text
1. grad is None 的参数
2. shared parameter，避免重复计数
3. tensor model parallel duplicate，避免 TP 副本重复计数
```

代码里明确检查了 `param_is_not_shared(param)` 和 `tensor_parallel.param_is_not_tensor_parallel_duplicate(...)`。([GitHub][4])

传输性质：

```text
片内传输：基本没有大 copy，主要是 Python 遍历和引用收集
片间传输：无
累加：无
dtype：
  收集到的通常是 main param 的 grad
  默认 distributed optimizer + bf16 下通常是 FP32
```

---

# 8. `get_grad_stats_parallel_group()`

这个函数决定 grad norm / num zeros 这类统计量在哪个进程组上 reduce。

普通 MegatronOptimizer 默认返回 model parallel group：

```python
return parallel_state.get_model_parallel_group()
```

DistributedOptimizer 覆盖了这个函数，注释说明：distributed optimizer 下，gradient statistics 会在 distributed optimizer instance 的所有 rank 上 reduce，而不是只在非 distributed optimizer 场景的 model-parallel ranks 上 reduce。([GitHub][4])

传输性质：

```text
这个函数本身不传输
它只决定后续 get_grad_norm_fp32 的 scalar all-reduce group
```

---

# 9. `get_grad_norm_fp32()`

默认 `norm_type=2`，也就是 global L2 norm。它会先本地计算各个 grad 的 L2 norm，然后做 scalar/tiny tensor all-reduce。

L2 路径大致是：

```python
grad_norm, _ = multi_tensor_applier(l2_norm_impl, ...)
total_norm = grad_norm ** norm_type
torch.distributed.all_reduce(total_norm, op=SUM, group=grad_stats_parallel_group)
total_norm = total_norm ** (1.0 / norm_type)
```

`get_grad_norm_fp32()` 的注释明确写的是 “Calculate the p-norm of gradients in FP32 precision”。([GitHub][6])

传输性质：

```text
片内计算：
  读取本 rank 持有的 grad shard
  计算 sum(g^2) / L2 norm

片间传输：
  FP32 scalar all-reduce SUM
  不是大 tensor 通信

默认 dtype：
  输入 grad：通常 FP32
  本地 norm 统计：FP32
  跨 rank all-reduce tensor：FP32

累加精度：
  本地 multi_tensor_l2norm 按 FP32 norm 语义
  跨 rank all_reduce 是 FP32 SUM
```

注意：这一步和前面的 DP gradient reduce-scatter 不是一回事。
`reduce-scatter` 传的是整个梯度 bucket；`get_grad_norm_fp32` 只传一个或少量 FP32 scalar。

---

# 10. `clip_grad_by_total_norm_fp32()`

这个函数按前面算出的 `grad_norm` 计算 clipping 系数：

```python
clip_coeff = max_norm / (total_norm + 1.0e-6)
```

如果 `clip_coeff < 1.0`，就对所有 grad 原地乘这个系数：

```python
multi_tensor_applier(multi_tensor_scale_impl, ..., [grads, grads], clip_coeff)
```

普通路径里它要求 `param.grad` 是 CUDA FP32 tensor；precision-aware decoupled-grad 路径允许 `decoupled_grad` 是 FP32 或 BF16。([GitHub][6])

传输性质：

```text
片内传输：有
  GPU 本地读取 grad -> 乘 clip_coeff -> 写回 grad

片间传输：无

累加：无
  只是逐元素 scale

默认 dtype：
  grad：FP32
  clip_coeff：Python float 或 FP32 scalar
  输出 grad：FP32
```

所以它不会产生 DP 网络通信，也不会做跨 GPU 累加。

---

# 11. `step_with_ready_grads()`

`step_with_ready_grads()` 先调用内部 optimizer：

```python
self.optimizer.step()
```

然后把更新后的 main params 拷回 model params / param buffer。MixedPrecisionOptimizer 代码里：

```python
self.optimizer.step()
...
self._copy_main_params_to_model_params()
```

DistributedOptimizer 又覆写了 `step_with_ready_grads()`，在 super 之后启动 parameter all-gather。([GitHub][4])

传输性质分两段：

```text
1. inner optimizer.step()
   片内计算，无 DP 片间通信
   默认 main param / grad / Adam state 是 FP32

2. 参数 all-gather
   片间通信
   dtype 是 param buffer dtype，bf16 模型下通常是 BF16
```

默认 bf16 + distributed optimizer 下，optimizer 不是直接更新完整 BF16 model weight，而是更新本 rank 的 FP32 main param shard。Megatron 的 distributed optimizer 初始化中，对 bf16/fp16 参数会创建 `shard_fp32_from_float16_groups`，然后 optimizer param group 使用这些 FP32 shard。([GitHub][5])

---

# 12. `copy_main_params_to_model_params()`

这个函数在 distributed optimizer 下不是简单把 FP32 main param 直接写回每个 model param。它的注释说：下一步会通过 DDP grad buffer 做 all-gather，所以这里负责把更新后的 main shards 写到 grad/param buffer 的正确位置。([GitHub][5])

默认 BF16 路径中，它会调用内部的 `copy_group_params()`：

```python
copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)
```

也就是说：

```text
FP32 main param shard -> BF16 param buffer shard
```

这里会发生 FP32 到 BF16 的 cast，因为目标 `bucket.param_data` 的 dtype 是参数 dtype。([GitHub][5])

传输性质：

```text
片内传输：有
  本 GPU 内 FP32 main shard -> BF16 param buffer shard

片间传输：无
  它只是为后续 all-gather 准备本 rank 的 param shard

累加：无
  只是 copy / cast
```

---

# 13. `copy_group_params()`

这个也是 `copy_main_params_to_model_params()` 内部 helper。它根据 `world_range` 找到当前 main param shard 在 `bucket.param_data` 里的位置，然后：

```python
shard_model_param.data.copy_(shard_main_param)
```

对于普通 BF16 参数，这就是：

```text
source: shard_main_param, FP32
target: bucket.param_data slice, BF16
operation: local copy + downcast
```

代码里 `copy_group_params()` 明确从 `bucket.param_data` 取 slice，然后 `copy_` main shard。([GitHub][5])

传输性质：

```text
片内传输：有
片间传输：无
累加：无
```

---

# 14. 参数 all-gather：optimizer step 后的 DP 片间传输

DistributedOptimizer 的 `step_with_ready_grads()` 在 super 更新完本地 shard 后，会调用：

```python
self.start_param_sync_for_bucket_group_subset()
```

如果没有 overlap param gather，就在 optimizer step 里同步触发 all-gather；如果打开 overlap，会推迟/异步到下一轮 forward pre-hook 附近。([GitHub][5])

DDP bucket 的 `start_param_sync()` 里，标准 distributed optimizer 路径使用：

```python
dist_all_gather_func(bucket.param_data, local_data_view, ...)
```

也就是把每个 DP rank 的 local param shard all-gather 到完整 `bucket.param_data`。([GitHub][2])

传输性质：

```text
片间传输：有，大规模参数 all-gather
传输 dtype：bucket.param_data.dtype

默认 --bf16：
  bucket.param_data dtype = BF16
  all-gather 传输 BF16 参数

累加：无
  all-gather 是拼接/收集，不做 SUM
```

这也是为什么 distributed optimizer 可以做到：

```text
梯度 reduce-scatter：默认 FP32
参数 all-gather：BF16
```

二者不是一个 buffer dtype。

---

# 15. `logical_and_across_model_parallel_group()`

这个函数不是 optimizer 数学更新的一部分，而是 training loop 在 `optimizer.step()` 之后为了同步 `update_successful`。训练代码里：

```python
update_successful = logical_and_across_model_parallel_group(update_successful)
```

目的是当某些 rank 没有 trainable params 或某些 rank update 失败时，在 model-parallel group 内得到一致判断。([GitHub][7])

函数实现是：

```python
input = torch.tensor([0 or 1], dtype=torch.int, device=cuda)
torch.distributed.all_reduce(input, op=MIN, group=model_parallel_group)
```

所以它的语义是 logical AND：只要某个 rank 是 0，MIN 后就是 0。([GitHub][8])

传输性质：

```text
片间传输：有，但只是 int scalar all-reduce
group：model_parallel_group，不是 DP grad group
dtype：int32 / torch.int
reduce op：MIN
累加：无，是 MIN reduce
```

---

# 16. `reduce_max_stat_across_model_parallel_group()`

这个函数也不是 optimizer 数学更新的一部分，而是把 `grad_norm`、`num_zeros_in_grad`、`learning_rate` 这类统计值同步到 model-parallel group 中，方便 logging。training loop 里：

```python
grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)
```

([GitHub][7])

函数实现是：

```python
stat = torch.tensor([stat], dtype=torch.float32, device=cuda)
torch.distributed.all_reduce(stat, op=MAX, group=model_parallel_group)
```

如果某个 rank 没有有效 stat，会用 `-1.0` 作为哨兵。([GitHub][8])

传输性质：

```text
片间传输：有，但只是 FP32 scalar all-reduce
group：model_parallel_group
dtype：FP32
reduce op：MAX
累加：无，是 MAX reduce
```

---

# 17. 汇总表

| profiler 函数                                   | 做什么                                   |                  片内传输 |                片间传输 | 默认 `--bf16` dtype                     | 累加 / reduce 精度               |
| --------------------------------------------- | ------------------------------------- | --------------------: | ------------------: | ------------------------------------- | ---------------------------- |
| backward hook                                 | `param.grad` 加到 `param.main_grad`     |                     有 |        无/可能触发异步通信注册 | `param.grad` BF16，`main_grad` 默认 FP32 | 本地 add 到 FP32 buffer         |
| `start_grad_sync`                             | DP grad reduce-scatter / all-reduce   | 有，bucket scale / view |     有，大 tensor NCCL | 默认 FP32 grad buffer                   | 默认 FP32 SUM/AVG              |
| `prepare_grads`                               | optimizer step 前处理梯度                  |                     有 |                 通常无 | FP32 main grad                        | 无；fp16 scaler 可能有 scalar MAX |
| `_copy_model_grads_to_main_grads`             | DDP grad shard -> optimizer main grad |                     有 |                   无 | 默认 FP32 -> FP32                       | 无                            |
| `_copy_group_grads`                           | 逐 param group 拷 grad shard            |                     有 |                   无 | 默认 FP32                               | 无                            |
| `get_grad_norm`                               | grad norm 入口                          |                     有 |                 间接有 | FP32                                  | 交给 `get_grad_norm_fp32`      |
| `get_main_grads_for_grad_norm`                | 收集参与 norm 的 grads，过滤重复                |                    很小 |                   无 | 默认 FP32                               | 无                            |
| `get_grad_stats_parallel_group`               | 决定 grad norm reduce group             |                     无 |                   无 | 无                                     | 无                            |
| `get_grad_norm_fp32`                          | 计算 global L2 grad norm                |                     有 | 有，scalar all-reduce | FP32 scalar                           | FP32 SUM 后 sqrt              |
| `clip_grad_by_total_norm_fp32`                | 按 global norm 缩放 grad                 |                     有 |                   无 | 默认 FP32 grad                          | 无 reduce，只是 scale            |
| `step_with_ready_grads`                       | inner optimizer step + 后续 param sync  |                     有 |                 间接有 | FP32 main param 更新                    | Adam update 通常 FP32          |
| `copy_main_params_to_model_params`            | main param shard -> param buffer      |                     有 |                   无 | FP32 -> BF16                          | 无                            |
| `copy_group_params`                           | 逐 group 写 param buffer shard          |                     有 |                   无 | FP32 -> BF16                          | 无                            |
| param all-gather                              | 收集各 DP rank 参数 shard                  |                     有 |     有，大 tensor NCCL | BF16 param buffer                     | 无累加，只 gather                 |
| `logical_and_across_model_parallel_group`     | 同步 update_successful                  |                  无/很小 | 有，scalar all-reduce | int                                   | MIN                          |
| `reduce_max_stat_across_model_parallel_group` | 同步 grad_norm 等日志值                     |                  无/很小 | 有，scalar all-reduce | FP32                                  | MAX                          |

---

# 18. 你最关心的精度结论

在你的默认 `--bf16` 训练里，最关键的是这几条：

```text
1. backward 产生的 param.grad 通常是 BF16
2. param.grad 被累加到 param.main_grad
3. param.main_grad 默认是 FP32 DDP grad buffer 的 view
4. DP reduce-scatter 传输 bucket.grad_data，默认 FP32
5. _copy_model_grads_to_main_grads 把 reduce-scatter 后的 grad shard 给 FP32 main param.grad
6. get_grad_norm_fp32 用 FP32 统计 global L2 norm，并做 FP32 scalar all-reduce
7. clip_grad_by_total_norm_fp32 在本 GPU 上缩放 FP32 grad
8. optimizer.step 更新 FP32 main param shard
9. copy_group_params 把 FP32 main param shard cast/copy 到 BF16 param buffer
10. param all-gather 传输 BF16 参数，无累加
```

所以这条路径可以简化成：

```text
梯度方向：
BF16 local grad
  -> FP32 main_grad / grad buffer
  -> FP32 reduce-scatter
  -> FP32 optimizer grad
  -> FP32 grad norm / clipping
  -> FP32 optimizer update

参数方向：
FP32 updated main param shard
  -> BF16 param buffer shard
  -> BF16 all-gather
  -> BF16 model param for next forward
```

---

# 19. 和非 distributed optimizer 的区别

如果不启用 `--use-distributed-optimizer`，就不是这条流程：

```text
gradient DP 通信：all-reduce full grad buffer
optimizer：每个 DP rank 都更新完整参数
param all-gather：通常没有
_copy_group_grads / copy_group_params：不会以 distributed optimizer 的方式出现
```

你 profiler 里出现 `_copy_group_grads`、`copy_group_params`、`params-all-gather`，基本说明你看到的是 distributed optimizer / ZeRO-1 风格的 shard optimizer 流程。

[1]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/distributed/distributed_data_parallel.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/distributed/param_and_grad_buffer.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/optimizer/optimizer.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/optimizer/distrib_optimizer.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/optimizer/clip_grads.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/training/training.py "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/training/utils.py "raw.githubusercontent.com"
