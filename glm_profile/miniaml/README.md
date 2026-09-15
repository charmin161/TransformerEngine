# GLM-5.2-NVFP4 / 4×B200：一次启动的最小 Nsight Systems 分析

编写日期：2026-09-15。目标采集端：nsys 2025.6.3；用户已有的查看端：2026.1.2。

## 这次做什么

沿用已跑通的 vLLM API 模型/后端配置，模型只加载一次。先运行五组短负载的参考计时，再分别采集五个 CUDA 时间线窗口。模型加载与每轮预热不纳入请求采集窗口。

本包不是 OpenCompass 质量评测的替代品。它调用本机已有的 `vllm bench serve`，使用合成文本、`/v1/completions`、固定输出长度；不重现原 chat 模板、thinking 或真实 HumanEval 的专家路由分布。你的 OpenCompass 配置原样保留用于之后的质量回归。

| 用例 | 目标输入 tokens | 输出 tokens | 并发上限 | timing 请求数 | trace 请求数 |
|---|---:|---:|---:|---:|---:|
| P512 | 512 | 1 | 1 | 3 | 1 |
| P8192 | 8192 | 1 | 1 | 3 | 1 |
| D512 | 512 | 64 | 1 | 3 | 1 |
| D8192 | 8192 | 64 | 1 | 3 | 1 |
| B4 | 512 | 64 | 4 | 12 | 4 |

每次 benchmark 在计时/采集之前另执行少量预热。trace 的正式请求合计 8 条，此外还有 timing 与预热请求。3 个请求只够初步观察，不用于可靠 P99 或正式性能回归结论。

## 保留和修改的配置

保留：
- 原模型路径、TP=4、EP、`quantization=modelopt`、`reasoning-parser=glm45`。
- `max-model-len=819200`、现有 CUDA Graph/compile 策略与现有 attention/MoE 后端。
- 原虚拟环境/容器、CUDA_VISIBLE_DEVICES、已有编译缓存与运行环境。

改变：
- 增加 `--profiler-config '{"profiler":"cuda"}'`。
- 设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。
- 显式关闭 prefix caching，防止重复预热命中 KV，使 prefill 被低估。
- 移除 `--disable-log-stats`；支持时关闭逐请求正文日志。
- 不使用 OpenCompass 的 600000 输出上限、n=5、重试或裁判请求。

没有添加 `--enforce-eager`；没有清缓存；没有更换注意力/专家后端；没有重装任何软件。
`max-model-len` 保留原值是为了尽量保持原运行基线，不表示这次会发送 819200 tokens；实际最长测试为约 8192 输入+64 输出。这并不保证降低服务预留的 KV 显存池，启动资源需求仍以原配置和实际 vLLM 行为为准。

## 前提

在 B200 推理服务器、原先能启动 API 的同一环境中执行。可视化服务器不运行这些脚本。
正常停止你自己原来的 API；确认四张 GPU 没有其他任务。不要用通配 `pkill python` 或 GPU reset。不要与 Nsight Compute、另一轮 Nsight Systems 或 DCGM 的 profiling counters 同时采集；如需暂停共享监控，先与管理员协调。

如果在容器中运行，`nsys` 与模型程序需要能在同一运行环境执行。保留原来容器及 GPU 分配，不为此切换驱动/虚拟环境。

先确认 nsys 的 GPU Metrics 编号：

```bash
nsys --version
nsys profile --gpu-metrics-devices=help
```

GPU Metrics 的编号以 nsys 为准，不能在有设备重映射的环境中盲目等同于 CUDA_VISIBLE_DEVICES 编号。本包默认 0,1,2,3；不一致时使用 `METRIC_DEVICES` 指定。

## 最少操作：两个终端

将整个文件夹放到推理服务器有空间的工作目录。两个终端进入相同目录，使用原推理环境。

### 终端 A：预检查，然后只启动一次模型

```bash
bash 01_start_server.sh
```

必要时覆盖 nsys 的 GPU 编号：

```bash
METRIC_DEVICES=0,1,2,3 bash 01_start_server.sh
```

不要同时执行上面两个启动命令。其他可覆盖项：`PYTHON`、`MODEL`、`MODEL_NAME`、`PORT`、`RESULT_ROOT`。RESULT_ROOT 推荐绝对路径；不需要修改模型配置。

脚本在加载 GLM 前会：
1. 检查本机 vLLM/benchmark 帮助，验证当前版本支持所用参数；不自动升级软件。
2. 检查本地端口、结果和临时目录的磁盘空间，记录环境/拓扑。
3. 用小矩阵和 CUDA Graph 做短采集，再导出小 SQLite 确认确实有 kernel 数据。
4. 默认尝试 GPU Metrics。权限/支持/数据检查不通过时退回 CUDA 时间线；时间线检查也失败则不加载模型。

GPU Metrics 策略可选：

```bash
GPU_METRICS=auto bash 01_start_server.sh  # 默认：指标失败则退回时间线
GPU_METRICS=1 bash 01_start_server.sh     # 必须有指标，否则在加载模型前停止
GPU_METRICS=0 bash 01_start_server.sh     # 明确只采时间线
```

每次只执行其中一个启动命令。auto 退回后依然可以定位耗时，但不能凭 GPU-Util 或 kernel 名称确认 compute/memory bound。确认方式：查看结果目录的 `gpu_metrics_mode.txt` 和 `preflight/verified_gpu_metric_rows.txt`，并在模型报告中确认四张卡的指标实际可用。

### 终端 B：执行参考计时和五个短窗口

```bash
python3 run_suite.py
```

脚本读取 `active_run.txt`，等待 `/health`，确认模型别名，再调用启动时记录的 Python 环境内的 vLLM benchmark。无须手动输入模型参数，无须再次加载模型。

各次请求日志写入 `results/.../logs/`。如需观察服务状态：

```bash
RUN=$(cat active_run.txt)
tail -f "$RUN/server.log"
```

单个 benchmark 进程默认总超时 180 秒，包含本地 tokenizer 加载、预热与测试；服务就绪等待单独控制，默认 3600 秒。它们是故障保护阈值，不是性能预期。不自动重试，不在出错后继续加并发。

### 完成并导出

终端 B 显示“全部完成”后，在终端 A 按一次 Ctrl+C 正常结束服务，让 nsys 完成收尾。不要 kill -9，不要立即关闭终端。报告以真正生成的 `.nsys-rep` 文件为准；各次采集编号/文件数量需结合日志核对，不硬编码后缀。

```bash
python3 summarize.py
RUN=$(cat active_run.txt)
find "$RUN/traces" -maxdepth 1 -name '*.nsys-rep' -print
```

在 nsys 完成导出后生成文本摘要：

```bash
RUN=$(cat active_run.txt)
find "$RUN/traces" -maxdepth 1 -name '*.nsys-rep' -print0 |
while IFS= read -r -d '' report; do
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "$report" \
    > "${report%.nsys-rep}.stats.txt"
done
```

`nsys stats` 可能导出 SQLite 并使用额外磁盘；留足空间。不要把四个 rank 的 kernel 总时长直接当请求墙钟延迟。

## 文件说明

```text
active_run.txt
results/<本次实验>/
  environment.json           Python、版本、关键环境变量
  gpu_environment.txt       GPU 与拓扑
  gpu_metrics_mode.txt      实际是否启用 GPU Metrics
  server_command.sh         生效的模型命令（记录文件）
  full_command.sh           nsys + 模型的完整命令（记录文件）
  server.log                模型启动、后端选择、统计、nsys 收尾
  preflight/                CLI 帮助、小矩阵采集与验证
  bench/*.json              每请求长度/延迟和聚合结果
  logs/*.log                每次 vLLM bench 的命令与完整输出
  metrics/*.prom            /metrics 前后快照
  manifest_*.csv            用例顺序、时间、结果位置、是否成功
  traces/*.nsys-rep         真正用于 GUI 的模型报告
  summary.csv               运行 summarize.py 后生成
```

`preflight/smoke*.nsys-rep` 只是工具验证，不是 GLM 数据。
`metrics/*.prom` 是累积指标快照，包含预热等请求，不可直接视为某个纯 GPU 阶段时间。
合成输入 decode/re-encode 可能导致少量 token 偏差；本包记录实际长度，并在明显偏离目标或输出长度不等时停止，不偷偷继续。

## GUI 怎么看

把模型 `.nsys-rep`、manifest 和 summary 复制到 Ubuntu 22.04 可视化服务器。使用 RDP 普通用户，不要 sudo su 后开 GUI：

```bash
/mnt_d/minyusong/nvidia/profiler/host-linux-x64/nsys-ui /实际复制路径/报告.nsys-rep
```

先看 P8192，再看 D8192，最后对比 D512/B4。展开四张 GPU 的 kernel 时间线和 GPU Metrics，不能只看 GPU0。

- P512/P8192 的 output=1 是 prefill 主导，不代表客户端 TTFT 就是纯 GPU prefill。
- D512/D8192 的 trace 包含开头 prefill；在后面重复的稳定 decode 区间查看执行。
- B4 的并发上限4不保证整个窗口每步恰好4条活跃序列，应从稳定区间验证。
- 阶段 busy 比例按每张 GPU kernel 时间区间的并集计算，不是多 stream/multi-rank 时间求和；也不等于 FLOPS 利用率。
- SMs Active、SM Issue、Tensor Active 与 DRAM 采样指标回答的问题不同。SMs Active 高不能单独证明 compute-bound；DRAM 百分比也不直接等于实际 GB/s。
- 高 DRAM 活跃且计算有余量是带宽瓶颈线索；计算流水线接近饱和是计算瓶颈线索。计算与 DRAM 都低且有空洞，需查 CPU 提交、同步、通信或规模不足。最终 kernel 限制可随后用小范围 NCU 确认。

## timing 与 trace 的边界

`timing` 是采集范围未打开时的参考时间，但进程仍由 nsys 启动，可能存在注入开销；不称为完全原生无 profiler 基线。
`trace` 使用 CUDA Graph node tracing，可能显著扰动执行，只用于解释时间线，不当作最终性能成绩。

正式优化验收需要另外运行不挂 nsys 的相同诊断配置。这不是本次必须追加的操作。届时可启动：

```bash
CAPTURE=0 bash 01_start_server.sh
# 另一个终端：
python3 run_suite.py --phase timing
```

native timing 不会清缓存或换后端，仍关闭 prefix cache并保留统计，便于口径对应。

## 出错后的处理

脚本在 trace 异常或超时后尝试 `/stop_profile`，停止后续用例。这个客户端保护不能保证取消卡住的 CUDA kernel 或恢复 GPU/通信死锁。

服务仍能响应时，必要时手动停止采集：

```bash
curl --max-time 60 -X POST http://127.0.0.1:8972/stop_profile
```

随后正常结束你自己的服务。若接口不响应，停止继续发请求，在终端 A Ctrl+C，保留 server.log；不要反复启动第二份模型，不要批量杀其他 Python 进程或重置 GPU。

报参数不支持时，先查 preflight 帮助和实际 vLLM 版本。本包接口按 vLLM 0.26.0 文档核对，不宣称用户当前已安装0.26.0，也不会为了参数兼容自动升级其环境。

只补采某个短窗口无需重新加载模型，前提是原服务和 nsys 还在正常运行：

```bash
python3 run_suite.py --phase trace --case D8192
```

运行 suite 时不要另开 OpenCompass、其他 benchmark 或同时启动第二个 suite。

## 已做的验证

Bash/Python 语法检查、模拟 HTTP/vLLM benchmark 的十组调度、结果汇总、实际长度不符检测、客户端超时、失败后停止后续用例及 emergency stop_profile 已通过。未在真实 B200、用户 vLLM 及 nsys 2025.6.3 上执行；小程序预检查用于在加载 GLM 前尽早暴露兼容问题，不能保证排除所有多卡/驱动问题。

## 主要依据

- NVIDIA Nsight Systems 2025.6 User Guide：capture range repeat、Graph tracing、GPU Metrics 与开销说明。
  https://archive.docs.nvidia.com/nsight-systems/2025.6/UserGuide/index.html
- vLLM 0.26.0 Profiling：spawn、CUDA profiler、API start/stop、bench --profile。
  https://docs.vllm.ai/en/v0.26.0/contributing/profiling/
- vLLM 0.26.0 bench serve CLI。
  https://docs.vllm.ai/en/v0.26.0/cli/bench/serve/
- vLLM benchmark 源码：warmup 位于 start_profile 之前、请求结果字段。
  https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/benchmarks/serve.py
- vLLM Production Metrics。
  https://docs.vllm.ai/en/v0.26.0/usage/metrics/
- NVIDIA Nsight Compute Triage。
  https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html
