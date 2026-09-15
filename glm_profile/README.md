# GLM-5.2-NVFP4 / 4×B200 / TP4：最小诊断脚本

适用接口按 vLLM 0.26.0 文档核对，先核实你实际的版本。没有安装器，没有自动升级/部署，没有猜测的完整模型配置。脚本已做 Shell/Python 语法检查，**未在你的 B200 上运行验证**。

目标：保留已跑通的 HumanEval 环境；先取得不挂 profiler 的计时，再抓预热后的短时间线。随机负载不是 HumanEval 质量评测，也不保证复现真实 MoE 路由。

## 0. 两台机器分别做什么

- 推理机：现有 vLLM/PyTorch/OpenCompass 不动；运行 Nsight Systems CLI（nsys）；需要热点硬件计数器时再用 Nsight Compute CLI（ncu）。
- 本地可视化机：Nsight Systems Full Version（nsys-ui）查看 `.nsys-rep`，无需在这台机器安装模型或 vLLM。
- 当前 Nsight Systems 2026.5.1 的 Ubuntu 官方支持列表是 22.04 / 24.04 / 26.04，不包含 20.04。不要把“可能启动”当成官方支持。最快的隔离方案是另一台受支持机器或 Ubuntu 24.04 桌面虚拟机；不必先对现有机器原地升级，更不要替换系统 glibc。
- GUI 版本至少不低于采集版本；同版本最简单。安装包按每台机器各自的 CPU 架构选 x86_64 或 ARM server / SBSA，不能因为两台机器都跑 Linux 就混用。
- 推理机离线时，在联网机器下载匹配版本和架构的官方独立安装包及依赖再转移；不要为了安装 profiler 执行 CUDA/驱动全家桶升级。

## 1. 清点环境

进入你跑通 HumanEval 的同一个 venv / 容器，再运行：

```bash
bash env_probe.sh logs/env.txt
```

已有 nsys/ncu 可能不在 PATH。脚本会检查常见安装目录。先核对驱动、工具、CPU 架构和 OS；选择支持 B200 且适配当前驱动的版本，不要求追最新版本。

Ubuntu/Debian 的独立 deb 安装示例（替换为真实绝对路径）：

```bash
NSYS_DEB=/absolute/path/to/downloaded-nsight-package.deb
dpkg-deb -f "$NSYS_DEB" Package Version Architecture Depends
sudo apt install --simulate "$NSYS_DEB"
# 确认不会升级/替换驱动与现有 CUDA 后再安装；离线需预先准备依赖。
sudo apt install "$NSYS_DEB"
nsys --version
nsys status -e
```

推理机选 CLI Only，本地选 Full Version。不是 Debian/Ubuntu 则用对应 RPM 或官方 .run；`.run` 可通过 `sh /真实路径/安装器.run` 打开交互安装。工具不在 PATH 时，加入安装目录的 `bin`，不是重新安装 vLLM。

## 2. 先验证小程序，不加载 GLM

在空闲且属于你的 GPU 上，使用原推理环境：

```bash
mkdir -p traces
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o traces/smoke python cuda_smoke.py
nsys stats traces/smoke.nsys-rep
```

报告中应有矩阵乘 CUDA kernel；复制这份小报告到本地并用 nsys-ui 打开，先验证完整采集—传输—查看链路。

硬件计数器权限和 CUDA 时间线权限不是同一件事：

```bash
nsys profile --gpu-metrics-devices=help
nsys profile --gpu-metrics-devices=all --gpu-metrics-set=help
```

确认权限和支持后，再给采集命令添加 `--gpu-metrics-devices=0,1,2,3 --gpu-metrics-frequency=10000`。设备编号以 nsys 枚举为准，只采你获分配的 GPU。CPU sampling 检查失败不必然意味着 CUDA tracing 失败；本例已关闭 CPU sampling/context switch。

遇到 ERR_NVGPUCTRPERM 或类似权限错误，请管理员按服务器策略授权。不要在运行任务的机器上卸载驱动模块、重启或盲目切换驱动；容器内 root 也不一定解决宿主机计数器权限。已有 DCGM profiling 可能占用计数器，需要管理员协调，不必为本任务新装 DCGM。

## 3. 先判定现有 OpenCompass 的调用方式

**A：原本启动 vLLM 服务，OpenCompass 通过 API 请求。** 使用下文 A。

**B：OpenCompass 配置使用 VLLM / VLLMwithChatTemplate 等直接创建 vllm.LLM。** 使用下文 B，不必为了 profiling 先改成服务架构。

只看到“OpenCompass 跑通”不能证明已有 HTTP 服务。端口不通也不能单独证明是离线模式，需结合原配置确认。

## A. 已有 API 服务

### A1. 不挂 profiler 的四个用例

保留原始启动命令和 HumanEval 配置。创建一份诊断启动配置，除了必要的缓存控制不要修改 TP、KV dtype、attention/MoE 后端、CUDA Graph、MTP 或 token budget。

固定长度负载会重复预热/请求；为避免 prefix KV 命中，诊断配置显式关闭 prefix cache：`--no-enable-prefix-caching`，并记录这一差异，不能把它称作原始配置。不要同时保留相反的启用参数。启动基线和采集版本时使用一致的 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（或保留原先明确配置的值）。

在推理机上运行客户端，避免额外的远程网络变量：

```bash
# 按实际模型路径、端口、/v1/models 返回的模型名修改。
export MODEL_DIR=/wireless/public/models/GLM-5.2-NVFP4
export BASE_URL=http://127.0.0.1:8972
export SERVED_MODEL=GLM5.2-NVFP4
curl -fsS "$BASE_URL/v1/models"

bash bench_case.sh 512 1 1 20 p512
bash bench_case.sh 8192 1 1 20 p8k
bash bench_case.sh 512 256 1 20 d1
bash bench_case.sh 512 256 8 32 d8
```

参数依次为：输入 token 数、输出 token 数、并发、请求数、结果标签。JSON 保存到 bench/，日志到 logs/。脚本固定长度（v0.26 的 random-range-ratio=0）、temperature=0、ignore-eos；ignore-eos 只用于固定性能负载，不能用于正常 HumanEval 质量评测。20/32 请求用于初诊，不足以可靠估计 P99。

并发 1 输出 1 是 prefill 主导的端到端首 token 测试，**不是纯 GPU prefill 时间**。并发 8 不代表任意一个调度 step 都正好有 8 条序列。

### A2. Nsight Systems 采集

停止你自己的基线服务；不要在相同四张卡上同时启动两个 GLM 副本。

把刚才的诊断启动命令复制到 `start_glm_profile.sh`，在真实 `vllm serve ...` 命令中添加：

```bash
--profiler-config '{"profiler":"cuda"}'
```

脚本必须前台运行，不要 `nohup`、`&`；不要只给 OpenCompass/API 客户端套 nsys。

终端 A：

```bash
bash capture_server.sh traces/glm52 bash ./start_glm_profile.sh
# 已验证 GPU counters 权限后，可改为：
# GPU_METRICS=1 GPU_METRICS_DEVICES=0,1,2,3 \
#   bash capture_server.sh traces/glm52 bash ./start_glm_profile.sh
```

终端 B：重新设置 A1 的三个环境变量，服务就绪后：

```bash
bash bench_case.sh 8192 1 1 1 trace_prefill --profile
bash bench_case.sh 512 64 1 1 trace_decode --profile
```

`--profile` 控制采集区间；加载、初始化和脚本预热不应混入目标窗口。最后等请求与 stop-profile 调用结束，再在终端 A 用 Ctrl+C 正常结束服务并写出报告，不要 kill -9。文件名可能自动带后缀：

```bash
find traces -maxdepth 1 -name '*.nsys-rep'
nsys stats /actual/path/to/report.nsys-rep
```

采集时的 benchmark JSON 只能用于诊断，不能混入正常性能成绩。

手动采集某条真实 HumanEval 请求的方式：POST `/start_profile` → 通过原入口发送完整渲染后的请求并等待生成结束 → POST `/stop_profile`。保留原 chat template、stop、sampling 参数；不要把它和固定长度随机负载的成绩混为一谈。

## B. 原本直接使用 vllm.LLM（离线）

无需先新建 HTTP 服务。将原 OpenCompass 配置里**实际传给 LLM 的全部引擎 kwargs**保存为 `engine_args.json`：包含 `model`、`tensor_parallel_size=4`、以及实际生效的 tokenizer/量化/KV/backend/编译等非默认参数。不能将整份 OpenCompass 字典原样复制：`type`、`abbr`、`run_cfg`、评测配置等不是 LLM kwargs。这个包不提供猜测的完整官方配置。

`offline_probe.py` 调用的是 vLLM 的 LLM / SamplingParams / start_profile / stop_profile；不引入新的测评框架。默认随机 token 诊断，不是 HumanEval。它一次加载模型后可跑多个用例。计时是完整阻塞 `LLM.generate` 的墙钟时间，不提供客户端流式 TTFT/ITL。

```bash
# 确保四张卡上没有另一个 GLM 服务副本。
python offline_probe.py --engine-config engine_args.json \
  --disable-prefix-cache \
  --case 512:1:1 --case 8192:1:1 \
  --case 512:256:1 --case 512:256:8 \
  --repeats 3 --result bench/offline_baseline.json
```

单次参数为 INPUT:OUTPUT:BATCH。这里 batch 是一次提交的请求数，不保证每一步的活跃序列数相同。脚本把实际使用的引擎 kwargs 和显式关闭 prefix cache 的改动保存在 JSON 中。结果已存在时会拒绝覆盖。

基线结束释放显存，再采两段短 trace：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  -o traces/offline_glm52 \
  python offline_probe.py --engine-config engine_args.json \
    --disable-prefix-cache --profile --repeats 1 \
    --case 8192:1:1 --case 512:64:1 \
    --result bench/offline_profiled.json
```

已验证权限后，同样可给 nsys 添加 GPU metrics 选项。脚本加载和预热在 start_profile 之前；实际 case 会用 NVTX 标签区分。

vLLM 自带 `vllm bench latency` 也可做离线基线，但必须把原有引擎配置逐项带入。它支持 `--input-len`、`--output-len`、`--batch-size`、`--num-iters-warmup`、`--num-iters`、`--output-json`。不要仅复制 model + TP4 就当成复现了官方配置。

## 4. 查看与解释

本地用 scp/现有传输工具取回实际 `.nsys-rep`，运行：

```bash
nsys-ui /actual/local/path/report.nsys-rep
```

优先看四张 GPU 的 CUDA kernels、CUDA API、GPU Metrics（有权限时）、进程/线程；先看单请求 prefill，再看连续 decode 的重复波形。选择明确窗口，排除加载、预热、客户端空闲和启动/结束边界。四张卡都要查看，并检查报告的丢事件/采集警告。

- GPU busy：该 GPU 在窗口中执行至少一个 kernel 的时间占比；按区间并集算，不要把多 stream 或四卡时间直接相加。
- nvidia-smi GPU-Util：采样期间的忙碌比例，不是 Tensor Core/FLOPS 利用率。
- memory.used：显存容量；nvidia-smi memory utilization 也不等于 HBM 已用带宽比例。
- nsys GPU Metrics：SM/Tensor 活跃度、DRAM 等采样指标；属于设备级，避免其他进程污染。SM/Tensor 活跃百分比也不能直接当 MFU。
- CUDA memcpy 轨道不代表所有显存访问；GEMM/attention 内部读写 HBM 不会都显示成 memcpy。
- 算力和 DRAM 都低时，可能是 CPU、launch、同步、通信、小规模 kernel 或访存延迟，不能强行归为 compute-bound / memory-bound。
- NCU 的 Memory Throughput 概览可能反映 L1/L2/DRAM 中某个瓶颈，需展开看 DRAM，不能把该概览百分比当 HBM 带宽。

统计表的 kernel 总时长/百分比不等于端到端关键路径占比，多 GPU/多 stream 可能重叠。用时间线解释端到端耗时。

## 5. Nsight Compute 只验证计数器，再聚焦真实热点

安装支持 B200 且兼容当前驱动的独立 Nsight Compute；首日可不安装 GUI。首次只跑小矩阵，检查权限/兼容性：

```bash
ncu --profile-from-start off --launch-count 1 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  -o traces/ncu_smoke python cuda_smoke.py
```

这个结果只说明工具可用，不能代表 GLM 的瓶颈。模型级证据应来自前面的 nsys 轨迹和 GPU metrics；需要严格确认时，抽出实际热点 kernel 的真实形状、布局、精度与路由，再用 ncu。

不要直接 `ncu --set full` 包住整个 TP4 服务/全量 HumanEval。NCU replay/串行化会改变执行，涉及必须并行的通信 kernel 时甚至可能卡住。没有硬件计数器证据时，请写“初步怀疑/待确认”，不要宣布已证明 compute/memory bound。

## 6. 两天交付物

第一天：env.txt、固定的原始/诊断启动配置、可打开的 smoke 报告、四组未挂 profiler 的结果。

第二天：两到三份短 trace、各 GPU 的时间线观察、按阶段的 Top-5 热点、硬件计数器证据（若权限允许）、初步瓶颈结论。

建议结果列：场景 / 输入输出长度 / 并发或 batch / 客户端 E2E 与 TTFT（在线）或 generate latency（离线）/ TPOT（在线输出>1）/ 每 GPU 活跃度 / DRAM 指标 / Top-5 kernel / 结论与置信程度。

保留现有 HumanEval 分数；本阶段不改变模型数值路径，不扩展质量测评集。

## 官方资料（2026-09-11 核对）

- Nsight Systems 下载、平台与 GUI 版本兼容：https://developer.nvidia.com/nsight-systems/get-started
- Nsight Systems 安装：https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
- Nsight Systems CLI / GPU Metrics：https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- vLLM 0.26 profiling：https://docs.vllm.ai/en/v0.26.0/contributing/profiling/
- vLLM 0.26 bench serve：https://docs.vllm.ai/en/v0.26.0/cli/bench/serve/
- vLLM 0.26 bench latency：https://docs.vllm.ai/en/v0.26.0/cli/bench/latency/
- vLLM 0.26 LLM API：https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/llm/
- Nsight Compute 下载：https://developer.nvidia.com/tools-overview/nsight-compute/get-started
- Nsight Compute 分析指南：https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html
- Nsight Compute CLI：https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- 计数器权限：https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
- nvidia-smi 字段语义：https://docs.nvidia.com/deploy/nvidia-smi/index.html
