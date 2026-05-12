可以配置，但我先给结论：**技术上能做；但结合你前面的 `illegal memory access` 现象，我不建议把 Megatron MoE 训练和外部 `sm_active_all_gpus_limited` 长期放在同一个 MPS server 下混跑。** MPS 不是隔离机制，某个 client 异常退出或触发 GPU fault 时，可能污染同一 MPS server 下的其他 client。你要继续实验，就必须做到：每台机器 MPS 环境完全一致、filler 不要无设限、filler 必须优雅退出。

下面给一套可执行配置。

---

## 1. 总体原则

你的集群是：

```text
llm_11 slots=4
llm_12 slots=4
llm_13 slots=4
llm_14 slots=4
llm_15 slots=4
llm_16 slots=4
llm_17 slots=4
llm_18 slots=4
```

所以应该是：

```text
每台机器启动 1 个 MPS control daemon
每台机器的 MPS daemon 覆盖本机 0,1,2,3 四张卡
DeepSpeed worker 继承同一组 CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY
每台机器再单独启动 1 个 sm_active_all_gpus_limited，覆盖本机 4 张卡
```

NVIDIA 文档明确说，每个 node 只应该运行一个 `nvidia-cuda-mps-control` daemon；DeepSpeed 的 `.deepspeed_env` 则可以把环境变量传播到每个 worker 进程。([NVIDIA Docs][1])

---

## 2. `.deepspeed_env` 建议这样改

你现在的 `.deepspeed_env` 是：

```bash
PATH=/cpfs01/liuwengang/Pai-Megatron-Patch-main/venv_pai/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LD_LIBRARY_PATH=/cpfs01/liuwengang/Pai-Megatron-Patch-main/venv_pai/lib:/usr/local/cuda-13.0/lib64
GLOO_SOCKET_IFNAME=eth0
TP_SOCKET_IFNAME=eth0
NCCL_SOCKET_IFNAME=bond0,bond1,bond2,bond3
MASTER_ADDR=10.207.212.186
MASTER_PORT=6001
CUDA_LAUNCH_BLOCKING=0
LD_PRELOAD=/usr/local/cuda-13.0/lib64/libcudart.so.13
```

建议加上 MPS pipe/log：

```bash
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${USER}
CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${USER}
```

完整变成：

```bash
PATH=/cpfs01/liuwengang/Pai-Megatron-Patch-main/venv_pai/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LD_LIBRARY_PATH=/cpfs01/liuwengang/Pai-Megatron-Patch-main/venv_pai/lib:/usr/local/cuda-13.0/lib64
GLOO_SOCKET_IFNAME=eth0
TP_SOCKET_IFNAME=eth0
NCCL_SOCKET_IFNAME=bond0,bond1,bond2,bond3
MASTER_ADDR=10.207.212.186
MASTER_PORT=6001
CUDA_LAUNCH_BLOCKING=0
LD_PRELOAD=/usr/local/cuda-13.0/lib64/libcudart.so.13
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${USER}
CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${USER}
```

**不要把这个放进去：**

```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=...
```

否则 Megatron 训练本身也会被限流。这个变量只应该给 filler 辅助程序单独设置。

---

## 3. 每台机器启动 MPS 的脚本

新建：

```bash
start_mps_all_nodes.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

HOSTS=(
	llm_11
	llm_12
	llm_13
	llm_14
	llm_15
	llm_16
	llm_17
	llm_18
)

for HOST in "${HOSTS[@]}"; do
	echo "Starting MPS on ${HOST}..."
	ssh "${HOST}" "bash -lc '
		set -e

		export CUDA_VISIBLE_DEVICES=0,1,2,3
		export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${USER}
		export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${USER}

		pkill -TERM -f nvidia-cuda-mps-server || true
		pkill -TERM -f nvidia-cuda-mps-control || true
		sleep 2
		pkill -KILL -f nvidia-cuda-mps-server || true
		pkill -KILL -f nvidia-cuda-mps-control || true

		rm -rf \${CUDA_MPS_PIPE_DIRECTORY} \${CUDA_MPS_LOG_DIRECTORY}
		mkdir -p \${CUDA_MPS_PIPE_DIRECTORY} \${CUDA_MPS_LOG_DIRECTORY}

		nvidia-cuda-mps-control -d

		sleep 1

		echo \"MPS processes on ${HOST}:\"
		ps -ef | grep nvidia-cuda-mps | grep -v grep || true
	'"
done
```

执行：

```bash
chmod +x start_mps_all_nodes.sh
./start_mps_all_nodes.sh
```

注意这里启动 MPS daemon 时设置了：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

那么 DeepSpeed worker 侧也必须使用同样的 GPU 可见集合。NVIDIA 文档提醒，MPS daemon 设置了 `CUDA_VISIBLE_DEVICES` 时，client 侧设备 ordinal 可能被 remap，因此不要让 daemon 和 client 使用不一致的 `CUDA_VISIBLE_DEVICES`。([NVIDIA Docs][1])

---

## 4. 启动 Megatron-LM 训练

你的训练命令基本可以保持：

```bash
run_cmd="deepspeed --hostfile hostfile_11-18 pretrain_gpt.py \
 ${megatron_options} ${dataset_options} ${pr_options} ${load_option} ${activation_checkpoint_options} \
 ${do_option} ${sp_option} ${moe_options} ${offload_option} ${vp_option} ${comm_overlap_option} ${packing_options} ${uneven_split_option} ${attn_backend_option}
 | tee -a ${SAVED_LOG_PATH}/output_qwen3_moe.log"
```

但我建议在启动前显式保证当前 shell 的 MPS env 和 `.deepspeed_env` 一致：

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${USER}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${USER}
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

然后：

```bash
eval "$run_cmd"
```

如果 DeepSpeed 自己在每个 node 上设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`，没问题；关键是不要出现：

```text
MPS daemon: CUDA_VISIBLE_DEVICES=0,1,2,3
DeepSpeed worker: CUDA_VISIBLE_DEVICES=0 或 1 或 2 或 3
filler: CUDA_VISIBLE_DEVICES=0,1,2,3
```

这种不一致会非常容易引入 MPS ordinal remap 混乱。

---

## 5. 每台机器启动 `sm_active_all_gpus_limited`

新建：

```bash
start_filler_all_nodes.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

HOSTS=(
	llm_11
	llm_12
	llm_13
	llm_14
	llm_15
	llm_16
	llm_17
	llm_18
)

FILLER_BIN="/path/to/sm_active_all_gpus_limited"
FILLER_LOG_DIR="/tmp/sm_active_filler_logs_${USER}"

for HOST in "${HOSTS[@]}"; do
	echo "Starting filler on ${HOST}..."
	ssh "${HOST}" "bash -lc '
		set -e

		export CUDA_VISIBLE_DEVICES=0,1,2,3
		export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${USER}
		export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${USER}

		mkdir -p ${FILLER_LOG_DIR}

		nohup env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=2 \
			${FILLER_BIN} 360000 100 1000 1000 4 \
			> ${FILLER_LOG_DIR}/filler.log 2>&1 &

		echo \$! > ${FILLER_LOG_DIR}/filler.pid
		echo \"filler pid: \$(cat ${FILLER_LOG_DIR}/filler.pid)\"
	'"
done
```

执行：

```bash
chmod +x start_filler_all_nodes.sh
./start_filler_all_nodes.sh
```

我这里建议先从非常低的 MPS 限额开始：

```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=2
```

而不是 10、20。你之前已经遇到 `illegal memory access`，所以不要一开始就激进。

参数：

```bash
sm_active_all_gpus_limited 360000 100 1000 1000 4
```

含义：

```text
360000  运行很久
100     filler 内部目标 100% SM
1000    每个 kernel busy 1000 ms
1000    period 1000 ms
4       blocksPerSm = 4
```

真正限流交给：

```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=2
```

---

## 6. 停止 filler：不要直接 Ctrl+C

你前面已经验证过：**MPS 环境里直接 Ctrl+C 杀辅助程序，可能让 Megatron 训练跟着报 `illegal memory access`**。

先给远程停止脚本：

```bash
stop_filler_all_nodes.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

HOSTS=(
	llm_11
	llm_12
	llm_13
	llm_14
	llm_15
	llm_16
	llm_17
	llm_18
)

FILLER_LOG_DIR="/tmp/sm_active_filler_logs_${USER}"

for HOST in "${HOSTS[@]}"; do
	echo "Stopping filler on ${HOST}..."
	ssh "${HOST}" "bash -lc '
		set +e

		if [ -f ${FILLER_LOG_DIR}/filler.pid ]; then
			FILLER_PID=\$(cat ${FILLER_LOG_DIR}/filler.pid)
			kill -TERM \${FILLER_PID}
			sleep 3
			kill -KILL \${FILLER_PID} 2>/dev/null || true
			rm -f ${FILLER_LOG_DIR}/filler.pid
		else
			pkill -TERM -f sm_active_all_gpus_limited || true
			sleep 3
			pkill -KILL -f sm_active_all_gpus_limited || true
		fi
	'"
done
```

但更关键的是：你的 `sm_active_all_gpus_limited` 程序最好加 signal handler，`SIGTERM/SIGINT` 时等当前 kernel 完成、`cudaDeviceSynchronize()` 后再退出。否则 MPS server 仍然可能被未完成的 GPU work 污染。

---

## 7. 停止 MPS

训练和 filler 都停掉后再停 MPS：

```bash
stop_mps_all_nodes.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

HOSTS=(
	llm_11
	llm_12
	llm_13
	llm_14
	llm_15
	llm_16
	llm_17
	llm_18
)

for HOST in "${HOSTS[@]}"; do
	echo "Stopping MPS on ${HOST}..."
	ssh "${HOST}" "bash -lc '
		set +e

		export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${USER}
		export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${USER}

		timeout 5s bash -lc \"echo quit | nvidia-cuda-mps-control\" || true

		pkill -TERM -f nvidia-cuda-mps-server || true
		pkill -TERM -f nvidia-cuda-mps-control || true
		sleep 2
		pkill -KILL -f nvidia-cuda-mps-server || true
		pkill -KILL -f nvidia-cuda-mps-control || true

		rm -rf \${CUDA_MPS_PIPE_DIRECTORY} \${CUDA_MPS_LOG_DIRECTORY}

		ps -ef | grep nvidia-cuda-mps | grep -v grep || true
	'"
done
```

---

## 8. 建议的完整执行顺序

```bash
# 1. 所有节点启动 MPS
./start_mps_all_nodes.sh

# 2. 确认 .deepspeed_env 包含 MPS pipe/log，但不包含 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
cat .deepspeed_env

# 3. 启动 Megatron 训练
eval "$run_cmd"

# 4. 另开一个终端，等训练稳定若干 step 后，再启动 filler
./start_filler_all_nodes.sh

# 5. 观察 step time / tokens/s / SM Active
# 如果训练异常，立即停止 filler
./stop_filler_all_nodes.sh

# 6. 训练结束后再停止 MPS
./stop_mps_all_nodes.sh
```

---

## 9. 强烈建议你加一个优雅退出版 filler

你的当前外部 filler 最大问题不是参数，而是**被 Ctrl+C/kill 时可能带崩 MPS server**。

至少加：

```cpp
#include <atomic>
#include <csignal>

std::atomic<bool> shouldStop(false);

void HandleSignal(int signal)
{
	shouldStop.store(true);
}
```

`main()` 里：

```cpp
std::signal(SIGINT, HandleSignal);
std::signal(SIGTERM, HandleSignal);
```

循环条件改成：

```cpp
while (!shouldStop.load() && std::chrono::steady_clock::now() < endTime)
```

退出前每张卡执行：

```cpp
cudaDeviceSynchronize();
cudaFree(deviceBlockResults);
cudaDeviceReset();
```

否则你用 MPS 混跑时，停止 filler 就有概率污染同一 MPS server 下的 Megatron/NCCL client。

---

## 10. 我对这套方案的风险判断

可以配置成上面这样，但我仍然建议你把它当成**实验方案**，不是生产训练方案。

特别是你已经出现过：

```text
MPS + Megatron MoE + filler -> illegal memory access
Ctrl+C filler -> Megatron 跟着挂
```

所以最保守的参数应该是：

```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=1
sm_active_all_gpus_limited 360000 100 1000 1000 2
```

然后逐步到：

```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=2
sm_active_all_gpus_limited 360000 100 1000 1000 4
```

不要一开始就：

```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10
```

一句话：**多机情况下，每个节点启动一套 MPS daemon；`.deepspeed_env` 只传播 MPS pipe/log；filler 每节点单独启动，并只给 filler 设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`；不要把限流变量传给 Megatron；不要直接 Ctrl+C 杀 filler。**

[1]: https://docs.nvidia.com/deploy/mps/appendix-common-tasks.html?utm_source=chatgpt.com "Appendix: Common Tasks — Multi-Process Service"
