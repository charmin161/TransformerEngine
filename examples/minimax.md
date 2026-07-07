可以这样判断：

**单节点 4×B200：先试 `--tensor-parallel-size 4`。**
模型卡写 `--tensor-parallel-size 8` 是 NVIDIA 给出的示例/测试命令，不代表必须 8 卡；vLLM 官方并行建议是：如果模型能放进单节点多卡，就把 `tensor_parallel_size` 设成该节点 GPU 数，比如 4 卡设 TP=4。([Hugging Face][1]) ([vLLM][2])

但 MiniMax-M3 是 428B/A23B MoE，且原生 1M context。**4×B200 是否能跑，主要取决于你设置的 `--max-model-len` 和 KV cache 余量**，不是只看权重能否放下。vLLM recipe 明确提示：完整 1M context 需要很大的 KV cache；可以用 `--max-model-len` 限制上下文，或者用多节点扩展。([vLLM Recipes][3])

---

## 1. 单节点 4×B200 推荐命令

先用 text-only、小上下文、eager/profiler 友好的方式跑通：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve nvidia/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 32768 \
  --language-model-only \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
```

这里几个点：

`--tensor-parallel-size 4`：单节点 4 卡 TP。MiniMax-M3 的 attention heads=64、KV heads=4，所以 TP=4 是结构上合理的；TP=3 这类就不合适。

`--block-size 128`：必须保留。MiniMax-M3 的 MSA sparse/index cache 要求 vLLM KV block size 对齐 128，recipe 也明确说这是 mandatory。([vLLM Recipes][3])

`--max-model-len 32768`：先别直接 1M。你只是抓前向激活，不需要一开始吃满 KV cache。确认显存后再改成 131072、262144、1048576。

`--language-model-only`：只做文本抓数时建议加，跳过 vision encoder，释放显存；vLLM recipe 也建议 text-only workload 用它。([vLLM Recipes][3])

如果这条 OOM，再试：

```bash
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192
```

---

## 2. 两节点 × 每节点 4×B200 怎么跑

你有两种方式：

### 方案 A：推荐，TP=4 + PP=2

也就是：**每个节点内部做 4 路 TP，两个节点之间做 2 路 pipeline parallel**。

vLLM 官方多节点建议就是：`tensor_parallel_size = 每节点 GPU 数`，`pipeline_parallel_size = 节点数`。例如 2 节点、每节点 8 卡时用 TP=8、PP=2；你的情况对应 TP=4、PP=2。([vLLM][2])

**Node 0/head：**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=eth0        # 按你的 IB/RoCE 网卡改，比如 ib0/bond0
export GLOO_SOCKET_IFNAME=eth0

vllm serve nvidia/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <NODE0_IP> \
  --block-size 128 \
  --max-model-len 131072 \
  --language-model-only \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
```

**Node 1/worker：**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

vllm serve nvidia/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr <NODE0_IP> \
  --headless \
  --block-size 128 \
  --max-model-len 131072 \
  --language-model-only \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
```

这是 vLLM multiprocessing 多节点模式，官方文档也给了 `--nnodes`、`--node-rank`、`--master-addr`、worker 节点 `--headless` 的模式。([vLLM][2])

**为什么更推荐 TP=4+PP=2，而不是跨节点 TP=8？**

因为跨节点 TP=8 会让 attention/MLP/MoE 的 TP 通信跨机器，all-reduce 压力更大。TP=4+PP=2 通常把高频张量并行通信限制在节点内，跨节点主要是层间 pipeline activation 传输，调试和稳定性更好。

---

### 方案 B：TP=8 跨两个节点

这个方式更贴近模型卡的 TP=8，但通信更重：

```bash
vllm serve nvidia/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray \
  --block-size 128 \
  --max-model-len 131072 \
  --language-model-only \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
```

这种要先起 Ray cluster。vLLM 文档说多节点默认/常用是 Ray，所有节点要环境一致、模型路径一致、Python 包一致。([vLLM][2])

我建议你调试抓数阶段优先用 **方案 A：TP=4+PP=2**。

---

## 3. Torch profiler 需要加哪些参数

当前 vLLM 推荐用 `--profiler-config`，设置 `profiler=torch` 和 `torch_profiler_dir`。官方说明支持 `torch_profiler_record_shapes`、`torch_profiler_with_memory`、`torch_profiler_with_stack`、`torch_profiler_with_flops`、`torch_profiler_use_gzip` 等参数。([vLLM][4])

### 单节点 profiler 命令

```bash
mkdir -p /tmp/vllm_profile_minimax_m3

vllm serve nvidia/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --enforce-eager \
  -cc.mode=0 \
  -cc.cudagraph_mode=NONE \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "/tmp/vllm_profile_minimax_m3",
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_memory": true,
    "torch_profiler_with_stack": true,
    "torch_profiler_with_flops": false,
    "torch_profiler_use_gzip": true,
    "wait_iterations": 0,
    "warmup_iterations": 1,
    "active_iterations": 2,
    "max_iterations": 4,
    "ignore_frontend": true
  }' \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
```

然后用 API 控制 profiler：

```bash
curl -X POST http://localhost:8000/start_profile
```

发一两个请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/MiniMax-M3-NVFP4",
    "messages": [
      {"role": "user", "content": "用一句话解释 MoE routing。"}
    ],
    "max_tokens": 32
  }'
```

停止并 flush：

```bash
curl -X POST http://localhost:8000/stop_profile
```

vLLM 官方示例也是先 `/start_profile`，再请求，再 `/stop_profile`；trace 可以用 Perfetto 打开。([vLLM][4])

---

## 4. 这些 profiler/debug 参数的作用

| 参数                                              | 作用                            | 建议                        |
| ----------------------------------------------- | ----------------------------- | ------------------------- |
| `--profiler-config '{"profiler":"torch", ...}'` | 启用 PyTorch profiler           | 必加                        |
| `torch_profiler_dir`                            | trace 输出目录                    | 必须是绝对路径                   |
| `torch_profiler_record_shapes=true`             | 记录 tensor shape               | 抓矩阵乘形状时建议开                |
| `torch_profiler_with_memory=true`               | 记录显存分配                        | 调 OOM/activation 时建议开     |
| `torch_profiler_with_stack=true`                | 记录 Python stack               | 定位 vLLM 函数调用时建议开          |
| `torch_profiler_with_flops=false`               | FLOPs 统计                      | 先关，开了更慢                   |
| `warmup_iterations=1`                           | 预热但丢弃                         | 减少冷启动噪声                   |
| `active_iterations=2`                           | 真正记录几个 engine iteration       | 抓数时小一点                    |
| `max_iterations=4`                              | 限制 profiler 范围                | 防止 trace 巨大               |
| `ignore_frontend=true`                          | 不抓前端 AsyncLLM 全流程             | 更聚焦 worker/model forward  |
| `--enforce-eager`                               | 关闭 vLLM compile 和 CUDA Graphs | 抓 Python hook/forward 更稳定 |
| `-cc.mode=0`                                    | 关闭 torch.compile              | 保守加上                      |
| `-cc.cudagraph_mode=NONE`                       | 关闭 CUDA Graph                 | 保守加上                      |
| `--max-num-seqs 1`                              | 单请求                           | 降低 trace 复杂度              |
| `--max-num-batched-tokens 8192`                 | 限制 batch token                | 控制 profiler 文件大小          |
| `--max-model-len 8192`                          | 降低 KV cache                   | 抓结构时够用                    |

vLLM 文档明确说 `--enforce-eager` 会关闭 vLLM 的 torch.compile 集成，并包括关闭 CUDAGraphs；如果只想关 CUDAGraph，可用 `-cc.cudagraph_mode=NONE`。([vLLM][5])

---

## 5. 两节点 profiler 命令怎么加

两节点时，两个节点都加 profiler config。建议每个节点写不同目录，避免本地路径混淆。

**Node 0：**

```bash
mkdir -p /tmp/vllm_profile_node0

vllm serve nvidia/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <NODE0_IP> \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --enforce-eager \
  -cc.mode=0 \
  -cc.cudagraph_mode=NONE \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "/tmp/vllm_profile_node0",
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_memory": true,
    "torch_profiler_with_stack": true,
    "torch_profiler_with_flops": false,
    "torch_profiler_use_gzip": true,
    "warmup_iterations": 1,
    "active_iterations": 2,
    "max_iterations": 4,
    "ignore_frontend": true
  }' \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
```

**Node 1：**

```bash
mkdir -p /tmp/vllm_profile_node1

vllm serve nvidia/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr <NODE0_IP> \
  --headless \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --enforce-eager \
  -cc.mode=0 \
  -cc.cudagraph_mode=NONE \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "/tmp/vllm_profile_node1",
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_memory": true,
    "torch_profiler_with_stack": true,
    "torch_profiler_with_flops": false,
    "torch_profiler_use_gzip": true,
    "warmup_iterations": 1,
    "active_iterations": 2,
    "max_iterations": 4,
    "ignore_frontend": true
  }' \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
```

然后只对 head node 调：

```bash
curl -X POST http://<NODE0_IP>:8000/start_profile
# 发送请求
curl -X POST http://<NODE0_IP>:8000/stop_profile
```

---

## 6. 如果你的目标是“抓激活值”，Profiler 不够

Torch profiler 主要告诉你：

```text
哪个 op 被调用
op shape
CUDA kernel 时间
stack trace
memory
```

但它**不会自动保存 qkv_proj、o_proj、MoE w13/w2 的真实激活 tensor 值**。你之前的目标是抓前向激活值和 expert id，所以应组合两套东西：

```text
Torch profiler：确认调用路径、kernel、shape、耗时。
源码插桩 / forward hook：真正保存 hidden_states、router_logits、topk_ids、w13 input/output、w2 input/output。
```

另外，vLLM 有 `--enable-return-routed-experts` 参数，含义是返回 routed experts，默认关闭。([vLLM][6])
但对你这个 Blackwell + NVFP4 + fused MoE 场景，我不建议把它当作唯一依据，因为 fused backend 可能在 kernel 内部完成 routing/top-k，Python 层不一定能稳定拿到你想要的所有中间量。更稳的是 patch `FusedMoE` / MiniMax-M3 MoE forward 内部。

---

## 7. 最终建议配置

你现在这个任务不是做吞吐 benchmark，而是抓数。建议分三步：

**第一步：单节点 TP=4，短上下文跑通。**

```bash
--tensor-parallel-size 4
--block-size 128
--max-model-len 8192
--max-num-seqs 1
--language-model-only
--enforce-eager
-cc.mode=0
-cc.cudagraph_mode=NONE
```

**第二步：加 torch profiler。**

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm_profile_minimax_m3","torch_profiler_record_shapes":true,"torch_profiler_with_memory":true,"torch_profiler_with_stack":true,"ignore_frontend":true,"active_iterations":2,"max_iterations":4}'
```

**第三步：如果 4 卡显存不够或需要更长 context，再上两节点。**

优先用：

```bash
--tensor-parallel-size 4
--pipeline-parallel-size 2
--nnodes 2
```

而不是一开始跨节点 TP=8。

[1]: https://huggingface.co/nvidia/MiniMax-M3-NVFP4/blame/main/README.md "README.md · nvidia/MiniMax-M3-NVFP4 at main"
[2]: https://docs.vllm.ai/en/stable/serving/parallelism_scaling/ "Parallelism and Scaling - vLLM"
[3]: https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3 "MiniMaxAI/MiniMax-M3 | vLLM Recipes"
[4]: https://docs.vllm.ai/en/stable/contributing/profiling/ "Profiling vLLM - vLLM"
[5]: https://docs.vllm.ai/en/v0.14.0/design/debug_vllm_compile/?utm_source=chatgpt.com "How to debug the vLLM-torch.compile integration"
[6]: https://docs.vllm.ai/en/v0.20.1/configuration/engine_args/?utm_source=chatgpt.com "Engine Arguments"
