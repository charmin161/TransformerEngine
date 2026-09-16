**先不要重启 API，也不要重新运行 `01_start_server.sh`。这次确定的是：压测客户端进程超过了 180 秒，不是已经证明“512 tokens 的 GPU 推理耗时超过 180 秒”。**

我检查了上一版脚本：这个超时覆盖**客户端导入依赖、加载 tokenizer、生成输入、预热、正式请求及保存结果**。同时，脚本没有强制实时刷新子进程输出，导致现在看不到明确的停顿位置。**这是我之前脚本的诊断缺口，但目前不能据此认定超时本身是误报。**

下面先检查现有服务，再只诊断 P512，**不重新加载模型、不增加并发、不直接把超时放大后重跑全套**。

## 一、先明确：这份日志说明了什么

| 日志信息                                | 可以确定的结论                                                      |
| ----------------------------------- | ------------------------------------------------------------ |
| 已进入 `[timing] P512`                 | 脚本之前的 `/health` 和模型名称检查已经通过                                  |
| `timed out after 180 seconds`       | 父进程等待整个 benchmark 子进程退出时超时，不是服务端返回了一个 180 秒推理指标              |
| `--num-warmups 1`、`--num-prompts 3` | 本组计划执行 1 条预热和 3 条正式请求，不是只发一次请求                               |
| `--max-concurrency 1`               | 在途推理请求上限为 1；`request-rate=inf` 不会取消这个并发限制                    |
| 命令没有 `--profile`                    | 这一组没有要求 benchmark 开启采集窗口；但服务仍由 nsys 启动，不能因此完全排除 profiler 的影响 |

vLLM 的 benchmark 确实在正式测试前加载 tokenizer、构造请求并进行预热；只有传入 `--profile` 才会执行 profiler 启动流程。

另外：

```bash
--ready-check-timeout-sec 0
```

**不是“请求超时为 0”，而是跳过 benchmark 自己的额外就绪检查。不要修改它来解决这次问题。**([vLLM][1])

## 二、先检查已经启动的 API，不发送新的推理负载

在 **B200 推理机、原来的环境**中执行：

```bash
cd /wireless/minyusong/glm_profiles

RUN=/wireless/minyusong/glm_profiles/results/20260915_154321_371628
BASE=http://127.0.0.1:8972

# 1. 健康检查
curl --noproxy '*' \
  --connect-timeout 3 --max-time 5 \
  -sS -i "$BASE/health"

# 2. 当前运行/等待请求数
curl --noproxy '*' \
  --connect-timeout 3 --max-time 5 \
  -sS "$BASE/metrics" \
  > "$RUN/metrics/diag_current.prom"

grep -E '^vllm:num_requests_(running|waiting)(\{|[[:space:]])' \
  "$RUN/metrics/diag_current.prom"

# 3. 服务端最近的输出
tail -120 "$RUN/server.log"

# 4. 原 benchmark 日志
tail -120 \
  "$RUN/logs/20260915_171827_2048676_timing_P512.log"
```

重点看两件事。

**第一，`/health` 是否返回 200。** vLLM 的这个接口调用引擎健康检查，**不会实际执行一条生成请求**，所以返回 200 仍不足以证明推理链路正常。

**第二，是否仍有未完成的请求。** `num_requests_running` 和 `num_requests_waiting` 分别反映正在处理和等待调度的请求。若持续非零，先不要叠加下一轮测试；指标抓取失败或没有匹配结果，也不能当成零。([vLLM][2])

**如果健康检查失败，或服务日志已经出现 worker 异常、通信错误、OOM 等，先停在这里，保留日志，不继续发请求。**

## 三、服务正常且没有遗留请求时，只发一个输出 token

这一步绕过**客户端的 vLLM 导入、tokenizer 加载和随机数据构造**，直接检查已有 API 能否完成一次最小生成。

仍在上面那个终端执行一次：

```bash
curl --noproxy '*' \
  --connect-timeout 3 \
  --max-time 60 \
  -sS -N \
  "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM5.2-NVFP4",
    "prompt": "The capital of France is",
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "ignore_eos": true,
    "skip_special_tokens": false,
    "stream": true,
    "stream_options": {"include_usage": true}
  }' \
  -w '\nHTTP=%{http_code} total=%{time_total}s\n' \
  | tee "$RUN/logs/direct_probe.log"
```

这里使用流式请求，是为了保持与你当前 benchmark 相同的响应方式；vLLM 的 `openai` benchmark 后端就是向 `/v1/completions` 发送流式请求，并读取 usage。

### 如何判断结果

**正常完成：**有正常的 `data:` 响应，usage 中 `completion_tokens` 为 1，最后出现：

```text
data: [DONE]
```

这说明服务能完成**这条短请求**。接下来优先诊断客户端初始化、随机输入处理，或特定输入对应的执行路径；**它还不能证明 512-token 场景已经正常**。

**只有 HTTP 200，但一直没有生成结果或 `[DONE]`：**仍不能算推理完成。

**请求超时：**不要重复发送，也不要马上改成 600 秒继续压测。此时最重要的是同时间段的 `server.log` 和运行/等待请求数。

这里的 60 秒只是客户端保护上限，不是预期推理时间；这条短文本请求也**不是 P512 的性能成绩**。

## 四、最小请求正常后，运行“带阶段日志”的 P512

我做了一个**只增加客户端文件的诊断包**，不包含服务启动脚本，不需要覆盖旧文件：

**[下载：GLM 客户端超时诊断包](sandbox:/mnt/data/glm52_client_diagnostic.zip)**

它仍然调用你环境中的 `vllm bench serve`，保留原来的请求长度、并发和结果校验，只增加：

**实时输出、各步骤的 BEGIN/END 标记，以及每 45 秒一次的 Python 堆栈快照。**

实时输出使用 Python 的 `-u`／`PYTHONUNBUFFERED`；定时堆栈使用标准库 `faulthandler`，不需要安装额外工具。([Python documentation][3])

包内是：

```text
run_suite_debug.py
bench_entry_debug.py
README.md
VALIDATION.txt
```

已完成语法检查，以及本地模拟的正常返回、错误退出、超时堆栈和服务进程保留测试；**未在你的 B200 环境实测，因此这是诊断工具，不是已经确认根因的修复包。**

### 执行方法

将压缩包传到原脚本目录：

```bash
cd /wireless/minyusong/glm_profiles

unzip glm52_client_diagnostic.zip
```

**保持终端 A 中的服务运行**，在客户端终端执行：

```bash
python3 -u run_suite_debug.py \
  --run-dir /wireless/minyusong/glm_profiles/results/20260915_154321_371628 \
  --phase timing \
  --case P512 \
  --case-timeout 180
```

这次依然只运行 **P512：1 条预热＋3 条正式请求，并发上限 1**。仍保留 180 秒总超时，没有开启 trace，也不会启动第二份模型。

你会看到类似这些阶段标记，具体耗时以实测为准：

```text
BEGIN import vLLM CLI
END import vLLM CLI

BEGIN import vLLM benchmark module
END import vLLM benchmark module

BEGIN get_tokenizer
END get_tokenizer

BEGIN get_samples
END get_samples

BEGIN _align_prompts_to_server_tokenizer
END _align_prompts_to_server_tokenizer

BEGIN benchmark
Warming up with 1 requests...
Warmup run completed.
Starting main benchmark run...
```

**最后一个出现 BEGIN、却没有出现 END 的步骤，就是需要重点检查的阶段。**

| 停在哪个阶段                               | 接下来优先检查什么                              |
| ------------------------------------ | -------------------------------------- |
| 导入 vLLM CLI／benchmark 模块             | Python 依赖导入、扩展库初始化、文件系统访问；此时尚不能归因于模型推理 |
| `get_tokenizer`                      | 本地 tokenizer 和配置加载                     |
| `get_samples`                        | 随机输入生成及 encode/decode                  |
| `_align_prompts_to_server_tokenizer` | 服务端 `/tokenize`、`/detokenize` 的请求处理    |
| `Warming up...`                      | 预热生成请求没有完成，结合服务端日志判断                   |
| `Starting main benchmark run...` 之后  | 正式请求或后续统计阶段，需要结合后续日志和服务状态              |

这些阶段对应 vLLM 0.26.0 的实际 benchmark 执行流程，其中随机数据还可能经过服务端 tokenizer 对齐。

注意，诊断日志里的：

```text
Timeout (0:00:45)!
```

是 `faulthandler` 的**定时堆栈快照提示，不是程序已经被终止**。真正达到总上限时，会另外打印 `[client-watchdog]` 并报告 `TimeoutExpired`。([Python documentation][4])

## 五、现在先不做的改动

**不要重装 vLLM、切换 attention/MoE 后端、关闭 CUDA Graph、清理编译缓存，或直接扩大测试并发。** 当前证据还不足以支持这些改动。

也不要只把：

```bash
--case-timeout 180
```

改成更大值，就把问题当成解决了。**只有日志确认程序在正常推进，且主要慢在客户端初始化或首次预热，才有理由单独延长这个诊断窗口。**

本轮首先要区分的是：

> **客户端还没把请求发出去，还是服务已经收到请求但没有完成。**

把**单 token 请求的输出、诊断日志最后一个阶段及对应的 `server.log` 末尾**贴回来，才能继续针对具体阻塞点处理，而不是让你反复重启这个大模型。

[1]: https://docs.vllm.ai/en/v0.26.0/cli/bench/serve/ "vllm bench serve - vLLM"
[2]: https://docs.vllm.ai/en/v0.26.0/usage/metrics/ "Production Metrics - vLLM"
[3]: https://docs.python.org/3/using/cmdline.html "1. Command line and environment — Python 3.14.7 documentation"
[4]: https://docs.python.org/3/library/faulthandler.html "faulthandler — Dump the Python traceback — Python 3.14.7 documentation"
