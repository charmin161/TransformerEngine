# GLM-5.2 客户端超时诊断（只增加客户端文件）

目的：判断原 run_suite.py 的 180 秒总超时发生在客户端模块导入、tokenizer、合成输入、服务端 /tokenize 对齐、预热还是正式 HTTP 推理阶段。此包不声称已经修复根因。

## 不要重启模型

保留正在运行的 API/终端 A；不要重新执行 01_start_server.sh。
压缩包只含 run_suite_debug.py、bench_entry_debug.py 和说明，不覆盖原文件、不安装或升级依赖、不修改服务端启动参数。

先停止重复运行旧的压测客户端。如果服务器已有未结束的请求，不要继续叠加测试。

## 先确认服务状态

在与原 API 相同的机器/容器里运行（不是可视化机）：

```bash
RUN=/wireless/minyusong/glm_profiles/results/20260915_154321_371628
BASE=http://127.0.0.1:8972

curl --noproxy '*' --connect-timeout 3 --max-time 5 -sS -i "$BASE/health"
curl --noproxy '*' --connect-timeout 3 --max-time 5 -sS "$BASE/metrics" \
  > "$RUN/metrics/diag_current.prom"
grep -E '^vllm:num_requests_(running|waiting)(\{|[[:space:]])' \
  "$RUN/metrics/diag_current.prom"
tail -120 "$RUN/server.log"
```

health=200 只说明健康检查通过，不证明一次生成能完成。metrics 抓取失败/没有匹配项不等于 running=0。
若 running/waiting 持续非零、health 无响应或已有错误，先保留日志并停止继续发推理请求。

## 一条不依赖客户端 vLLM/tokenizer 的检查请求

仅在服务仍响应且无未结束请求时执行一次。它只要求 1 个输出 token，无自动重试，不启动第二份模型。

```bash
curl --noproxy '*' --connect-timeout 3 --max-time 60 \
  -sS -N "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM5.2-NVFP4","prompt":"The capital of France is", "max_tokens":1,"temperature":0.0,"top_p":1.0,"ignore_eos":true,"skip_special_tokens":false,"stream":true,"stream_options":{"include_usage":true}}' \
  -w '\nHTTP=%{http_code} total=%{time_total}s\n' \
  | tee "$RUN/logs/direct_probe.log"
```

检查正常 SSE 数据、completion_tokens=1、[DONE]，不是只检查 HTTP=200。curl 的超时是客户端保护，不保证恢复或强制取消服务器中的 kernel。这个短文本不是 512-token 性能基线。
请求不完成时，不进入下一轮压测；查看同时间段 server.log 和 metrics。

## 客户端诊断

将两个 .py 文件放在一起；可直接解压到原脚本目录，无需覆盖任何旧文件：

```bash
cd /wireless/minyusong/glm_profiles
unzip /实际上传路径/glm52_client_diagnostic.zip

python3 -u run_suite_debug.py \
  --run-dir /wireless/minyusong/glm_profiles/results/20260915_154321_371628 \
  --phase timing --case P512 --case-timeout 180
```

依然使用 environment.json 中记录的原虚拟环境 Python 执行实际 benchmark。
默认只跑 timing/P512（1 条预热 + 3 条正式、在途并发上限 1），不会默认启动 trace 或执行其余场景。

新增内容：
- 子进程 Python -u / PYTHONUNBUFFERED=1；输出实时写入日志并同步到终端。
- 模块导入、get_tokenizer、get_samples、可用时的服务端 tokenizer 对齐、benchmark 的 BEGIN/END 标记。
- 每 45 秒输出 Python 线程堆栈，20 秒打印客户端存活信息；不是额外发请求。
- 总超时默认仍为 180 秒；失败仅终止此次客户端进程组，不结束 API/其他任务。
- 保留原 JSON、长度校验、manifest 与 metrics 快照逻辑；新时间戳文件不覆盖旧结果。

日志中的 `Timeout (0:00:45)!` 来自 faulthandler 的定时快照，不代表程序已经被终止。真正总超时会明确打印 `[client-watchdog]`，并抛出 TimeoutExpired。

诊断版改变了客户端导入顺序并增加日志，仅用于定位阻塞，不把它的耗时当正式性能成绩。异步等待时，线程堆栈可能只显示 selectors/asyncio；要结合最近的阶段标记和服务端日志判断，不据此单独认定服务卡死。

| 最后阶段 | 下一步看什么 |
|---|---|
| import vLLM CLI / benchmark module 没有 END | Python 依赖导入堆栈、本地/共享文件系统、扩展库初始化 |
| get_tokenizer 有 BEGIN 没有 END | tokenizer/配置加载对应堆栈 |
| get_samples 有 BEGIN 没有 END | 合成数据生成/encode/decode 对应堆栈 |
| _align_prompts_to_server_tokenizer 有 BEGIN 没有 END | /tokenize、/detokenize 处理；尚不等于 GPU 推理等待 |
| Warming up... 没有 completed | 预热请求未结束；对照服务日志、running/waiting、响应状态 |
| Starting main benchmark run... 后停住 | 正式请求或统计阶段；结合后续日志与服务响应 |

不要只把超时提高后重跑全部用例。仅当日志证实在正常推进且慢的是初始化/预热时，才考虑单独延长诊断窗口。

## 验证范围

详见 VALIDATION.txt：做了 Python 语法检查与本地 mock 的正常返回、非零退出、超时堆栈、API 不被客户端清理杀死测试。
没有 B200，也没有用户实际的 vLLM 环境；没有实测真实模型。

## 官方参考（实现核对）
- https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/benchmarks/serve.py
- https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/entrypoints/serve/instrumentator/health.py
- https://docs.python.org/3/library/faulthandler.html
- https://docs.python.org/3/using/cmdline.html
