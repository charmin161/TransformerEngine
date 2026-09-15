#!/usr/bin/env bash
# 在已跑通 API 的同一虚拟环境/容器、同一 GPU 分配下运行。不安装或更新软件。
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}
MODEL=${MODEL:-/wireless/public/models/GLM-5.2-NVFP4}
MODEL_NAME=${MODEL_NAME:-GLM5.2-NVFP4}
PORT=${PORT:-8972}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-819200}
GPU_METRICS=${GPU_METRICS:-auto}       # auto | 0 | 1
METRIC_DEVICES=${METRIC_DEVICES:-0,1,2,3}  # nsys 的设备编号，不一定等于容器中的 CUDA 编号
GRAPH_TRACE=${GRAPH_TRACE:-node}      # 首轮短采集需要 kernel 明细
CAPTURE=${CAPTURE:-1}                # 0: 原生不挂 nsys，只运行 timing 对照
ROOT=${RESULT_ROOT:-$HERE/results}
RUN="$ROOT/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN"/{preflight,bench,logs,metrics,traces}
printf '%s\n' "$RUN" > "$HERE/active_run.txt"
# tee 不响应 Ctrl+C，让 nsys 在用户正常停止后仍有机会输出导出日志。
exec > >(trap '' INT; tee -a "$RUN/server.log") 2>&1
printf '实验目录：%s\n' "$RUN"
command -v "$PYTHON" >/dev/null
command -v nsys >/dev/null
[[ -d "$MODEL" ]] || { echo "模型目录不存在：$MODEL"; exit 2; }
[[ "$CAPTURE" == 0 || "$CAPTURE" == 1 ]] || exit 2
[[ "$GPU_METRICS" == auto || "$GPU_METRICS" == 0 || "$GPU_METRICS" == 1 ]] || exit 2
[[ "$GRAPH_TRACE" == node || "$GRAPH_TRACE" == graph ]] || exit 2
# 保持原有 CUDA_VISIBLE_DEVICES / 后端 / 编译缓存；不强行改 GPU、驱动或 TMPDIR。
export MODEL MODEL_NAME PORT MAX_MODEL_LEN GPU_METRICS METRIC_DEVICES GRAPH_TRACE CAPTURE RUN
"$PYTHON" - <<'PY'
import importlib.metadata as md, json, os, pathlib, shutil, socket, sys
run = pathlib.Path(os.environ['RUN'])
port = int(os.environ['PORT'])
with socket.socket() as s:
    try:
        s.bind(('127.0.0.1', port))
    except OSError as e:
        raise SystemExit(f'端口 {port} 已占用。请先正常结束旧 API，不要在同四张卡上启动第二份模型。{e}')
for p in (run, pathlib.Path(os.environ.get('TMPDIR', '/tmp'))):
    free = shutil.disk_usage(p).free / 2**30
    print(f'{p}: 可用磁盘 {free:.1f} GiB')
    if free < 5:
        raise SystemExit('剩余磁盘不足 5 GiB，停止。请先选择有空间的目录/处理临时磁盘。')
versions = {}
for name in ('vllm', 'torch', 'transformers', 'flashinfer-python', 'flashinfer-cubin', 'opencompass'):
    try: versions[name] = md.version(name)
    except md.PackageNotFoundError: versions[name] = None
info = {key: os.environ[key] for key in ('MODEL', 'MODEL_NAME', 'PORT', 'MAX_MODEL_LEN', 'GRAPH_TRACE', 'CAPTURE')}
info.update(python=sys.executable, versions=versions, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            extra_environment={k: os.environ[k] for k in ('VLLM_ATTENTION_BACKEND','VLLM_MOE_BACKEND','CUDA_LAUNCH_BLOCKING','VLLM_WORKER_MULTIPROC_METHOD','TMPDIR') if k in os.environ})
(run/'environment.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
print(json.dumps(info, ensure_ascii=False, indent=2))
if os.environ.get('CUDA_LAUNCH_BLOCKING') == '1':
    raise SystemExit('检测到 CUDA_LAUNCH_BLOCKING=1。它会改变异步执行；本次先退出该调试设置再运行。')
PY
nsys --version
(nvidia-smi; nvidia-smi topo -m) > "$RUN/gpu_environment.txt" 2>&1 || true
# 在加载大模型之前，验证服务端、benchmark 和 nsys 的参数。
if ! "$PYTHON" -m vllm.entrypoints.openai.api_server --help=all > "$RUN/preflight/server_help.txt" 2>&1; then
  "$PYTHON" -m vllm.entrypoints.openai.api_server --help > "$RUN/preflight/server_help.txt" 2>&1
fi
"$PYTHON" -m vllm.entrypoints.cli.main bench serve --help > "$RUN/preflight/bench_help.txt" 2>&1
nsys profile --help > "$RUN/preflight/nsys_help.txt" 2>&1
for flag in profiler-config prefix-caching; do
  grep -q -- "$flag" "$RUN/preflight/server_help.txt" || {
    echo "本机服务端帮助未发现 $flag。尚未加载模型，请查看 preflight/server_help.txt；不要直接升级环境。"; exit 2;
  }
done
for flag in --max-concurrency --num-warmups --profile --save-detailed --extra-body --random-input-len --random-output-len; do
  grep -q -- "$flag" "$RUN/preflight/bench_help.txt" || {
    echo "本机 benchmark 不支持 $flag；尚未加载模型，查看 preflight/bench_help.txt。"; exit 2;
  }
done
COMMON=(--trace=cuda,nvtx --sample=none --cpuctxsw=none "--cuda-graph-trace=$GRAPH_TRACE" --capture-range=cudaProfilerApi)
if grep -q -- --cuda-event-trace "$RUN/preflight/nsys_help.txt"; then COMMON+=(--cuda-event-trace=false); fi
if grep -q -- --trace-fork-before-exec "$RUN/preflight/nsys_help.txt"; then COMMON+=(--trace-fork-before-exec=true); fi
METRICS=()
if [[ "$GPU_METRICS" != 0 ]]; then METRICS=("--gpu-metrics-devices=$METRIC_DEVICES" --gpu-metrics-frequency=1000); fi

if [[ "$CAPTURE" == 1 ]]; then
  echo '先运行小矩阵 CUDA 采集检查；此时不加载 GLM。'
  SMOKE="$RUN/preflight/smoke"
  set +e
  timeout --signal=TERM --kill-after=10s 180s \
    nsys profile "${COMMON[@]}" "${METRICS[@]}" --capture-range-end=stop \
    -o "$RUN/preflight/smoke" "$PYTHON" "$HERE/cuda_smoke.py" \
    > "$RUN/preflight/smoke.log" 2>&1
  RC=$?
  set -e
  BAD_METRICS=0
  if grep -Eiq '(ERR_NVGPUCTRPERM|ERR_NVGPUCTR|permission.*denied|insufficient.*permission|not supported.*GPU|GPU.*not supported|GPU Metrics.*(fail|error|unavailable)|metrics.*(permission|not available))' "$RUN/preflight/smoke.log"; then BAD_METRICS=1; fi
  if [[ "$RC" != 0 || "$BAD_METRICS" == 1 ]]; then
    if [[ "$GPU_METRICS" == auto ]]; then
      echo '含 GPU Metrics 的检查未通过，先退回仅 CUDA 时间线；详情见 preflight/smoke.log。'
      METRICS=()
      SMOKE="$RUN/preflight/smoke_no_metrics"
      timeout --signal=TERM --kill-after=10s 180s \
        nsys profile "${COMMON[@]}" --capture-range-end=stop \
        -o "$RUN/preflight/smoke_no_metrics" "$PYTHON" "$HERE/cuda_smoke.py" \
        > "$RUN/preflight/smoke_no_metrics.log" 2>&1 || {
          tail -80 "$RUN/preflight/smoke_no_metrics.log"; echo '小程序采集也失败；停止，不加载 GLM。'; exit 2;
        }
    else
      tail -80 "$RUN/preflight/smoke.log"
      echo '采集检查失败；停止，不加载 GLM。'; exit 2
    fi
  fi
  REPORT=$(find "$RUN/preflight" -maxdepth 1 -name "$(basename "$SMOKE")*.nsys-rep" -print -quit)
  if [[ -z "$REPORT" ]]; then
    echo '检查后没有找到 .nsys-rep；请先查看 preflight 日志，不加载 GLM。'; exit 2
  fi
  timeout --signal=TERM --kill-after=10s 120s nsys export --type=sqlite \
    --output="$RUN/preflight/smoke_check.sqlite" "$REPORT" \
    > "$RUN/preflight/smoke_export.log" 2>&1 || {
      cat "$RUN/preflight/smoke_export.log"; echo '小报告导出失败，先处理此问题，不加载 GLM。'; exit 2;
    }
  "$PYTHON" - <<'PYVERIFY'
import os, pathlib, sqlite3
run = pathlib.Path(os.environ['RUN'])
path = run/'preflight/smoke_check.sqlite'
if not path.exists():
    raise SystemExit('没有找到导出的 SQLite，不能确认采集成功。')
with sqlite3.connect(str(path)) as db:
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    kernels = 0
    for table in tables:
        if 'KERNEL' not in table: continue
        cols = {r[1] for r in db.execute('PRAGMA table_info("' + table.replace('"', '""') + '")')}
        if {'start', 'end'}.issubset(cols):
            kernels += db.execute('SELECT COUNT(*) FROM "' + table.replace('"', '""') + '"').fetchone()[0]
    if kernels < 1:
        raise SystemExit('小报告中没有 CUDA kernel 记录。请先检查 preflight 日志/driver/CUPTI 兼容性，不加载 GLM。')
    metrics = db.execute('SELECT COUNT(*) FROM GPU_METRICS').fetchone()[0] if 'GPU_METRICS' in tables else 0
(run/'preflight/verified_gpu_metric_rows.txt').write_text(str(metrics))
print(f'小报告确认：CUDA kernel 记录 {kernels} 条，GPU Metrics 记录 {metrics} 条。')
PYVERIFY
  if [[ ${#METRICS[@]} -gt 0 && $(cat "$RUN/preflight/verified_gpu_metric_rows.txt") == 0 ]]; then
    if [[ "$GPU_METRICS" == 1 ]]; then
      echo '要求采集 GPU Metrics，但小报告中没有指标；先处理权限/计数器问题，不加载 GLM。'; exit 2
    fi
    echo '小报告没有 GPU Metrics 数据：主实验只保留已验证的 CUDA 时间线。'
    METRICS=()
  fi
fi
printf '%s\n' "${METRICS[*]:-disabled}" > "$RUN/gpu_metrics_mode.txt"
# spawn 是 Nsight 下 vLLM 官方建议。其余计算配置不改。
export VLLM_WORKER_MULTIPROC_METHOD=spawn
SERVER=("$PYTHON" -m vllm.entrypoints.openai.api_server
  --model "$MODEL" --tensor-parallel-size 4 --quantization modelopt
  --trust-remote-code --port "$PORT" --host 127.0.0.1
  --enable-expert-parallel --reasoning-parser glm45
  --served-model-name "$MODEL_NAME" --max-model-len "$MAX_MODEL_LEN"
  --no-enable-prefix-caching)
if [[ "$CAPTURE" == 1 ]]; then SERVER+=(--profiler-config '{"profiler":"cuda"}'); fi
# 不恢复 --disable-log-stats：这次要保留运行统计。请求正文不需要打印。
if grep -q -- --disable-log-requests "$RUN/preflight/server_help.txt"; then SERVER+=(--disable-log-requests); fi
printf '%q ' "${SERVER[@]}" > "$RUN/server_command.sh"; printf '\n' >> "$RUN/server_command.sh"
if [[ "$CAPTURE" == 1 ]]; then
  COMMAND=(nsys profile "${COMMON[@]}" "${METRICS[@]}" --capture-range-end=repeat -o "$RUN/traces/glm52" "${SERVER[@]}")
else
  COMMAND=("${SERVER[@]}")
fi
printf '%q ' "${COMMAND[@]}" > "$RUN/full_command.sh"; printf '\n' >> "$RUN/full_command.sh"
echo '开始启动一次 GLM API。另开同一环境终端运行：python3 run_suite.py'
echo '所有测试结束后，在本终端按一次 Ctrl+C 正常收尾；不要 kill -9。'
exec "${COMMAND[@]}"
