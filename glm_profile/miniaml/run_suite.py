#!/usr/bin/env python3
"""一次服务启动：使用本机 vLLM bench 测五种负载，再采五个短窗口。
不加载模型，不安装依赖，不在出错后重试。Python 标准库负责调度与超时。
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
CASES = [
    ('P512', 512, 1, 1),
    ('P8192', 8192, 1, 1),
    ('D512', 512, 64, 1),
    ('D8192', 8192, 64, 1),
    ('B4', 512, 64, 4),
]
# 不继承 HTTP 代理访问 localhost。
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def http(base: str, path: str, *, method: str = 'GET', timeout: float = 10) -> str:
    request = urllib.request.Request(base + path, method=method,
        data=b'' if method == 'POST' else None)
    with OPENER.open(request, timeout=timeout) as r:
        return r.read().decode('utf-8', errors='replace')

def stop_profile(base: str, logfile: Path) -> bool:
    try:
        text = http(base, '/stop_profile', method='POST', timeout=60)
        logfile.write_text(text or 'HTTP 200')
        return True
    except Exception as e:
        logfile.write_text(repr(e))
        print('警告：/stop_profile 未成功。停止所有请求，在服务终端按 Ctrl+C；'
              '客户端超时不能保证取消卡住的 GPU kernel。', flush=True)
        return False

def save_metrics(base: str, file: Path) -> None:
    try:
        file.write_text(http(base, '/metrics'))
    except Exception as e:
        file.with_suffix('.error.txt').write_text(repr(e))

def execute(cmd: list[str], log: Path, timeout: float, env: dict[str, str]) -> None:
    """超时仅结束本次 benchmark 的进程组，不结束 vLLM 或其他用户的进程。"""
    with log.open('w') as f:
        f.write('COMMAND: ' + json.dumps(cmd, ensure_ascii=False) + '\n')
        f.flush()
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'benchmark 返回 {code}，日志：{log}')
        except BaseException:
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
            raise

def validate_result(path: Path, count: int, input_len: int, output_len: int) -> dict:
    data = json.loads(path.read_text())
    if data.get('completed') != count or data.get('failed', 0) != 0:
        raise RuntimeError(f'存在失败请求或计数不符：{path}')
    if any(data.get('errors', [])):
        raise RuntimeError(f'请求含 error：{path}')
    if len(data.get('output_lens', [])) != count or any(x != output_len for x in data['output_lens']):
        raise RuntimeError(f'实际输出 token 数不等于 {output_len}。'
                           '请检查本机 random-range-ratio/ignore-eos 语义和 JSON，停止后续采集。')
    inputs = data.get('input_lens', [])
    if len(inputs) != count:
        raise RuntimeError(f'缺失 input_lens，无法验证负载：{path}')
    # Tokenizer 的特殊 token / decode-encode 会造成小偏差；始终保留实测长度。
    if any(abs(x - input_len) > max(16, input_len * .02) for x in inputs):
        raise RuntimeError(f'输入长度偏离目标 {input_len}：{inputs}。'
                           '请检查本机长度采样默认值，不继续生成不可比较的报告。')
    return data

def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir or Path((HERE / 'active_run.txt').read_text().strip())
    run_dir = run_dir.resolve()
    info = json.loads((run_dir / 'environment.json').read_text())
    base = 'http://127.0.0.1:' + str(info['PORT'])
    if args.phase in ('trace', 'all') and info['CAPTURE'] != '1':
        raise RuntimeError('本次服务没有挂 nsys；请用 --phase timing。')
    help_text = (run_dir/'preflight/bench_help.txt').read_text()
    bench = [info['python'], '-m', 'vllm.entrypoints.cli.main', 'bench', 'serve']
    env = os.environ.copy()
    env['NO_PROXY'] = env['no_proxy'] = '127.0.0.1,localhost'
    # 只使用本地 tokenizer 与合成数据，避免 benchmark 意外联网。
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    print(f'结果目录：{run_dir}\n等待 API /health 就绪，最长 {args.wait_seconds}s。', flush=True)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            http(base, '/health', timeout=3)
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise RuntimeError('服务尚未就绪；请检查 server.log。没有发出测试请求。')
            time.sleep(3)
    models = json.loads(http(base, '/v1/models'))
    if info['MODEL_NAME'] not in [x.get('id') for x in models.get('data', [])]:
        raise RuntimeError('服务返回的模型别名与本次启动参数不一致。')
    selected = CASES if not args.case else [x for x in CASES if x[0] in args.case]
    if len(selected) != (len(CASES) if not args.case else len(set(args.case))):
        raise ValueError('未知 case；可选：' + ', '.join(x[0] for x in CASES))
    phases = ['timing', 'trace'] if args.phase == 'all' else [args.phase]
    # 独立子目录，不覆盖旧结果；trace 可复用当前服务追加采集。
    batch_id = dt.datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    manifest = run_dir / f'manifest_{batch_id}.csv'
    with manifest.open('w', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=['phase','case','input_target','output_target','concurrency','num_prompts','start_utc','end_utc','result','status'])
        writer.writeheader()
        mf.flush()
        for phase in phases:
            for label, ilen, olen, concurrency in selected:
                if shutil.disk_usage(run_dir).free < 5 * 2**30:
                    raise RuntimeError('结果磁盘剩余不足 5 GiB，停止发请求。')
                # 同一输出长度：timing 采三小轮，trace 仅一轮，最多四条在途。
                count = concurrency * (3 if phase == 'timing' else 1)
                tag = f'{batch_id}_{phase}_{label}'
                result = run_dir/'bench'/f'{tag}.json'
                log = run_dir/'logs'/f'{tag}.log'
                row = dict(phase=phase, case=label, input_target=ilen, output_target=olen,
                           concurrency=concurrency, num_prompts=count, start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                           end_utc='', result=str(result.relative_to(run_dir)), status='running')
                command = bench + [
                    '--backend','openai','--base-url',base,'--endpoint','/v1/completions',
                    '--model',info['MODEL_NAME'],'--tokenizer',info['MODEL'],
                    '--trust-remote-code','--dataset-name','random',
                    '--random-input-len',str(ilen),'--random-output-len',str(olen),
                    '--random-prefix-len','0','--num-prompts',str(count),
                    '--max-concurrency',str(concurrency),'--request-rate','inf',
                    '--num-warmups',str(concurrency),'--seed','42','--ignore-eos',
                    '--extra-body','{"temperature":0.0,"top_p":1.0,"skip_special_tokens":false}',
                    '--percentile-metrics','ttft,tpot,itl,e2el','--metric-percentiles','50',
                    '--save-result','--save-detailed','--result-dir',str(result.parent),
                    '--result-filename',result.name,'--disable-tqdm']
                # 不硬编码 random-range-ratio：旧版常用 1 表示固定长度，新版为 0。
                # 使用本机默认值后，validate_result 会核验真正输入/输出长度。
                if '--ready-check-timeout-sec' in help_text:
                    command += ['--ready-check-timeout-sec','0']
                if phase == 'trace':
                    command += ['--profile']
                print(f'[{phase}] {label}: {ilen}→{olen}, C={concurrency}, N={count}; 日志 {log.name}', flush=True)
                save_metrics(base, run_dir/'metrics'/f'{tag}_before.prom')
                try:
                    execute(command, log, args.case_timeout, env)
                    if phase == 'trace':
                        text = log.read_text(errors='replace')
                        if 'Profiler started' not in text or 'Profiler stopped' not in text:
                            raise RuntimeError('benchmark 日志未确认 profiler 成功启停；查看 ' + str(log))
                    validate_result(result, count, ilen, olen)
                    row['status'] = 'ok'
                except BaseException:
                    row['status'] = 'failed'
                    if phase == 'trace':
                        stop_profile(base, run_dir/'logs'/f'{tag}_emergency_stop.log')
                    if log.exists():
                        print('\n'.join(log.read_text(errors='replace').splitlines()[-30:]), flush=True)
                    raise
                finally:
                    row['end_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
                    writer.writerow(row)
                    mf.flush()
                    save_metrics(base, run_dir/'metrics'/f'{tag}_after.prom')
                # 留少量非采集间隙，避免连续 start/stop 紧贴；不算入 benchmark。
                time.sleep(1)
    print(f'全部完成。请求与采集顺序：{manifest}\n在服务终端按一次 Ctrl+C 正常收尾，'
          '让 nsys 导出报告；不要 kill -9。', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--phase', choices=['all','timing','trace'], default='all')
    parser.add_argument('--case', action='append', help='可重复：P512/P8192/D512/D8192/B4')
    parser.add_argument('--wait-seconds', type=int, default=3600)
    parser.add_argument('--case-timeout', type=int, default=180,
                        help='单个 benchmark 进程的墙钟上限，含本地 tokenizer 加载/预热/采集')
    args = parser.parse_args()
    if args.wait_seconds < 1 or args.case_timeout < 1:
        parser.error('timeouts must be positive')
    try:
        run(args)
    except KeyboardInterrupt:
        print('用户中断；不再发送请求。查看日志并在服务端正常结束采集。', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'停止：{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
