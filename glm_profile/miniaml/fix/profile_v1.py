#!/usr/bin/env python3
"""Bounded GLM/vLLM HTTP profiling client. Python standard library only.
Run with: python3 -I -S profile_http_only_v1_1.py --run-dir EXISTING_RUN
Version 1.1: separate bounded profiler-control timeout and progress logs.
Does NOT import vllm/torch/transformers, start a model, or change installed files.
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import contextlib
import csv
import datetime as dt
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import sys
import threading
import time
import urllib.parse
import uuid

CASES = {
    'P512': (512, 1, 1), 'P8192': (8192, 1, 1),
    'D512': (512, 64, 1), 'D8192': (8192, 64, 1), 'B4': (512, 64, 4),
}
START = time.monotonic()
PRINT_LOCK = threading.Lock()

def log(message):
    with PRINT_LOCK:
        print(f'[http-only +{time.monotonic()-START:.3f}s] {message}', flush=True)

def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

class Client:
    def __init__(self, base_url, key=''):
        url = urllib.parse.urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname:
            raise ValueError('base-url 必须是 HTTP(S) 地址。')
        if url.path not in ('', '/') or url.query or url.fragment or url.username:
            raise ValueError('base-url 请使用 http://127.0.0.1:8972，不包含 /v1。')
        self.url = url
        self.key = key
        self.lock = threading.Lock()
        self.sockets = set()

    def abort(self):
        with self.lock:
            socks = list(self.sockets)
        for sock in socks:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    @contextlib.contextmanager
    def response(self, method, path, payload=None, timeout=30.0):
        cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        conn = cls(self.url.hostname, self.url.port, timeout=min(timeout, 5.0))
        sock = None
        timer = None
        expired = threading.Event()
        begin = time.perf_counter()
        headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        def expire():
            expired.set()
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
        try:
            # http.client ignores HTTP_PROXY/HTTPS_PROXY: this is a direct connection.
            conn.connect()
            sock = conn.sock
            remaining = timeout - (time.perf_counter()-begin)
            if remaining <= 0:
                raise TimeoutError(f'{path}: 连接阶段超时')
            sock.settimeout(remaining)
            with self.lock:
                self.sockets.add(sock)
            timer = threading.Timer(remaining, expire)
            timer.daemon = True
            timer.start()
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            if response.status >= 300:
                detail = response.read(8192).decode('utf-8', 'replace')
                raise RuntimeError(f'{method} {path}: HTTP {response.status}: {detail}')
            yield response
            if expired.is_set():
                raise TimeoutError(f'{path}: 请求总时间超过 {timeout:g}s')
        except Exception as exc:
            if expired.is_set() or isinstance(exc, (socket.timeout, TimeoutError)):
                raise TimeoutError(f'{path}: 请求超时，上限 {timeout:g}s；未重试') from exc
            raise
        finally:
            if timer:
                timer.cancel()
            if sock is not None:
                with self.lock:
                    self.sockets.discard(sock)
            conn.close()

    def raw(self, method, path, payload=None, timeout=30):
        with self.response(method, path, payload, timeout) as response:
            return response.read(4*1024*1024)

    def json(self, method, path, payload=None, timeout=30):
        body = self.raw(method, path, payload, timeout)
        return json.loads(body) if body else None

    def idle(self, output, label):
        deadline = time.monotonic()+15
        while True:
            text = self.raw('GET', '/metrics', timeout=5).decode('utf-8')
            values = {'running': [], 'waiting': []}
            for line in text.splitlines():
                match = re.match(r'^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([^\s]+)', line)
                if match:
                    values[match.group(1)].append(float(match.group(2)))
            if not all(values.values()) or not all(math.isfinite(v) for vs in values.values() for v in vs):
                raise RuntimeError('/metrics 未返回完整有效的 running/waiting 指标；不把缺失当成空闲。')
            if all(v == 0 for vs in values.values() for v in vs):
                (output / (label+'.prom')).write_text(text, encoding='utf-8')
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'服务仍有遗留/其他请求 {values}；停止，不叠加新请求。')
            time.sleep(0.5)

    def complete(self, ids, out_len, timeout, request_id):
        payload = {
            'model': ARGS.model, 'prompt': ids, 'add_special_tokens': False,
            'max_tokens': out_len, 'temperature': 0.0, 'top_p': 1.0,
            'ignore_eos': True, 'skip_special_tokens': False,
            'stream': True, 'stream_options': {'include_usage': True},
            'return_token_ids': True, 'request_id': request_id,
            # Unique per request: prevent prefix reuse even if server cache is on.
            'cache_salt': uuid.uuid4().hex,
        }
        start = time.perf_counter()
        usage, done, events, generated = None, False, [], []
        finish_reason = None
        with self.response('POST', '/v1/completions', payload, timeout) as response:
            if 'text/event-stream' not in response.getheader('Content-Type', ''):
                raise RuntimeError('生成接口未返回 text/event-stream。')
            parts = []
            while True:
                line = response.readline(1024*1024)
                if not line:
                    break
                if len(line) >= 1024*1024:
                    raise RuntimeError('异常超长 SSE 行')
                line = line.decode('utf-8').rstrip('\r\n')
                if line.startswith('data:'):
                    parts.append(line[5:].lstrip())
                    continue
                if line != '' or not parts:
                    continue
                data, parts = '\n'.join(parts), []
                now = time.perf_counter()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(f'SSE error: {event["error"]}')
                if event.get('usage') is not None:
                    usage = event['usage']
                for choice in event.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise RuntimeError('本工具只接受 n=1 的单序列响应。')
                    token_ids = choice.get('token_ids') or []
                    if token_ids:
                        if not all(type(t) is int for t in token_ids):
                            raise RuntimeError('响应 token_ids 类型异常。')
                        generated.extend(token_ids)
                        events.append({'elapsed_ms': (now-start)*1000, 'tokens': len(token_ids)})
                    finish_reason = choice.get('finish_reason') or finish_reason
        e2e = (time.perf_counter()-start)*1000
        if not done:
            raise RuntimeError('SSE 没有 [DONE]；不作为成功推理。')
        if not usage or usage.get('prompt_tokens') != len(ids) or usage.get('completion_tokens') != out_len:
            raise RuntimeError(f'usage 长度不符：expected={len(ids)}->{out_len}, actual={usage}')
        if len(generated) != out_len or not events:
            raise RuntimeError(f'return_token_ids 缺失或数量不符：{len(generated)} != {out_len}；不推测 TTFT。')
        ttft = events[0]['elapsed_ms']
        tpot = (events[-1]['elapsed_ms']-ttft)/(out_len-1) if out_len > 1 else None
        return {
            'request_id': request_id, 'prompt_tokens': len(ids), 'completion_tokens': out_len,
            'ttft_stream_ms': ttft, 'tpot_stream_est_ms': tpot, 'e2e_ms': e2e,
            'finish_reason': finish_reason, 'usage': usage, 'token_events': events,
            'generated_token_ids': generated,
        }

def profile_control(client, endpoint, timeout, record_path):
    """Send one control request. A timeout leaves server state unknown, not cancelled."""
    begin = time.monotonic()
    finished = threading.Event()
    result = {
        'endpoint': endpoint,
        'attempt_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'timeout_seconds': timeout,
        'status': 'pending',
    }
    save(record_path, result)
    log(f'POST {endpoint}；控制请求上限 {timeout:g}s；此期间不发送新的生成请求。')

    def heartbeat():
        while not finished.wait(10):
            log(f'{endpoint} 尚未返回，已用 {time.monotonic()-begin:.1f}s；'
                '只等待原请求，不重试、不发送生成请求。')

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        client.raw('POST', endpoint, timeout=timeout)
        result.update(status='completed',
                      returned_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        log(f'{endpoint} 已成功返回；耗时 {time.monotonic()-begin:.3f}s。')
        return result
    except BaseException as exc:
        result.update(status='unknown', error_type=type(exc).__name__, error=str(exc))
        log(f'{endpoint} 未确认完成；服务端状态未知。不要直接重发控制请求或开始下一窗口。')
        raise
    finally:
        finished.set()
        thread.join(timeout=1)
        result['elapsed_seconds'] = time.monotonic()-begin
        save(record_path, result)


def batch(client, prompts, out_len, n, c, phase, label, case_deadline, output):
    records = []
    sent = 0
    executor = futures.ThreadPoolExecutor(max_workers=c)
    pending = {}
    begin = time.perf_counter()
    abort = threading.Event()
    def deadline_abort():
        abort.set()
        client.abort()
    timer = threading.Timer(case_deadline, deadline_abort)
    timer.daemon = True
    timer.start()
    try:
        def submit(index):
            remaining = case_deadline-(time.perf_counter()-begin)
            if remaining <= 0 or abort.is_set():
                raise TimeoutError('当前 batch 超时；不再提交请求。')
            rid = f'http-{RUN_ID}-{phase}-{label}-{index}'
            future = executor.submit(client.complete, prompts[index % len(prompts)], out_len,
                                     min(ARGS.request_timeout, remaining), rid)
            pending[future] = index
        for _ in range(min(c, n)):
            submit(sent)
            sent += 1
        while pending:
            remaining = case_deadline-(time.perf_counter()-begin)
            if remaining <= 0:
                raise TimeoutError(f'{phase}/{label} batch 超过 {case_deadline:g}s')
            complete, _ = futures.wait(pending, timeout=min(20, remaining), return_when=futures.FIRST_COMPLETED)
            if not complete:
                log(f'{phase}/{label}: 已发 {sent}/{n}，等待在途请求，不额外增加并发。')
                continue
            # Consume all completed futures before replacing any failed slot.
            completed_records = []
            for future in complete:
                pending.pop(future)
                completed_records.append(future.result())
            for record in completed_records:
                records.append(record)
                tpot = record['tpot_stream_est_ms']
                log(f'{phase}/{label}: {len(records)}/{n}; {record["prompt_tokens"]}->{record["completion_tokens"]}; '
                    f'TTFT={record["ttft_stream_ms"]:.3f}ms; E2E={record["e2e_ms"]:.3f}ms; '
                    f'TPOT_est={"N/A" if tpot is None else f"{tpot:.3f}ms"}')
            while len(pending) < c and sent < n:
                submit(sent)
                sent += 1
        return records, time.perf_counter()-begin
    except BaseException:
        client.abort()
        for future in pending:
            future.cancel()
        save(output / f'{phase}_{label}_partial.json', {'successful_requests': records, 'sent': sent})
        raise
    finally:
        timer.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

def tokenize_prompts(client, output):
    # One server tokenize call; tokenization is completed before any capture window.
    text = '\n'.join(
        f'Record {i}: function inspect_{i}(items) computes a sum, compares indices, '
        f'and checks a boundary. The system stores a document with value {i*17+13}. '
        'Explain the result carefully and write a correct Python function.'
        for i in range(700)
    )
    log('POST /tokenize：使用已运行服务的 tokenizer，客户端不加载模型或 tokenizer。')
    tokenized = client.json('POST', '/tokenize', {
        'model': ARGS.model, 'prompt': text, 'add_special_tokens': False,
    }, timeout=30)
    tokens = tokenized.get('tokens') if isinstance(tokenized, dict) else None
    needed = max(CASES[name][0] for name in SELECTED)+3*127
    if not tokens or not all(type(t) is int and t >= 0 for t in tokens) or len(tokens) < needed:
        raise RuntimeError(f'/tokenize 返回 token 不足/无效；需要至少 {needed}，不自动猜测或重试。')
    max_len = tokenized.get('max_model_len')
    if max_len and max(CASES[name][0]+CASES[name][1] for name in SELECTED) > max_len:
        raise RuntimeError('选定测试长度超过服务 max_model_len。')
    prompts = {name: [tokens[j*127:j*127+CASES[name][0]] for j in range(CASES[name][2])]
               for name in SELECTED}
    save(output / 'prompts.json', {'source': 'fixed_text_server_tokenize_v1', 'model': ARGS.model,
         'source_sha256': hashlib.sha256(text.encode()).hexdigest(), 'prompts': prompts})
    log('固定长度 token ID 输入已构造并保存。')
    return prompts

def write_summary(output, rows):
    fields = ['phase', 'case', 'input_tokens', 'output_tokens', 'concurrency', 'requests',
              'ttft_stream_ms_mean', 'tpot_stream_est_ms_mean', 'e2e_ms_mean',
              'batch_wall_seconds', 'output_tokens_per_second']
    with (output/'summary_http.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    global ARGS, SELECTED, RUN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8972')
    parser.add_argument('--model', default='GLM5.2-NVFP4')
    parser.add_argument('--case', choices=['all']+list(CASES), default='P512')
    parser.add_argument('--phase', choices=['timing', 'trace', 'all'], default='timing')
    parser.add_argument('--request-timeout', type=float, default=60)
    parser.add_argument('--batch-timeout', type=float, default=180)
    parser.add_argument('--profile-timeout', type=float, default=300,
                        help='Profiler control timeout only; request/batch bounds unchanged.')
    ARGS = parser.parse_args()
    if not ARGS.run_dir.is_dir():
        parser.error('run-dir 必须是当前已启动模型的实验目录；不会创建新的模型运行目录。')
    if not 1 <= ARGS.request_timeout <= 180 or not 1 <= ARGS.batch_timeout <= 600:
        parser.error('request-timeout 范围为 1..180 秒；batch-timeout 范围为 1..600 秒。')
    if not 1 <= ARGS.profile_timeout <= 600:
        parser.error('profile-timeout 范围为 1..600 秒；不会改变生成请求超时。')
    SELECTED = list(CASES) if ARGS.case == 'all' else [ARGS.case]
    RUN_ID = dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    output = ARGS.run_dir / ('http_only_'+RUN_ID)
    output.mkdir()
    save(output/'config.json', {**vars(ARGS), 'run_dir': str(ARGS.run_dir), 'cases': SELECTED,
        'python': sys.executable, 'stdlib_only': True, 'prefix_policy': 'unique cache_salt per request',
        'workload': 'fixed text token IDs, NOT vllm bench random dataset',
        'timing_note': 'client streaming observations; server remains under nsys',
        'token_id_return': True})
    client = Client(ARGS.base_url, os.environ.get('OPENAI_API_KEY', ''))
    rows = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        log(f'结果目录：{output}')
        log('客户端仅标准库；不会导入 vllm/transformers/torch，也不会启动或重启模型。')
        client.raw('GET', '/health', timeout=5)
        models = client.json('GET', '/v1/models', timeout=5)
        if ARGS.model not in [x.get('id') for x in models.get('data', [])]:
            raise RuntimeError('模型别名未出现在 /v1/models 中。')
        client.idle(output, 'initial')
        prompts = tokenize_prompts(client, output)
        phases = ['timing', 'trace'] if ARGS.phase == 'all' else [ARGS.phase]
        for phase in phases:
            for name in SELECTED:
                length, out_len, concurrency = CASES[name]
                client.idle(output, f'{phase}_{name}_before')
                log(f'{phase}/{name}：{length}->{out_len}, C<={concurrency}; 先预热 {concurrency} 条，采集尚未开启。')
                warmups, _ = batch(client, prompts[name], out_len, concurrency, concurrency,
                                   phase+'_warmup', name, ARGS.batch_timeout, output)
                save(output/f'{phase}_{name}_warmup.json', warmups)
                client.idle(output, f'{phase}_{name}_after_warmup')
                start_completed = False
                try:
                    if phase == 'trace':
                        save(output/f'{phase}_{name}_capture.json', {'start_attempt_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
                        profile_control(client, '/start_profile', ARGS.profile_timeout,
                                        output/f'{phase}_{name}_start_control.json')
                        start_completed = True
                    n = concurrency if phase == 'trace' else 3*concurrency
                    records, wall = batch(client, prompts[name], out_len, n, concurrency,
                                          phase, name, ARGS.batch_timeout, output)
                    save(output/f'{phase}_{name}.json', {'requests': records, 'batch_wall_seconds': wall})
                finally:
                    if start_completed:
                        log(f'{name}: 生成阶段已结束；停止采集，不关闭 API 服务。')
                        profile_control(client, '/stop_profile', ARGS.profile_timeout,
                                        output/f'{phase}_{name}_stop_control.json')
                        save(output/f'{phase}_{name}_capture_stopped.json', {'stop_returned_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
                row = {'phase': phase, 'case': name, 'input_tokens': length, 'output_tokens': out_len,
                       'concurrency': concurrency, 'requests': n,
                       'ttft_stream_ms_mean': statistics.mean(r['ttft_stream_ms'] for r in records),
                       'tpot_stream_est_ms_mean': statistics.mean(r['tpot_stream_est_ms'] for r in records) if out_len>1 else '',
                       'e2e_ms_mean': statistics.mean(r['e2e_ms'] for r in records),
                       'batch_wall_seconds': wall, 'output_tokens_per_second': n*out_len/wall}
                rows.append(row)
                write_summary(output, rows)
                client.idle(output, f'{phase}_{name}_after')
        log(f'全部完成。汇总：{output / "summary_http.csv"}')
        log('API 未停止。全部 trace 完成后，再到原 nsys 服务终端按一次 Ctrl+C 正常导出报告。')
        return 0
    except BaseException as exc:
        client.abort()
        save(output/'error.json', {'type': type(exc).__name__, 'error': str(exc)})
        log(f'停止：{type(exc).__name__}: {exc}')
        log('不重试、不继续叠加请求、不重启模型。客户端断开不保证恢复服务端 kernel/驱动死锁。')
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

if __name__ == '__main__':
    raise SystemExit(main())
