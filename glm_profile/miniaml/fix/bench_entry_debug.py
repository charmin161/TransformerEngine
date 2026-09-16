#!/usr/bin/env python3
"""Diagnostic-only vLLM benchmark launcher; does not start an API/model.
Uses standard-library logging around the existing local vLLM implementation.
"""
from __future__ import annotations
import faulthandler
import functools
import importlib
import os
import sys
import time

T0 = time.monotonic()

def mark(text: str) -> None:
    print(f'[bench-debug +{time.monotonic()-T0:.3f}s] {text}', flush=True)

def wrap_sync(module, name: str) -> None:
    original = getattr(module, name, None)
    if not callable(original):
        mark(f'Optional stage hook not available: {name}')
        return
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        start = time.monotonic()
        mark(f'BEGIN {name}')
        try:
            value = original(*args, **kwargs)
        except BaseException:
            mark(f'ERROR {name}; elapsed={time.monotonic()-start:.3f}s')
            raise
        mark(f'END {name}; elapsed={time.monotonic()-start:.3f}s')
        return value
    setattr(module, name, wrapped)

def wrap_async(module, name: str) -> None:
    original = getattr(module, name, None)
    if not callable(original):
        mark(f'Optional stage hook not available: {name}')
        return
    @functools.wraps(original)
    async def wrapped(*args, **kwargs):
        start = time.monotonic()
        mark(f'BEGIN {name}')
        try:
            value = await original(*args, **kwargs)
        except BaseException:
            mark(f'ERROR {name}; elapsed={time.monotonic()-start:.3f}s')
            raise
        mark(f'END {name}; elapsed={time.monotonic()-start:.3f}s')
        return value
    setattr(module, name, wrapped)

def main() -> None:
    # Refuse to launch a model or another CLI subcommand through this wrapper.
    if sys.argv[1:3] != ['bench', 'serve']:
        raise SystemExit('This launcher only accepts: bench serve [existing flags]')
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(line_buffering=True, write_through=True)
    seconds = float(os.environ.get('BENCH_DUMP_SECONDS', '45'))
    if not 1 <= seconds <= 3600:
        raise SystemExit('BENCH_DUMP_SECONDS must be between 1 and 3600')
    mark(f'PID={os.getpid()}; Python={sys.executable}')
    mark(f'Python stack snapshot every {seconds:g}s; this does NOT terminate the process')
    # Timer covers module imports, local tokenizer loading and HTTP waits.
    # A dump containing selectors/asyncio is not, by itself, proof of a server hang.
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(seconds, repeat=True, file=sys.stderr, exit=False)
    try:
        mark('BEGIN import vLLM CLI')
        cli = importlib.import_module('vllm.entrypoints.cli.main')
        mark('END import vLLM CLI')
        mark('BEGIN import vLLM benchmark module')
        bench = importlib.import_module('vllm.benchmarks.serve')
        mark('END import vLLM benchmark module')
        for name in ('get_tokenizer', 'get_samples'):
            wrap_sync(bench, name)
        for name in ('_align_prompts_to_server_tokenizer', 'benchmark'):
            wrap_async(bench, name)
        # Keep the original CLI parsing and benchmark implementation.
        sys.argv[0] = 'vllm'
        mark('BEGIN CLI main (remaining imports, argument parsing, benchmark)')
        cli.main()
        mark('END CLI main')
    finally:
        faulthandler.cancel_dump_traceback_later()

if __name__ == '__main__':
    main()
