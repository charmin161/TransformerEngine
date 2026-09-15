#!/usr/bin/env python3
"""Small vLLM 0.26 offline diagnostic; not an OpenCompass/HumanEval evaluation.

Supply a JSON object containing the COMPLETE LLM engine kwargs from the working
configuration (including model and tensor_parallel_size). The program never
imports an OpenCompass config or guesses its non-default engine settings.

Only --disable-prefix-cache and --profile explicitly override engine settings.
The measured time is blocking LLM.generate wall time, NOT streaming TTFT/TPOT.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer.')
    return result


def parse_case(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(x) for x in value.split(':'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use INPUT:OUTPUT:BATCH, e.g. 512:256:1') from exc
    if len(parts) != 3 or min(parts) < 1:
        raise argparse.ArgumentTypeError('Use three positive integers: INPUT:OUTPUT:BATCH')
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-config', type=Path, required=True)
    parser.add_argument('--case', type=parse_case, action='append', required=True)
    parser.add_argument('--warmups', type=positive_int, default=2)
    parser.add_argument('--repeats', type=positive_int, default=3)
    parser.add_argument('--disable-prefix-cache', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Call worker CUDA profiler APIs around each measured case.')
    parser.add_argument('--result', type=Path, default=Path('bench/offline.json'))
    args = parser.parse_args()
    if args.result.exists():
        parser.error(f'Result already exists; choose a new --result: {args.result}')
    if args.profile and args.repeats != 1:
        parser.error('Use --repeats 1 with --profile to keep traces small.')
    config = json.loads(args.engine_config.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not isinstance(config.get('model'), str):
        parser.error('Engine JSON must be an object with a string model field.')
    if config.get('tensor_parallel_size') != 4:
        parser.error('For this 4-GPU diagnostic, explicitly set tensor_parallel_size to 4.')
    if not Path(config['model']).is_dir():
        parser.error('model must point to your existing LOCAL model directory.')
    if args.disable_prefix_cache:
        config['enable_prefix_caching'] = False
    if config.get('enable_prefix_caching') is not False:
        parser.error('Controlled repeated prompts require --disable-prefix-cache or explicit enable_prefix_caching=false.')
    if args.profile:
        config['profiler_config'] = {'profiler': 'cuda'}
    elif config.get('profiler_config'):
        parser.error('Remove profiler_config for unprofiled timing, or use --profile.')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    from vllm import LLM, SamplingParams
    from importlib.metadata import version
    import torch

    llm = LLM(**config)
    tokenizer = llm.get_tokenizer()
    special_ids = set(tokenizer.all_special_ids)
    vocab = sorted(set(tokenizer.get_vocab().values()) - special_ids)
    if not vocab:
        raise RuntimeError('No non-special tokenizer IDs found.')
    result = {
        'vllm_version': version('vllm'),
        'profiled': args.profile,
        'timing_scope': 'blocking LLM.generate wall time; excludes model load, prompt construction and warmups',
        'workload': 'synthetic token IDs; not representative of HumanEval routing or quality',
        'engine_kwargs': config,
        'worker_multiproc_method': os.environ['VLLM_WORKER_MULTIPROC_METHOD'],
        'cases': [],
    }
    for index, (isl, osl, batch) in enumerate(args.case):
        rng = random.Random(12345 + index)
        prompts = [{'prompt_token_ids': rng.choices(vocab, k=isl)} for _ in range(batch)]
        sampling = SamplingParams(temperature=0.0, max_tokens=osl, ignore_eos=True)
        label = f'i{isl}_o{osl}_b{batch}'
        for _ in range(args.warmups):
            llm.generate(prompts, sampling, use_tqdm=False)
        durations: list[float] = []
        counts: list[int] = []
        if args.profile:
            llm.start_profile()
        try:
            torch.cuda.nvtx.range_push(label)
            try:
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    outputs = llm.generate(prompts, sampling, use_tqdm=False)
                    duration = time.perf_counter() - start
                    actual_counts = [len(item.outputs[0].token_ids) for item in outputs]
                    if len(actual_counts) != batch or any(n != osl for n in actual_counts):
                        raise RuntimeError(f'Unexpected output token counts: {actual_counts}; expected {batch} x {osl}.')
                    durations.append(duration)
                    counts.append(sum(actual_counts))
            finally:
                torch.cuda.nvtx.range_pop()
        finally:
            if args.profile:
                llm.stop_profile()
        row = {
            'label': label, 'input_tokens_per_request': isl,
            'output_tokens_per_request': osl, 'submitted_batch_size': batch,
            'actual_output_tokens_total_per_run': counts,
            'latency_seconds_per_run': durations,
            'median_latency_seconds': statistics.median(durations),
            'output_tokens_per_second_per_run': [n / t for n, t in zip(counts, durations)],
        }
        result['cases'].append(row)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f'Results: {args.result}. Profiled timings are diagnostic only, not benchmark scores.')


if __name__ == '__main__':
    main()
