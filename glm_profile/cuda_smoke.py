#!/usr/bin/env python3
"""Verify CUDA tracing and counter access; NOT a GLM performance benchmark."""
import sys

def main() -> int:
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the active Python environment.')
        torch.manual_seed(0)
        x = torch.randn((2048, 2048), device='cuda:0', dtype=torch.bfloat16)
        for _ in range(3):
            y = x @ x
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            for _ in range(10):
                y = x @ x
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        print('CUDA smoke test complete:', tuple(y.shape))
        return 0
    except Exception as exc:
        print(f'CUDA smoke test failed: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
