#!/usr/bin/env python3
"""只验证采集兼容性；此程序的性能不代表 GLM，且不加载模型。"""
import time
import torch

if not torch.cuda.is_available():
    raise SystemExit('当前 Python 环境没有可用 CUDA。')
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
for _ in range(5):
    torch.mm(a, b, out=c)
torch.cuda.synchronize()
# 用实际 CUDA Graph 验证 node tracing；峰值显存远低于加载模型。
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    torch.mm(a, b, out=c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
try:
    end = time.monotonic() + 0.2
    while time.monotonic() < end:
        graph.replay()
        torch.cuda.synchronize()
finally:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
print('CUDA graph smoke finished; this is not a GLM benchmark.')
