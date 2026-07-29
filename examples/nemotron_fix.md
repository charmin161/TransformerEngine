## 结论：回到“已经能稳定推理”的旧环境，在它上面做源码覆盖和最小补丁

不要继续用当前新环境做这次实验。

你现在的目标是评测 **attention score 2:4 裁剪对下游任务准确率、pass rate、生成结果的影响**，而不是验证新版 SGLang。对于这种实验，最重要的是固定其余变量。新环境连原始模型推理都可能因为 FlashInfer、CUBIN、SGLang kernel 或 autotune 链路发生段错误，即使最终勉强启动，也无法确认结果差异来自 2:4，还是来自依赖栈变化。

SGLang 的依赖确实是紧耦合的。例如官方 `v0.5.13` 明确固定了 `flashinfer_python==0.6.12`、`flashinfer_cubin==0.6.12`、`sglang-kernel==0.4.3`；当前 `main` 已经变成 `flashinfer_python==0.6.15.post1`、`sglang-kernel==0.4.5`，并同时升级了 CUTLASS DSL、Transformers 等组件。混用新代码与旧二进制依赖很容易产生难以定位的 JIT、ABI 或预编译 CUBIN 问题。

我不能仅凭你现在的信息确认新环境的段错误一定发生在 FlashInfer autotune context，但**没有必要先解决它**。这不是完成 2:4 质量实验的最短路径。

| 方案                                   | 已知基础是否稳定 | Attention 是否容易修改 |     实验可信度 | 建议              |
| ------------------------------------ | -------: | ---------------: | --------: | --------------- |
| 新 SGLang 环境继续排段错误                    |        否 |                是 |         低 | 放弃当前路线          |
| 旧环境直接修改 FlashInfer                   |        是 |   否，核心 kernel 难改 |         中 | 不选              |
| 旧环境 + 放开 Triton + 修改旧版 Triton kernel |        是 |                是 |         高 | **推荐**          |
| 旧环境 + `torch_native` 显式 attention    |        是 |                是 | 低，550B 太慢 | 仅作小规模 reference |

---

# 一、不要修改原来的旧环境，建立 Python 源码覆盖层

不需要复制整个虚拟环境，也不要在旧环境里执行新的 `pip install sglang`。最稳妥的方法是：

* 继续使用旧环境中的 Python、PyTorch、FlashInfer、ModelOpt、SGLang kernel；
* 只复制旧环境里已经安装的 `sglang` Python 包；
* 通过 `PYTHONPATH` 优先加载复制出来的可修改源码。

假设旧环境是之前堆栈中的：

```bash
OLD_ENV=/wireless/minyusong/nemotron_3_ultra/sglang_nemotron
PYTHON="$OLD_ENV/bin/python"

OVERLAY=/wireless/minyusong/nemotron_3_ultra/sglang_old_2of4_overlay
```

先记录当前工作环境版本：

```bash
"$PYTHON" - <<'PY'
import importlib.metadata as md
import torch

packages = [
    "sglang",
    "flashinfer-python",
    "flashinfer-cubin",
    "sglang-kernel",
    "torch",
    "transformers",
    "nvidia-modelopt",
]

for name in packages:
    try:
        print(f"{name:24s} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name:24s} NOT INSTALLED")

print("torch cuda:", torch.version.cuda)
PY
```

复制实际加载的 SGLang 包：

```bash
SGLANG_PKG=$(
  "$PYTHON" - <<'PY'
from pathlib import Path
import sglang
print(Path(sglang.__file__).resolve().parent)
PY
)

echo "Original SGLang package: $SGLANG_PKG"

rm -rf "$OVERLAY"
mkdir -p "$OVERLAY"
cp -a "$SGLANG_PKG" "$OVERLAY/sglang"
```

启用覆盖：

```bash
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
```

确认确实加载复制后的代码：

```bash
"$PYTHON" - <<'PY'
import sglang
from sglang.srt.layers.attention import triton_backend
from sglang.srt.layers.attention.triton_ops import (
    extend_attention,
    decode_attention,
)

print("sglang          :", sglang.__file__)
print("triton_backend  :", triton_backend.__file__)
print("extend_attention:", extend_attention.__file__)
print("decode_attention:", decode_attention.__file__)
PY
```

预期路径都应位于：

```text
/wireless/minyusong/nemotron_3_ultra/sglang_old_2of4_overlay/sglang/
```

这样：

```text
旧环境原件                         保持不动，可随时恢复原始推理
sglang_old_2of4_overlay           专门做 Triton 和 2:4 修改
PyTorch/FlashInfer/ModelOpt/CUBIN 仍然使用旧环境中已经验证过的版本
```

---

# 二、只补旧环境中的两个 Triton 兼容点

你旧环境报错路径是：

```text
sglang/srt/arg_groups/nemotron_h_hook.py
```

说明应该修改旧版结构，而不是把新版 `sglang/kernels/ops/attention/` 文件复制回来。

## 补丁 1：放开 NemotronH 的 Triton 限制

文件：

```text
$OVERLAY/sglang/srt/arg_groups/nemotron_h_hook.py
```

找到：

```python
assert server_args.attention_backend != "triton", (
    "NemotronHForCausalLM does not support triton attention backend,"
    "as the first layer might not be an attention layer"
)
```

改成：

```python
if server_args.attention_backend == "triton":
    logger.warning(
        "Experimental: allowing Triton full-attention backend for NemotronH."
    )
```

这只是取消保守检查，还必须做第二个补丁。

## 补丁 2：不要假设第 0 层是 full attention

文件：

```text
$OVERLAY/sglang/srt/layers/attention/triton_backend.py
```

找到类似：

```python
elif (
    model_runner.hybrid_gdn_config is not None
    or model_runner.kimi_linear_config is not None
    or model_runner.linear_attn_model_spec is not None
):
    self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
    self.swa_v_head_dim = None
else:
    self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
    self.swa_v_head_dim = None
```

改为更稳健的版本：

```python
elif (
    getattr(model_runner, "mamba2_config", None) is not None
    or getattr(model_runner, "hybrid_gdn_config", None) is not None
    or getattr(model_runner, "kimi_linear_config", None) is not None
    or getattr(model_runner, "linear_attn_model_spec", None) is not None
):
    # Hybrid models may begin with a Mamba/linear-attention layer.
    # Do not assume global layer 0 owns an MHA KV buffer.
    self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
    self.swa_v_head_dim = None
else:
    self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
    self.swa_v_head_dim = None
```

这两个补丁解决的是两个不同层次的问题：

```text
nemotron_h_hook.py
    解决：禁止启动

triton_backend.py
    解决：启动后错误地读取 layer 0 的 KV buffer
```

不能只注释断言。

---

# 三、继续使用旧版本自身的 Triton kernel

旧环境应修改：

```text
$OVERLAY/sglang/srt/layers/attention/triton_ops/extend_attention.py
$OVERLAY/sglang/srt/layers/attention/triton_ops/decode_attention.py
```

不要把新版的：

```text
sglang/kernels/ops/attention/extend_attention.py
sglang/kernels/ops/attention/decode_attention.py
```

整体复制到旧环境。新版已经增加了 page-aware strides、score modifiers、不同的 CUDA Graph 接口和额外参数，直接移植会引入新的不兼容。

你只需要把相同的数学逻辑重新应用到旧版文件：

```text
qk
→ 原 causal/padding/custom mask
→ 每4个 score 删除最小的2个
→ online softmax
→ PV
```

旧版必须修改的三个位置是：

```text
extend_attention.py::_fwd_kernel
    1. prefix score 分支
    2. current-extend score 分支

decode_attention.py
    3. Nemotron GQA grouped decode kernel
```

若旧版中存在 deterministic unified kernel，而你没有开启 deterministic inference，可以暂时不改它。

---

# 四、旧环境中的 FlashInfer 不需要删除，也不需要修改

切换 full attention 到 Triton，并不意味着整个模型完全不使用 FlashInfer。

推荐保持：

```text
12个 full-attention 层  → 修改后的 Triton attention
Mamba-2 层              → 旧环境原本可工作的 Mamba backend
NVFP4 MoE                → 旧环境原本可工作的 flashinfer_trtllm
```

启动参数中可以继续保留：

```bash
--mamba-backend flashinfer
--moe-runner-backend flashinfer_trtllm
```

你只是在 hybrid wrapper 的 full-attention 分支中使用 Triton，并没有触碰已验证可用的 NVFP4 MoE 和 Mamba 实现。

这正是旧环境的价值：不需要重新验证 550B 权重加载、ModelOpt NVFP4、MoE runner、Mamba state cache 和 FlashInfer binary。

---

# 五、先做一次最小 smoke test，再打开 CUDA Graph

## 阶段 1：只验证 Triton dense 能运行

第一轮先关闭 2:4，同时临时关闭 CUDA Graph，避免在 graph capture 阶段同时调试：

```bash
export SGLANG_ATTN_2OF4=0
rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
```

在原来已经成功的启动命令上，只改变或增加：

```bash
--attention-backend triton \
--page-size 1 \
--kv-cache-dtype bf16 \
--disable-radix-cache \
--chunked-prefill-size -1 \
--disable-cuda-graph \
--disable-piecewise-cuda-graph
```

并暂时删除 speculative decoding 参数。

只发送一个：

```text
短 prompt
max_tokens=2
temperature=0
```

成功标准：

```text
服务加载完成
出现 Triton extend 日志
出现 grouped GQA decode 日志
能够返回2个token
```

这一步不是正式评测，只用于证明补丁和 kernel 路由成立。

## 阶段 2：打开 decode CUDA Graph

Smoke test 成功后，停止服务，删除：

```bash
--disable-cuda-graph
```

可以继续保留：

```bash
--disable-piecewise-cuda-graph
```

这样通常可以避免 prefill piecewise graph 带来的额外复杂性，同时保留正常 decode CUDA Graph。

SGLang 的 attention backend 在 CUDA Graph 下并不会被跳过；捕获阶段记录实际 Triton kernel launch，replay 时再次执行该 kernel。官方文档也明确要求 attention backend 实现 capture/replay metadata，并说明 decode backend 会被 CUDA Graph 捕获。([GitHub][1])

因此你的：

```text
QK
→ 4选2 mask
→ softmax
→ PV
```

全部仍会在 graph replay 中执行。

Python `print()` 可能只在 warmup 或 capture 阶段出现，而不会每个 token 都重复出现；这只是因为 replay 不再重新执行 Python 调度，并不代表 attention 被跳过。

---

# 六、正式下游评测应比较 Triton dense 与 Triton 2:4

不要直接把：

```text
旧环境 FlashInfer dense
```

和：

```text
Triton 2:4
```

作为唯一对照，因为其中同时改变了 backend 和 2:4 策略。

更干净的实验是：

### A：Triton dense

```bash
export SGLANG_ATTN_2OF4=0
```

### B：Triton 2:4

```bash
export SGLANG_ATTN_2OF4=1
```

两组保持完全相同：

```text
checkpoint
TP/EP
KV cache dtype
CUDA Graph batch sizes
prompt
chat template
temperature
seed
max_tokens
任务 grader
```

正式评测建议先关闭 speculative decoding，因为 speculative acceptance 变化会额外影响生成过程。先测 target model 自身的 2:4 质量，之后再单独测 MTP/EAGLE acceptance rate。

最快的执行顺序是：

1. 原始旧环境 FlashInfer：已有可运行性基准，不再修改。
2. Triton dense：先跑 20～50 个样本，确认它与 FlashInfer 没有明显任务级偏差。
3. Triton dense：跑完整下游基准，得到严格对照分。
4. Triton 2:4：跑相同完整基准。
5. 最后再重新开启 speculative decoding，检查 acceptance rate 和最终任务分数。

---

# 七、为什么先关闭 Radix cache 和 chunked prefill

你的旧版 extend kernel 将 attention 拆成：

```text
prefix score
+
current extend score
```

若 prefix 长度不是 4 的倍数，例如：

```text
prefix_len = 6
```

全局 quartet 应为：

```text
[0,1,2,3]
[4,5,6,7]
[8,9,10,11]
```

但两个阶段分别分组会得到：

```text
prefix：[0,1,2,3] [4,5,...]
extend：[6,7,8,9] ...
```

这不再是严格的全局 K 维 2:4。

因此第一轮下游质量评测使用：

```bash
--disable-radix-cache
--chunked-prefill-size -1
```

使初始 prefill 的：

```text
prefix_len = 0
```

从 K=0 开始严格每四个一组。

这不会关闭 decode CUDA Graph，只会牺牲 prefix reuse 和 chunked prefill。

---

# 八、不要把这次结果解释成 Rubin 的真实速度

当前仿真只做了：

```text
一半 score → -inf
一半 softmax probability → 0
```

后面的代码仍然：

```python
加载全部 V
执行 dense P @ V
```

因此它适合测：

```text
下游准确率
pass rate
perplexity/NLL
生成结果分叉
重复与退化
```

不适合测：

```text
Rubin Sparse MMA 的真实加速
V 读取减少
端到端吞吐提升
```

这个实现甚至可能因额外的两次 `argmin` 而变慢。

---

## 最终抉择

采用：

```text
旧的、已经证明能推理的依赖环境
+
独立 PYTHONPATH 源码覆盖层
+
放开 NemotronH Triton 断言
+
修复 Triton 对 Mamba hybrid 的 layer-0 假设
+
修改旧版 Triton score kernel
```

放弃：

```text
继续在当前 main 环境中排查 FlashInfer/autotune 段错误
```

这是变量最少、可回滚、最快得到可信下游质量结果的路线。

[1]: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md?utm_source=chatgpt.com "sglang/docs/advanced_features/attention_backend.md at main · sgl-project/sglang · GitHub"
