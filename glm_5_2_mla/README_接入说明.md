# GLM-5.2 / vLLM 0.29.0：主 Q 在原生 FP8 转换前做 MXFP8 伪量化

## 实验定义与边界

保留原生 FP8 KV cache、原生 FP8 attention、原始 q_scale/k_scale，以及已有 K NVFP4 伪量化。
新增路径是：高精度 absorbed NoPE + RoPE 后 PE -> 每 32 维 MXFP8 量化/反量化 -> 原生 E4M3 FP8 打包。
这不是纯 MXFP8 QK，更不是原生 MXFP8×NVFP4 MMA。

适用范围：用户此前的 GLM-5.2-NVFP4、vLLM 0.29.0、FlashInfer sparse MLA、TP=8 跨两节点。
当前接入针对普通 TP、无 PCP/DCP/投机解码，eager 或 breakable PIECEWISE。
不声称支持 FULL CUDA Graph、其他模型或其他 attention backend。

本实现不改 fused_q / CuTeDSL / indexer，也不修改原始 ql_nope、q_pe。
在图外 eager attention 区域，重新从高精度源计算 Q，只覆盖既有 FP8 Q 输出中的实际 decode 行。
旧 FP8 Q 是输出缓冲区，不作为输入读取；所以不是先 FP8 再 MXFP8。
主路径和混合 batch 的通用 MLA 回退路径都要接入，不能只改其中一个。

验证：CPU 数学参考测试与 Python 语法检查已执行；CUDA/Triton 不可用，GPU 编译、GPU 测试、模型评测、图重放均未执行。见 validation.json。

## 0. 先撤掉上一版错误的后端量化入口

在 flashinfer_mla_sparse.py / FlashInferMLASparseImpl.forward_mqa 中，删除上一版新增的：
- 要求 Q/cache 为 BF16 的 dtype 检查；
- 对现有 q 调用 fake_quant_mxfp8_decode_q_ 的代码；
- 不再使用的该量化器 import。

保留原生实现的检查，不要删除 vLLM 原有逻辑。
保留你在 common/kernels.py 中的 K NVFP4 代码，本方案不改该文件。
不要再添加 --kv-cache-dtype bfloat16；之前为了旧方案加过的这一覆盖需撤掉，恢复原来的 FP8 路径。

## 1. 安装新 helper

把 mxfp8_q_prepack.py 放到：

```text
vllm/v1/attention/ops/mxfp8_q_prepack.py
```

后续三个源码文件顶部均增加：

```python
from vllm.v1.attention.ops.mxfp8_q_prepack import (
    Q_MXFP8_MODE,
    repack_mxfp8_decode_q_,
)
```

这三个文件是：

```text
vllm/models/deepseek_v32/attention.py
vllm/model_executor/layers/attention/mla_attention.py
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
```

其中 backend 文件其实只需要导入 Q_MXFP8_MODE，repack helper 不在 backend 调用。

## 2. 传递真实请求阶段：FlashInfer metadata

在 FlashInferMLASparseMetadata 的默认字段末尾增加（已有就不重复）：

```python
    q_fq_is_prefilling: torch.Tensor | None = None
```

在 FlashInferMLASparseMetadataBuilder 中增加以下两个方法。
已有上一版 build 方法时，替换该方法，不要保留两个同名定义。
TRTLLM builder 会继承这些方法。

```python
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> FlashInferMLASparseMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        if Q_MXFP8_MODE == "off":
            return metadata
        phase = common_attn_metadata.is_prefilling
        if phase is None:
            raise RuntimeError(
                "Real requests require is_prefilling; do not infer phase "
                "from query length, position, or num_decode_tokens"
            )
        metadata.q_fq_is_prefilling = phase.to(
            device=metadata.req_id_per_token.device,
            dtype=torch.bool,
        )
        return metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> FlashInferMLASparseMetadata:
        # Only synthetic capture metadata may use a dummy phase vector.
        # Runtime build still requires the real request phase.
        if Q_MXFP8_MODE != "off" and common_attn_metadata.is_prefilling is None:
            common_attn_metadata = common_attn_metadata.replace(
                is_prefilling=torch.zeros(
                    common_attn_metadata.num_reqs,
                    dtype=torch.bool,
                    device=self.device,
                )
            )
        return self.build(0, common_attn_metadata)
```

必须保证 phase 与 req_id_per_token 来自同一轮请求重排；原生 builder 使用同一 CommonAttentionMetadata。
这里不根据 query_len==1、slot_mapping、position 或 num_decode_tokens 猜测阶段。
phase 的每步 GPU 地址允许变化，仅因为下面 helper 在 eager breakpoint 内执行，不能把它挪进 FULL graph。

## 3. 主路径：DeepseekV32Attention._sparse_indexer_and_attn

文件 vllm/models/deepseek_v32/attention.py。
该方法已有 @eager_break_during_capture。
在 `_use_sparse_mha(...)` 分支的 `return` 后，现有这句之前插入：

```python
        if self._fp8_kv_needs_view:
```

新增代码：

```python
        if Q_MXFP8_MODE != "off":
            if not self._fp8_query:
                raise RuntimeError("This patch expects the original FP8 query/cache path")
            if self.use_pcp or self.impl.dcp_world_size != 1:
                raise RuntimeError("This integration is scoped to ordinary TP, without PCP/DCP")
            phase = getattr(attn_metadata, "q_fq_is_prefilling", None)
            if phase is None:
                raise RuntimeError("Missing MXFP8 request-phase metadata")
            repack_mxfp8_decode_q_(
                ql_nope,                 # original [T,H,512], NOT FP8
                q_pe,                    # original [T,H,64], PRE-RoPE
                mqa_q[:num_actual],      # output only, original FP8 carrier
                self._q_scale,
                attn_metadata.req_id_per_token[:num_actual],
                phase,
                positions=positions,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                apply_mxfp8=(Q_MXFP8_MODE == "on"),
            )
```

不要从 mqa_q 反量化回去。不修改 q_pe 原 tensor，所以 dense-MHA prefill 和其他引用不受原位污染。
主路径重做 RoPE 用 FP32 寄存器，不增加一次 BF16 中间写回。

## 4. 混合 batch 回退：MLAAttention.forward_impl

文件 vllm/model_executor/layers/attention/mla_attention.py。
混合 prefill/decode batch 可以由上面的 `_use_sparse_mha` 分支进入这里，decode 的 Q 会重新生成。
找到已有代码：

```python
            if fp8_attention and self.impl.supports_quant_query_input:
                assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]
                assert mqa_ql_nope.shape[1] == mqa_q_pe.shape[1]
                mqa_q = self._decode_concat_quant_fp8_op(
                    mqa_ql_nope, mqa_q_pe, self._q_scale
                )
            else:
                mqa_q = (mqa_ql_nope, mqa_q_pe)
```

在上述 `mqa_q = self._decode_concat_quant_fp8_op(...)` 之后、`else:` 之前加入：

```python
                if Q_MXFP8_MODE != "off" and hasattr(
                    attn_metadata, "q_fq_is_prefilling"
                ):
                    if self.use_pcp or self.impl.dcp_world_size != 1:
                        raise RuntimeError("MXFP8 integration requires ordinary TP")
                    phase = attn_metadata.q_fq_is_prefilling
                    if phase is None:
                        raise RuntimeError("Missing MXFP8 request-phase metadata")
                    repack_mxfp8_decode_q_(
                        mqa_ql_nope,       # absorbed 512-D high-precision source
                        mqa_q_pe,          # ALREADY rotated; no second RoPE
                        mqa_q,             # write-only FP8 output
                        self._q_scale,
                        attn_metadata.req_id_per_token[:mqa_q.shape[0]],
                        phase,
                        apply_mxfp8=(Q_MXFP8_MODE == "on"),
                    )
```

这里不传 positions/cos_sin_cache：mqa_q_pe 已经旋转过，而且可能已按原路径写回 BF16。
不重新对该 PE 做 RoPE，也不试图恢复原路径已经发生的 BF16 舍入。
只覆盖选择了 actual decode 的行，保留 short prefill/extend 的原生 FP8 Q。
该 guard 将此修改限制到有新增 FlashInfer phase 字段的 metadata。

## 5. 启动配置与对照实验

保留用户原先 FP8 cache 解析方式、TP=8、两节点、modelopt、expert parallel、batch size 和 PIECEWISE 配置。

两个节点/所有 worker 使用相同代码，并在启动前设置：

```bash
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_Q_MXFP8_MODE=control
```

`VLLM_Q_MXFP8_MODE` 是本补丁自定义环境变量，不是 vLLM 自带选项。
每次更改 mode 后重启所有相关服务进程。

- off：原生基线，不执行新重打包。
- control：重打包但不做 MXFP8，用来检查新 kernel 的 RoPE/除法/舍入与原生 CuTeDSL/Triton 的差异。
- on：同一个重打包 kernel，加 MXFP8。

先用少量相同输入对比 off 与 control，再比较 control 与 on。
两组对照的 q_scale、物理 cache dtype、K 伪量化、图模式和评测参数都保持不变。
不能保证 off/control bitwise 相等：CuTeDSL 的乘倒数与这里的除法、FMA、原生转换等可能在边界有差异。
不要在高频 forward 中增加 .item()/.cpu()/print 整个张量；诊断抓数限定少量 token/层。

本 patch 每层增加一个重打包 kernel（走到相应路径时）。它并不重新扫描历史 KV，也不扩充 FP8 cache。
不需要全局 --enforce-eager；helper 通过 stream-capture 检查拒绝在 FULL graph 捕获中运行。
PIECEWISE 正确重放仍需在目标环境验证，不以 CPU 测试替代。

## 6. 量化方向与语义

Q 为 [T,H,576]；用户 TP=8 时 H=8。
NoPE: [T,H,512] -> [T,H,16,32]。
PE: [T,H,64] -> [T,H,2,32]。
每个 token/head 分别沿最后维度求 amax，不跨 head/token。
512 恰好被 32 整除，所以独立量化这两部分与拼接后按连续 32 维量化等价。

PE 的 r1/r2 是偶数/奇数位置，必须先交错成 `[r1[0],r2[0],r1[1],r2[1],...]`。
分别对 32 维 r1/r2 求 scale 会产生另一种分组。

采用 E4M3 + E8M0：FP32 `amax * (1/448)` 向上舍入到 E8M0，E4M3 RNE，反量化回 FP32。
然后按原生 `_q_scale` 再转 E4M3。
原生 attention 的 bmm1_scale、bmm2_scale 和 K cache 均不更改。
实际执行路径仍是普通 FP8×FP8，不把每组 MX scale 传给 attention。
有些数值经两阶段转换可能与原生 FP8 一样；不能仅从最终 dtype 判断伪量化是否生效。

## 7. 测试

```bash
python test_cpu_reference.py
python test_gpu.py
```

后者必须在可用的 vLLM/Triton/CUDA 环境执行。
GPU 测试覆盖原始 PE / 已旋转 PE、control / on、非连续 Q 输入/输出、prefill 行不变和 CPU 参考一致性。
模型集成与 PIECEWISE capture/replay 另外验证，尤其使用混合 batch，检查两个入口均被覆盖。

## 核对依据

官方 vLLM v0.29.0（2026-10-08 访问）：
- vllm/models/deepseek_v32/attention.py
- vllm/models/deepseek_v32/common/kernels.py
- vllm/models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py
- vllm/model_executor/layers/attention/mla_attention.py
- vllm/model_executor/layers/attention/sparse_mla_attention.py
- vllm/v1/attention/backend.py
- vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
- vllm/compilation/breakable_cudagraph.py
NVIDIA TransformerEngine common/cast/mxfp8/quantize_mxfp8.cuh 的 scale 策略。

这些是公开版本，不是用户本地安装文件的完整副本。用户本地需按代码上下文定位，不能按旧行号硬替换。
