# vLLM 0.29.0：仅真实 decode 的 MLA Q MXFP8 伪量化

## 范围与验证边界

此接入示例针对 `FlashInferMLASparseImpl`（Blackwell 的 FlashInfer TRTLLM 稀疏 MLA 路径），不是原生 MXFP8×NVFP4 CUDA attention kernel。它修改最终 Q 的数值，返回/保留 BF16、FP16 或 FP32；不修改 attention 的归约、累加或 softmax 实现。

用户原有的 K/V NVFP4 伪量化需另外保留。本包不自动改动用户的源码。先在现有启动命令中指定：

```bash
--dtype bfloat16 --kv-cache-dtype bfloat16 --enforce-eager
```

首次验证不要启用投机解码。此版本的 metadata phase tensor 是按步分配的，**不应直接宣称支持 CUDA Graph replay**；本接入仅面向 eager 精度实验。之后做 CUDA Graph 版本时，需要 builder 持有固定地址的 phase buffer，每一步原位更新，并验证捕获、填充行和混合 batch。

当前已执行：Python 语法检查、CPU PyTorch reference 测试（见 `reference_test_results.json`）。当前环境没有 CUDA/Triton，因此未执行 Triton 编译、GPU kernel、vLLM 端到端测试。GPU 测试脚本是提供给目标环境运行的，不是已完成的验证。

## 1. 放置 quantizer 文件

把 `mxfp8_q_fake_quant.py` 放到源码：

```text
vllm/v1/attention/ops/mxfp8_q_fake_quant.py
```

不要同时在模型入口和后端入口量化一次，以免重复操作。

## 2. 传递真正的 prefill/decode 阶段

编辑：

```text
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
```

在 `FlashInferMLASparseMetadata` dataclass 的现有默认字段之后，追加：

```python
    q_fq_is_prefilling: torch.Tensor | None = None
```

在 `FlashInferMLASparseMetadataBuilder` 类中新增以下方法。Blackwell 的 `FlashInferMLASparseTRTLLMMetadataBuilder` 会继承此方法，不用再添加一遍：

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
        phase = common_attn_metadata.is_prefilling
        if phase is None:
            raise RuntimeError(
                "Strict decode-only Q fake quant needs CommonAttentionMetadata.is_prefilling"
            )
        metadata.q_fq_is_prefilling = phase.to(
            device=metadata.req_id_per_token.device,
            dtype=torch.bool,
        )
        return metadata
```

`is_prefilling=True` 表示请求尚在 prompt 处理阶段，不能因该请求走了 MQA kernel 就当作真实生成阶段。`req_id_per_token` 将 Q 的 packed token 行映射到请求。二者必须来自同一轮、同一 token/request 重排后的 metadata。缺失 phase 时直接报错，不静默按 shape 猜测阶段。

## 3. 在最终 Q 拼接后注入

在同一后端文件的 import 区新增：

```python
from vllm.v1.attention.ops.mxfp8_q_fake_quant import fake_quant_mxfp8_decode_q_
```

在 `FlashInferMLASparseImpl.forward_mqa` 中，现有 tuple-to-tensor 拼接完成之后、`num_actual_toks = q.shape[0]` 之前插入：

```python
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                f"Q is {q.dtype}; use BF16/FP16 physical cache for this fake-quant experiment"
            )
        if kv_c_and_k_pe_cache.dtype != q.dtype:
            raise TypeError(
                f"Q/cache dtype mismatch: {q.dtype} vs {kv_c_and_k_pe_cache.dtype}"
            )
        phase = attn_metadata.q_fq_is_prefilling
        if phase is None:
            raise RuntimeError("Q fake-quant phase metadata was not populated")
        fake_quant_mxfp8_decode_q_(
            q,
            attn_metadata.req_id_per_token[:q.shape[0]],
            phase,
        )
```

此处 Q 已完成 NoPE 吸收投影和 RoPE。`fused_q` 选择 Triton 或 CuTeDSL 都不影响这个后端入口。若实际运行的是其他 backend，上述文件不会生效，先在模型初始化完成后核对 `type(self.impl).__module__` 和 `type(self.impl).__name__`。

## 4. 维度和量化规则

GLM-5.2 配置：`q_lora_rank=2048`、`qk_nope_head_dim=192`、`qk_rope_head_dim=64`、`kv_lora_rank=512`、总 attention heads 为 64。

```text
q_nope: [T, H, 192]
W_UK_T: [H, 192, 512]
ql_nope: [T, H, 512]       # q_nope 吸收 W_UK_T 后
q_pe_rotated: [T, H, 64]
Q: [T, H, 576]            # 后端拼接后的输入
```

沿最后的 576 维分组：每一个 token、每一个 head 独立，18 组×32。前 16 组是 NoPE，后 2 组是连续交错 RoPE。不能沿 tokens/head 归约求 amax。输入非连续时使用真实 stride，不依赖 flatten/reshape 的物理连续性。

scale 策略为 NVIDIA TE 风格：FP32 `amax * (1/448)` 向上舍入为 E8M0 power-of-two scale，再将块内值缩放、E4M3 RNE、反量化。代码用位运算构造 scale，避免近似 log2 在边界选错指数。全零块返回零；含 NaN/Inf 的块保留原值，避免把上游异常藏掉。

E8M0 的有限 scale 编码为 `2**-127` 到 `2**127`，编码 0 不是零。本 helper 不存储压缩 payload 或真实 scale buffer。也不保证与任意第三方 MXFP8 量化器 bitwise 一致；量化器选择 scale 的规则应明确记录。

保持原有 attention softmax scale 不变，不能因为 latent QK 维度变为 576 而改成 `1/sqrt(576)`。

## 5. 原 K RoPE 分组修正（可选但建议单独对照）

用户原有 `r1`/`r2` 分别保存偶/奇通道。分别做 16 元素量化并不等于最终缓存连续 16 元素量化。要按最终缓存布局执行，替换两行独立 `r1`/`r2` 量化为：

```python
k_rope = tl.reshape(tl.join(r1, r2), (2 * KPE_HALF_ROT_DIM,))
k_rope = _nvfp4_fake_quant(k_rope, 1.0, 2 * KPE_HALF_ROT_DIM, 16)
r1, r2 = tl.split(tl.reshape(k_rope, (KPE_HALF_ROT_DIM, 2)))
```

保留 `kv_c = _nvfp4_fake_quant(kv_c, 1.0, KV_DIM, 16)`。先只加 Q 量化测增量，再单独改变 K 的分组或 global scale，避免同时改变多个因素。

## 6. global_scale=1.0

用户代码中 `g=global_scale` 是全局反量化尺度的倒数：

```text
s_block = FP8_RNE(g * amax_block / 6)
x_hat = E2M1_RNE(x * g / s_block) * s_block / g
```

固定 g=1 仍有 per-16 动态 block scale，但冻结了 per-tensor 二级缩放。常见 max-based 二级缩放对应 `g=6*448/amax_tensor`；传统写法的 global dequant scale 则是其倒数 `amax_tensor/(6*448)`。全零 tensor 可取 g=1。

g 无法简单约掉，因为 `s_block` 中有 FP8 舍入。选择 per-tensor amax 还需定义 tensor 的范围（当前步、整层校准等）。不能在写一个新 token 时用新 g 去重新解释之前缓存的 packed 数据；本伪量化版本写回高精度值，不存 global-scale metadata。

注意 MLA 的 latent `kv_c` 同时是 QK 中的 K 和 PV 中的 V。用户在缓存写入处量化 `kv_c` 会同时影响 QK 和 PV，且原代码也可能影响 prefill 输出/缓存。本次补丁仅保证“新增 Q 量化”限制在真实 decode，并不把原实验变为纯 QK 隔离实验。

## 7. 运行测试

CPU reference：

```bash
python test_mxfp8_reference.py
```

目标 GPU 环境（需要同一个可用的 vLLM/Triton 环境）：

```bash
python test_mxfp8_gpu.py
```

GPU 测试覆盖原位写回、prefill 不变、decode 数值改变、连续/非连续布局、BF16/FP16/FP32 与 CPU reference 对照。通过后仍需进行模型端到端的少量请求验证，以及 token/head/group 数据抓取；不能用 CPU reference 测试替代真正的 GPU/vLLM 测试。

## 核对的官方源码

```text
vllm-project/vllm tag v0.29.0:
  vllm/models/deepseek_v32/attention.py
  vllm/models/deepseek_v32/common/kernels.py
  vllm/model_executor/layers/attention/mla_attention.py
  vllm/model_executor/layers/attention/sparse_mla_attention.py
  vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
  vllm/v1/attention/backend.py
  vllm/v1/worker/gpu_model_runner.py
NVIDIA/TransformerEngine:
  transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh
  transformer_engine/common/util/ptx_arch_spec.cuh
  transformer_engine/common/recipe/__init__.py
Hugging Face nvidia/GLM-5.2-NVFP4/config.json
```
