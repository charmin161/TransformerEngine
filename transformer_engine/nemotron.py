# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nemotron_h.py

"""Inference-only NemotronH model."""
# [xzj] 增加抓数代码

from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

# [xzj] 导入数据抓取器
from sglang.srt.layers.catcher import SGLangDataCatcher, PhaseType, CaptureConfig

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.compilation.piecewise_context_manager import (
    get_forward_context,
    is_in_piecewise_cuda_graph,
)
from sglang.srt.configs import NemotronHConfig
from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MLP, MOE
from sglang.srt.distributed import (
    get_moe_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import ReLU2
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba2AttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    replace_prefix,
    replace_substrings,
)
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    get_current_device_stream_fast,
    is_cuda,
    make_layers,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.utils import logger

_is_cuda = is_cuda()

# [xzj] ==================== NemotronH 全局抓数配置 ====================
# Attention 层采集配置
_capture_config_attn = CaptureConfig(
    warmup_steps=0,          # 预热
    start_step=1,            # warmup 后第一个 step 为 step 1
    stop_step=2048,          # 采集到 step 2048
    step_interval=1,         # 每个 step 都采集
    start_layer=0,           # 从第 0 层开始
    stop_layer=100,          # 到第 100 层结束（根据模型实际层数调整）
    layer_interval=1,        # 每个 layer 都采集
)

# MOE/MLP 层采集配置
_capture_config_moe = CaptureConfig(
    warmup_steps=0,
    start_step=1,
    stop_step=2048,
    step_interval=1,
    start_layer=0,
    stop_layer=100,
    layer_interval=5,        # MOE 数据量大，可以适当增大间隔
)

# [xzj] 全局计数器和状态变量 - 共享 step 计数器
_nemotron_layer_call_counter = 0       # layer 调用次数（用于调试）
_nemotron_current_step = 0             # 当前 step ID（全局共享）
_nemotron_last_layer_idx = -1          # 上一次调用的 layer_idx（用于检测新 step）
_nemotron_current_phase = None         # 当前 phase
_nemotron_current_m_a = 0              # 当前 m_a
_nemotron_current_step_id = 0          # 当前 step_id

# [xzj] 抓数开关
ZHUASHU_NEMOTRON = True

# [xzj] 阶段采集开关（分别控制 prefill 和 decode）
ZHUASHU_NEMOTRON_PREFILL = True
ZHUASHU_NEMOTRON_DECODE = True

# [xzj] 全局抓取器实例 - 延迟初始化
_global_catcher_attn = None
_global_catcher_moe = None


def _get_catcher_attn():
    """获取或初始化 Attention 抓取器"""
    global _global_catcher_attn
    if _global_catcher_attn is None:
        _global_catcher_attn = SGLangDataCatcher(
            save_dir="/wireless/minyusong/nemotron_3_ultra/captured_data/nemotron_h_attn",
            max_samples_per_request=100,
            max_samples_per_layer=20,
            capture_config=_capture_config_attn,
        )
        print(f"[NemotronH Capture] Attention catcher initialized, saving to: /wireless/minyusong/nemotron_3_ultra/captured_data/nemotron_h_attn")
    return _global_catcher_attn


def _get_catcher_moe():
    """获取或初始化 MOE/MLP 抓取器"""
    global _global_catcher_moe
    if _global_catcher_moe is None:
        _global_catcher_moe = SGLangDataCatcher(
            save_dir="/wireless/minyusong/nemotron_3_ultra/captured_data/nemotron_h_moe",
            max_samples_per_request=100,
            max_samples_per_layer=20,
            capture_config=_capture_config_moe,
        )
        print(f"[NemotronH Capture] MOE/MLP catcher initialized, saving to: /wireless/minyusong/nemotron_3_ultra/captured_data/nemotron_h_moe")
    return _global_catcher_moe


# [xzj] 异步保存函数
def _async_save_nemotron_data(catcher, layer_idx, phase, m_a, tp_rank, tp_size, step_id, tensor_dict, data_type, *gpu_refs):
    """异步保存数据的函数

    Args:
        gpu_refs: GPU tensor 引用，用于防止异步传输期间内存被释放
    """
    try:
        print(f"[NemotronH Capture] Saving data: layer={layer_idx}, phase={phase.value}, step={step_id}, data_type={data_type}, tensors={list(tensor_dict.keys())}")
        result = catcher.capture(
            batch=None,
            tensor_dict=tensor_dict,
            layer_idx=layer_idx,
            force_phase=phase,
            allow_no_batch=True,
            extra_info={
                "m_a": m_a,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "step_id": step_id,
                "data_type": data_type,
            }
        )
        print(f"[NemotronH Capture] Saved successfully: {result}")
    except Exception as e:
        print(f"[NemotronH Capture] Error saving data: {e}")
        import traceback
        traceback.print_exc()


import threading


# [xzj] ==================== NemotronH 全局抓数配置结束 ====================


class NemotronHMLP(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        reduce_results: bool = True,
        prefix: str = "",
        layer_idx: int = -1,  # [xzj] 添加 layer_idx 参数用于抓数
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx  # [xzj] 保存 layer_idx

        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = ReLU2()

    def forward(self, x: torch.Tensor):
        # [xzj] MLP 抓数代码 - 抓取输入
        # 只在非CUDA Graph模式下打印调试信息
        if not is_in_piecewise_cuda_graph():
            print(f"[NemotronH Capture] MLP forward called, ZHUASHU_NEMOTRON={ZHUASHU_NEMOTRON}, shape={x.shape}")
        if ZHUASHU_NEMOTRON:
            global _nemotron_layer_call_counter, _nemotron_current_step, _nemotron_last_layer_idx
            global _nemotron_current_phase, _nemotron_current_m_a, _nemotron_current_step_id

            _nemotron_layer_call_counter += 1

            # 获取 layer_idx（MLP 没有 layer_idx 属性，需要从外部获取或设为 -1）
            layer_idx = getattr(self, 'layer_idx', -1)

            # 检测是否开始新的 step
            # 当 layer_idx 比之前的小，说明进入了新的 step
            if _nemotron_last_layer_idx != -1 and layer_idx < _nemotron_last_layer_idx:
                _nemotron_current_step += 1

            # 判断 phase
            m_a = x.shape[0]
            if m_a == 1:
                phase = PhaseType.DECODE
            else:
                phase = PhaseType.PREFILL

            _nemotron_current_phase = phase
            _nemotron_current_m_a = m_a

            # 判断是否在 warmup 阶段
            in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
            should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

            # 判断是否采集当前 phase
            should_capture_phase = (phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or \
                                   (phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)

            if not is_in_piecewise_cuda_graph():
                print(f"[NemotronH Capture] MLP check: layer_idx={layer_idx}, step={_nemotron_current_step}, m_a={m_a}, phase={phase.value}, in_warmup={in_warmup}, should_capture_layer={should_capture_layer}, should_capture_phase={should_capture_phase}")

            if not in_warmup and should_capture_layer and should_capture_phase:
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception as e:
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step - _capture_config_moe.warmup_steps + 1
                _nemotron_current_step_id = step_id

                catcher = _get_catcher_moe()
                x_gpu_copy = x.detach().clone()
                x_cpu = x_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, phase, m_a, tp_rank, tp_size, step_id,
                          {"mlp_input": x_cpu}, "mlp_input",
                          x_gpu_copy),
                    daemon=True
                ).start()

            _nemotron_last_layer_idx = layer_idx
        # [xzj] 抓数代码结束

        x, _ = self.up_proj(x)
        x = self.act_fn(x)

        # [xzj] 抓取 up_proj 输出（activation 输入）
        if ZHUASHU_NEMOTRON:
            layer_idx = getattr(self, 'layer_idx', -1)
            in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
            should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)
            should_capture_phase = (_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or \
                                   (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)

            if not in_warmup and should_capture_layer and should_capture_phase:
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception:
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step_id
                catcher = _get_catcher_moe()
                x_gpu_copy = x.detach().clone()
                x_cpu = x_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                          {"mlp_up_proj_output": x_cpu}, "mlp_up_proj_output",
                          x_gpu_copy),
                    daemon=True
                ).start()
        # [xzj] 抓数代码结束

        x, _ = self.down_proj(x)

        # [xzj] 抓取 down_proj 输出（MLP 最终输出）
        if ZHUASHU_NEMOTRON:
            layer_idx = getattr(self, 'layer_idx', -1)
            in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
            should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)
            should_capture_phase = (_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or \
                                   (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)

            if not in_warmup and should_capture_layer and should_capture_phase:
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception:
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step_id
                catcher = _get_catcher_moe()
                x_gpu_copy = x.detach().clone()
                x_cpu = x_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                          {"mlp_output": x_cpu}, "mlp_output",
                          x_gpu_copy),
                    daemon=True
                ).start()
        # [xzj] 抓数代码结束

        return x


_alt_stream = None


def _get_or_create_alt_stream(device_module):
    global _alt_stream
    if _alt_stream is None:
        _alt_stream = device_module.Stream()
    return _alt_stream


class NemotronHMoE(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx  # [xzj] 保存 layer_idx 用于抓数

        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.device_module = torch.get_device_module()

        self.ep_group = get_moe_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.use_latent_moe = getattr(config, "moe_latent_size", None) is not None
        self.moe_hidden_size = (
            config.moe_latent_size if self.use_latent_moe else config.hidden_size
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            activation=config.mlp_hidden_act,
            layer_id=layer_idx,
            is_gated=False,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            use_grouped_topk=True,
            topk_group=config.topk_group,
            num_expert_group=config.n_group,
            renormalize=config.norm_topk_prob,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
        )
        if config.n_shared_experts:
            self.shared_experts = NemotronHMLP(
                config,
                intermediate_size=config.moe_shared_expert_intermediate_size
                * config.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                layer_idx=layer_idx,  # [xzj] 传递 layer_idx
            )
        else:
            self.shared_experts = None

        if self.use_latent_moe:
            self.fc1_latent_proj = ReplicatedLinear(
                input_size=config.hidden_size,
                output_size=self.moe_hidden_size,
                bias=config.mlp_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1_latent_proj",
            )
            self.fc2_latent_proj = ReplicatedLinear(
                input_size=self.moe_hidden_size,
                output_size=config.hidden_size,
                bias=config.mlp_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2_latent_proj",
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # torch.compile cannot trace CUDA streams, so use the non-overlapping
        # path when inside piecewise CUDA graph compilation.
        if _is_cuda and not is_in_piecewise_cuda_graph():
            return self._forward_core_shared_routed_overlap(hidden_states)
        else:
            return self._forward_core_normal(hidden_states)

    def _forward_core_normal(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # [xzj] MoE 抓数代码 - 抓取输入
        layer_idx = getattr(self, 'layer_idx', -1)
        if not is_in_piecewise_cuda_graph():
            print(f"[NemotronH Capture] MoE _forward_core_normal called, ZHUASHU_NEMOTRON={ZHUASHU_NEMOTRON}, layer_idx={layer_idx}, shape={hidden_states.shape}")
        if ZHUASHU_NEMOTRON:
            global _nemotron_layer_call_counter, _nemotron_current_step, _nemotron_last_layer_idx
            global _nemotron_current_phase, _nemotron_current_m_a, _nemotron_current_step_id

            _nemotron_layer_call_counter += 1

            # 检测是否开始新的 step
            # 当 layer_idx 比之前的小，说明进入了新的 step
            if _nemotron_last_layer_idx != -1 and layer_idx < _nemotron_last_layer_idx:
                _nemotron_current_step += 1

            # 判断 phase
            m_a = hidden_states.shape[0]
            if m_a == 1:
                phase = PhaseType.DECODE
            else:
                phase = PhaseType.PREFILL

            _nemotron_current_phase = phase
            _nemotron_current_m_a = m_a

            # 判断是否在 warmup 阶段
            in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
            should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

            if not is_in_piecewise_cuda_graph():
                print(f"[NemotronH Capture] MoE check: layer_idx={layer_idx}, step={_nemotron_current_step}, m_a={m_a}, phase={phase.value}, in_warmup={in_warmup}, should_capture_layer={should_capture_layer}")

            if not in_warmup and should_capture_layer and ((phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                if not is_in_piecewise_cuda_graph():
                    print(f"[NemotronH Capture] MoE CAPTURING input! layer={layer_idx}, step={_nemotron_current_step}")
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception as e:
                    if not is_in_piecewise_cuda_graph():
                        print(f"[NemotronH Capture] Error getting TP rank: {e}")
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step - _capture_config_moe.warmup_steps + 1
                _nemotron_current_step_id = step_id

                catcher = _get_catcher_moe()
                hidden_states_gpu_copy = hidden_states.detach().clone()
                hidden_states_cpu = hidden_states_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, phase, m_a, tp_rank, tp_size, step_id,
                          {"moe_input": hidden_states_cpu}, "moe_input",
                          hidden_states_gpu_copy),
                    daemon=True
                ).start()

            _nemotron_last_layer_idx = layer_idx
        # [xzj] 抓数代码结束

        # router_scores: [num_tokens, num_experts]
        # bf16 gemm on tensor cores with fp32 accumulation/output for sigmoid/topk.
        router_logits = torch.mm(
            hidden_states, self.gate.weight.t(), out_dtype=torch.float32
        )

        # [xzj] 抓取 router logits
        if ZHUASHU_NEMOTRON:
            in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
            should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

            if not in_warmup and should_capture_layer and ((_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception:
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step_id
                catcher = _get_catcher_moe()
                router_logits_gpu_copy = router_logits.detach().clone()
                router_logits_cpu = router_logits_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                          {"moe_router_logits": router_logits_cpu}, "moe_router_logits",
                          router_logits_gpu_copy),
                    daemon=True
                ).start()
        # [xzj] 抓数代码结束

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None
        topk_output = self.topk(hidden_states, router_logits)

        # [xzj] 抓取 topk 输出
        if ZHUASHU_NEMOTRON:
            in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
            should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

            if not in_warmup and should_capture_layer and ((_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception:
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step_id
                catcher = _get_catcher_moe()
                # topk_output 是一个元组，包含 topk_weights 和 topk_ids
                if hasattr(topk_output, 'topk_weights') and hasattr(topk_output, 'topk_ids'):
                    topk_weights = topk_output.topk_weights
                    topk_ids = topk_output.topk_ids
                    topk_weights_gpu_copy = topk_weights.detach().clone()
                    topk_ids_gpu_copy = topk_ids.detach().clone()
                    topk_weights_cpu = topk_weights_gpu_copy.to('cpu', non_blocking=True)
                    topk_ids_cpu = topk_ids_gpu_copy.to('cpu', non_blocking=True)
                    threading.Thread(
                        target=_async_save_nemotron_data,
                        args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                              {"moe_topk_weights": topk_weights_cpu, "moe_topk_ids": topk_ids_cpu}, "moe_topk_output",
                              topk_weights_gpu_copy, topk_ids_gpu_copy),
                        daemon=True
                    ).start()
        # [xzj] 抓数代码结束

        if self.use_latent_moe:
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        final_hidden_states = self.experts(hidden_states, topk_output)

        # [xzj] 抓取 experts 输出
        if ZHUASHU_NEMOTRON:
            in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
            should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

            if not in_warmup and should_capture_layer and ((_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception:
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step_id
                catcher = _get_catcher_moe()
                final_hidden_states_gpu_copy = final_hidden_states.detach().clone()
                final_hidden_states_cpu = final_hidden_states_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                          {"moe_experts_output": final_hidden_states_cpu}, "moe_experts_output",
                          final_hidden_states_gpu_copy),
                    daemon=True
                ).start()
        # [xzj] 抓数代码结束

        return final_hidden_states, shared_output

    def _forward_core_shared_routed_overlap(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # [xzj] MoE 抓数代码 - 抓取输入（overlap 模式）
        layer_idx = getattr(self, 'layer_idx', -1)
        if ZHUASHU_NEMOTRON:
            global _nemotron_layer_call_counter, _nemotron_current_step, _nemotron_last_layer_idx
            global _nemotron_current_phase, _nemotron_current_m_a, _nemotron_current_step_id

            _nemotron_layer_call_counter += 1

            # 检测是否开始新的 step
            # 当 layer_idx 比之前的小，说明进入了新的 step
            if _nemotron_last_layer_idx != -1 and layer_idx < _nemotron_last_layer_idx:
                _nemotron_current_step += 1

            # 判断 phase
            m_a = hidden_states.shape[0]
            if m_a == 1:
                phase = PhaseType.DECODE
            else:
                phase = PhaseType.PREFILL

            _nemotron_current_phase = phase
            _nemotron_current_m_a = m_a

            # 判断是否在 warmup 阶段
            in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
            should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

            if not in_warmup and should_capture_layer and ((phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception:
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step - _capture_config_moe.warmup_steps + 1
                _nemotron_current_step_id = step_id

                catcher = _get_catcher_moe()
                hidden_states_gpu_copy = hidden_states.detach().clone()
                hidden_states_cpu = hidden_states_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, phase, m_a, tp_rank, tp_size, step_id,
                          {"moe_input": hidden_states_cpu}, "moe_input_overlap",
                          hidden_states_gpu_copy),
                    daemon=True
                ).start()

            _nemotron_last_layer_idx = layer_idx
        # [xzj] 抓数代码结束

        alt_stream = _get_or_create_alt_stream(self.device_module)

        alt_stream.wait_stream(get_current_device_stream_fast())

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        with self.device_module.stream(alt_stream):
            # router_scores: [num_tokens, num_experts]
            # bf16 gemm on tensor cores with fp32 accumulation/output for sigmoid/topk.
            router_logits = torch.mm(
                hidden_states, self.gate.weight.t(), out_dtype=torch.float32
            )

            # [xzj] 抓取 router logits（overlap 模式）
            if ZHUASHU_NEMOTRON:
                in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
                should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

                if not in_warmup and should_capture_layer and ((_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                    try:
                        from sglang.srt.distributed.parallel_state import (
                            get_moe_tensor_parallel_rank,
                            get_moe_tensor_parallel_world_size,
                        )
                        tp_rank = get_moe_tensor_parallel_rank()
                        tp_size = get_moe_tensor_parallel_world_size()
                    except Exception:
                        tp_rank = 0
                        tp_size = 1

                    step_id = _nemotron_current_step_id
                    catcher = _get_catcher_moe()
                    router_logits_gpu_copy = router_logits.detach().clone()
                    router_logits_cpu = router_logits_gpu_copy.to('cpu', non_blocking=True)
                    threading.Thread(
                        target=_async_save_nemotron_data,
                        args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                              {"moe_router_logits": router_logits_cpu}, "moe_router_logits_overlap",
                              router_logits_gpu_copy),
                        daemon=True
                    ).start()
            # [xzj] 抓数代码结束

            topk_output = self.topk(hidden_states, router_logits)

            # [xzj] 抓取 topk 输出（overlap 模式）
            if ZHUASHU_NEMOTRON:
                in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
                should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

                if not in_warmup and should_capture_layer and ((_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                    try:
                        from sglang.srt.distributed.parallel_state import (
                            get_moe_tensor_parallel_rank,
                            get_moe_tensor_parallel_world_size,
                        )
                        tp_rank = get_moe_tensor_parallel_rank()
                        tp_size = get_moe_tensor_parallel_world_size()
                    except Exception:
                        tp_rank = 0
                        tp_size = 1

                    step_id = _nemotron_current_step_id
                    catcher = _get_catcher_moe()
                    if hasattr(topk_output, 'topk_weights') and hasattr(topk_output, 'topk_ids'):
                        topk_weights = topk_output.topk_weights
                        topk_ids = topk_output.topk_ids
                        topk_weights_gpu_copy = topk_weights.detach().clone()
                        topk_ids_gpu_copy = topk_ids.detach().clone()
                        topk_weights_cpu = topk_weights_gpu_copy.to('cpu', non_blocking=True)
                        topk_ids_cpu = topk_ids_gpu_copy.to('cpu', non_blocking=True)
                        threading.Thread(
                            target=_async_save_nemotron_data,
                            args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                                  {"moe_topk_weights": topk_weights_cpu, "moe_topk_ids": topk_ids_cpu}, "moe_topk_output_overlap",
                                  topk_weights_gpu_copy, topk_ids_gpu_copy),
                            daemon=True
                        ).start()
            # [xzj] 抓数代码结束

            if self.use_latent_moe:
                hidden_states, _ = self.fc1_latent_proj(hidden_states)
            final_hidden_states = self.experts(hidden_states, topk_output)

            # [xzj] 抓取 experts 输出（overlap 模式）
            if ZHUASHU_NEMOTRON:
                in_warmup = _nemotron_current_step < _capture_config_moe.warmup_steps
                should_capture_layer = _capture_config_moe.should_capture_layer(layer_idx)

                if not in_warmup and should_capture_layer and ((_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                    try:
                        from sglang.srt.distributed.parallel_state import (
                            get_moe_tensor_parallel_rank,
                            get_moe_tensor_parallel_world_size,
                        )
                        tp_rank = get_moe_tensor_parallel_rank()
                        tp_size = get_moe_tensor_parallel_world_size()
                    except Exception:
                        tp_rank = 0
                        tp_size = 1

                    step_id = _nemotron_current_step_id
                    catcher = _get_catcher_moe()
                    final_hidden_states_gpu_copy = final_hidden_states.detach().clone()
                    final_hidden_states_cpu = final_hidden_states_gpu_copy.to('cpu', non_blocking=True)
                    threading.Thread(
                        target=_async_save_nemotron_data,
                        args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                              {"moe_experts_output": final_hidden_states_cpu}, "moe_experts_output_overlap",
                              final_hidden_states_gpu_copy),
                        daemon=True
                    ).start()
            # [xzj] 抓数代码结束

        get_current_device_stream_fast().wait_stream(alt_stream)

        return final_hidden_states, shared_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        # routed_scaling_factor is fused into the experts call (applied by the
        # MoE runner / topk), so final_hidden_states is already scaled.
        final_hidden_states, shared_output = self._forward_core(hidden_states)

        if self.use_latent_moe:
            final_hidden_states, _ = self.fc2_latent_proj(final_hidden_states)

        if shared_output is not None:
            final_hidden_states += shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class NemotronHMLPDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        hybrid_override_pattern = config.hybrid_override_pattern
        mlp_index = hybrid_override_pattern[: layer_idx + 1].count("-") - 1
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[mlp_index]
        else:
            intermediate_size = config.intermediate_size

        self.mixer = NemotronHMLP(
            config,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            bias=config.mlp_bias,
            prefix=f"{prefix}.mixer",
            layer_idx=layer_idx,  # [xzj] 传递 layer_idx
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer.forward(hidden_states)
        return hidden_states, residual


class NemotronHMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_config = config.get_nemotron_h_config_for_layer(layer_idx)

        self.mixer = NemotronHMoE(
            layer_config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer.forward(hidden_states)
        return hidden_states, residual


class NemotronHMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_idx
        self.mixer = MambaMixer2(
            cache_params=config.mamba2_cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.mamba_n_groups,
            rms_norm_eps=config.layer_norm_epsilon,
            activation=config.mamba_hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def _forward_mamba(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Core Mamba forward logic for the eager path; returns the result."""
        attn_backend = get_attn_backend()
        assert isinstance(attn_backend, HybridLinearAttnBackend)
        assert isinstance(attn_backend.linear_attn_backend, Mamba2AttnBackend)
        return attn_backend.linear_attn_backend.forward(
            mixer=self.mixer,
            layer_id=self.layer_id,
            hidden_states=hidden_states,
            output=None,
            forward_batch=forward_batch,
            use_triton_causal_conv=True,
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        if is_in_breakable_cuda_graph():
            output = torch.empty_like(hidden_states)
            breakable_nemotron_mamba2_with_output(hidden_states, output, self.layer_id)
            return output, residual

        if is_in_piecewise_cuda_graph():
            output = torch.empty_like(hidden_states)
            nemotron_mamba2_with_output(hidden_states, output, self.layer_id)
            return output, residual
        else:
            output = self._forward_mamba(hidden_states, forward_batch)
            return output, residual


class NemotronHAttention(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx  # [xzj] 保存 layer_idx 用于抓数
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            sliding_window_size=config.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        # [xzj] NemotronH Attention 抓数代码
        if not is_in_piecewise_cuda_graph():
            print(f"[NemotronH Capture] Attention forward called, ZHUASHU_NEMOTRON={ZHUASHU_NEMOTRON}, shape={hidden_states.shape}")
        # 注意：所有 global 声明必须在函数开头
        if ZHUASHU_NEMOTRON:
            global _nemotron_layer_call_counter, _nemotron_current_step, _nemotron_last_layer_idx
            global _nemotron_current_phase, _nemotron_current_m_a, _nemotron_current_step_id

            _nemotron_layer_call_counter += 1

            # 获取 layer_idx
            layer_idx = getattr(self, 'layer_idx', None)
            if layer_idx is None:
                layer_idx = -1

            # 检测是否开始新的 step
            # 当 layer_idx 比之前的小，说明进入了新的 step
            if _nemotron_last_layer_idx != -1 and layer_idx < _nemotron_last_layer_idx:
                _nemotron_current_step += 1

            # 判断 phase: m_a 是 batch 中的 token 数
            m_a = hidden_states.shape[0]
            if m_a == 1:
                phase = PhaseType.DECODE
            else:
                phase = PhaseType.PREFILL

            _nemotron_current_phase = phase
            _nemotron_current_m_a = m_a

            # 判断是否在 warmup 阶段
            in_warmup = _nemotron_current_step < _capture_config_attn.warmup_steps

            # 检查 layer_interval
            should_capture_layer = _capture_config_attn.should_capture_layer(layer_idx)

            if not is_in_piecewise_cuda_graph():
                print(f"[NemotronH Capture] Attention check: layer_idx={layer_idx}, step={_nemotron_current_step}, m_a={m_a}, phase={phase.value}, in_warmup={in_warmup}, should_capture_layer={should_capture_layer}")

            if not in_warmup and should_capture_layer and ((phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                if not is_in_piecewise_cuda_graph():
                    print(f"[NemotronH Capture] Attention CAPTURING input! layer={layer_idx}, step={_nemotron_current_step}")
                # 获取 TP rank 信息
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception as e:
                    if not is_in_piecewise_cuda_graph():
                        print(f"[NemotronH Capture] Error getting TP rank: {e}")
                    tp_rank = 0
                    tp_size = 1

                # 计算 step_id
                step_id = _nemotron_current_step - _capture_config_attn.warmup_steps + 1
                _nemotron_current_step_id = step_id

                # 异步保存：先在 GPU 上 clone，再异步传输到 CPU
                catcher = _get_catcher_attn()
                hidden_states_gpu_copy = hidden_states.detach().clone()
                hidden_states_cpu = hidden_states_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, phase, m_a, tp_rank, tp_size, step_id,
                          {"attn_input": hidden_states_cpu}, "attn_input",
                          hidden_states_gpu_copy),
                    daemon=True
                ).start()

            _nemotron_last_layer_idx = layer_idx
        # [xzj] 抓数代码结束

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # [xzj] 抓取 QKV 输出
        if ZHUASHU_NEMOTRON:
            # 使用之前已经设置的 global 变量值
            layer_idx = getattr(self, 'layer_idx', None)
            if layer_idx is None:
                layer_idx = -1

            in_warmup = _nemotron_current_step < _capture_config_attn.warmup_steps
            should_capture_layer = _capture_config_attn.should_capture_layer(layer_idx)

            if not in_warmup and should_capture_layer and ((_nemotron_current_phase == PhaseType.PREFILL and ZHUASHU_NEMOTRON_PREFILL) or (_nemotron_current_phase == PhaseType.DECODE and ZHUASHU_NEMOTRON_DECODE)):
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_moe_tensor_parallel_rank,
                        get_moe_tensor_parallel_world_size,
                    )
                    tp_rank = get_moe_tensor_parallel_rank()
                    tp_size = get_moe_tensor_parallel_world_size()
                except Exception:
                    tp_rank = 0
                    tp_size = 1

                step_id = _nemotron_current_step_id

                catcher = _get_catcher_attn()
                q_gpu_copy = q.detach().clone()
                k_gpu_copy = k.detach().clone()
                v_gpu_copy = v.detach().clone()
                q_cpu = q_gpu_copy.to('cpu', non_blocking=True)
                k_cpu = k_gpu_copy.to('cpu', non_blocking=True)
                v_cpu = v_gpu_copy.to('cpu', non_blocking=True)
                threading.Thread(
                    target=_async_save_nemotron_data,
                    args=(catcher, layer_idx, _nemotron_current_phase, _nemotron_current_m_a, tp_rank, tp_size, step_id,
                          {"q": q_cpu, "k": k_cpu, "v": v_cpu}, "qkv_output",
                          q_gpu_copy, k_gpu_copy, v_gpu_copy),
                    daemon=True
                ).start()
        # [xzj] 抓数代码结束

        attn_output = self.attn.forward(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class NemotronHAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_config = config.get_nemotron_h_config_for_layer(layer_idx)

        self.mixer = NemotronHAttention(
            layer_config,
            layer_idx,
            quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer.forward(
            hidden_states=hidden_states, forward_batch=forward_batch
        )
        return hidden_states, residual


Layers = (
    NemotronHAttentionDecoderLayer,
    NemotronHMLPDecoderLayer,
    NemotronHMambaDecoderLayer,
    NemotronHMoEDecoderLayer,
)
ALL_DECODER_LAYER_TYPES: dict[str, type] = {
    ATTENTION: NemotronHAttentionDecoderLayer,
    MLP: NemotronHMLPDecoderLayer,
    MAMBA: NemotronHMambaDecoderLayer,
    MOE: NemotronHMoEDecoderLayer,
}


class NemotronHModel(nn.Module):
    def __init__(
        self,
        *,
        config: NemotronHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        lora_config = None
        self.config = config
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.hybrid_override_pattern[idx]]
            return layer_class(config, idx, quant_config=quant_config, prefix=prefix)

        self.layers, self.start_layer, self.end_layer = make_layers(
            len(config.hybrid_override_pattern),
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )
        if self.pp_group.is_last_rank:
            self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.norm_f = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if not isinstance(layer, Layers):
                raise ValueError(f"Unknown layer type: {type(layer)}")
            hidden_states, residual = layer.forward(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states


class NemotronHForCausalLM(nn.Module):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "out_proj",
        "in_proj",
        "up_proj",
        "gate_up_proj",
        "down_proj",
        "fc1_latent_proj",
        "fc2_latent_proj",
    ]

    remap_prefix = {"backbone": "model"}
    remap_substr = {
        "A_log": "A",
        "embeddings": "embed_tokens",
        "k_proj.k_scale": "attn.k_scale",
        "v_proj.v_scale": "attn.v_scale",
    }

    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "backbone.": "model.",
        }
    )

    def __init__(
        self,
        *,
        config: NemotronHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        lora_config = None
        self.config = config
        self.quant_config = quant_config
        self.model = self._init_model(
            config=config, quant_config=quant_config, prefix=prefix
        )
        self.pp_group = get_pp_group()

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.unpadded_vocab_size = config.vocab_size
                if lora_config:
                    self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
                self.lm_head = ParallelLMHead(
                    self.unpadded_vocab_size,
                    config.hidden_size,
                    org_num_embeddings=config.vocab_size,
                    padding_size=(
                        DEFAULT_VOCAB_PADDING_SIZE
                        # We need bigger padding if using lora for kernel
                        # compatibility
                        if not lora_config
                        else lora_config.lora_vocab_padding_size
                    ),
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        if self.pp_group.world_size > 1 and self.config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=self.lm_head.weight.shape,
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)

    def _init_model(
        self,
        config: NemotronHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return NemotronHModel(
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.embed_tokens

    def get_stacked_multiply(self, module_name):
        """Non-gated MoE uses stacked_multiply=1 for gate_up_proj_moe."""
        if module_name == "gate_up_proj_moe":
            return 1  # Non-gated: only w1, no w3
        # Fall back to defaults for everything else
        from sglang.srt.lora.utils import get_stacked_multiply

        return get_stacked_multiply(module_name)

    def get_hidden_dim(self, module_name, layer_idx):
        """Return (input_dim, output_dim) for LoRA buffers, per layer type."""
        config = self.config
        layer_type = config.layers_block_type[layer_idx]
        hidden_size = config.hidden_size
        head_dim = getattr(
            config, "head_dim", hidden_size // config.num_attention_heads
        )

        if module_name == "qkv_proj":
            return (
                hidden_size,
                head_dim
                * (config.num_attention_heads + config.num_key_value_heads * 2),
            )
        elif module_name == "o_proj":
            return (
                head_dim * config.num_attention_heads,
                hidden_size,
            )
        elif module_name == "out_proj":
            # Mamba out_proj: RowParallelLinear from mamba_intermediate to hidden_size
            mamba_intermediate = config.mamba_num_heads * config.mamba_head_dim
            return mamba_intermediate, hidden_size
        elif module_name == "gate_up_proj":
            if layer_type == "mamba":
                # Mamba in_proj gate component: output = mamba_num_heads * mamba_head_dim
                mamba_intermediate = config.mamba_num_heads * config.mamba_head_dim
                return hidden_size, mamba_intermediate * 2
            elif layer_type == "moe":
                # Shared expert: only has up_proj (no gate), but gets stacked
                shared_inter = (
                    config.moe_shared_expert_intermediate_size * config.n_shared_experts
                )
                return hidden_size, shared_inter * 2
            else:
                # MLP layer
                return hidden_size, config.intermediate_size * 2
        elif module_name == "up_proj":
            if layer_type == "moe":
                shared_inter = (
                    config.moe_shared_expert_intermediate_size * config.n_shared_experts
                )
                return hidden_size, shared_inter
            else:
                return hidden_size, config.intermediate_size
        elif module_name == "down_proj":
            if layer_type == "moe":
                shared_inter = (
                    config.moe_shared_expert_intermediate_size * config.n_shared_experts
                )
                return shared_inter, hidden_size
            else:
                return config.intermediate_size, hidden_size
        elif module_name == "in_proj":
            # Mamba in_proj: gate_proj + x_proj, each mamba_intermediate wide
            mamba_intermediate = config.mamba_num_heads * config.mamba_head_dim
            return hidden_size, mamba_intermediate * 2
        elif module_name == "x_proj":
            # Mamba x_proj: projects from hidden_size to mamba_intermediate
            mamba_intermediate = config.mamba_num_heads * config.mamba_head_dim
            return hidden_size, mamba_intermediate
        elif module_name == "gate_up_proj_moe":
            # Non-gated MoE: only w1, no w3. stacked_multiply=1.
            # For latent MoE, experts operate in moe_latent_size space.
            moe_hidden = getattr(config, "moe_latent_size", None) or hidden_size
            return moe_hidden, config.moe_intermediate_size
        elif module_name == "down_proj_moe":
            moe_hidden = getattr(config, "moe_latent_size", None) or hidden_size
            return config.moe_intermediate_size, moe_hidden
        elif module_name == "fc1_latent_proj":
            moe_latent = getattr(config, "moe_latent_size", None) or hidden_size
            return hidden_size, moe_latent
        elif module_name == "fc2_latent_proj":
            moe_latent = getattr(config, "moe_latent_size", None) or hidden_size
            return moe_latent, hidden_size
        elif module_name == "embed_tokens":
            return config.vocab_size, hidden_size
        elif module_name == "lm_head":
            return hidden_size, config.vocab_size
        else:
            raise NotImplementedError(
                f"get_hidden_dim not implemented for {module_name}"
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model.forward(
            input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> None:
        # - FusedMoe.w1 (aka gate_proj) should be up_proj since that's
        #   what the activation is applied to
        # - FusedMoe.w3 (aka up_proj) should be ignored since we're
        #   using non-gated MoE
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="",
            num_experts=self.config.max_n_routed_experts,
        )

        params_dict = dict(self.named_parameters())

        # Stream weights directly from the generator to avoid buffering
        # the entire checkpoint (~75 GB) into a Python list. On unified-
        # memory systems (e.g. DGX Spark, 119 GB) the old buffered path
        # caused OOM: skeleton 81.6 GB + buffer 75 GB = 157 GB peak.
        for name, loaded_weight in weights:
            name = replace_prefix(name, self.remap_prefix)
            name = replace_substrings(name, self.remap_substr)
            if is_mtp:
                if "mtp" not in name:
                    continue

                name = name.replace("mtp.layers.", "model.layers.")

                if "embeddings" in name:
                    name = name.replace("embeddings", "model.embed_tokens")
                    if name.startswith("backbone."):
                        name = name.replace("backbone.", "")

            if not is_mtp and "mtp" in name:
                continue

            if "scale" in name:
                if name not in params_dict:
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "embed_tokens" in name and not self.pp_group.is_first_rank:
                continue

            if (
                "norm_f" in name or "lm_head" in name
            ) and not self.pp_group.is_last_rank:
                continue

            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")


class NemotronHPuzzleForCausalLM(NemotronHForCausalLM):
    pass


EntryClass = [NemotronHForCausalLM, NemotronHPuzzleForCausalLM]


@register_custom_op(mutates_args=["output"])
@register_split_op()
def nemotron_mamba2_with_output(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
) -> None:
    """Split op for Mamba2 forward in piecewise CUDA graph mode."""
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    mamba_layer = attention_layers[layer_id]

    # In piecewise CUDA graph mode, hidden_states may be padded to the
    # captured graph size. Slice to actual token count for Mamba forward.
    attn_backend = get_attn_backend()
    metadata = attn_backend.linear_attn_backend.forward_metadata
    num_actual_tokens = metadata.num_prefill_tokens + (
        metadata.num_decodes * metadata.draft_token_num
        if metadata.is_target_verify
        else metadata.num_decodes
    )
    if hidden_states.shape[0] != num_actual_tokens:
        hidden_states = hidden_states[:num_actual_tokens]

    ret = mamba_layer._forward_mamba(hidden_states, forward_batch)

    # Copy result back; output may be larger (padded) so only fill actual tokens
    output[:num_actual_tokens].view(ret.shape).copy_(ret)
    if output.shape[0] != num_actual_tokens:
        output[num_actual_tokens:].zero_()


breakable_nemotron_mamba2_with_output = eager_on_graph(True)(
    nemotron_mamba2_with_output
)
