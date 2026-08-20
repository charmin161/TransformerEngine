(EngineCore pid=4076649) INFO 08-20 16:38:29 [core.py:116] Initializing a V1 LLM engine (v0.26.0) with config: model='/wireless/public/models/GLM-5.2-NVFP4', speculative_config=None, tokenizer='/wireless/public/models/GLM-5.2-NVFP4', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=819200, download_dir=None, load_format=auto, tensor_parallel_size=4, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=modelopt_fp4, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=fp8_e4m3, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='glm45', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=0, served_model_name=GLM5.2-NVFP4, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'ir_enable_torch_wrap': True, 'splitting_ops': ['vllm::unified_attention_with_output', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::qwen_gdn_attention_core', 'vllm::gdn_attention_core_xpu', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::deepseek_v4_attention', 'vllm::hpc_rope_norm_forward', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [2730, 8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': True, 'enable_qk_norm_rope_fusion': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False, 'fuse_qk_norm_rope_kvcache': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native']), enable_flashinfer_autotune=True, enable_cutedsl_warmup=True, enable_bf16x3_router_gemm=False, moe_backend='auto', linear_backend='auto')
(EngineCore pid=4076649) WARNING 08-20 16:38:29 [multiproc_executor.py:1070] Reducing Torch parallelism from 144 threads to 1 to avoid unnecessary CPU contention. Set OMP_NUM_THREADS in the external environment to tune this value as needed.
(EngineCore pid=4076649) INFO 08-20 16:38:29 [multiproc_executor.py:140] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=10.207.212.181 (local), world_size=4, local_world_size=4
(Worker pid=4155524) INFO 08-20 16:42:11 [parallel_state.py:1615] world_size=4 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:52621 backend=nccl
(Worker pid=4155525) INFO 08-20 16:42:11 [parallel_state.py:1615] world_size=4 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:52621 backend=nccl
(Worker pid=4155526) INFO 08-20 16:42:11 [parallel_state.py:1615] world_size=4 rank=2 local_rank=2 distributed_init_method=tcp://127.0.0.1:52621 backend=nccl
(Worker pid=4155527) INFO 08-20 16:42:11 [parallel_state.py:1615] world_size=4 rank=3 local_rank=3 distributed_init_method=tcp://127.0.0.1:52621 backend=nccl
(Worker pid=4155524) INFO 08-20 16:42:21 [pynccl.py:113] vLLM is using nccl==2.28.9
(Worker pid=4155524) INFO 08-20 16:42:23 [cuda_communicator.py:264] Using ['CUSTOM', 'SYMM_MEM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=4155524) INFO 08-20 16:42:24 [cuda_communicator.py:264] Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'ep:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=4155524) INFO 08-20 16:42:24 [parallel_state.py:1946] rank 0 in world size 4 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0, EPLB rank N/A
(Worker pid=4155524) INFO 08-20 16:42:26 [topk_topp_sampler.py:55] Using FlashInfer for top-p & top-k sampling.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:27 [gpu_model_runner.py:5250] Starting to load model /wireless/public/models/GLM-5.2-NVFP4...
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [cuda.py:482] Using FLASHINFER_MLA_SPARSE attention backend out of potential backends: ['FLASHINFER_MLA_SPARSE'].
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [selector.py:202] Using HND KV cache layout for FLASHINFER_MLA_SPARSE backend.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [mla_attention.py:451] Using standard fp8 KV cache format. To use DeepSeek's fp8_ds_mla KV cache format, please set `--attention-backend FLASHMLA_SPARSE`
(Worker_TP1_EP1 pid=4155525) INFO 08-20 16:42:29 [selector.py:202] Using HND KV cache layout for FLASHINFER_MLA_SPARSE backend.
(Worker_TP3_EP3 pid=4155527) INFO 08-20 16:42:29 [selector.py:202] Using HND KV cache layout for FLASHINFER_MLA_SPARSE backend.
(Worker_TP2_EP2 pid=4155526) INFO 08-20 16:42:29 [selector.py:202] Using HND KV cache layout for FLASHINFER_MLA_SPARSE backend.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [selector.py:174] Using TRTLLM_RAGGED MLA prefill backend.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [deep_gemm.py:175] deep_gemm not found in site-packages, trying vendored vllm.third_party.deep_gemm
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [deep_gemm.py:202] DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [deep_gemm.py:120] DeepGEMM E8M0 enabled on current platform.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [expert_map_manager.py:245] [EP Rank 0/4] Expert parallelism is enabled. Expert placement strategy: linear. Local/global number of experts: 64/256. Experts local to global index map: 0->0, 1->1, 2->2, 3->3, 4->4, 5->5, 6->6, 7->7, 8->8, 9->9, 10->10, 11->11, 12->12, 13->13, 14->14, 15->15, 16->16, 17->17, 18->18, 19->19, 20->20, 21->21, 22->22, 23->23, 24->24, 25->25, 26->26, 27->27, 28->28, 29->29, 30->30, 31->31, 32->32, 33->33, 34->34, 35->35, 36->36, 37->37, 38->38, 39->39, 40->40, 41->41, 42->42, 43->43, 44->44, 45->45, 46->46, 47->47, 48->48, 49->49, 50->50, 51->51, 52->52, 53->53, 54->54, 55->55, 56->56, 57->57, 58->58, 59->59, 60->60, 61->61, 62->62, 63->63.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:29 [nvfp4.py:285] Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTEDSL_BATCHED', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN', 'HUMMING', 'EMULATION'].
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:32 [weight_utils.py:869] Filesystem type for checkpoints: AUTOFS. Checkpoint size: 432.90 GiB. Available RAM: 748.13 GiB.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 16:42:32 [weight_utils.py:892] Auto-prefetch is disabled because the filesystem (AUTOFS) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/47 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:   2% Completed | 1/47 [00:41<31:40, 41.31s/it]
Loading safetensors checkpoint shards:   4% Completed | 2/47 [01:31<34:52, 46.49s/it]
Loading safetensors checkpoint shards:   6% Completed | 3/47 [02:25<36:33, 49.84s/it]
Loading safetensors checkpoint shards:   9% Completed | 4/47 [02:57<30:47, 42.96s/it]
Loading safetensors checkpoint shards:  11% Completed | 5/47 [03:49<32:11, 45.99s/it]
Loading safetensors checkpoint shards:  13% Completed | 6/47 [04:47<34:18, 50.21s/it]
Loading safetensors checkpoint shards:  15% Completed | 7/47 [05:37<33:30, 50.26s/it]
Loading safetensors checkpoint shards:  17% Completed | 8/47 [06:14<29:50, 45.91s/it]
(Worker_TP2_EP2 pid=4155526) INFO 08-20 16:49:43 [interface.py:635] Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER backend.
Loading safetensors checkpoint shards:  19% Completed | 9/47 [07:11<31:17, 49.41s/it]
(Worker_TP3_EP3 pid=4155527) INFO 08-20 16:49:53 [interface.py:635] Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER backend.
Loading safetensors checkpoint shards:  21% Completed | 10/47 [08:03<30:57, 50.21s/it]
Loading safetensors checkpoint shards:  23% Completed | 11/47 [08:35<26:44, 44.57s/it]
Loading safetensors checkpoint shards:  26% Completed | 12/47 [09:29<27:43, 47.53s/it]
Loading safetensors checkpoint shards:  28% Completed | 13/47 [10:19<27:21, 48.27s/it]
Loading safetensors checkpoint shards:  30% Completed | 14/47 [11:02<25:42, 46.74s/it]
Loading safetensors checkpoint shards:  32% Completed | 15/47 [11:38<23:13, 43.56s/it]
Loading safetensors checkpoint shards:  34% Completed | 16/47 [12:34<24:26, 47.31s/it]
Loading safetensors checkpoint shards:  36% Completed | 17/47 [13:33<25:16, 50.56s/it]
Loading safetensors checkpoint shards:  38% Completed | 18/47 [14:11<22:44, 47.05s/it]
Loading safetensors checkpoint shards:  40% Completed | 19/47 [15:15<24:14, 51.94s/it]
Loading safetensors checkpoint shards:  43% Completed | 20/47 [16:09<23:40, 52.60s/it]
Loading safetensors checkpoint shards:  45% Completed | 21/47 [16:51<21:23, 49.36s/it]
Loading safetensors checkpoint shards:  47% Completed | 22/47 [19:01<30:41, 73.64s/it]
Loading safetensors checkpoint shards:  49% Completed | 23/47 [21:03<35:12, 88.01s/it]
Loading safetensors checkpoint shards:  51% Completed | 24/47 [22:07<31:00, 80.90s/it]
Loading safetensors checkpoint shards:  53% Completed | 25/47 [22:51<25:34, 69.76s/it]
Loading safetensors checkpoint shards:  55% Completed | 26/47 [24:10<25:22, 72.52s/it]
Loading safetensors checkpoint shards:  57% Completed | 27/47 [25:17<23:40, 71.02s/it]
Loading safetensors checkpoint shards:  60% Completed | 28/47 [26:04<20:14, 63.92s/it]
Loading safetensors checkpoint shards:  62% Completed | 29/47 [29:21<31:05, 103.66s/it]
Loading safetensors checkpoint shards:  64% Completed | 30/47 [30:17<25:17, 89.28s/it]
Loading safetensors checkpoint shards:  66% Completed | 31/47 [31:02<20:18, 76.13s/it]
Loading safetensors checkpoint shards:  68% Completed | 32/47 [31:23<14:51, 59.44s/it]
Loading safetensors checkpoint shards:  70% Completed | 33/47 [32:04<12:35, 53.97s/it]
Loading safetensors checkpoint shards:  72% Completed | 34/47 [32:57<11:39, 53.80s/it]
Loading safetensors checkpoint shards:  74% Completed | 35/47 [33:37<09:53, 49.49s/it]
Loading safetensors checkpoint shards:  77% Completed | 36/47 [34:34<09:30, 51.85s/it]
Loading safetensors checkpoint shards:  79% Completed | 37/47 [35:09<07:49, 46.91s/it]
Loading safetensors checkpoint shards:  81% Completed | 38/47 [35:43<06:27, 43.00s/it]
Loading safetensors checkpoint shards:  83% Completed | 39/47 [35:49<04:15, 31.89s/it]
Loading safetensors checkpoint shards:  85% Completed | 40/47 [36:03<03:05, 26.45s/it]
Loading safetensors checkpoint shards:  87% Completed | 41/47 [36:06<01:56, 19.46s/it]
Loading safetensors checkpoint shards:  89% Completed | 42/47 [36:28<01:40, 20.07s/it]
Loading safetensors checkpoint shards:  91% Completed | 43/47 [37:04<01:39, 24.99s/it]
Loading safetensors checkpoint shards:  94% Completed | 44/47 [37:37<01:21, 27.30s/it]
Loading safetensors checkpoint shards:  96% Completed | 45/47 [37:38<00:39, 19.64s/it]
Loading safetensors checkpoint shards: 100% Completed | 47/47 [37:39<00:00, 10.60s/it]
Loading safetensors checkpoint shards: 100% Completed | 47/47 [37:39<00:00, 48.07s/it]
(Worker_TP0_EP0 pid=4155524) 
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:20:12 [default_loader.py:430] Loading weights took 2259.09 seconds
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:20:12 [kv_cache.py:134] Checkpoint does not provide a q scaling factor. Setting it to k_scale. This only matters for FP8 Attention backends (flash-attn or flashinfer).
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:20:12 [kv_cache.py:151] Using KV cache scaling factor 1.0 for fp8_e4m3. If this is unintended, verify that k/v_scale scaling factors are properly set in the checkpoint.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:20:18 [nvfp4.py:544] Using MoEPrepareAndFinalizeNoDPEPMonolithic
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:20:27 [gpu_model_runner.py:5347] Model loading took 106.69 GiB memory and 2278.911946 seconds
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:20:27 [interface.py:635] Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER backend.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:20:27 [interface.py:635] Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER backend.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:17 [backends.py:1094] Using cache directory: /root/.cache/vllm/torch_compile_cache/cf4970ccaf/rank_0_0/backbone for vLLM's torch.compile
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:17 [backends.py:1155] Dynamo bytecode transform time: 48.90 s
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:17 [flashinfer_all_reduce.py:121] Auto-selected flashinfer allreduce backend: mnnvl
(Worker_TP0_EP0 pid=4155524) /wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
(Worker_TP0_EP0 pid=4155524)   return func(*args, **kwargs)
[rank0]:[W820 17:21:17.820265420 ProcessGroupNCCL.cpp:5188] Guessing device ID based on global rank. This can cause a hang if rank to GPU mapping is heterogeneous. You can specify device_id in init_process_group()
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:18 [flashinfer_all_reduce.py:170] Initialized FlashInfer Allreduce norm fusion workspace with backend=mnnvl
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/79acdf58eca046c7cc8e0bddfd4c91c8700d6c14e547349ab19057a6c3af3618/rank_3_0/model, reason: Kernel index 1 not found in id_to_kernel
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321] 
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321] While executing %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 0, grid: [(%s72, 32, 1)], tma_descriptor_metadata: {}, kwargs: {positions: %l_positions_, q: %view_3, cos_sin_cache: %l_self_modules_layers_modules_0_modules_self_attn_modules_mla_attn_modules_rotary_emb_buffers_cos_sin_cache_, q_fp8: %empty_like, weights: %getitem_16, weights_out: %empty_like_1}})
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321] Original traceback:
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1476, in forward
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]     hidden_states, residual = layer(
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1300, in forward
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]     hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1174, in forward
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]     return self.mla_attn(positions, hidden_states, llama_4_scaling)
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/layers/mla.py", line 170, in forward
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]     self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 760, in forward
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]     q_fp8, weights = fused_indexer_q_rope_quant(
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py", line 228, in fused_indexer_q_rope_quant
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321]     _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321] 
(Worker_TP3_EP3 pid=4155527) WARNING 08-20 17:21:21 [decorators.py:321] Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/79acdf58eca046c7cc8e0bddfd4c91c8700d6c14e547349ab19057a6c3af3618/rank_0_0/model, reason: Kernel index 1 not found in id_to_kernel
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321] 
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321] While executing %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 0, grid: [(%s72, 32, 1)], tma_descriptor_metadata: {}, kwargs: {positions: %l_positions_, q: %view_3, cos_sin_cache: %l_self_modules_layers_modules_0_modules_self_attn_modules_mla_attn_modules_rotary_emb_buffers_cos_sin_cache_, q_fp8: %empty_like, weights: %getitem_16, weights_out: %empty_like_1}})
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321] Original traceback:
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1476, in forward
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]     hidden_states, residual = layer(
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1300, in forward
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]     hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1174, in forward
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]     return self.mla_attn(positions, hidden_states, llama_4_scaling)
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/layers/mla.py", line 170, in forward
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]     self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 760, in forward
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]     q_fp8, weights = fused_indexer_q_rope_quant(
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py", line 228, in fused_indexer_q_rope_quant
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321]     _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321] 
(Worker_TP0_EP0 pid=4155524) WARNING 08-20 17:21:21 [decorators.py:321] Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/79acdf58eca046c7cc8e0bddfd4c91c8700d6c14e547349ab19057a6c3af3618/rank_2_0/model, reason: Kernel index 1 not found in id_to_kernel
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321] 
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321] While executing %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 0, grid: [(%s72, 32, 1)], tma_descriptor_metadata: {}, kwargs: {positions: %l_positions_, q: %view_3, cos_sin_cache: %l_self_modules_layers_modules_0_modules_self_attn_modules_mla_attn_modules_rotary_emb_buffers_cos_sin_cache_, q_fp8: %empty_like, weights: %getitem_16, weights_out: %empty_like_1}})
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321] Original traceback:
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1476, in forward
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]     hidden_states, residual = layer(
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1300, in forward
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]     hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1174, in forward
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]     return self.mla_attn(positions, hidden_states, llama_4_scaling)
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/layers/mla.py", line 170, in forward
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]     self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 760, in forward
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]     q_fp8, weights = fused_indexer_q_rope_quant(
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py", line 228, in fused_indexer_q_rope_quant
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321]     _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321] 
(Worker_TP2_EP2 pid=4155526) WARNING 08-20 17:21:21 [decorators.py:321] Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/79acdf58eca046c7cc8e0bddfd4c91c8700d6c14e547349ab19057a6c3af3618/rank_1_0/model, reason: Kernel index 1 not found in id_to_kernel
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321] 
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321] While executing %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 0, grid: [(%s72, 32, 1)], tma_descriptor_metadata: {}, kwargs: {positions: %l_positions_, q: %view_3, cos_sin_cache: %l_self_modules_layers_modules_0_modules_self_attn_modules_mla_attn_modules_rotary_emb_buffers_cos_sin_cache_, q_fp8: %empty_like, weights: %getitem_16, weights_out: %empty_like_1}})
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321] Original traceback:
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1476, in forward
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]     hidden_states, residual = layer(
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1300, in forward
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]     hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 1174, in forward
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]     return self.mla_attn(positions, hidden_states, llama_4_scaling)
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/layers/mla.py", line 170, in forward
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]     self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py", line 760, in forward
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]     q_fp8, weights = fused_indexer_q_rope_quant(
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]   File "/wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py", line 228, in fused_indexer_q_rope_quant
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321]     _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321] 
(Worker_TP1_EP1 pid=4155525) WARNING 08-20 17:21:21 [decorators.py:321] Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)
(EngineCore pid=4076649) INFO 08-20 17:21:28 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:45 [backends.py:1155] Dynamo bytecode transform time: 24.60 s
(Worker_TP0_EP0 pid=4155524) /wireless/minyusong/glm_5_2/glm_5_2/lib/python3.12/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
(Worker_TP0_EP0 pid=4155524)   return func(*args, **kwargs)
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:48 [backends.py:378] Cache the graph of compile range (1, 2730) for later use
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:48 [backends.py:378] Cache the graph of compile range (2731, 8192) for later use
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:54 [backends.py:393] Compiling a graph for compile range (1, 2730) takes 5.57 s
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:21:54 [backends.py:393] Compiling a graph for compile range (2731, 8192) takes 5.60 s
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:16 [decorators.py:708] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/79acdf58eca046c7cc8e0bddfd4c91c8700d6c14e547349ab19057a6c3af3618/rank_0_0/model
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:16 [monitor.py:53] torch.compile took 55.25 s in total
[2026-08-20 17:22:25.251] [info] TRT-LLM fused MoE cooperative launch SM allocation: 144 SMs used for MoE, 8 SMs reserved for overlapping kernels (total SMs: 152)
[2026-08-20 17:22:25.287] [info] TRT-LLM fused MoE cooperative launch SM allocation: 144 SMs used for MoE, 8 SMs reserved for overlapping kernels (total SMs: 152)
[2026-08-20 17:22:25.357] [info] TRT-LLM fused MoE cooperative launch SM allocation: 144 SMs used for MoE, 8 SMs reserved for overlapping kernels (total SMs: 152)
[2026-08-20 17:22:25.405] [info] TRT-LLM fused MoE cooperative launch SM allocation: 144 SMs used for MoE, 8 SMs reserved for overlapping kernels (total SMs: 152)
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:28 [monitor.py:81] Initial profiling/warmup run took 11.46 s
(EngineCore pid=4076649) INFO 08-20 17:22:28 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:29 [indexer.py:306] DSA indexer decode path: use_flattening=False (next_n=1, use_fp4_indexer_cache=False)
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:22:30 [gpu_model_runner.py:6612] Profiling CUDA graph memory: PIECEWISE=51 (largest=512), FULL=51 (largest=512)
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:22:30 [gpu_model_runner.py:6612] Profiling CUDA graph memory: PIECEWISE=51 (largest=512), FULL=51 (largest=512)
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:30 [gpu_model_runner.py:6612] Profiling CUDA graph memory: PIECEWISE=51 (largest=512), FULL=51 (largest=512)
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:22:30 [gpu_model_runner.py:6612] Profiling CUDA graph memory: PIECEWISE=51 (largest=512), FULL=51 (largest=512)
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:22:34 [custom_all_reduce.py:213] Registering 0 cuda graph addresses
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:22:34 [custom_all_reduce.py:213] Registering 0 cuda graph addresses
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:34 [custom_all_reduce.py:213] Registering 0 cuda graph addresses
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:22:34 [custom_all_reduce.py:213] Registering 0 cuda graph addresses
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:22:36 [gpu_model_runner.py:6737] Estimated CUDA graph memory: 4.99 GiB total
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:22:36 [gpu_model_runner.py:6737] Estimated CUDA graph memory: 4.99 GiB total
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:22:36 [gpu_model_runner.py:6737] Estimated CUDA graph memory: 4.99 GiB total
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:36 [gpu_model_runner.py:6737] Estimated CUDA graph memory: 4.99 GiB total
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:22:36 [gpu_worker.py:575] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.8929 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9471. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:22:36 [gpu_worker.py:575] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.8929 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9471. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:22:37 [gpu_worker.py:575] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.8929 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9471. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:37 [gpu_worker.py:560] Available KV cache memory: 50.62 GiB
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:37 [gpu_worker.py:575] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.8929 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9471. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=4076649) INFO 08-20 17:22:37 [kv_cache_utils.py:2177] GPU KV cache size: 1,139,456 tokens
(EngineCore pid=4076649) INFO 08-20 17:22:37 [kv_cache_utils.py:2178] Maximum concurrency for 819,200 tokens per request: 1.39x
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:22:37 [gpu_worker.py:774] Compile and warming up model for size 8192
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:22:37 [gpu_worker.py:774] Compile and warming up model for size 8192
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:22:37 [gpu_worker.py:774] Compile and warming up model for size 8192
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:22:37 [gpu_worker.py:774] Compile and warming up model for size 8192
(Worker_TP3_EP3 pid=4155527) 2026-08-20 17:22:38,159 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker_TP2_EP2 pid=4155526) 2026-08-20 17:22:38,160 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker_TP0_EP0 pid=4155524) 2026-08-20 17:22:38,164 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker_TP1_EP1 pid=4155525) 2026-08-20 17:22:38,168 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:   0%|                                                                                                | 0/21 [00:00<?, ?profile/s]([AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:   0%|                                                                                                | 0/21 [00:00<?, ?profile/s]([AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:   0%|                                                                                                | 0/21 [00:00<?, ?profile/s]([AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:  14%|████████████▌                                                                           | 3/21 [01:00<06:02, 20.15s/profile](EngineCore pid=4076649) INFO 08-20 17:23:38 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:  24%|████████████████████▉                                                                   | 5/21 [01:59<06:28, 24.25s/profile](EngineCore pid=4076649) INFO 08-20 17:24:38 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:  33%|█████████████████████████████▎                                                          | 7/21 [02:51<05:51, 25.14s/profile](EngineCore pid=4076649) INFO 08-20 17:25:38 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe: 100%|███████████████████████████████████████████████████████████████████████████████████████| 21/21 [03:21<00:00,  9.60s/profile]
[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe: 100%|███████████████████████████████████████████████████████████████████████████████████████| 21/21 [03:24<00:00,  9.73s/profile]
[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe: 100%|███████████████████████████████████████████████████████████████████████████████████████| 21/21 [03:48<00:00, 10.88s/profile]
[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe: 100%|███████████████████████████████████████████████████████████████████████████████████████| 21/21 [03:57<00:00, 11.30s/profile]
(Worker_TP3_EP3 pid=4155527) 2026-08-20 17:26:35,708 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP2_EP2 pid=4155526) 2026-08-20 17:26:35,708 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP1_EP1 pid=4155525) 2026-08-20 17:26:35,708 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP0_EP0 pid=4155524) 2026-08-20 17:26:35,933 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:26:35 [kernel_warmup.py:65] Warming up ll_bf16 router GEMM kernels.
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:26:35 [kernel_warmup.py:65] Warming up ll_bf16 router GEMM kernels.
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:26:35 [kernel_warmup.py:65] Warming up ll_bf16 router GEMM kernels.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:26:35 [kernel_warmup.py:65] Warming up ll_bf16 router GEMM kernels.
(EngineCore pid=4076649) INFO 08-20 17:26:38 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:26:46 [cutedsl_warmup.py:101] Skipping CuTeDSL warmup because no compile units were requested.
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:26:46 [cutedsl_warmup.py:101] Skipping CuTeDSL warmup because no compile units were requested.
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:26:47 [cutedsl_warmup.py:101] Skipping CuTeDSL warmup because no compile units were requested.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:26:47 [cutedsl_warmup.py:101] Skipping CuTeDSL warmup because no compile units were requested.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:26:47 [gpu_model_runner.py:6798] Rank 0: Torch profiler disabled for CUDA graph capture
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:49<00:00,  1.03it/s]
Capturing CUDA graphs (decode, FULL):   6%|██████▊                                                                                                            | 3/51 [00:01<00:24,  1.95it/s](EngineCore pid=4076649) INFO 08-20 17:27:38 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
Capturing CUDA graphs (decode, FULL):  98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▊  | 50/51 [00:29<00:00,  1.61it/s](Worker_TP2_EP2 pid=4155526) INFO 08-20 17:28:07 [custom_all_reduce.py:213] Registering 0 cuda graph addresses                                                                                 
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:28:07 [custom_all_reduce.py:213] Registering 0 cuda graph addresses
Capturing CUDA graphs (decode, FULL): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:30<00:00,  1.67it/s]
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:28:07 [custom_all_reduce.py:213] Registering 0 cuda graph addresses
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:28:07 [custom_all_reduce.py:213] Registering 0 cuda graph addresses
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:28:08 [gpu_worker.py:793] CUDA graph pool memory: 4.13 GiB (actual), 4.99 GiB (estimated), difference: 0.86 GiB (20.9%).
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:28:08 [gpu_worker.py:857] Free memory on device (181.84/184.31 GiB) on startup. Desired GPU memory utilization is (0.92, 169.56 GiB). Actual usage is 106.69 GiB for weight, 6.31 GiB for peak activation, 0.95 GiB for non-torch memory, and 4.13 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=55124106138` (51.34 GiB) to fit into requested memory, or `--kv-cache-memory=68302635520` (63.61 GiB) to fully utilize gpu memory. Current kv cache memory in use is 50.62 GiB.
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:28:08 [gpu_worker.py:793] CUDA graph pool memory: 4.13 GiB (actual), 4.99 GiB (estimated), difference: 0.86 GiB (20.9%).
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:28:08 [gpu_worker.py:857] Free memory on device (181.84/184.31 GiB) on startup. Desired GPU memory utilization is (0.92, 169.56 GiB). Actual usage is 106.69 GiB for weight, 6.31 GiB for peak activation, 0.95 GiB for non-torch memory, and 4.13 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=55124106138` (51.34 GiB) to fit into requested memory, or `--kv-cache-memory=68302635520` (63.61 GiB) to fully utilize gpu memory. Current kv cache memory in use is 50.62 GiB.
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:28:08 [gpu_worker.py:793] CUDA graph pool memory: 4.13 GiB (actual), 4.99 GiB (estimated), difference: 0.86 GiB (20.9%).
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:28:08 [gpu_worker.py:857] Free memory on device (181.84/184.31 GiB) on startup. Desired GPU memory utilization is (0.92, 169.56 GiB). Actual usage is 106.69 GiB for weight, 6.31 GiB for peak activation, 0.95 GiB for non-torch memory, and 4.13 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=55124106138` (51.34 GiB) to fit into requested memory, or `--kv-cache-memory=68302635520` (63.61 GiB) to fully utilize gpu memory. Current kv cache memory in use is 50.62 GiB.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:28:08 [gpu_model_runner.py:6844] Graph capturing finished in 82 secs, took 4.13 GiB
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:28:08 [gpu_worker.py:793] CUDA graph pool memory: 4.13 GiB (actual), 4.99 GiB (estimated), difference: 0.86 GiB (20.9%).
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:28:08 [gpu_worker.py:857] Free memory on device (181.84/184.31 GiB) on startup. Desired GPU memory utilization is (0.92, 169.56 GiB). Actual usage is 106.69 GiB for weight, 6.31 GiB for peak activation, 0.95 GiB for non-torch memory, and 4.13 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=55124106138` (51.34 GiB) to fit into requested memory, or `--kv-cache-memory=68302635520` (63.61 GiB) to fully utilize gpu memory. Current kv cache memory in use is 50.62 GiB.
(Worker_TP3_EP3 pid=4155527) INFO 08-20 17:28:15 [jit_monitor.py:79] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP0_EP0 pid=4155524) INFO 08-20 17:28:15 [jit_monitor.py:79] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP2_EP2 pid=4155526) INFO 08-20 17:28:15 [jit_monitor.py:79] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP1_EP1 pid=4155525) INFO 08-20 17:28:15 [jit_monitor.py:79] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(EngineCore pid=4076649) INFO 08-20 17:28:16 [core.py:340] init engine (profile, create kv cache, warmup model) took 469.04 s (compilation: 55.52 s)
(EngineCore pid=4076649) INFO 08-20 17:28:19 [vllm.py:1109] Asynchronous scheduling is enabled.
(EngineCore pid=4076649) INFO 08-20 17:28:19 [kernel.py:295] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(EngineCore pid=4076649) INFO 08-20 17:28:27 [compilation.py:329] Enabled custom fusions: act_quant, allreduce_rms
(APIServer pid=3986999) INFO 08-20 17:28:27 [api_server.py:673] Supported tasks: ['generate']
(APIServer pid=3986999) WARNING 08-20 17:28:30 [model.py:1546] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 1.0, 'top_p': 0.95}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=3986999) INFO 08-20 17:28:32 [hf.py:540] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
(APIServer pid=3986999) INFO 08-20 17:28:32 [api_server.py:677] Starting vLLM server on http://127.0.0.1:8972
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:37] Available routes are:
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /generative_scoring, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/chat/completions/derender, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /v1/completions/derender, Methods: POST
(APIServer pid=3986999) INFO 08-20 17:28:32 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=3986999) INFO:     Started server process [3986999]
(APIServer pid=3986999) INFO:     Waiting for application startup.
(APIServer pid=3986999) INFO:     Application startup complete.

推理配置为：
python3 -m vllm.entrypoints.openai.api_server \
    --model /wireless/public/models/GLM-5.2-NVFP4 \
    --tensor-parallel-size 4 \
    --quantization modelopt \
    --trust-remote-code \
    --disable-log-stats \
    --port 8972 \
    --host "127.0.0.1"    \
    --enable-expert-parallel            \
    --reasoning-parser glm45            \
    --served-model-name GLM5.2-NVFP4    \
    --max-model-len 819200
