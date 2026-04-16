# Kimi-K2.5-NVFP4 vLLM Local Scheme

## 1. Start the server

```bash
export MODEL_PATH=/absolute/path/to/Kimi-K2.5-NVFP4
export SERVED_MODEL_NAME=Kimi-K2.5-NVFP4-local
export TP_SIZE=4
bash /mnt/data/start_vllm_kimi_k25.sh
```

## 2. Ask one question through the OpenAI-compatible API

```bash
python3 /mnt/data/chat_client.py \
	--base-url http://127.0.0.1:8000 \
	--model Kimi-K2.5-NVFP4-local \
	--prompt "Hello, introduce yourself briefly." \
	--max-tokens 128
```

## 3. Profile one offline request (cleaner than profiling the HTTP server)

```bash
python3 /mnt/data/profile_one_offline_request.py \
	--model /absolute/path/to/Kimi-K2.5-NVFP4 \
	--tensor-parallel-size 4 \
	--prompt "Explain the difference between dense MLP and MoE in one paragraph." \
	--max-tokens 128 \
	--warmup \
	--profile-dir ./vllm_profile
```

## Notes

- The API server route is useful for normal chatting and integration tests.
- The offline profiler route is better when you want a cleaner single-request trace.
- The profiler trace directory is controlled by `VLLM_TORCH_PROFILER_DIR`.
