import argparse
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams



def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Profile one offline vLLM generation request with torch profiler."
	)
	parser.add_argument("--model", required=True, help="Local model path or HF repo name.")
	parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
	parser.add_argument("--prompt", required=True, help="Prompt to run.")
	parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens.")
	parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
	parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
	parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling.")
	parser.add_argument("--max-model-len", type=int, default=262144, help="Maximum model length.")
	parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization.")
	parser.add_argument("--profile-dir", default="./vllm_profile", help="Directory to store profiler traces.")
	parser.add_argument("--summary-file", default="./vllm_profile/profile_summary.txt", help="Summary output file.")
	parser.add_argument("--warmup", action="store_true", help="Run one warmup request before profiling.")
	parser.add_argument("--sleep-after-stop", type=float, default=10.0, help="Seconds to wait after stop_profile for trace flush.")
	return parser.parse_args()



def main() -> int:
	args = parse_args()

	profile_dir = Path(args.profile_dir).resolve()
	profile_dir.mkdir(parents=True, exist_ok=True)
	os.environ["VLLM_TORCH_PROFILER_DIR"] = str(profile_dir)

	sampling_params = SamplingParams(
		max_tokens=args.max_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		top_k=args.top_k,
	)

	llm = LLM(
		model=args.model,
		tensor_parallel_size=args.tensor_parallel_size,
		trust_remote_code=True,
		max_model_len=args.max_model_len,
		gpu_memory_utilization=args.gpu_memory_utilization,
	)

	prompts = [args.prompt]

	if args.warmup:
		print("[INFO] Running warmup request...")
		warmup_outputs = llm.generate(prompts, sampling_params)
		for output in warmup_outputs:
			if output.outputs:
				print("[WARMUP OUTPUT]", output.outputs[0].text)

	print("[INFO] Starting profiler...")
	llm.start_profile()

	outputs = llm.generate(prompts, sampling_params)

	print("[INFO] Stopping profiler...")
	llm.stop_profile()

	for output in outputs:
		print("\n===== prompt =====\n")
		print(output.prompt)
		print("\n===== generated =====\n")
		if output.outputs:
			print(output.outputs[0].text)
		else:
			print("<empty>")

	summary_file = Path(args.summary_file).resolve()
	summary_file.parent.mkdir(parents=True, exist_ok=True)
	with summary_file.open("w", encoding="utf-8") as handle:
		handle.write("Profile completed.\n")
		handle.write(f"Profile directory: {profile_dir}\n")
		handle.write("Open the trace in TensorBoard or Chrome trace viewer if available.\n")

	print(f"[INFO] Trace directory: {profile_dir}")
	print(f"[INFO] Summary file: {summary_file}")
	print(f"[INFO] Sleeping {args.sleep_after_stop} seconds for trace flush...")
	time.sleep(args.sleep_after_stop)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
