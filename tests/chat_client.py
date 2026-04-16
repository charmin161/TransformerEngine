import argparse
import json
import sys
import urllib.error
import urllib.request


def build_payload(model: str, prompt: str, max_tokens: int, temperature: float) -> dict:
	return {
		"model": model,
		"messages": [
			{"role": "user", "content": prompt},
		],
		"max_tokens": max_tokens,
		"temperature": temperature,
	}



def main() -> int:
	parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible chat client for vLLM.")
	parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL.")
	parser.add_argument("--model", default="Kimi-K2.5-NVFP4-local", help="Served model name.")
	parser.add_argument("--prompt", required=True, help="User prompt.")
	parser.add_argument("--max-tokens", type=int, default=256, help="Maximum output tokens.")
	parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
	args = parser.parse_args()

	url = args.base_url.rstrip("/") + "/v1/chat/completions"
	payload = build_payload(
		model=args.model,
		prompt=args.prompt,
		max_tokens=args.max_tokens,
		temperature=args.temperature,
	)
	data = json.dumps(payload).encode("utf-8")
	request = urllib.request.Request(
		url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)

	try:
		with urllib.request.urlopen(request) as response:
			body = response.read().decode("utf-8")
	except urllib.error.HTTPError as exc:
		body = exc.read().decode("utf-8", errors="replace")
		print(f"HTTP {exc.code}: {body}", file=sys.stderr)
		return 1
	except urllib.error.URLError as exc:
		print(f"Connection error: {exc}", file=sys.stderr)
		return 1

	parsed = json.loads(body)
	print(json.dumps(parsed, ensure_ascii=False, indent=2))

	try:
		content = parsed["choices"][0]["message"].get("content", "")
	except (KeyError, IndexError, AttributeError):
		content = ""

	if content:
		print("\n===== assistant =====\n")
		print(content)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
