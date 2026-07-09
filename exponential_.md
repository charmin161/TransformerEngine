在 DeepSeek V4-Pro 公开参考推理代码里，这个采样运算的数据量是：

```text
logits / probs / exponential_noise shape = [batch_size, vocab_size]
```

不是 `[batch_size, seq_len, vocab_size]`。

原因是 `ParallelHead.get_logits()` 里只取了最后一个位置：

```python
return F.linear(x[:, -1].float(), self.weight)
```

所以每个 decode step 只对“当前要生成的下一个 token”做一次采样。DeepSeek V4-Pro 配置里的 `vocab_size = 129280`，因此交互式推理默认 `batch_size=1` 时，`sample()` 里的 `probs` 和 `exponential_` 噪声张量通常是：

```text
[1, 129280]
```

也就是每步大约 12.9 万个 float32 元素。单个 `[1,129280]` float32 tensor 大约：

```text
129280 * 4 bytes ≈ 505 KiB
```

`probs` 一个，`exponential_noise` 一个，临时参与 `div_ + argmax`。如果 `batch_size=4`，就是：

```text
[4, 129280] ≈ 2.0 MiB / tensor
```

如果是多卡 tensor parallel，`ParallelHead` 会先在每个 rank 上算本地 `part_vocab_size = vocab_size // world_size`，然后 `all_gather` 拼回完整 vocab logits，所以 `sample()` 看到的仍然是完整 `[B, 129280]`。([Hugging Face][1])

DeepSeek V4-Pro 的 `generate.py` 里，`sample()` 代码就是：

```python
logits = logits / max(temperature, 1e-5)
probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
```

代码注释明确说它是 Gumbel-max trick，等价于 multinomial sampling，但在 GPU 上更快，因为避免 `torch.multinomial` 的 GPU-to-CPU sync。([Hugging Face][2])

下面给你一个可以直接跑的验证脚本。注意：**两个方法不会逐样本输出相同 token**，因为随机数不同；验证的是“统计分布相同”。

```python
# compare_multinomial_vs_exponential_race.py
import argparse
import time
import torch


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_probs(vocab_size: int, device: torch.device, seed: int, mode: str):
    """
    构造一个离散概率分布。
    mode='peaked' 更像 LLM logits：少数 token 概率较高，大量 token 概率很低。
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    if mode == "uniform":
        probs = torch.ones(vocab_size, device=device, dtype=torch.float32)
        probs /= probs.sum()
        return probs

    logits = torch.randn(vocab_size, device=device, generator=g, dtype=torch.float32)

    if mode == "peaked":
        # 人为制造一些高概率 token，便于统计观察。
        k = min(50, vocab_size)
        logits[:k] += torch.linspace(8.0, 2.0, k, device=device)

    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs


@torch.inference_mode()
def sample_multinomial_counts(probs: torch.Tensor, trials: int):
    """
    torch.multinomial 的参考采样：
        sample ~ Categorical(probs)
    """
    sync(probs.device)
    t0 = time.time()

    samples = torch.multinomial(probs, num_samples=trials, replacement=True)
    counts = torch.bincount(samples, minlength=probs.numel()).to(torch.float32)

    sync(probs.device)
    return counts, time.time() - t0


@torch.inference_mode()
def sample_exponential_race_counts(probs: torch.Tensor, trials: int, chunk_size: int):
    """
    DeepSeek V4-Pro sample() 使用的等价形式：
        E_i ~ Exp(1)
        sample = argmax_i p_i / E_i

    为避免一次分配 [trials, vocab_size]，这里分 chunk 做。
    """
    device = probs.device
    vocab_size = probs.numel()
    counts = torch.zeros(vocab_size, device=device, dtype=torch.float32)

    sync(device)
    t0 = time.time()

    for start in range(0, trials, chunk_size):
        cur = min(chunk_size, trials - start)

        # 对应 DeepSeek 里的 torch.empty_like(probs).exponential_(1)
        # 这里为了并行做 cur 次采样，shape 是 [cur, vocab_size]
        noise = torch.empty((cur, vocab_size), device=device, dtype=torch.float32).exponential_(1.0)

        # 原地把 noise 变成 probs / Exp(1)，避免额外分配一个 scores tensor
        noise.reciprocal_()
        noise.mul_(probs)

        samples = noise.argmax(dim=-1)
        counts += torch.bincount(samples, minlength=vocab_size).to(torch.float32)

    sync(device)
    return counts, time.time() - t0


def summarize(probs, counts_m, counts_e, trials, topk):
    freq_m = counts_m / trials
    freq_e = counts_e / trials

    top_p, top_idx = torch.topk(probs, k=min(topk, probs.numel()))

    print("\n=== Top tokens comparison ===")
    print("rank | token_id | expected_p | multinomial_freq | exp_race_freq")
    print("-" * 72)
    for r, idx in enumerate(top_idx.tolist()):
        print(
            f"{r:4d} | {idx:8d} | "
            f"{probs[idx].item():10.6f} | "
            f"{freq_m[idx].item():16.6f} | "
            f"{freq_e[idx].item():13.6f}"
        )

    top_mass_expected = probs[top_idx].sum().item()
    top_mass_m = freq_m[top_idx].sum().item()
    top_mass_e = freq_e[top_idx].sum().item()

    print("\n=== Aggregate metrics ===")
    print(f"trials: {trials}")
    print(f"vocab_size: {probs.numel()}")
    print(f"top{len(top_idx)} expected mass:    {top_mass_expected:.6f}")
    print(f"top{len(top_idx)} multinomial mass: {top_mass_m:.6f}")
    print(f"top{len(top_idx)} exp-race mass:    {top_mass_e:.6f}")

    # 两个经验分布之间的差异；trials 越大越小
    l1_between = (freq_m - freq_e).abs().sum().item()
    max_abs_between = (freq_m - freq_e).abs().max().item()

    print(f"L1(freq_multinomial, freq_exp_race): {l1_between:.6f}")
    print(f"max_abs_diff_between_methods:        {max_abs_between:.6f}")

    # 和真实 probs 的误差；大 vocab + 有限 trials 时，全量 L1 会比较大，这是正常采样噪声
    max_abs_m = (freq_m - probs).abs().max().item()
    max_abs_e = (freq_e - probs).abs().max().item()
    print(f"max_abs(freq_multinomial - probs):   {max_abs_m:.6f}")
    print(f"max_abs(freq_exp_race - probs):      {max_abs_e:.6f}")


def one_step_deepseek_shape_demo(batch_size: int, vocab_size: int, device: torch.device):
    """
    模拟 DeepSeek sample() 单步输入形状：
        logits: [B, V]
        probs:  [B, V]
        noise:  [B, V]
        output: [B]
    """
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

    multinomial_out = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
    exp_race_out = probs.div(torch.empty_like(probs).exponential_(1.0)).argmax(dim=-1)

    print("\n=== One-step DeepSeek-shape demo ===")
    print(f"logits shape:          {tuple(logits.shape)}")
    print(f"probs shape:           {tuple(probs.shape)}")
    print(f"multinomial_out shape: {tuple(multinomial_out.shape)}")
    print(f"exp_race_out shape:    {tuple(exp_race_out.shape)}")
    print("说明：两者输出 shape 相同，但单次随机结果不要求逐元素相同。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vocab-size", type=int, default=129280, help="DeepSeek V4-Pro vocab_size 是 129280")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", type=str, default="peaked", choices=["peaked", "random", "uniform"])
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu instead.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("PyTorch:", torch.__version__)
    print("device:", device)

    one_step_deepseek_shape_demo(args.batch_size, args.vocab_size, device)

    probs = make_probs(args.vocab_size, device, args.seed, args.mode)

    counts_m, tm = sample_multinomial_counts(probs, args.trials)
    counts_e, te = sample_exponential_race_counts(probs, args.trials, args.chunk_size)

    print("\n=== Timing ===")
    print(f"torch.multinomial time: {tm:.4f} s")
    print(f"exp-race time:          {te:.4f} s")
    print("注意：这里的 exp-race 是 Python 分块实现，不代表 DeepSeek 融合/实际框架中的最优性能。")

    summarize(probs, counts_m, counts_e, args.trials, args.topk)


if __name__ == "__main__":
    main()
```

推荐你先这样跑一个小 vocab 的强统计验证：

```bash
python compare_multinomial_vs_exponential_race.py \
  --device cuda \
  --vocab-size 100 \
  --trials 500000 \
  --chunk-size 4096 \
  --mode peaked
```

再跑接近 DeepSeek V4-Pro 采样形状的版本：

```bash
python compare_multinomial_vs_exponential_race.py \
  --device cuda \
  --vocab-size 129280 \
  --batch-size 1 \
  --trials 50000 \
  --chunk-size 128 \
  --mode peaked
```

如果你在 B200 上显存充足，可以把 `--chunk-size` 提到 `512` 或 `1024`。对 DeepSeek 形状来说：

```text
chunk_size=128:
128 * 129280 * 4 bytes ≈ 63 MiB
```

脚本里 `exponential_race` 的核心等价关系是：

```python
noise = torch.empty((cur, vocab_size), device=device).exponential_(1.0)
sample = (probs / noise).argmax(dim=-1)
```

而 DeepSeek 单步中就是：

```python
probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
```

`torch.multinomial` 的输入行可以不归一化为 1，但必须是非负、有限、且每行总和非零；`Tensor.exponential_()` 则是用指数分布样本填充 tensor。([docs.pytorch.org][3])

[1]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json "config.json · deepseek-ai/DeepSeek-V4-Pro at main"
[2]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/generate.py "inference/generate.py · deepseek-ai/DeepSeek-V4-Pro at main"
[3]: https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html?utm_source=chatgpt.com "torch.multinomial — PyTorch 2.12 documentation"
