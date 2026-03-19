import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer


def get_fp4_e2m1_value_table(device: torch.device) -> torch.Tensor:
	return torch.tensor(
		[
			0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
			-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
		],
		device=device,
		dtype=torch.float32,
	)


def check_environment() -> None:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required.")

	is_available, reason = te.is_nvfp4_available(return_reason=True)
	if not is_available:
		raise RuntimeError(f"NVFP4 is not available on this system: {reason}")


def build_test_input(
	rows: int = 16,
	columns: int = 16,
	anchor_value: float = 6.0,
	target_value: float = 0.875,
) -> tuple[torch.Tensor, torch.Tensor]:
	if columns != 16:
		raise ValueError("This demo intentionally uses one 16-element NVFP4 block per row.")

	input_tensor = torch.full(
		(rows, columns),
		target_value,
		device="cuda",
		dtype=torch.float32,
	)
	input_tensor[:, : columns // 2] = anchor_value
	target_mask = input_tensor == target_value
	return input_tensor.contiguous(), target_mask


def build_quantizer(stochastic_rounding: bool) -> NVFP4Quantizer:
	return NVFP4Quantizer(
		rowwise=True,
		columnwise=False,
		with_amax_reduction=False,
		with_rht=False,
		with_post_rht_amax=False,
		with_2d_quantization=False,
		stochastic_rounding=stochastic_rounding,
		with_random_sign_mask=False,
	)


def unpack_fp4_rowwise_data(
	packed_rowwise_data: torch.Tensor,
	logical_columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	first_codes = (packed_rowwise_data & 0x0F).to(torch.uint8)
	second_codes = ((packed_rowwise_data >> 4) & 0x0F).to(torch.uint8)

	unpacked_codes = torch.stack((first_codes, second_codes), dim=-1).reshape(
		packed_rowwise_data.shape[0],
		-1,
	)
	unpacked_codes = unpacked_codes[:, :logical_columns]

	fp4_value_table = get_fp4_e2m1_value_table(packed_rowwise_data.device)
	unpacked_fp4_values = fp4_value_table[unpacked_codes.long()]
	return unpacked_codes, unpacked_fp4_values


def decode_rowwise_scales(
	rowwise_scale_inv_bytes: torch.Tensor,
	logical_rows: int,
	logical_columns: int,
) -> torch.Tensor:
	logical_blocks_per_row = logical_columns // 16
	scale_bytes = rowwise_scale_inv_bytes[:logical_rows, :logical_blocks_per_row].contiguous()

	try:
		scale_values = scale_bytes.view(torch.float8_e4m3fn).to(torch.float32)
	except Exception as error:
		raise RuntimeError(
			"Failed to reinterpret rowwise_scale_inv bytes as torch.float8_e4m3fn. "
			"Your PyTorch build may not support view(torch.float8_e4m3fn)."
		) from error

	return scale_values


def reconstruct_manual_float_values(
	unpacked_fp4_values: torch.Tensor,
	rowwise_scale_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	per_element_scales = rowwise_scale_values.repeat_interleave(16, dim=1)
	per_element_scales = per_element_scales[:, : unpacked_fp4_values.shape[1]]
	manual_float_values = unpacked_fp4_values * per_element_scales
	return per_element_scales, manual_float_values


@torch.no_grad()
def quantize_and_decode(
	input_tensor: torch.Tensor,
	stochastic_rounding: bool,
) -> dict:
	quantizer = build_quantizer(stochastic_rounding=stochastic_rounding)
	quantized_tensor = quantizer(input_tensor)
	dequantized_tensor = quantized_tensor.dequantize(dtype=torch.float32)

	packed_rowwise_data = quantized_tensor._rowwise_data
	rowwise_scale_inv_bytes = quantized_tensor._rowwise_scale_inv
	amax_rowwise = quantized_tensor._amax_rowwise

	unpacked_codes, unpacked_fp4_values = unpack_fp4_rowwise_data(
		packed_rowwise_data=packed_rowwise_data,
		logical_columns=input_tensor.shape[1],
	)

	rowwise_scale_values = decode_rowwise_scales(
		rowwise_scale_inv_bytes=rowwise_scale_inv_bytes,
		logical_rows=input_tensor.shape[0],
		logical_columns=input_tensor.shape[1],
	)

	per_element_scales, manual_float_values = reconstruct_manual_float_values(
		unpacked_fp4_values=unpacked_fp4_values,
		rowwise_scale_values=rowwise_scale_values,
	)

	return {
		"quantized_tensor": quantized_tensor,
		"packed_rowwise_data": packed_rowwise_data,
		"rowwise_scale_inv_bytes": rowwise_scale_inv_bytes,
		"unpacked_codes": unpacked_codes,
		"unpacked_fp4_values": unpacked_fp4_values,
		"rowwise_scale_values": rowwise_scale_values,
		"per_element_scales": per_element_scales,
		"manual_float_values": manual_float_values,
		"dequantized_tensor": dequantized_tensor,
		"amax_rowwise": amax_rowwise,
	}


def summarize_unique_values(values: torch.Tensor) -> list[tuple[float, int]]:
	unique_values, counts = torch.unique(values, sorted=True, return_counts=True)
	return [
		(float(unique_value.item()), int(count.item()))
		for unique_value, count in zip(unique_values, counts)
	]


def describe_neighbour_values(scale_value: float, target_value: float) -> tuple[float, float, float] | None:
	fp4_value_table = get_fp4_e2m1_value_table(torch.device("cuda"))
	candidate_values = torch.unique((fp4_value_table * scale_value).to(torch.float32))
	candidate_values = candidate_values[candidate_values >= 0].sort().values

	lower_candidates = candidate_values[candidate_values <= target_value]
	upper_candidates = candidate_values[candidate_values >= target_value]

	if lower_candidates.numel() == 0 or upper_candidates.numel() == 0:
		return None

	lower_value = float(lower_candidates.max().item())
	upper_value = float(upper_candidates.min().item())

	if upper_value == lower_value:
		probability_of_rounding_up = 1.0
	else:
		probability_of_rounding_up = (target_value - lower_value) / (upper_value - lower_value)

	return lower_value, upper_value, probability_of_rounding_up


def print_first_row_debug(decoded_result: dict, input_tensor: torch.Tensor, title: str) -> None:
	row_index = 0

	print("=" * 132)
	print(title)
	print("=" * 132)
	print("Packed rowwise bytes for row 0:")
	print(decoded_result["packed_rowwise_data"][row_index].cpu())
	print()
	print("Decoded logical scales for row 0:")
	print(decoded_result["rowwise_scale_values"][row_index].cpu())
	print()
	print("col | input | fp4_code | fp4_value | scale | manual_float | te_dequant")
	print("-" * 74)

	for column_index in range(input_tensor.shape[1]):
		print(
			f"{column_index:>3d} | "
			f"{input_tensor[row_index, column_index].item():>5.3f} | "
			f"{int(decoded_result['unpacked_codes'][row_index, column_index].item()):>8d} | "
			f"{decoded_result['unpacked_fp4_values'][row_index, column_index].item():>9.3f} | "
			f"{decoded_result['per_element_scales'][row_index, column_index].item():>5.3f} | "
			f"{decoded_result['manual_float_values'][row_index, column_index].item():>12.3f} | "
			f"{decoded_result['dequantized_tensor'][row_index, column_index].item():>10.3f}"
		)

	max_abs_error = (
		decoded_result["manual_float_values"] - decoded_result["dequantized_tensor"]
	).abs().max().item()

	print()
	print(f"max_abs_error(manual_vs_te) = {max_abs_error:.8f}")
	print(f"amax_rowwise = {decoded_result['amax_rowwise'].item():.8f}")
	print()


@torch.no_grad()
def run_trials(
	input_tensor: torch.Tensor,
	target_mask: torch.Tensor,
	stochastic_rounding: bool,
	trial_count: int,
) -> dict:
	quantizer = build_quantizer(stochastic_rounding=stochastic_rounding)

	all_target_values = []
	per_trial_means = []

	for _ in range(trial_count):
		dequantized_tensor = quantizer(input_tensor).dequantize(dtype=torch.float32)
		target_values = dequantized_tensor[target_mask]
		all_target_values.append(target_values)
		per_trial_means.append(target_values.mean())

	all_target_values = torch.cat(all_target_values)
	per_trial_means = torch.stack(per_trial_means)

	return {
		"all_target_values": all_target_values,
		"per_trial_means": per_trial_means,
		"unique_histogram": summarize_unique_values(all_target_values),
		"global_mean": float(all_target_values.mean().item()),
		"trial_mean_std": float(per_trial_means.std(unbiased=False).item()),
	}


def main() -> None:
	check_environment()

	torch.manual_seed(1234)
	torch.cuda.manual_seed_all(1234)
	torch.set_printoptions(precision=6, sci_mode=False, linewidth=200)

	rows = 16
	columns = 16
	anchor_value = 6.0
	target_value = 0.875
	trial_count = 200

	input_tensor, target_mask = build_test_input(
		rows=rows,
		columns=columns,
		anchor_value=anchor_value,
		target_value=target_value,
	)

	print(f"Input shape = {tuple(input_tensor.shape)}")
	print(f"Anchor value = {anchor_value}")
	print(f"Target value = {target_value}")
	print(f"Repeated target count = {int(target_mask.sum().item())}")
	print()
	print("Input row 0:")
	print(input_tensor[0].cpu())
	print()

	deterministic_debug = quantize_and_decode(
		input_tensor=input_tensor,
		stochastic_rounding=False,
	)
	stochastic_debug = quantize_and_decode(
		input_tensor=input_tensor,
		stochastic_rounding=True,
	)

	print_first_row_debug(
		decoded_result=deterministic_debug,
		input_tensor=input_tensor,
		title="Deterministic quantization decode",
	)
	print_first_row_debug(
		decoded_result=stochastic_debug,
		input_tensor=input_tensor,
		title="Stochastic-rounding quantization decode",
	)

	first_scale = float(stochastic_debug["rowwise_scale_values"][0, 0].item())
	neighbour_description = describe_neighbour_values(
		scale_value=first_scale,
		target_value=target_value,
	)

	if neighbour_description is not None:
		lower_value, upper_value, probability_of_rounding_up = neighbour_description
		print("=" * 132)
		print("Neighbour analysis")
		print("=" * 132)
		print(f"Decoded first-row block scale = {first_scale:.6f}")
		print(f"Target value = {target_value:.6f}")
		print(f"Nearest lower representable value = {lower_value:.6f}")
		print(f"Nearest upper representable value = {upper_value:.6f}")
		print(f"Expected stochastic-rounding probability of rounding up = {probability_of_rounding_up:.6f}")
		print(f"Expected unbiased mean = {target_value:.6f}")
		print()

	deterministic_stats = run_trials(
		input_tensor=input_tensor,
		target_mask=target_mask,
		stochastic_rounding=False,
		trial_count=trial_count,
	)
	stochastic_stats = run_trials(
		input_tensor=input_tensor,
		target_mask=target_mask,
		stochastic_rounding=True,
		trial_count=trial_count,
	)

	print("=" * 132)
	print("Deterministic target statistics")
	print("=" * 132)
	print(f"Histogram = {deterministic_stats['unique_histogram']}")
	print(f"Global mean = {deterministic_stats['global_mean']:.6f}")
	print(f"Std of per-trial means = {deterministic_stats['trial_mean_std']:.6f}")
	print()

	print("=" * 132)
	print("Stochastic-rounding target statistics")
	print("=" * 132)
	print(f"Histogram = {stochastic_stats['unique_histogram']}")
	print(f"Global mean = {stochastic_stats['global_mean']:.6f}")
	print(f"Std of per-trial means = {stochastic_stats['trial_mean_std']:.6f}")
	print()

	print("=" * 132)
	print("Interpretation")
	print("=" * 132)
	print("1. manual_float should match te_dequant very closely.")
	print("2. Deterministic rounding should collapse the repeated target values to one nearby representable value.")
	print("3. Stochastic rounding should distribute those repeated target values over two neighbouring representable values.")
	print("4. The stochastic global mean should stay close to the original target value.")
	print()


if __name__ == "__main__":
	main()
