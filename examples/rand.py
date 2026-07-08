import torch


def compare_prefix(
    op_name: str,
    small_shape,
    large_shape,
    dtype=torch.float32,
    device="cpu",
    seed=1234,
):
    assert torch.tensor(small_shape).prod().item() <= torch.tensor(large_shape).prod().item()

    op = getattr(torch, op_name)

    torch.manual_seed(seed)
    a = op(*small_shape, dtype=dtype, device=device)

    torch.manual_seed(seed)
    b = op(*large_shape, dtype=dtype, device=device)

    a_flat = a.flatten()
    b_prefix = b.flatten()[: a_flat.numel()]

    equal = torch.equal(a_flat, b_prefix)
    close = torch.allclose(a_flat, b_prefix, rtol=0, atol=0)

    print("=" * 80)
    print(f"op        : torch.{op_name}")
    print(f"device    : {device}")
    print(f"dtype     : {dtype}")
    print(f"small     : {small_shape}, numel={a_flat.numel()}")
    print(f"large     : {large_shape}, numel={b.numel()}")
    print(f"equal     : {equal}")
    print(f"allclose0 : {close}")

    if not equal:
        diff_idx = torch.nonzero(a_flat != b_prefix, as_tuple=False).flatten()
        print(f"num_diff  : {diff_idx.numel()} / {a_flat.numel()}")

        show_n = min(10, diff_idx.numel())
        if show_n > 0:
            idx = diff_idx[:show_n]
            print("first different indices:")
            for i in idx.tolist():
                print(
                    f"  idx={i}, small={a_flat[i].item()}, large_prefix={b_prefix[i].item()}"
                )


def compare_same_numel_different_shape(
    op_name: str,
    shape1,
    shape2,
    dtype=torch.float32,
    device="cpu",
    seed=1234,
):
    assert torch.tensor(shape1).prod().item() == torch.tensor(shape2).prod().item()

    op = getattr(torch, op_name)

    torch.manual_seed(seed)
    a = op(*shape1, dtype=dtype, device=device)

    torch.manual_seed(seed)
    b = op(*shape2, dtype=dtype, device=device)

    a_flat = a.flatten()
    b_flat = b.flatten()

    equal = torch.equal(a_flat, b_flat)

    print("=" * 80)
    print(f"same numel, different shape")
    print(f"op        : torch.{op_name}")
    print(f"device    : {device}")
    print(f"dtype     : {dtype}")
    print(f"shape1    : {shape1}")
    print(f"shape2    : {shape2}")
    print(f"equal     : {equal}")

    if not equal:
        diff_idx = torch.nonzero(a_flat != b_flat, as_tuple=False).flatten()
        print(f"num_diff  : {diff_idx.numel()} / {a_flat.numel()}")

        show_n = min(10, diff_idx.numel())
        if show_n > 0:
            idx = diff_idx[:show_n]
            print("first different indices:")
            for i in idx.tolist():
                print(
                    f"  idx={i}, shape1={a_flat[i].item()}, shape2={b_flat[i].item()}"
                )


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    dtypes = [torch.float32]

    if torch.cuda.is_available():
        dtypes.extend([torch.float16, torch.bfloat16])

    for device in devices:
        for dtype in dtypes:
            if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
                continue

            for op_name in ["rand", "randn"]:
                compare_prefix(
                    op_name=op_name,
                    small_shape=(1024,),
                    large_shape=(2048,),
                    dtype=dtype,
                    device=device,
                    seed=2026,
                )

                compare_prefix(
                    op_name=op_name,
                    small_shape=(1024,),
                    large_shape=(1024 * 1024 + 17,),
                    dtype=dtype,
                    device=device,
                    seed=2026,
                )

                compare_same_numel_different_shape(
                    op_name=op_name,
                    shape1=(1024,),
                    shape2=(32, 32),
                    dtype=dtype,
                    device=device,
                    seed=2026,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
