"""Run on the actual vLLM CUDA environment, from this directory.

This file was not executed in the authoring environment (which is CPU-only).
"""
import torch
from mxfp8_q_fake_quant import fake_quant_mxfp8_decode_q_
from mxfp8_reference import mxfp8_reference


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Run this test on the vLLM GPU server.")
    torch.manual_seed(19)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for strided in (False, True):
            q = torch.randn(5, 8, 576, device='cuda', dtype=dtype)
            q[3, 0, :32] = 0
            q[4, 1, 512:] *= 0.03125
            if strided:
                q = q.transpose(0, 1).contiguous().transpose(0, 1)
            original = q.cpu().clone()
            # Request 0: three prefill tokens. Request 1: two decode tokens.
            ids = torch.tensor([0, 0, 0, 1, 1], device=q.device, dtype=torch.int32)
            flags = torch.tensor([True, False], device=q.device, dtype=torch.bool)
            expected = original.clone()
            expected[3:] = mxfp8_reference(original[3:])
            returned = fake_quant_mxfp8_decode_q_(q, ids, flags)
            assert returned.data_ptr() == q.data_ptr()
            torch.cuda.synchronize()
            actual = q.cpu()
            torch.testing.assert_close(actual[:3], original[:3], atol=0, rtol=0)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert (actual[3:] != original[3:]).any()
            print(f"PASS dtype={dtype}, strided={strided}, shape={tuple(q.shape)}")
    print("All tested GPU cases passed. This is not an end-to-end model test.")


if __name__ == '__main__':
    main()
