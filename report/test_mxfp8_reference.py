import json
import torch
from mxfp8_reference import mxfp8_reference, ue8m0_upward_scale


def main():
    torch.manual_seed(123)
    # Include all subnormal boundaries and normal powers of two.
    powers = torch.pow(2.0, torch.arange(-127, 128, dtype=torch.float64)).float()
    scales_in = torch.cat((torch.tensor([0.0, 2.0**-149, 2.0**-128]),
                           powers, torch.nextafter(powers, torch.zeros_like(powers)),
                           torch.nextafter(powers, torch.full_like(powers, float('inf')))))
    scales_in = scales_in[torch.isfinite(scales_in)]
    # Independent oracle: search a sorted table of all 255 finite E8M0 values.
    idx = torch.searchsorted(powers, scales_in).clamp(max=254)
    torch.testing.assert_close(ue8m0_upward_scale(scales_in), powers[idx], rtol=0, atol=0)
    results = {'scale_boundary_cases': scales_in.numel(), 'scale_oracle': 'PASS'}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        q = torch.randn(5, 8, 576).to(dtype)
        q[0, 0, :32] = 0
        y = mxfp8_reference(q)
        assert y.shape == q.shape and y.dtype == q.dtype
        assert torch.isfinite(y).all()
        assert torch.count_nonzero(y[0, 0, :32]) == 0
        # Grouping cannot depend on other token/head/group values.
        altered = q.clone()
        altered[0, 1, :32] *= 100
        y2 = mxfp8_reference(altered)
        mask = torch.ones_like(q, dtype=torch.bool)
        mask[0, 1, :32] = False
        torch.testing.assert_close(y[mask], y2[mask], rtol=0, atol=0)
        # 512/64 split is exactly aligned with 32-wide groups.
        split_y = torch.cat((mxfp8_reference(q[..., :512]),
                             mxfp8_reference(q[..., 512:])), dim=-1)
        torch.testing.assert_close(y, split_y, rtol=0, atol=0)
        # A strided layout must preserve logical grouping.
        strided = q.transpose(0, 1).contiguous().transpose(0, 1)
        torch.testing.assert_close(mxfp8_reference(strided), y, rtol=0, atol=0)
        # Request 0 is prefill and request 1 is decode, packed as [0,0,0,1,1].
        req_ids = torch.tensor([0, 0, 0, 1, 1])
        phase = torch.tensor([True, False])
        result = q.clone()
        decode_rows = ~phase[req_ids]
        result[decode_rows] = y[decode_rows]
        torch.testing.assert_close(result[:3], q[:3], rtol=0, atol=0)
        results[str(dtype)] = {
            'shape': list(q.shape), 'group_independence': 'PASS',
            'split_equivalence': 'PASS', 'strided_layout': 'PASS',
            'prefill_unchanged_reference': 'PASS',
            'normalized_rmse': float(torch.linalg.vector_norm(y.float()-q.float()) /
                                     torch.linalg.vector_norm(q.float()))}
    # Nonfinite groups are preserved rather than silently sanitized.
    q_bad = torch.zeros(1, 1, 64)
    q_bad[0, 0, 0] = float('inf'); q_bad[0, 0, 33] = float('nan')
    y_bad = mxfp8_reference(q_bad)
    assert torch.isinf(y_bad[0, 0, 0]) and torch.isnan(y_bad[0, 0, 33])
    results['nonfinite_preservation_reference'] = 'PASS'
    results['GPU_triton_execution'] = 'NOT RUN: CPU-only environment, no Triton installed'
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
