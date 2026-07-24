"""Triton fused Viterbi vs the reference PyTorch DP — must match in quality.

CUDA-only: marked `cuda` and skipped unless an NVIDIA GPU is present, so the
CPU CI suite stays green. On a CUDA box run it explicitly:

    pytest tests/test_triton_kernel.py -m cuda
    # or standalone: python tests/test_triton_kernel.py
"""
import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis.qtip import T_SEQ, codebook_lut, viterbi_tb  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA GPU")


@pytest.mark.cuda
@cuda
def test_triton_matches_reference_quality():
    """The fused Triton encoder must reach the same distortion as the PyTorch DP.

    Ties in fp32 path costs let the two implementations pick different
    equal-cost bit paths, so we assert equal *quality* (MSE), not bit-identity.
    """
    from trellis.viterbi_triton import viterbi_tb_triton

    dev = "cuda"
    lut = codebook_lut("mcg").to(dev)
    torch.manual_seed(0)

    for K in (2, 3):
        w = torch.randn(64, T_SEQ, device=dev)
        _, dq_ref = viterbi_tb(w, K, lut, dp_dtype=torch.float32)
        _, dq_tri = viterbi_tb_triton(w, K, lut)
        mse_ref = ((w - dq_ref) ** 2).mean().item()
        mse_tri = ((w - dq_tri) ** 2).mean().item()
        assert abs(mse_tri - mse_ref) / mse_ref < 0.01, (
            f"K={K}: triton quality {mse_tri:.5f} != reference {mse_ref:.5f}"
        )


def _benchmark():
    """Throughput report — informational, run manually on a CUDA box."""
    from trellis.viterbi_triton import viterbi_tb_triton

    assert torch.cuda.is_available(), "needs CUDA"
    dev = "cuda"
    lut = codebook_lut("mcg").to(dev)
    K = 2
    for n in (256, 2048):
        w = torch.randn(n, T_SEQ, device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        viterbi_tb_triton(w, K, lut)
        torch.cuda.synchronize()
        dt = time.time() - t0
        wps = n * T_SEQ / dt
        print(f"triton: {n} tiles in {dt:.2f}s = {wps / 1e6:.2f}M weights/s "
              f"(8B encode ~= {7.6e9 / wps / 60:.0f} min)")


if __name__ == "__main__":
    test_triton_matches_reference_quality()
    _benchmark()
    print("\nTRITON KERNEL OK")
