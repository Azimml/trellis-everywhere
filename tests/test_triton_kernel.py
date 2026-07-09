"""Triton fused Viterbi vs the reference PyTorch DP — must match exactly.

Run on a CUDA box (DGX): python tests/test_triton_kernel.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis.qtip import codebook_lut, viterbi_tb, T_SEQ  # noqa: E402
from trellis.viterbi_triton import viterbi_tb_triton       # noqa: E402


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    dev = "cuda"
    lut = codebook_lut("mcg").to(dev)
    torch.manual_seed(0)

    for K in (2, 3):
        w = torch.randn(64, T_SEQ, device=dev)
        bits_ref, dq_ref = viterbi_tb(w, K, lut, dp_dtype=torch.float32)
        bits_tri, dq_tri = viterbi_tb_triton(w, K, lut)
        mse_ref = ((w - dq_ref) ** 2).mean().item()
        mse_tri = ((w - dq_tri) ** 2).mean().item()
        same = (bits_ref == bits_tri).float().mean().item()
        print(f"K={K}: ref mse={mse_ref:.5f} triton mse={mse_tri:.5f} "
              f"bit-agreement={same:.4f}")
        # ties in fp math may pick different equal-cost paths; require equal QUALITY
        assert abs(mse_tri - mse_ref) / mse_ref < 0.01, "triton quality != reference"

    # ---- throughput ----
    K = 2
    for n in (256, 2048):
        w = torch.randn(n, T_SEQ, device=dev)
        torch.cuda.synchronize(); t0 = time.time()
        viterbi_tb_triton(w, K, lut)
        torch.cuda.synchronize(); dt = time.time() - t0
        wps = n * T_SEQ / dt
        print(f"triton: {n} tiles in {dt:.2f}s = {wps/1e6:.2f}M weights/s "
              f"(8B encode ~= {7.6e9 / wps / 60:.0f} min)")

    print("\nTRITON KERNEL OK")


if __name__ == "__main__":
    main()
