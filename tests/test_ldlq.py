"""LDLQ test ladder (plan §3). Stage 1-2 are cheap and catch wiring bugs."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis.qtip import codebook_lut, tcq_quantize_weight  # noqa: E402
from trellis.ldlq import (block_ldl, regularize_H, tcq_quantize_weight_ldlq)  # noqa: E402


def test_block_ldl_identity():
    """H=I -> chol(I)=I -> block_ldl zeros everything -> L=0 (no feedback)."""
    H = torch.eye(64)
    L = block_ldl(regularize_H(H, 0.0), b=16)
    assert torch.allclose(L, torch.zeros_like(L), atol=1e-6), L.abs().max()
    print("ok: block_ldl(I) == 0 (feedback vanishes, as the plan predicts)")


def test_ldlq_HI_matches_baseline():
    """Stage 1: with H=I (zero feedback), LDLQ must reproduce plain tcq.

    Uses fp32 DP (backend='torch') on CPU-sized data so there is NO bf16
    batch-order nondeterminism — the only remaining difference would be a
    wiring bug, so we can demand a tight bound.
    """
    # Test the LOOP directly with L==0 (feedback genuinely off). Using H=I is
    # unreliable: fp Hadamard roundoff leaves L ~2.8e-8, and Viterbi tie-breaks
    # amplify that to ~1e-3 — a numerical artifact, not a wiring bug. With L
    # exactly zero the compensated target == raw target, so LDLQ must reproduce
    # the flat baseline bit-exactly.
    import trellis.ldlq as ldlq_mod
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lut = codebook_lut("mcg").to(dev)
    w = (torch.randn(64, 256, device=dev) * 0.02)
    base = tcq_quantize_weight(w, K=3, lut=lut, seed=0, backend="torch",
                               dp_dtype=torch.float32)
    # monkeypatch block_ldl -> return exact zero L (feedback fully off)
    orig = ldlq_mod.block_ldl
    ldlq_mod.block_ldl = lambda H, b=16: torch.zeros(H.shape[0], H.shape[0], device=H.device)
    try:
        ld = tcq_quantize_weight_ldlq(w, K=3, lut=lut, H=torch.eye(256, device=dev),
                                      seed=0, sigma_reg=0.0, dp_dtype=torch.float32,
                                      backend="torch")
    finally:
        ldlq_mod.block_ldl = orig
    rel = ((base - ld) ** 2).mean().item() / base.square().mean().item()
    print(f"  L=0 (fp32 DP): LDLQ vs baseline rel-diff = {rel:.2e}")
    assert rel < 1e-8, f"L=0 LDLQ != baseline (rel {rel}) — wiring bug"
    print("ok: LDLQ with L=0 reproduces baseline tcq bit-exact (loop wired right)")


def test_ldlq_real_H_reduces_hessian_objective():
    """Stage 2: with a real (random SPD) H, feedback reduces tr(E H Eᵀ)."""
    torch.manual_seed(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lut = codebook_lut("mcg").to(dev)
    w = (torch.randn(64, 256, device=dev) * 0.02)
    A = torch.randn(256, 256, device=dev)
    H = (A.T @ A) / 256 + torch.eye(256, device=dev)          # SPD
    base = tcq_quantize_weight(w, K=3, lut=lut, seed=0,
                               backend="torch" if dev == "cpu" else "triton")
    ld = tcq_quantize_weight_ldlq(w, K=3, lut=lut, H=H, seed=0)

    def hess_obj(wq):
        E = (w - wq).float()
        return torch.einsum("nk,kj,nj->", E, H, E).item()
    ob, ol = hess_obj(base), hess_obj(ld)
    print(f"  tr(E H Eᵀ): baseline={ob:.4e}  ldlq={ol:.4e}  ({ol/ob:.3f}x)")
    assert ol < ob, "LDLQ did not reduce the Hessian objective it optimizes"
    print("ok: LDLQ reduces tr(E H Eᵀ) vs no-feedback baseline")


if __name__ == "__main__":
    test_block_ldl_identity()
    test_ldlq_HI_matches_baseline()
    test_ldlq_real_H_reduces_hessian_objective()
    print("\nLDLQ LADDER STAGES 1-2 PASSED")
