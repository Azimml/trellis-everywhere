"""Tests for the faithful QTIP/EXL3 layer (constants, TB-Viterbi, IP, random access)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis import qtip  # noqa: E402
from trellis.qtip import (  # noqa: E402
    L, S, T_SEQ, cb_1mad, codebook_lut, decode_recursive, decode_window,
    viterbi_tb, tcq_quantize_weight, IPContext, _block_had,
)
from trellis.quant import _uniform_group_fakequant  # noqa: E402


def test_codebook_constants():
    # hand-computed: x=0 -> LCG = 76625530 = 0x0491367A; bytes 122+54+145+4 = 325
    v = cb_1mad(torch.tensor([0])).item()
    expect = (325 - 510) / 147.800537109375
    assert abs(v - expect) < 1e-6, (v, expect)
    # every computed codebook must be ~N(0,1) over all states
    for name in ("1mad", "3inst", "mcg"):
        lut = codebook_lut(name)
        m, s = lut.mean().item(), lut.std().item()
        assert abs(m) < 0.05 and 0.8 < s < 1.35, (name, m, s)
    print("ok: codebook constants + distributions (1mad hand-check, all ~N(0,1))")


def test_random_access_decode_equals_recursion():
    """The property the whole cross-platform port rests on."""
    torch.manual_seed(0)
    lut = codebook_lut("mcg")
    for K in (2, 3, 4):
        bits = torch.randint(0, 1 << K, (8, 64))
        a = decode_recursive(bits, K, lut)
        b = decode_window(bits, K, lut)
        assert torch.equal(a, b), f"K={K}: window decode != recursion"
    print("ok: random-access window decode == sequential recursion (K=2,3,4)")


def test_tailbiting_wraps_and_bpw():
    torch.manual_seed(1)
    lut = codebook_lut("mcg")
    w = torch.randn(4, T_SEQ)
    K = 2
    bits, dq = viterbi_tb(w, K, lut)
    assert bits.shape == (4, T_SEQ) and (bits < (1 << K)).all()      # exactly K bpw
    # reconstruct state path from bits alone using the WRAPPED stream:
    # initial window must equal the end of the stream (tail-biting).
    n = w.shape[0]
    stream = bits  # [n, T] of K-bit symbols
    s = torch.zeros(n, dtype=torch.long)
    warm = (L - K) // K  # symbols needed to fill the initial window from the tail
    for t in range(T_SEQ - warm, T_SEQ):
        s = ((s << K) | stream[:, t]) & (S - 1)
    vals = torch.empty(n, T_SEQ)
    for t in range(T_SEQ):
        s = ((s << K) | stream[:, t]) & (S - 1)
        vals[:, t] = lut[s]
    assert torch.allclose(vals, dq), "decode from wrapped bitstream != encoder's dq"
    print("ok: tail-biting verified — K bpw exactly, stream wraps, bits alone reproduce dq")


def test_tcq_beats_scalar_on_gaussian():
    torch.manual_seed(2)
    lut = codebook_lut("mcg")
    w = torch.randn(64, T_SEQ)
    _, dq = viterbi_tb(w, 2, lut)
    mse_tcq = ((w - dq) ** 2).mean().item()
    dqu = _uniform_group_fakequant(w, bits=2, groupsize=128, sym=False).float()
    mse_sq = ((w - dqu) ** 2).mean().item()
    # QTIP Table 2: L=12,k=2 trellis hits ~0.073 MSE on N(0,1); L=16 slightly better.
    assert mse_tcq < 0.085, f"TCQ MSE {mse_tcq} worse than expected (~0.07)"
    assert mse_tcq < 0.4 * mse_sq, (mse_tcq, mse_sq)
    print(f"ok: K=2 TCQ mse={mse_tcq:.4f} vs scalar-2b mse={mse_sq:.4f} "
          f"(paper L=12 reference ~0.073)")


def test_ip_roundtrip_identity():
    torch.manual_seed(3)
    w = torch.randn(128, 256)
    ip = IPContext(w, cb_rms=1.0, seed=0)
    back = ip.restore(ip.z)
    assert torch.allclose(back, w, atol=1e-5), (back - w).abs().max()
    print("ok: incoherence processing restore(ip(w)) == w")


def test_full_layer_quant_sanity():
    torch.manual_seed(4)
    lut = codebook_lut("mcg")
    w = torch.randn(128, 256) * 0.02          # LLM-ish weight scale
    wq = tcq_quantize_weight(w, K=2, lut=lut, seed=0, tile_batch=256)
    rel = ((w - wq) ** 2).mean().item() / w.square().mean().item()
    dqu = _uniform_group_fakequant(w, bits=2, groupsize=128, sym=False).float()
    rel_sq = ((w - dqu) ** 2).mean().item() / w.square().mean().item()
    assert rel < 0.12, f"relative MSE too high: {rel}"
    assert rel < rel_sq, (rel, rel_sq)
    print(f"ok: full-layer 2-bit TCQ rel-MSE={rel:.4f} vs scalar {rel_sq:.4f}")


if __name__ == "__main__":
    test_codebook_constants()
    test_random_access_decode_equals_recursion()
    test_tailbiting_wraps_and_bpw()
    test_tcq_beats_scalar_on_gaussian()
    test_ip_roundtrip_identity()
    test_full_layer_quant_sanity()
    print("\nALL QTIP-LAYER TESTS PASSED")
