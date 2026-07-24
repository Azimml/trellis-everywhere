"""Correctness tests for the generic bitshift trellis engine.

Run: python -m pytest tests/ -q   (or plain `python tests/test_trellis.py`)
"""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis.trellis import BitshiftTrellis, gaussian_codebook  # noqa: E402


def brute_force_best(tr: BitshiftTrellis, w: torch.Tensor, s0: int = 0):
    """Enumerate every possible bit stream; return the minimum achievable MSE."""
    (T,) = w.shape
    best = float("inf")
    for stream in itertools.product(range(1 << tr.k), repeat=T):
        bits = torch.tensor([stream], dtype=torch.long)
        dq = tr.decode(bits, s0=s0)[0]
        mse = ((w - dq) ** 2).sum().item()
        best = min(best, mse)
    return best


def test_viterbi_is_optimal():
    torch.manual_seed(0)
    tr = BitshiftTrellis(L=4, k=2, codebook_values=gaussian_codebook(4, seed=1))
    for _trial in range(5):
        w = torch.randn(1, 5)
        bits, dq = tr.encode(w)
        got = ((w - dq) ** 2).sum().item()
        want = brute_force_best(tr, w[0])
        assert abs(got - want) < 1e-4, f"viterbi {got} != brute force {want}"
    print("ok: viterbi matches brute-force optimum (L=4,k=2,T=5, 5 trials)")


def test_roundtrip_decode_matches():
    torch.manual_seed(1)
    tr = BitshiftTrellis(L=8, k=2, codebook_values=gaussian_codebook(8, seed=2))
    w = torch.randn(16, 32)
    bits, dq = tr.encode(w)
    dq2 = tr.decode(bits)
    assert torch.equal(dq, dq2)
    print("ok: encode's dq == reference decode of its bits")


def test_block_restart_equals_independent_chunks():
    """encode(block=B) must equal encoding each B-chunk separately."""
    torch.manual_seed(2)
    tr = BitshiftTrellis(L=8, k=2, codebook_values=gaussian_codebook(8, seed=3))
    n, T, B = 4, 32, 8
    w = torch.randn(n, T)
    bits_blk, dq_blk = tr.encode(w, block=B)
    for c in range(T // B):
        wc = w[:, c * B:(c + 1) * B]
        bits_c, dq_c = tr.encode(wc)
        assert torch.equal(bits_blk[:, c * B:(c + 1) * B], bits_c)
        assert torch.equal(dq_blk[:, c * B:(c + 1) * B], dq_c)
    print("ok: block-restarted encode == independent per-chunk encodes")


def test_blockwise_decode_is_parallelizable():
    """The property the whole project rests on: with block restarts, each
    block decodes independently — order/parallelism cannot change values."""
    torch.manual_seed(3)
    tr = BitshiftTrellis(L=8, k=2, codebook_values=gaussian_codebook(8, seed=4))
    n, T, B = 4, 32, 8
    w = torch.randn(n, T)
    bits, _ = tr.encode(w, block=B)
    seq = tr.decode(bits, block=B)                       # sequential reference
    par = torch.cat([tr.decode(bits[:, c * B:(c + 1) * B])  # "parallel" blocks
                     for c in range(T // B)], dim=1)
    assert torch.equal(seq, par)
    print("ok: block decode == parallel independent block decode")


def test_quality_improves_with_state_bits():
    """Sanity: more trellis memory (bigger L) at fixed k must not hurt MSE."""
    torch.manual_seed(4)
    w = torch.randn(8, 64)
    mses = []
    for L in (2, 6, 10):
        tr = BitshiftTrellis(L=L, k=2, codebook_values=gaussian_codebook(L, seed=5))
        _, dq = tr.encode(w)
        mses.append(((w - dq) ** 2).mean().item())
    assert mses[0] >= mses[1] >= mses[2] - 1e-6, mses
    print(f"ok: MSE falls as L grows at fixed 2 bpw: {[f'{m:.4f}' for m in mses]}")


if __name__ == "__main__":
    test_viterbi_is_optimal()
    test_roundtrip_decode_matches()
    test_block_restart_equals_independent_chunks()
    test_blockwise_decode_is_parallelizable()
    test_quality_improves_with_state_bits()
    print("\nALL TRELLIS ENGINE TESTS PASSED")
