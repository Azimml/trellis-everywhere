"""LDLQ: Hessian-aware error feedback around the trellis tile quantizer.

Why: per-tile trellis error is incoherent WITHIN a matrix (a global/per-tile
rescale is a proven no-op), but it accumulates COHERENTLY across transformer
layers through the residual stream — which is what breaks the 8B while leaving
each matrix's weight-MSE healthy. LDLQ quantizes input-channel blocks in
reverse order and feeds the committed error forward through L = blockLDL(H),
minimizing tr(E H Eᵀ) instead of ‖E‖² — i.e. the error that actually matters
once projected through real activations H = E[xxᵀ].

Faithful to EXL3 (exl3_lib/quantize.py block_ldl 292-362, ldlq 365-481) and
QTIP BlockLDLQ (lib/algo/ldlq.py 37-74). See notes/ldlq_plan.md.

Convention: our weights are (n_out, k_in); LDLQ walks the INPUT axis, so we
operate on Wt = W.T of shape (k, n) and transpose back.
"""
from __future__ import annotations
import torch

from trellis.qtip import (T_SEQ, _block_had, _fit_block, IPContext, viterbi_tb)


def regularize_H(H: torch.Tensor, sigma_reg: float = 0.025) -> torch.Tensor:
    k = H.shape[0]
    dm = torch.diag(H).mean()
    H = H.clone()
    H[torch.arange(k), torch.arange(k)] += sigma_reg * dm
    return H


def block_ldl(H: torch.Tensor, b: int = 16) -> torch.Tensor:
    """Return L (k,k), block-unit-lower-triangular, scalar diagonal zeroed.

    Retries with escalating diagonal damping if Cholesky fails (ill-conditioned
    or tiny-sample H). H = L Lᵀ; then right-multiply each block column by its
    diagonal block's inverse so diagonal blocks become identity.
    """
    k = H.shape[0]
    assert k % b == 0, f"k={k} not divisible by block {b}"
    m = k // b
    idx = torch.arange(k, device=H.device)
    # Robust damping: base off max(diag) not mean (mean can be ~0 for a
    # low-activation layer, making relative damping useless), and floor with an
    # absolute epsilon so a degenerate H still becomes SPD.
    dmax = float(torch.diag(H).abs().max())
    base = max(dmax, 1.0) * 0.01
    Hc = H
    Lc = None
    for attempt in range(14):
        try:
            Lc = torch.linalg.cholesky(Hc)
            break
        except Exception:
            Hc = H.clone()
            Hc[idx, idx] += (2.0 ** attempt) * base
    if Lc is None:
        return None  # caller falls back to plain (no-feedback) TCQ for this layer

    Lb = Lc.reshape(m, b, m, b)
    diag = Lb[torch.arange(m), :, torch.arange(m), :]        # (m, b, b)
    dinv = torch.linalg.inv(diag)
    L = Lc.reshape(m, b, m, b).clone()
    for i in range(m):
        L[:, :, i, :] = L[:, :, i, :] @ dinv[i]
    L = L.reshape(k, k)
    idx = torch.arange(k)
    L[idx, idx] = 0.0
    return L


@torch.no_grad()
def ldlq(Wt: torch.Tensor, L: torch.Tensor, enc, buf: int = 128) -> torch.Tensor:
    """Reverse-order block error feedback. Wt (k,n) in IP space, L (k,k).

    enc(rows) quantizes a (16, n) compensated block. Returns Wq (k,n).
    fp32 accumulators + in-place addmm_ (same recurrence, faithful to the fp32
    reference; bf16 accumulation would break the H=I bit-exact invariant).
    """
    k, n = Wt.shape
    Wq = torch.zeros(k, n, dtype=torch.float32, device=Wt.device)
    prod = torch.zeros(k, n, dtype=torch.float32, device=Wt.device)  # cross-span cache
    Wt_f = Wt.float()
    Lf = L.float()
    for j in range(k, 0, -buf):
        i = max(j - buf, 0)
        bW, bWq, bL = Wt_f[i:j], Wq[i:j], Lf[i:j]  # bL: (span, k)
        span = j - i
        for bj in range(span, 0, -16):
            bi = bj - 16
            later_err = bW[bj:] - bWq[bj:]                    # (span-bj, n) intra-span
            later_L = bL[bj:, i + bi:i + bj]                  # (span-bj, 16)
            comp = prod[i + bi:i + bj]                        # VIEW (16,n); cross-span term
            comp.addmm_(later_L.t(), later_err)              # in-place += later_L.T @ later_err
            rows = bW[bi:bj] + comp                           # compensated target
            bWq[bi:bj] = enc(rows).float()
        b_err = bW - bWq                                      # (span, n)
        prod.addmm_(bL.t(), b_err)                            # (k,span)@(span,n) big GEMM
    return Wq.to(Wt.dtype)


def _transform_H(H: torch.Tensor, su: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Bring H (k,k, input axis) into IP space: apply su signs + block-Hadamard
    both sides (signs+Hadamard ONLY — the diagonal per-channel/global scales are
    folded into su/sv on the weight side and must NOT be applied to H)."""
    H = su.view(-1, 1) * H * su.view(1, -1)
    H = _block_had(H, dim=0, block=block)
    H = _block_had(H, dim=1, block=block)
    return H


@torch.no_grad()
def tcq_quantize_weight_ldlq(w: torch.Tensor, K: int, lut: torch.Tensor, H: torch.Tensor,
                             seed: int = 0, sigma_reg: float = 0.025,
                             dp_dtype=torch.bfloat16, backend: str = "auto") -> torch.Tensor:
    """LDLQ variant of tcq_quantize_weight. w (n_out,k_in); H (k_in,k_in) proxy
    Hessian in ORIGINAL space (E[xxᵀ] over calibration). Returns dequant fp.
    """
    n, k = w.shape
    assert n % 16 == 0 and k % 16 == 0
    cb_rms = lut.square().mean().sqrt().item()
    ip = IPContext(w.float(), cb_rms, seed=seed)
    z = ip.z                                                  # (n, k) IP space
    lut_d = lut.to(z.device)

    # Backend: torch fp32 DP (bit-exact, for CPU/regression) vs fused Triton
    # kernel (~100x faster; collapses the 256-step Python DP into one launch).
    # The H=I bit-exact invariant holds as long as baseline uses the SAME backend.
    use_triton = backend == "triton" or (backend == "auto" and z.is_cuda)
    if use_triton:
        from trellis.viterbi_triton import viterbi_tb_triton
        def _q(t): return viterbi_tb_triton(t, K, lut_d)[1]
    else:
        def _q(t): return viterbi_tb(t, K, lut_d, dp_dtype=dp_dtype)[1]

    def enc(rows):
        # rows: (16 input-channels, n_out) -> tiles (n_out/16, 16-out, 16-in).
        n_out = rows.shape[1]
        rt = rows.t().contiguous()                                    # (n_out, 16-in)
        t = rt.reshape(n_out // 16, 16, 1, 16).permute(0, 2, 1, 3).reshape(-1, T_SEQ)
        dq = _q(t)
        dqt = dq.reshape(n_out // 16, 1, 16, 16).permute(0, 2, 1, 3).reshape(n_out, 16)
        return dqt.t().contiguous()                                   # (16-in, n_out)

    # LDLQ walks input axis: operate on z.T (k, n)
    Ht = _transform_H(H.float().to(z.device), ip.su, block=_fit_block(k, 128))
    L = block_ldl(regularize_H(Ht, sigma_reg), b=16)
    if L is None:
        # Degenerate Hessian (e.g. rank-deficient from few calib seqs): fall back
        # to plain no-feedback TCQ for this one layer (EXL3 does the same). L=0.
        L = torch.zeros(k, k, device=z.device)
    zt_q = ldlq(z.t().contiguous(), L, enc)                   # (k, n)
    zq = zt_q.t().contiguous()                                # (n, k)
    return ip.restore(zq).to(w.dtype)
