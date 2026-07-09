"""Fused Triton Viterbi for the bitshift trellis (L=16), one program per tile.

Why: the pure-PyTorch DP materializes ~65K-state tensors through DRAM every
step — on GB10's LPDDR5x that's a ~50-hour encode for an 8B. This kernel keeps
the DP working set (a 2^(L-K)-entry cost vector) resident per-tile (L1/L2
ping-pong scratch), computes the MCG codebook *in registers* (no table reads),
and only streams out backpointers. Same math as qtip._viterbi_pass, exactly.

Layout notes (mirrors qtip.py):
  state s = (q << (L-K)) | e,   e = edge = s & (E-1),   E = 2^(L-K)
  predecessor edge of s = s >> K = (q << (L-2K)) | (e >> K)
Backtrace stays in PyTorch (cheap: T gathers over [n]).
"""
from __future__ import annotations
import torch

import triton
import triton.language as tl

L = 16
S = 1 << L


@triton.jit
def _mcg_val(s):
    """EXL3 'mcg' codebook, computed: s -> fp32 value (see qtip.cb_mcg)."""
    x = (s * 0xCBAC1FED).to(tl.uint32)
    res = (x & 0x8FFF8FFF) ^ 0x3B603B60
    hi = ((res >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    lo = (res & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    return (hi + lo).to(tl.float32)


@triton.jit
def _viterbi_tile_kernel(
    w_ptr,          # [n, T] fp32 targets (tile-major)
    bp_ptr,         # [n, T, E] uint8 backpointers (out)
    cost_ptr,       # [n, 2, E] fp32 scratch (ping-pong)
    init_edge_ptr,  # [n] int32; -1 = unconstrained, else forced initial edge
    T: tl.constexpr, K: tl.constexpr, E: tl.constexpr, Q: tl.constexpr,
    L: tl.constexpr,
):
    tile = tl.program_id(0)
    e = tl.arange(0, E)                       # output-edge lanes
    INF = 1e30

    init_edge = tl.load(init_edge_ptr + tile)
    cost = tl.where((init_edge < 0) | (e == init_edge), 0.0, INF)

    buf0 = cost_ptr + tile * 2 * E
    buf1 = buf0 + E
    h_new = e >> K                            # pred-edge low part, per lane

    for t in range(T):
        # write current costs to scratch so lanes can cross-read (ping-pong)
        cur = buf0 if t % 2 == 0 else buf1
        tl.store(cur + e, cost)
        tl.debug_barrier()
        w_t = tl.load(w_ptr + tile * T + t)

        best = tl.full((E,), INF, tl.float32)
        arg = tl.zeros((E,), tl.uint8)
        for q in tl.static_range(Q):
            pred = (q << (L - 2 * K)) | h_new
            c_prev = tl.load(cur + pred)
            s = ((q << (L - K)) | e).to(tl.uint32)
            d = w_t - _mcg_val(s)
            cand = c_prev + d * d
            take = cand < best
            best = tl.where(take, cand, best)
            arg = tl.where(take, tl.full((E,), q, tl.uint8), arg)
        cost = best
        tl.store(bp_ptr + tile * T * E + t * E + e, arg)
        tl.debug_barrier()

    # persist final costs for the host-side argmin / constrained pick
    fin = buf0 if T % 2 == 0 else buf1
    tl.store(fin + e, cost)


@torch.no_grad()
def viterbi_pass_triton(w: torch.Tensor, K: int,
                        overlap: torch.Tensor | None) -> torch.Tensor:
    """Drop-in replacement for qtip._viterbi_pass (MCG codebook only).

    w: [n, T] fp32 on CUDA. Returns optimal state path [n, T] int64.
    """
    n, T = w.shape
    dev = w.device
    E = 1 << (L - K)
    Q = 1 << K
    w = w.float().contiguous()
    states = torch.empty(n, T, dtype=torch.long, device=dev)
    # Bound the grid and the backpointer footprint (bp is n*T*E bytes): stream
    # tiles through in host-side batches. Each batch: bp [B,T,E], scratch [B,2,E].
    max_bp_bytes = 1 << 30                                 # ~1 GiB of backpointers per batch
    B = max(1, min(n, max_bp_bytes // (T * E)))
    for i in range(0, n, B):
        wb = w[i:i + B].contiguous()
        b = wb.shape[0]
        bp = torch.empty(b, T, E, dtype=torch.uint8, device=dev)
        scratch = torch.empty(b, 2, E, dtype=torch.float32, device=dev)
        if overlap is not None:
            init = overlap[i:i + b].to(torch.int32).contiguous()
        else:
            init = torch.full((b,), -1, dtype=torch.int32, device=dev)
        _viterbi_tile_kernel[(b,)](wb, bp, scratch, init, T=T, K=K, E=E, Q=Q,
                                   L=L, num_warps=8)
        final_cost = scratch[:, T % 2, :]                  # [b, E]
        e = (final_cost.argmin(dim=1) if overlap is None
             else overlap[i:i + b].long().clone())
        for t in range(T - 1, -1, -1):
            q = bp[:, t, :].gather(1, e.long().unsqueeze(1)).squeeze(1).long()
            s_t = (q << (L - K)) | e
            states[i:i + b, t] = s_t
            e = s_t >> K
    return states


@torch.no_grad()
def viterbi_tb_triton(w: torch.Tensor, K: int, lut: torch.Tensor):
    """Tail-biting encode using the fused kernel (mirrors qtip.viterbi_tb)."""
    n, T = w.shape
    roll = T // 2
    states1 = viterbi_pass_triton(torch.roll(w, roll, dims=1), K, None)
    overlap = states1[:, roll] >> K
    states2 = viterbi_pass_triton(w, K, overlap)
    bits = states2 & ((1 << K) - 1)
    return bits, lut.float().to(w.device)[states2]
