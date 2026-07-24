"""Faithful QTIP/EXL3-style trellis quantization (see notes/qtip-exl3-spec.md).

Key facts this module implements exactly:
- Bitshift trellis, L=16 states, V=1, K bits/weight: s_t = ((s_{t-1} << K) | b_t) & 0xFFFF.
  The state at step t IS the sliding 16-bit window of the packed bitstream —
  decode is a pure function of the stream (random-access), never a recursion.
- Computed codebooks with the published constants: 1MAD, 3INST, MCG (EXL3 default).
- Exact Viterbi DP over (L-K)-bit *edges* (EXL3 quantize.cu formulation) with the
  QTIP Algorithm-4 two-pass approximate tail-biting (roll by T/2), so storage is
  exactly K bits/weight per T=256 tile.
- EXL3-flavored incoherence processing: ±1 sign vectors + block-diagonal
  Hadamard-128 on both dims + RMS normalization to the codebook scale.
  (v0 omits LDLQ/Hessian feedback and the golden-section global scale: this is
  EXL3's Hessian-free fallback mode == QTIP paper Table 3 "no fine-tune" regime.)
"""
from __future__ import annotations

import torch

MASK32 = 0xFFFFFFFF
L = 16
S = 1 << L                 # 65536 states
T_SEQ = 256                # one 16x16 tile per trellis sequence
CODEBOOK_SCALE_EXL3 = 1.24371088   # exl3 quantize.py line 15


# --------------------------------------------------------------------------- codebooks
def _fp16_from_bits(bits16: torch.Tensor) -> torch.Tensor:
    return bits16.to(torch.uint16).view(torch.float16).float()


def cb_1mad(x: torch.Tensor) -> torch.Tensor:
    """QTIP Alg.1 (decode_1mad): LCG then sum of 4 bytes, standardized."""
    x = x.to(torch.int64)
    x = (x * 34038481 + 76625530) & MASK32
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    return (y.float() - 510.0) / 147.800537109375


def cb_3inst(x: torch.Tensor) -> torch.Tensor:
    """QTIP Alg.2 (decode_3inst): LCG, lop3 mask/xor, sum of fp16 halves."""
    x = x.to(torch.int64)
    x = (x * 89226354 + 64248484) & MASK32
    res = (x & 0x8FFF8FFF) ^ 0x3B603B60
    return _fp16_from_bits((res >> 16) & 0xFFFF) + _fp16_from_bits(res & 0xFFFF)


def cb_mcg(x: torch.Tensor) -> torch.Tensor:
    """EXL3 default 'mcg': multiplicative congruential (no add), same tail as 3INST."""
    x = x.to(torch.int64)
    x = (x * 0xCBAC1FED) & MASK32
    res = (x & 0x8FFF8FFF) ^ 0x3B603B60
    return _fp16_from_bits((res >> 16) & 0xFFFF) + _fp16_from_bits(res & 0xFFFF)


CODEBOOKS = {"1mad": cb_1mad, "3inst": cb_3inst, "mcg": cb_mcg}


def codebook_lut(name: str, device="cpu") -> torch.Tensor:
    """Evaluate the computed codebook over all 2^16 states -> [65536] fp32 LUT.

    (The decoder never stores this; kernels compute it per-state. The LUT is an
    encode-time convenience and is bit-identical to the computed function.)
    """
    states = torch.arange(S, device=device)
    return CODEBOOKS[name](states).float()


# --------------------------------------------------------------------------- decode
def decode_recursive(bits: torch.Tensor, K: int, lut: torch.Tensor,
                     s0: torch.Tensor | None = None) -> torch.Tensor:
    """Reference sequential decode: state recursion. bits [n, T] -> values [n, T]."""
    n, T = bits.shape
    s = torch.zeros(n, dtype=torch.long, device=bits.device) if s0 is None else s0.clone()
    out = torch.empty(n, T, dtype=torch.float32, device=bits.device)
    for t in range(T):
        s = ((s << K) | bits[:, t].long()) & (S - 1)
        out[:, t] = lut[s]
    return out


def decode_window(bits: torch.Tensor, K: int, lut: torch.Tensor,
                  s0: torch.Tensor | None = None) -> torch.Tensor:
    """Random-access decode: state_t = last 16 bits of the stream ending at t.

    No recursion — every position computed independently (this is what maps to
    one WebGPU/Metal/NEON lane per weight or per small chunk). Must equal
    decode_recursive exactly; test_qtip.py asserts it.
    """
    n, T = bits.shape
    dev = bits.device
    steps_needed = (L + K - 1) // K              # how many past symbols cover 16 bits
    b = bits.long()
    if s0 is None:
        s0 = torch.zeros(n, dtype=torch.long, device=dev)
    state = torch.zeros(n, T, dtype=torch.long, device=dev)
    for j in range(steps_needed):                # vectorized over ALL t simultaneously
        idx = torch.arange(T, device=dev) - (steps_needed - 1 - j)
        valid = idx >= 0
        sym = torch.where(valid.unsqueeze(0), b[:, idx.clamp(min=0)], torch.zeros_like(b[:, :1]))
        # positions before the stream start draw their bits from s0's window;
        # that seed contribution is applied in the (s0 != 0) recursion pass below.
        state = ((state << K) | sym) & (S - 1)
    if (s0 != 0).any():
        # early positions whose window extends before t=0 need s0 bits: recompute
        # them with the recursion (cheap: only first ceil(L/K)-1 positions).
        warm = min(steps_needed - 1, T)
        if warm > 0:
            state[:, :warm] = _states_recursive(b[:, :warm], K, s0)
    return lut[state]


def _states_recursive(bits: torch.Tensor, K: int, s0: torch.Tensor) -> torch.Tensor:
    n, T = bits.shape
    s = s0.clone()
    out = torch.empty(n, T, dtype=torch.long, device=bits.device)
    for t in range(T):
        s = ((s << K) | bits[:, t].long()) & (S - 1)
        out[:, t] = s
    return out


# --------------------------------------------------------------------------- encode
@torch.no_grad()
def viterbi_tb(w: torch.Tensor, K: int, lut: torch.Tensor,
               dp_dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail-biting trellis encode of w [n, T=256] -> (bits [n,T] ints, dq [n,T]).

    QTIP Algorithm 4: (1) Viterbi on the sequence rolled by T/2, unconstrained;
    (2) read the (L-K)-bit overlap at the roll point; (3) Viterbi on the true
    sequence constrained to start from and end at that overlap. Exactly K bpw.
    dp_dtype (e.g. torch.bfloat16) speeds up emission-cost math only; the
    returned dq always uses the fp32 codebook.
    """
    n, T = w.shape
    assert T == T_SEQ, f"expected T={T_SEQ}"
    roll = T // 2
    states1 = _viterbi_pass(torch.roll(w, roll, dims=1), K, lut, None, dp_dtype)
    # state at rolled position T/2 corresponds to original position 0; its top
    # L-K bits are the overlap the wrapped stream must honor.
    overlap = states1[:, roll] >> K
    states2 = _viterbi_pass(w, K, lut, overlap, dp_dtype)
    bits = states2 & ((1 << K) - 1)
    return bits, lut.float()[states2]


@torch.no_grad()
def _viterbi_pass(w: torch.Tensor, K: int, lut: torch.Tensor,
                  overlap: torch.Tensor | None, dp_dtype=None) -> torch.Tensor:
    """Exact DP over (L-K)-bit edges (EXL3 quantize.cu formulation), batched.

    Returns the optimal state path [n, T]. cost/backptr memory:
    cost [n, 2^(L-K)] fp32, backptr [T, n, 2^(L-K)] uint8.
    """
    n, T = w.shape
    dev = w.device
    E = 1 << (L - K)           # number of edges
    Q = 1 << K                 # candidates per edge
    INF = torch.finfo(torch.float32).max / 4

    if overlap is None:
        cost = torch.zeros(n, E, device=dev)
    else:                      # constrain first state's predecessor edge
        cost = torch.full((n, E), INF, device=dev)
        cost.scatter_(1, overlap.unsqueeze(1), 0.0)

    bp = torch.empty(T, n, E, dtype=torch.uint8, device=dev)
    # Predecessor edge of state (q<<(L-K))|e is that state >> K, i.e.
    # pe(q, e) = (q << (L-2K)) | (e >> K).  Viewing cost as [n, Q, E>>K] makes
    # the "gather" a pure broadcast: cand[n,q,h*2^K+l] = cost[n,q,h] + err —
    # no index gather at all (this was the old inner-loop bottleneck).
    H_ = E >> K                                                       # 2^(L-2K)
    dpt = dp_dtype or torch.float32
    lut_v = lut.to(dpt).view(1, Q, H_, 1 << K)                        # state value at [q,h,l]
    w = w.to(dpt)

    for t in range(T):
        err = (w[:, t].view(n, 1, 1, 1) - lut_v) ** 2                 # [n, Q, H, 2^K]
        cand = err.float() + cost.view(n, Q, H_, 1)                   # broadcast add
        best, arg = cand.min(dim=1)                                   # [n, H, 2^K]
        cost = best.reshape(n, E)
        bp[t] = arg.reshape(n, E).to(torch.uint8)

    # backtrace
    if overlap is None:
        e = cost.argmin(dim=1)                                       # [n]
    else:                       # constrain final edge == overlap (stream wraps)
        e = overlap.clone()
    rows = torch.arange(n, device=dev)
    states = torch.empty(n, T, dtype=torch.long, device=dev)
    for t in range(T - 1, -1, -1):
        q = bp[t, rows, e].long()
        s_t = (q << (L - K)) | e
        states[:, t] = s_t
        e = s_t >> K            # previous edge
    return states


# --------------------------------------------------------------------------- incoherence processing
def _hadamard(n: int, device) -> torch.Tensor:
    """Sylvester Hadamard matrix H_n / sqrt(n) (n power of 2): orthogonal, symmetric."""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def _fit_block(d: int, block: int) -> int:
    """Largest power-of-2 divisor of d, capped at `block` (handles dims like 576)."""
    b = 1
    while b < block and d % (b * 2) == 0:
        b *= 2
    return b


def _block_had(x: torch.Tensor, dim: int, block: int = 128) -> torch.Tensor:
    """Block-diagonal Hadamard along `dim` (EXL3 uses 128; shrinks to fit odd dims)."""
    x = x.movedim(dim, -1)
    d = x.shape[-1]
    block = _fit_block(d, block)
    H = _hadamard(block, x.device)
    y = (x.reshape(*x.shape[:-1], d // block, block) @ H).reshape(x.shape)
    return y.movedim(-1, dim)


class IPContext:
    """EXL3-flavored incoherence processing (spec §D).

    Order (matching exl3 quantize.py `regularize`):
      1. ±1 sign flips both dims,
      2. per-OUT-channel RMS scales (fold into sv), block-Hadamard along out dim,
      3. per-IN-channel RMS scales (fold into su), block-Hadamard along in dim,
      4. global RMS normalization to the codebook scale.
    The per-channel scales are what tame LLM outlier channels when using
    block-diagonal (not full-dim) Hadamard mixing.
    """

    def __init__(self, w: torch.Tensor, cb_rms: float, seed: int = 0, block: int = 128):
        g = torch.Generator(device="cpu").manual_seed(seed)
        n, k = w.shape
        self.block = block
        sv = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1).to(w.device)
        su = (torch.randint(0, 2, (k,), generator=g).float() * 2 - 1).to(w.device)

        z = w * sv.view(-1, 1) * su.view(1, -1)
        # per-out-channel RMS (normalized to mean 1), then mix the out dim
        out_s = z.square().mean(dim=1).sqrt().clamp_min(1e-8)
        out_s = out_s / out_s.mean()
        z = z / out_s.view(-1, 1)
        z = _block_had(z, dim=0, block=block)
        # per-in-channel RMS (normalized to mean 1), then mix the in dim
        in_s = z.square().mean(dim=0).sqrt().clamp_min(1e-8)
        in_s = in_s / in_s.mean()
        z = z / in_s.view(1, -1)
        z = _block_had(z, dim=1, block=block)

        self.sv, self.su = sv, su
        self.out_s, self.in_s = out_s, in_s
        self.scale = z.square().mean().sqrt().item() / cb_rms
        self.z = z / self.scale

    def restore(self, zq: torch.Tensor) -> torch.Tensor:
        y = zq * self.scale
        y = _block_had(y, dim=1, block=self.block)   # Hadamard/sqrt is its own inverse
        y = y * self.in_s.view(1, -1)
        y = _block_had(y, dim=0, block=self.block)
        y = y * self.out_s.view(-1, 1)
        return y * self.sv.view(-1, 1) * self.su.view(1, -1)


# --------------------------------------------------------------------------- layer quantizer
@torch.no_grad()
def tcq_quantize_weight(w: torch.Tensor, K: int, lut: torch.Tensor,
                        seed: int = 0, tile_batch: int = 16384,
                        dp_dtype=torch.bfloat16, backend: str = "auto") -> torch.Tensor:
    """Fake-quantize one weight matrix [n_out, n_in] with IP + tail-biting TCQ.

    Tiles: 16 consecutive output rows x 16 input channels -> T=256 sequences
    (QTIP ldlq.py slab reshape). Returns the dequantized fp matrix (same shape).
    backend: 'triton' (fused, MCG only, ~5x faster on GPU), 'torch' (portable),
    or 'auto' (triton on CUDA, else torch).
    """
    n, k = w.shape
    assert n % 16 == 0 and k % 16 == 0, f"shape {w.shape} not 16-divisible"
    cb_rms = lut.square().mean().sqrt().item()
    ip = IPContext(w.float(), cb_rms, seed=seed)
    z = ip.z                                                   # [n, k]
    tiles = z.view(n // 16, 16, k // 16, 16).permute(0, 2, 1, 3).reshape(-1, T_SEQ)

    use_triton = backend == "triton" or (backend == "auto" and w.is_cuda)
    if use_triton:
        from trellis.viterbi_triton import viterbi_tb_triton

        def enc(t):
            return viterbi_tb_triton(t, K, lut)[1]
    else:
        def enc(t):
            return viterbi_tb(t, K, lut, dp_dtype=dp_dtype)[1]

    out = torch.empty_like(tiles)
    for i in range(0, tiles.shape[0], tile_batch):
        out[i:i + tile_batch] = enc(tiles[i:i + tile_batch]).float()
    zq = out.view(n // 16, k // 16, 16, 16).permute(0, 2, 1, 3).reshape(n, k)
    return ip.restore(zq).to(w.dtype)
