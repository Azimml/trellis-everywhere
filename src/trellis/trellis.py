"""Generic bitshift-trellis quantizer (QTIP/EXL3 family), batched PyTorch.

The scheme
----------
A *bitshift trellis* has 2^L states. Decoding walks a state sequence
``s_1 .. s_T``; each step consumes k fresh bits ``b_t`` via

    s_t = ((s_{t-1} << k) | b_t) & (2**L - 1)

and emits one reconstructed weight ``value = codebook(s_t)``.  Storage is k
bits per weight (the fresh bits), yet the emitted value depends on the last
L bits of history — that overlap is why TCQ beats scalar quantization at
equal bits.  Encoding = Viterbi dynamic programming: find the bit stream
whose decoded values minimize squared error against the target weights.

Decode is inherently *sequential* in s (this is why fast decoders are CUDA
warp-shuffle tricks).  Our central research question: restart the state
every `block` weights (block-restarted trellis) so blocks decode in
parallel — and measure the quality cost.  `encode(..., block=N)` implements
exactly that; `block=None` is the classic sequential trellis.

This module is deliberately *generic* (any L, k, codebook callable) so unit
tests can verify optimality at tiny sizes against brute force. The faithful
QTIP codebooks (1MAD / 3INST / HYB) and incoherence processing layer on top
in qtip.py once their spec is pinned.

Conventions
-----------
* Emit-on-arrival: the value emitted at step t is codebook(s_t) — the state
  *after* consuming b_t. First step: s_1 = ((s_0 << k) | b_1) & mask with
  s_0 = 0 unless an initial state is given.
* Shapes: weights are encoded as independent rows [n, T] (n sequences of
  length T). Viterbi is fully vectorized over n and over states.
"""
from __future__ import annotations
import torch


class BitshiftTrellis:
    def __init__(self, L: int, k: int, codebook_values: torch.Tensor):
        """
        L: state bits (2^L states). k: fresh bits consumed per weight.
        codebook_values: [2^L] tensor mapping state -> emitted value (fp32).
        """
        assert 1 <= k <= L
        self.L, self.k = L, k
        self.S = 1 << L
        self.mask = self.S - 1
        assert codebook_values.shape == (self.S,)
        self.cb = codebook_values.float()

    # -------------------------------------------------- decode (reference)
    def decode(self, bits: torch.Tensor, s0: int | torch.Tensor = 0,
               block: int | None = None) -> torch.Tensor:
        """bits: [n, T] ints in [0, 2^k). Returns values [n, T] fp32.

        block=N restarts the state at every N-weight boundary (s -> s0),
        making blocks independently decodable — the parallel-friendly mode.
        Sequential python loop over T: this is the *reference* decoder used
        for correctness; fast kernels come later.
        """
        n, T = bits.shape
        dev = bits.device
        cb = self.cb.to(dev)
        s = torch.full((n,), 0, dtype=torch.long, device=dev)
        if isinstance(s0, torch.Tensor):
            s = s0.long().clone()
        elif s0:
            s = torch.full((n,), int(s0), dtype=torch.long, device=dev)
        out = torch.empty(n, T, dtype=torch.float32, device=dev)
        for t in range(T):
            if block is not None and t % block == 0 and t > 0:
                s.zero_() if not isinstance(s0, torch.Tensor) else s.copy_(s0.long())
                if not isinstance(s0, torch.Tensor) and s0:
                    s.fill_(int(s0))
            s = ((s << self.k) | bits[:, t].long()) & self.mask
            out[:, t] = cb[s]
        return out

    # -------------------------------------------------- encode (Viterbi)
    @torch.no_grad()
    def encode(self, w: torch.Tensor, s0: int = 0,
               block: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Viterbi-optimal encode. w: [n, T] fp32 -> (bits [n, T], dq [n, T]).

        block=N: encode each N-chunk independently (state restarts) — the
        parallel-friendly variant. block=None: one sequential trellis.
        Exact DP (no beam): dp over all 2^L states, vectorized on device.
        """
        if block is None:
            return self._encode_chunk(w, s0)
        n, T = w.shape
        assert T % block == 0, "T must be divisible by block"
        wb = w.reshape(n * (T // block), block)
        bits, dq = self._encode_chunk(wb, s0)
        return bits.reshape(n, T), dq.reshape(n, T)

    def _encode_chunk(self, w: torch.Tensor, s0: int) -> tuple[torch.Tensor, torch.Tensor]:
        n, T = w.shape
        dev = w.device
        S, k, L = self.S, self.k, self.L
        cb = self.cb.to(dev)                          # [S]
        # Predecessors of state s' are p with low (L-k) bits of p == high bits
        # of s'; equivalently p in {(s' >> k) + j << (L-k)} for j in 0..2^k-1.
        sp = torch.arange(S, device=dev)
        base = sp >> k                                 # [S]
        js = torch.arange(1 << k, device=dev) << (L - k)
        preds = base.unsqueeze(1) + js.unsqueeze(0)    # [S, 2^k] pred state ids

        # From-the-start reachability: after t steps only states whose history
        # above t*k bits matches s0's shifted bits are reachable. Handle by
        # initializing dp = +inf except s0 and *allowing* only real preds —
        # the DP below naturally propagates inf for unreachable states.
        INF = torch.finfo(torch.float32).max / 4
        dp = torch.full((n, S), INF, device=dev)
        dp[:, int(s0) & self.mask] = 0.0
        # backptr: which j (0..2^k-1) won, per (t, n, state) — k bits each.
        bp = torch.empty(T, n, S, dtype=torch.uint8 if k <= 8 else torch.int16, device=dev)

        err = (w.unsqueeze(2) - cb.view(1, 1, S)) ** 2  # [n, T, S] emission costs
        for t in range(T):
            # candidate costs: dp at each predecessor of every state
            cand = dp[:, preds]                        # [n, S, 2^k]
            best, arg = cand.min(dim=2)                # [n, S]
            dp = best + err[:, t, :]
            bp[t] = arg.to(bp.dtype)

        # trace back from the best final state
        state = dp.argmin(dim=1)                       # [n]
        bits = torch.empty(n, T, dtype=torch.long, device=dev)
        rows = torch.arange(n, device=dev)
        for t in range(T - 1, -1, -1):
            bits[:, t] = state & ((1 << k) - 1)        # fresh bits = low k bits of state
            j = bp[t, rows, state].long()
            state = (state >> k) + (j << (L - k))      # step back to predecessor
        dq = self.decode(bits, s0=s0)
        return bits, dq


def gaussian_codebook(L: int, seed: int = 0) -> torch.Tensor:
    """Placeholder codebook: 2^L iid N(0,1) values from a fixed seed.

    QTIP's real codebooks (1MAD/3INST/HYB) are *computed* hashes of the state
    so the decoder needs no 2^L-entry table in memory; statistically they
    approximate exactly this. Fine for engine correctness tests and early
    quality curves; swapped for faithful codebooks in qtip.py.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1 << L, generator=g)
