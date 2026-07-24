"""Weight quantizers for Trellis-WebGPU.

Phase 0 ships fake-quant baselines (uniform per-group round-to-nearest) so the
eval harness has something to measure *today*. These are the numbers the real
trellis quantizer must beat at equal bits/weight.

"Fake quant" = quantize then dequantize back to fp16 in place. It measures the
QUALITY cost of a scheme (perplexity), independent of any fast kernel — exactly
what Phase 1's go/no-go quality gate needs.

The real (sequential + block-parallel) trellis quantizer lands in trellis.py
once the faithful QTIP/EXL3 spec is extracted; both will implement `Quantizer`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Quantizer:
    """Interface: map a weight matrix [out, in] -> a fake-quantized fp16 matrix."""
    name: str = "identity"
    bpw: float = 16.0

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return w


def _uniform_group_fakequant(w: torch.Tensor, bits: int, groupsize: int, sym: bool) -> torch.Tensor:
    """Asymmetric (or symmetric) per-group uniform quant along the input dim.

    w: [out, in]. Groups of `groupsize` columns share a scale/zero-point — this
    is the same structure GGUF K-quants and GPTQ/AWQ use, so it's an honest
    baseline. Handles a non-divisible last group by falling back to per-row.
    """
    assert w.dim() == 2, "expected a 2D weight [out, in]"
    out_f, in_f = w.shape
    gs = in_f if (groupsize <= 0 or in_f % groupsize != 0) else groupsize
    w = w.float()
    wg = w.reshape(out_f, in_f // gs, gs)  # [out, ngroups, gs]

    qmax = (1 << bits) - 1
    if sym:
        amax = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = amax / (qmax // 2 if qmax >= 2 else 1)
        q = torch.clamp(torch.round(wg / scale), -(qmax // 2 + 1), qmax // 2)
        dq = q * scale
    else:
        wmin = wg.amin(dim=-1, keepdim=True)
        wmax = wg.amax(dim=-1, keepdim=True)
        scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
        zero = torch.round(-wmin / scale)
        q = torch.clamp(torch.round(wg / scale) + zero, 0, qmax)
        dq = (q - zero) * scale
    return dq.reshape(out_f, in_f).to(torch.float16)


class UniformGroupQuant(Quantizer):
    """Baseline: N-bit per-group uniform quantization (GGUF/GPTQ-style)."""

    def __init__(self, bits: int, groupsize: int = 128, sym: bool = False):
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.bpw = float(bits)  # (ignores scale/zp overhead; fine for a baseline)
        self.name = f"uniform-{bits}b-g{groupsize}{'-sym' if sym else ''}"

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        return _uniform_group_fakequant(w, self.bits, self.groupsize, self.sym)


@torch.no_grad()
def apply_fake_quant(model: nn.Module, q: Quantizer, skip: tuple[str, ...] = ("lm_head",)) -> int:
    """In-place fake-quantize every nn.Linear weight (except `skip` names).

    Returns the number of layers quantized. Embeddings and the final lm_head are
    skipped by default (standard practice — they're quality-sensitive and cheap).
    """
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(s in name for s in skip):
            dq = q.quantize_weight(mod.weight.data)
            mod.weight.data.copy_(dq.to(mod.weight.dtype))
            n += 1
    return n
