#!/usr/bin/env python
"""End-to-end model quality: TCQ (QTIP-style, K bpw) vs scalar quant vs fp16.

This is the project's quality gate, model-sized. Produces the table:
    fp16  |  uniform-Kb  |  TCQ-Kb (ours)
on wikitext-2 perplexity. Run small (SmolLM2-135M) for a fast signal, then
Qwen3-8B on the DGX for the headline.

Example:
  python scripts/run_tcq_eval.py --model HuggingFaceTB/SmolLM2-135M --K 2 --max-samples 30
"""
import argparse
import os
import sys
import time
import threading
import zlib


def start_ram_guard(limit_gb: float):
    """Abort the process if SYSTEM used-RAM exceeds limit_gb (shared box safety).

    Reads /proc/meminfo (MemTotal - MemAvailable) every 2s. On breach, prints a
    loud line and os._exit(137) so we never OOM-kill a co-tenant's job.
    """
    def total_avail():
        info = {}
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":")
            info[k] = int(v.strip().split()[0])  # kB
        used = (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024
        return used
    def loop():
        while True:
            try:
                used = total_avail()
                if used > limit_gb:
                    print(f"\n[RAM-GUARD] system RAM {used:.0f}G > limit {limit_gb:.0f}G "
                          f"— aborting to protect the shared box.", flush=True)
                    os._exit(137)
            except Exception:
                pass
            time.sleep(2)
    t = threading.Thread(target=loop, daemon=True)
    t.start()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from trellis.eval import load_model, get_wikitext2_test, eval_ppl
from trellis.quant import UniformGroupQuant, apply_fake_quant
from trellis.qtip import codebook_lut, tcq_quantize_weight


@torch.no_grad()
def collect_hessians(model, tok, nseq, device, skip=("lm_head",),
                     store_cpu=True, store_fp16=True, only=None):
    """If `only` is a tuple of substrings, only hook Linears whose name matches
    (e.g. only=('down_proj','gate_proj','up_proj') => MLP-only, far lower peak)."""
    """One forward pass caching H = E[xxᵀ] per Linear, for LDLQ.

    MEMORY-SAFE: accumulate H on-GPU per layer during the pass (needed for the
    matmul), but immediately move each finished H to CPU (and optionally fp16)
    so the resident set is ~half and off the unified-memory hot path. For an 8B
    the down_proj Hessians (12288²) dominate; keeping 252 of them fp32-on-device
    is ~40GB — this halves+offloads that. Caller frees each after use.
    """
    ids = get_wikitext2_test(tok)
    acc, handles = {}, []
    lin = [(n, m) for n, m in model.named_modules()
           if isinstance(m, nn.Linear) and not any(s in n for s in skip)
           and (only is None or any(o in n for o in only))]
    for name, mod in lin:
        acc[name] = [torch.zeros(mod.weight.shape[1], mod.weight.shape[1], device=device), 0]
        def mk(nm):
            def hook(m, inp):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                acc[nm][0] += x.T @ x; acc[nm][1] += x.shape[0]
            return hook
        handles.append(mod.register_forward_pre_hook(mk(name)))
    with torch.no_grad():
        for i in range(nseq):
            b = ids[:, i * 2048:(i + 1) * 2048].to(device)
            if b.shape[1] < 2048:
                break
            model(b)
    for h in handles:
        h.remove()
    out = {}
    for n, (a, c) in acc.items():
        H = a / max(c, 1)
        if store_fp16:
            H = H.half()
        if store_cpu:
            H = H.cpu()
        out[n] = H
    acc.clear()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def apply_tcq(model: nn.Module, K: int, lut: torch.Tensor, tile_batch: int,
              device: str, backend: str = "auto", skip=("lm_head",),
              hessians: dict | None = None) -> int:
    n = 0
    t0 = time.time()
    layers = [(name, m) for name, m in model.named_modules()
              if isinstance(m, nn.Linear) and not any(s in name for s in skip)]
    total = len(layers)
    for i, (name, mod) in enumerate(layers):
        w = mod.weight.data.to(device)
        seed = zlib.crc32(name.encode())
        if hessians is not None and name in hessians:
            from trellis.ldlq import tcq_quantize_weight_ldlq
            H = hessians[name].to(device=device, dtype=torch.float32)  # rehydrate one
            wq = tcq_quantize_weight_ldlq(w.float(), K=K, lut=lut.to(device),
                                          H=H, seed=seed)
            del H, hessians[name]                     # free this Hessian immediately
        else:
            wq = tcq_quantize_weight(w.float(), K=K, lut=lut.to(device),
                                     seed=seed, tile_batch=tile_batch, backend=backend)
        mod.weight.data.copy_(wq.to(mod.weight.dtype))
        del wq
        n += 1
        if i % 10 == 0 or i == total - 1:
            el = time.time() - t0
            print(f"  [tcq] {i + 1}/{total} layers  ({el:.0f}s elapsed, "
                  f"~{el / (i + 1) * (total - i - 1):.0f}s left)", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, default=2, help="trellis bits per weight")
    ap.add_argument("--codebook", default="mcg", choices=["mcg", "3inst", "1mad"])
    ap.add_argument("--tile-batch", type=int, default=16384)
    ap.add_argument("--backend", default="auto", choices=["auto", "triton", "torch"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--skip-uniform", action="store_true")
    ap.add_argument("--ldlq", type=int, default=0, metavar="NSEQ",
                    help="if >0, use LDLQ with a Hessian from NSEQ calibration seqs")
    ap.add_argument("--ldlq-mlp-only", action="store_true",
                    help="collect+apply Hessians only for MLP layers (low memory; "
                         "attention TCQ is fine per diagnosis)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ram-limit-gb", type=float, default=110.0,
                    help="abort if system used-RAM exceeds this (shared-box safety)")
    args = ap.parse_args()

    start_ram_guard(args.ram_limit_gb)

    dev = args.device
    print(f"[cfg] model={args.model} K={args.K} cb={args.codebook} "
          f"tile_batch={args.tile_batch} dev={dev}", flush=True)

    model, tok = load_model(args.model, device=dev)
    ids = get_wikitext2_test(tok)
    import copy
    # Only keep a full-model backup if we actually need to restore (uniform pass).
    base_sd = copy.deepcopy(model.state_dict()) if not args.skip_uniform else None

    fp16 = eval_ppl(model, ids, args.seqlen, dev, args.max_samples)
    print(f"\n== fp16 ppl = {fp16:.4f}\n", flush=True)
    rows = [("fp16", fp16)]

    if not args.skip_uniform:
        q = UniformGroupQuant(args.K, 128, sym=False)
        apply_fake_quant(model, q)
        ppl_u = eval_ppl(model, ids, args.seqlen, dev, args.max_samples)
        rows.append((q.name, ppl_u))
        print(f"== {q.name} ppl = {ppl_u:.4f}\n", flush=True)
        model.load_state_dict(base_sd)

    hessians = None
    tag = f"tcq-{args.K}b-{args.codebook}"
    if args.ldlq > 0:
        # Diagnosis: only the MLP (esp. down_proj) needs Hessian feedback;
        # attention TCQ is fine. MLP-only keeps far fewer/smaller Hessians live.
        only = ("down_proj", "gate_proj", "up_proj") if args.ldlq_mlp_only else None
        print(f"[ldlq] collecting Hessians from {args.ldlq} seqs"
              f"{' (MLP-only)' if only else ''}...", flush=True)
        t0 = time.time()
        # store_fp16=False: fp16 rounding of a diag-skewed 12288-dim H makes it
        # indefinite by ~O(dmax), forcing heavy Cholesky damping that weakens
        # feedback on exactly the massive-activation layers that need it most.
        hessians = collect_hessians(model, tok, args.ldlq, dev, only=only,
                                    store_fp16=False)
        print(f"[ldlq] {len(hessians)} Hessians in {time.time() - t0:.0f}s", flush=True)
        tag += f"-ldlq{args.ldlq}" + ("-mlp" if only else "")

    lut = codebook_lut(args.codebook)
    t0 = time.time()
    nl = apply_tcq(model, args.K, lut, args.tile_batch, dev, backend=args.backend,
                   hessians=hessians)
    print(f"[tcq] quantized {nl} layers in {time.time() - t0:.0f}s", flush=True)
    ppl_t = eval_ppl(model, ids, args.seqlen, dev, args.max_samples)
    rows.append((tag, ppl_t))

    print("\n=== RESULT (wikitext-2 ppl, lower is better) ===")
    for name, p in rows:
        print(f"  {name:24s} {p:12.4f}")


if __name__ == "__main__":
    main()
