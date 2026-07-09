#!/usr/bin/env python
"""Phase-0 baseline runner: fp16 perplexity vs uniform group-quant at N bits.

Examples
--------
# Laptop sanity (fits 4 GB):
python scripts/run_eval.py --model HuggingFaceTB/SmolLM2-135M --bits 2 3 4 --groupsize 128

# The headline gate, once a big GPU is wired up (unchanged code):
python scripts/run_eval.py --model meta-llama/Llama-3.1-8B --bits 2 3 4 --max-samples 40
"""
import argparse
import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from trellis.eval import load_model, get_wikitext2_test, eval_ppl
from trellis.quant import UniformGroupQuant, apply_fake_quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap eval windows (use on big models / slow cards)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[load] {args.model} on {args.device}")
    model, tok = load_model(args.model, device=args.device)
    ids = get_wikitext2_test(tok)
    print(f"[data] wikitext-2 test: {ids.numel():,} tokens "
          f"({ids.numel() // args.seqlen} windows @ seqlen {args.seqlen})")

    base_sd = copy.deepcopy(model.state_dict())  # to restore between quant configs

    fp16 = eval_ppl(model, ids, args.seqlen, args.device, args.max_samples)
    rows = [("fp16", 16.0, fp16)]
    print(f"\n  fp16 baseline ppl = {fp16:.4f}\n")

    for b in args.bits:
        model.load_state_dict(base_sd)  # restore clean weights
        q = UniformGroupQuant(b, args.groupsize, args.sym)
        n = apply_fake_quant(model, q)
        ppl = eval_ppl(model, ids, args.seqlen, args.device, args.max_samples)
        rows.append((q.name, q.bpw, ppl))
        print(f"  {q.name:24s} ({n} layers)  ppl = {ppl:.4f}  (+{ppl - fp16:.4f})")

    print("\n=== SUMMARY (wikitext-2 ppl; lower is better) ===")
    print(f"{'scheme':26s} {'bpw':>5s} {'ppl':>10s} {'Δ vs fp16':>12s}")
    for name, bpw, ppl in rows:
        print(f"{name:26s} {bpw:5.1f} {ppl:10.4f} {ppl - fp16:12.4f}")


if __name__ == "__main__":
    main()
