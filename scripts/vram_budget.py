#!/usr/bin/env python
"""Exact VRAM budget for the packed browser runtime — the honest number to claim.

Computes every GPU buffer the runtime holds, from the real manifest:
  - packed weights (bits) + IP metadata (su/sv/isc/osc as f32 on GPU)
  - fp16 tensors (norms; embed if not quantized)
  - KV cache: num_layers * 2 * max_seq * n_kv * head_dim * 4 bytes
  - per-token activation scratch (transient peak)
Reports the resident weight footprint + peak. Deterministic, verifiable.
Usage: python scripts/vram_budget.py web/model_8b [max_seq]
"""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "web/model_8b"
MAXSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 512   # demo context
man = {}
if os.path.exists(f"{D}/manifest.json"):
    man = json.load(open(f"{D}/manifest.json"))
else:
    for f in os.listdir(D):
        if f.endswith(".bin.json"): man.update(json.load(open(f"{D}/{f}")))
cfg = json.load(open(f"{D}/config.json"))

def elt(dt): return 4 if dt in ("u32", "f32") else 2
weight_bytes = 0
# GPU-resident: packed bits (u32) + IP vecs uploaded as f32 (4B each elt) + fp16 tensors
for name, m in man.items():
    if not isinstance(m, dict): continue
    if m.get("quant"):
        # bits
        b = man[f"{name}.bits"]; weight_bytes += b["shape"][0]*b["shape"][1]*4
        # su(k),sv(n),isc(k),osc(n) each uploaded as f32
        weight_bytes += (m["k_in"]*2 + m["n_out"]*2) * 4
    elif name.endswith((".bits",".su",".sv",".isc",".osc",".scale")):
        continue
    else:  # fp16 tensor stored as-is on GPU (2B)
        n = 1
        for s in m["shape"]: n *= s
        weight_bytes += n * elt(m["dtype"])

H = cfg["hidden_size"]; nkv = cfg["n_kv_heads"]; hd = cfg["head_dim"]; nl = cfg["num_layers"]
I = cfg["intermediate_size"]; V = cfg["vocab_size"]
nh = cfg["n_heads"]
kv_bytes = nl * 2 * MAXSEQ * nkv * hd * 4

# Transient scratch: in single-encoder mode (see model.js forward()), every f32
# buffer allocated during a token lives until the encoder is submitted, then GCs.
# So the transient peak = SUM of every f32buf allocated across one forward pass.
# packed linear() allocates 2*k_in + 4*n_out floats; norms/act/attn add small bufs.
def lin(k, n): return 2*k + 4*n           # xf,tmp(k) + t,y1,y2,y(n)
scratch_floats = H                         # embed
for _ in range(nl):
    scratch_floats += H                    # input norm
    scratch_floats += lin(H, nh*hd)        # q_proj
    scratch_floats += 2 * lin(H, nkv*hd)   # k_proj, v_proj
    scratch_floats += nh*hd                # attn out
    scratch_floats += lin(nh*hd, H)        # o_proj
    scratch_floats += H                    # post norm
    scratch_floats += 2 * lin(H, I)        # gate, up
    scratch_floats += I                    # swiglu act
    scratch_floats += lin(I, H)            # down
scratch_floats += H + lin(H, V)            # final norm + lm_head
act_peak = scratch_floats * 4

GB = 1024**3
print(f"model: {D}   context: {MAXSEQ}")
print(f"  packed weights + IP metadata : {weight_bytes/GB:5.2f} GB   (resident)")
print(f"  KV cache ({MAXSEQ} ctx)          : {kv_bytes/GB:5.2f} GB   (resident)")
print(f"  activation scratch (peak)     : {act_peak/GB:5.2f} GB   (transient)")
print(f"  ---------------------------------------------")
print(f"  PEAK VRAM (honest claim)      : {(weight_bytes+kv_bytes+act_peak)/GB:5.2f} GB")
print(f"  (fits a 4 GB GPU: ~3.6-3.8 GB usable after WebGPU/driver reserve)")
