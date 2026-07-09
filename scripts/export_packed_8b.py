#!/usr/bin/env python
"""Export Qwen3-8B as a browser-ready packed 3-bit model, sharded.

- Linears: K=3 trellis. MLP linears use LDLQ (MLP-only Hessians, 64 seqs) —
  the exact recipe that gave wikitext2 ppl 10.34 (1.15x fp16). Attention plain.
- Embedding: K=3 trellis (Qwen3-8B does NOT tie embeddings; fp16 embed+head
  would be 2.5GB — must be quantized to fit a 4GB-VRAM browser GPU).
- lm_head: K=4 trellis (logit quality).
- Output: web/model_8b/  config.json, manifest.json, shards (layer_NN.bin,
  emb.bin, head.bin, norms.bin), refs.json (top-5 of the QUANTIZED torch model
  = the browser's exact ground truth, plus original-model top5 for context).
- Resumable: existing shard+fragment files are skipped on relaunch.
- All Viterbi encoding through the fused Triton kernel.
"""
import gc, json, os, sys, time, zlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis.qtip import codebook_lut, IPContext, T_SEQ, _fit_block
from trellis.ldlq import _transform_H, block_ldl, regularize_H, ldlq
from trellis.viterbi_triton import viterbi_tb_triton
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("EXPORT_MODEL", "Qwen/Qwen3-8B")
K_LIN, K_EMB, K_HEAD = 3, 3, 4
NSEQ_H = 64
OUT = os.path.join(os.path.dirname(__file__), "..",
                   "web", os.environ.get("EXPORT_OUT", "model_8b"))
dev = "cuda"

# ---------------- RAM guard (shared box) ----------------
def start_ram_guard(limit_gb=105.0):
    import threading
    def loop():
        while True:
            try:
                info = {}
                for ln in open("/proc/meminfo"):
                    k, v = ln.split(":"); info[k] = int(v.strip().split()[0])
                used = (info["MemTotal"] - info["MemAvailable"]) / 1048576
                if used > limit_gb:
                    print(f"[RAM-GUARD] {used:.0f}G > {limit_gb}G — aborting", flush=True)
                    os._exit(137)
            except Exception:
                pass
            time.sleep(2)
    threading.Thread(target=loop, daemon=True).start()

# ---------------- bit packing (reversed, browser layout) ----------------
def pack_bits_np(bits, K):
    """bits: [ntiles, 256] int numpy -> u32 [ntiles, nwords] reversed pack."""
    nt, T = bits.shape
    total = T * K; nwords = (total + 31) // 32
    out = np.zeros((nt, nwords), np.uint32)
    b = bits.astype(np.int64)
    for t in range(T):
        for kb in range(K):
            pos = (T - 1 - t) * K + kb
            out[:, pos >> 5] |= ((b[:, t] >> kb) & 1).astype(np.uint32) << (pos & 31)
    return out

# ---------------- shard writer ----------------
class Shard:
    def __init__(self, path):
        self.path = path; self.blob = bytearray(); self.frag = {}
    def put(self, name, arr, dtype):
        a = arr.astype(dtype) if isinstance(arr, np.ndarray) else arr.detach().cpu().numpy().astype(dtype)
        self.frag[name] = {"offset": len(self.blob), "shape": list(a.shape),
                           "dtype": {np.dtype(np.float16): "f16", np.dtype(np.uint32): "u32",
                                     np.dtype(np.float32): "f32"}[a.dtype],
                           "shard": os.path.basename(self.path)}
        self.blob.extend(a.tobytes())
    def flush(self):
        with open(self.path, "wb") as f: f.write(bytes(self.blob))
        json.dump(self.frag, open(self.path + ".json", "w"))
        print(f"  [shard] {os.path.basename(self.path)} {len(self.blob)/1e6:.1f}MB "
              f"({len(self.frag)} tensors)", flush=True)

def shard_done(path): return os.path.exists(path) and os.path.exists(path + ".json")

# ---------------- quantizers (bits-capturing) ----------------
lut = codebook_lut("mcg").to(dev)
CB_RMS = lut.square().mean().sqrt().item()

@torch.no_grad()
def quant_plain_bits(w, K, tile_batch=16384):
    """Plain trellis (no Hessian). Returns (bits u32 np [nt,nwords], ip, W_hat)."""
    n, k = w.shape
    ip = IPContext(w.float(), CB_RMS, seed=zlib.crc32(b"x"))
    z = ip.z
    tiles = z.view(n // 16, 16, k // 16, 16).permute(0, 2, 1, 3).reshape(-1, T_SEQ)
    bits_all = torch.empty(tiles.shape[0], T_SEQ, dtype=torch.int16)
    dq_all = torch.empty_like(tiles)
    for i in range(0, tiles.shape[0], tile_batch):
        b_, d_ = viterbi_tb_triton(tiles[i:i + tile_batch].contiguous(), K, lut)
        bits_all[i:i + tile_batch] = b_.short().cpu(); dq_all[i:i + tile_batch] = d_
    zq = dq_all.view(n // 16, k // 16, 16, 16).permute(0, 2, 1, 3).reshape(n, k)
    W_hat = ip.restore(zq.to(dev))
    packed = pack_bits_np(bits_all.numpy(), K)
    return packed, ip, W_hat

@torch.no_grad()
def quant_ldlq_bits(w, Hm, K):
    """LDLQ trellis with bits capture. Returns (bits u32 np, ip, W_hat)."""
    n, k = w.shape
    ip = IPContext(w.float(), CB_RMS, seed=zlib.crc32(b"x"))
    z = ip.z
    Ht = _transform_H(Hm.float().to(dev), ip.su, block=_fit_block(k, 128))
    L = block_ldl(regularize_H(Ht, 0.025), b=16)
    if L is None: L = torch.zeros(k, k, device=dev)
    tiles_in = k // 16; tiles_out = n // 16
    bits_store = torch.empty(tiles_out, tiles_in, T_SEQ, dtype=torch.int16)
    kidx = [tiles_in]  # ldlq walks k-blocks strictly descending; one enc call each

    def enc(rows):
        kidx[0] -= 1
        n_out = rows.shape[1]
        rt = rows.t().contiguous()
        t = rt.reshape(n_out // 16, 16, 1, 16).permute(0, 2, 1, 3).reshape(-1, T_SEQ)
        b_, dq = viterbi_tb_triton(t, K, lut)
        bits_store[:, kidx[0]] = b_.short().cpu()
        dqt = dq.reshape(n_out // 16, 1, 16, 16).permute(0, 2, 1, 3).reshape(n_out, 16)
        return dqt.t().contiguous()

    zt_q = ldlq(z.t().contiguous(), L, enc)
    assert kidx[0] == 0, f"enc call count mismatch: {kidx[0]}"
    zq = zt_q.t().contiguous()
    W_hat = ip.restore(zq)
    packed = pack_bits_np(bits_store.view(-1, T_SEQ).numpy(), K)
    return packed, ip, W_hat

def put_quant(sh, name, packed, ip, n, k, K):
    sh.put(f"{name}.bits", packed, np.uint32)
    sh.put(f"{name}.su", ip.su.cpu().numpy(), np.float16)
    sh.put(f"{name}.sv", ip.sv.cpu().numpy(), np.float16)
    sh.put(f"{name}.isc", ip.in_s.cpu().numpy(), np.float16)
    sh.put(f"{name}.osc", ip.out_s.cpu().numpy(), np.float16)
    sh.put(f"{name}.scale", np.array([ip.scale], np.float32), np.float32)
    sh.frag[name] = {"quant": True, "n_out": n, "k_in": k, "K": K,
                     "block": _fit_block(k, 128), "tiles_out": n // 16, "tiles_in": k // 16,
                     "shard": os.path.basename(sh.path)}

# ---------------- MLP Hessian collection (streamed, CPU fp32) ----------------
@torch.no_grad()
def collect_mlp_hessians(model, tok, nseq):
    from trellis.eval import get_wikitext2_test
    ids = get_wikitext2_test(tok)
    acc, handles = {}, []
    only = ("down_proj", "gate_proj", "up_proj")
    lin = [(n, m) for n, m in model.named_modules()
           if isinstance(m, nn.Linear) and any(o in n for o in only)]
    for name, mod in lin:
        acc[name] = [torch.zeros(mod.weight.shape[1], mod.weight.shape[1], device=dev), 0]
        def mk(nm):
            def hook(m, inp):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                acc[nm][0] += x.T @ x; acc[nm][1] += x.shape[0]
            return hook
        handles.append(mod.register_forward_pre_hook(mk(name)))
    for i in range(nseq):
        b = ids[:, i * 2048:(i + 1) * 2048].to(dev)
        if b.shape[1] < 2048: break
        model(b)
    for h in handles: h.remove()
    out = {n: (a / max(c, 1)).float().cpu() for n, (a, c) in acc.items()}
    acc.clear(); torch.cuda.empty_cache()
    return out

# ---------------- main ----------------
@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    start_ram_guard(105.0)
    t00 = time.time()
    print(f"[load] {MODEL} bf16 on {dev}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = model.config

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    orig_top5 = torch.topk(model(ids).logits[0, -1].float(), 5).indices.tolist()
    print(f"[refs] original top5: {orig_top5}", flush=True)

    print(f"[hessians] MLP-only, {NSEQ_H} seqs...", flush=True)
    tH = time.time()
    hess = collect_mlp_hessians(model, tok, NSEQ_H)
    print(f"[hessians] {len(hess)} in {time.time()-tH:.0f}s", flush=True)

    nl = cfg.num_hidden_layers
    for l in range(nl):
        spath = os.path.join(OUT, f"layer_{l:02d}.bin")
        if shard_done(spath):
            print(f"[skip] layer {l} (resume)", flush=True); continue
        sh = Shard(spath); t0 = time.time()
        for sub in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            name = f"model.layers.{l}.{sub}"
            mod = dict(model.named_modules())[name]
            w = mod.weight.data.float()
            n, k = w.shape
            hname = f"{name}"
            if "mlp" in sub and hname in hess:
                packed, ip, W_hat = quant_ldlq_bits(w, hess.pop(hname), K_LIN)
            else:
                packed, ip, W_hat = quant_plain_bits(w, K_LIN)
            put_quant(sh, f"{name}.weight", packed, ip, n, k, K_LIN)
            mod.weight.data.copy_(W_hat.to(mod.weight.dtype))  # in-place -> quant refs later
            del w, W_hat, packed, ip; gc.collect(); torch.cuda.empty_cache()
        # per-layer norms (incl Qwen3 q_norm/k_norm) fp16
        for nsub in ("input_layernorm", "post_attention_layernorm",
                     "self_attn.q_norm", "self_attn.k_norm"):
            key = f"model.layers.{l}.{nsub}.weight"
            sd = model.state_dict()
            if key in sd: sh.put(key, sd[key].float().cpu().numpy(), np.float16)
        sh.flush()
        print(f"[layer {l+1}/{nl}] {time.time()-t0:.0f}s  (total {time.time()-t00:.0f}s)", flush=True)

    # ---- embedding (K=3) ----
    epath = os.path.join(OUT, "emb.bin")
    if not shard_done(epath):
        print("[embed] quantizing (K=3)...", flush=True)
        t0 = time.time()
        emb = model.get_input_embeddings().weight.data.float()
        packed, ip, W_hat = quant_plain_bits(emb, K_EMB)
        sh = Shard(epath)
        put_quant(sh, "model.embed_tokens.weight", packed, ip, emb.shape[0], emb.shape[1], K_EMB)
        model.get_input_embeddings().weight.data.copy_(W_hat.to(model.dtype))
        sh.flush(); del emb, W_hat, packed, ip; gc.collect(); torch.cuda.empty_cache()
        print(f"[embed] {time.time()-t0:.0f}s", flush=True)
    else:
        print("[skip] embed (resume)", flush=True)

    # ---- lm_head (K=4) ----
    hpath = os.path.join(OUT, "head.bin")
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    if tied:
        print("[head] tied to embedding — runtime reuses embed weight (no separate head)", flush=True)
    elif not shard_done(hpath):
        print("[head] quantizing (K=4)...", flush=True)
        t0 = time.time()
        head = model.get_output_embeddings().weight.data.float()
        packed, ip, W_hat = quant_plain_bits(head, K_HEAD)
        sh = Shard(hpath)
        put_quant(sh, "lm_head.weight", packed, ip, head.shape[0], head.shape[1], K_HEAD)
        model.get_output_embeddings().weight.data.copy_(W_hat.to(model.dtype))
        sh.flush(); del head, W_hat, packed, ip; gc.collect(); torch.cuda.empty_cache()
        print(f"[head] {time.time()-t0:.0f}s", flush=True)
    else:
        print("[skip] head (resume)", flush=True)

    # ---- final norm ----
    npath = os.path.join(OUT, "norms.bin")
    if not shard_done(npath):
        sh = Shard(npath)
        sh.put("model.norm.weight", model.model.norm.weight.float().cpu().numpy(), np.float16)
        sh.flush()

    # ---- refs on the QUANTIZED torch model (browser ground truth) ----
    out = model(ids)
    top = torch.topk(out.logits[0, -1].float(), 5)
    refs = dict(prompt=prompt, input_ids=ids[0].tolist(),
                top5_ids=top.indices.tolist(),
                top5_toks=[tok.decode([i]) for i in top.indices.tolist()],
                orig_top5=orig_top5)
    json.dump(refs, open(os.path.join(OUT, "refs.json"), "w"), indent=1)
    print(f"[refs] quantized-model top5: {refs['top5_ids']} {refs['top5_toks']}", flush=True)

    # ---- merge manifest + config + tokenizer ----
    man = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".bin.json"):
            man.update(json.load(open(os.path.join(OUT, f))))
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"))
    rope = getattr(cfg, "rope_theta", None) or cfg.rope_parameters.get("rope_theta", 1e6)
    jscfg = dict(hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
                 num_layers=nl, n_heads=cfg.num_attention_heads,
                 n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                 vocab_size=cfg.vocab_size, rope_theta=float(rope),
                 rms_eps=cfg.rms_norm_eps, tie_embeddings=False, qk_norm=True, K=K_LIN)
    json.dump(jscfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    tok.save_pretrained(OUT)
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".bin"))
    print(f"[done] {total/1e9:.2f} GB in shards -> {OUT}  ({(time.time()-t00)/60:.0f} min)", flush=True)

if __name__ == "__main__":
    main()
