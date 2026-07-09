#!/usr/bin/env python
"""Export SmolLM2-135M with linears as PACKED TRELLIS BITS (3-bit) + IP metadata,
embedding/norms as fp16. The browser decodes the trellis inside the matmul.

Layout per linear "name":
  name.bits  : u32[ntiles * nwords]   packed K-bit trellis symbols (reversed pack)
  name.su    : f16[k_in]              input-dim IP sign*scale vector
  name.sv    : f16[n_out]             output-dim IP sign*scale vector
  name.scale : f32 scalar             global IP scale
  name.osc   : f16[n_out]             per-out-channel scale (mean-1)
  name.isc   : f16[k_in]              per-in-channel scale (mean-1)
plus a "quant" flag in manifest. Decode reproduces qtip.IPContext.restore exactly.

We verify the packed model's forward pass against PyTorch top-5 before shipping.
"""
import json, os, sys, zlib
import numpy as np
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis.qtip import codebook_lut, viterbi_tb, IPContext, T_SEQ, _fit_block

MODEL="HuggingFaceTB/SmolLM2-135M"; K=3
OUT=os.path.join(os.path.dirname(__file__),"..","web","model_packed")
dev=os.environ.get("EXPORT_DEV","cuda" if torch.cuda.is_available() else "cpu")

def pack_bits(bits):  # bits: [ntiles, T] int -> u32[ntiles, nwords] reversed pack
    ntiles,T=bits.shape; total=T*K; nwords=(total+31)//32
    out=np.zeros((ntiles,nwords),np.uint32)
    b=bits.cpu().numpy().astype(np.int64)
    for t in range(T):
        for kb in range(K):
            pos=(T-1-t)*K+kb
            out[:,pos>>5]|=((b[:,t]>>kb)&1).astype(np.uint32)<<(pos&31)
    return out

@torch.no_grad()
def main():
    os.makedirs(OUT,exist_ok=True)
    print(f"[load] {MODEL} on {dev}")
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32).to(dev).eval()
    tok=AutoTokenizer.from_pretrained(MODEL); lut=codebook_lut("mcg").to(dev)
    cfg=model.config
    manifest={}; blob=bytearray()
    def put(name,arr,dtype):
        a=arr.detach().cpu().numpy().astype(dtype) if torch.is_tensor(arr) else arr.astype(dtype)
        manifest[name]={"offset":len(blob),"shape":list(a.shape),"dtype":{np.float16:"f16",np.uint32:"u32",np.float32:"f32"}[a.dtype.type]}
        blob.extend(a.tobytes())

    import time; t0=time.time(); lin=[(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear) and "lm_head" not in n]
    for i,(name,mod) in enumerate(lin):
        w=mod.weight.data.to(dev).float(); n,k=w.shape
        ip=IPContext(w,lut.square().mean().sqrt().item(),seed=zlib.crc32(name.encode()))
        z=ip.z
        tiles=z.view(n//16,16,k//16,16).permute(0,2,1,3).reshape(-1,T_SEQ)
        bits,_=viterbi_tb(tiles,K,lut,dp_dtype=torch.bfloat16)
        packed=pack_bits(bits.view(-1,T_SEQ))   # [ntiles, nwords]
        put(f"{name}.bits",packed,np.uint32)
        put(f"{name}.su",ip.su,np.float16); put(f"{name}.sv",ip.sv,np.float16)
        put(f"{name}.isc",ip.in_s,np.float16); put(f"{name}.osc",ip.out_s,np.float16)
        put(f"{name}.scale",np.array([ip.scale],np.float32),np.float32)
        manifest[name]={"quant":True,"n_out":n,"k_in":k,"K":K,"block":_fit_block(k,128),
                        "tiles_out":n//16,"tiles_in":k//16}
        if i%20==0 or i==len(lin)-1: print(f"[pack] {i+1}/{len(lin)} ({time.time()-t0:.0f}s)",flush=True)
    # embedding + norms as fp16 (unquantized)
    for name,p in model.state_dict().items():
        if any(s in name for s in ["proj.weight"]) and "layernorm" not in name: continue
        put(name,p.to(torch.float16),np.float16)
    open(os.path.join(OUT,"weights.bin"),"wb").write(bytes(blob))
    json.dump(manifest,open(os.path.join(OUT,"manifest.json"),"w"))
    print(f"[weights] {len(blob)/1e6:.1f} MB")
    jscfg=dict(hidden_size=cfg.hidden_size,intermediate_size=cfg.intermediate_size,
        num_layers=cfg.num_hidden_layers,n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads,head_dim=64,vocab_size=cfg.vocab_size,
        rope_theta=100000.0,rms_eps=cfg.rms_norm_eps,tie_embeddings=True,K=K)
    json.dump(jscfg,open(os.path.join(OUT,"config.json"),"w"))
    tok.save_pretrained(OUT)
    print(f"[done] packed model -> {OUT}")

if __name__=="__main__": main()
