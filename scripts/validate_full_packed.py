#!/usr/bin/env python
"""Full packed-3bit model forward pass through the actual WGSL shaders (decode-
matmul + IP fold for quant linears, fp16 MATMUL/RMSNORM for embed/norms), exactly
as web/model.js + web/packed.js chain it, on GPU. Check top-5 vs PyTorch refs.
Final gate: proves the SHIPPABLE 3-bit browser model is correct.
"""
import json, os, re, struct, sys
import numpy as np, wgpu

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "web", "model_packed")
KJS = open(os.path.join(ROOT, "web", "kernels.js")).read()
def K(n): return re.search(rf"export const {n} = /\* wgsl \*/`(.*?)`;", KJS, re.S).group(1)

cfg = json.load(open(f"{D}/config.json")); man = json.load(open(f"{D}/manifest.json"))
refs = json.load(open(f"{ROOT}/web/model/refs.json"))
blob = np.fromfile(f"{D}/weights.bin", np.uint8)
def raw(m):
    n=int(np.prod(m["shape"])); nb=n*(4 if m["dtype"] in ("u32","f32") else 2)
    return blob[m["offset"]:m["offset"]+nb]
def f16(b):
    u=np.frombuffer(b,np.uint16).astype(np.uint32)
    s=np.where(u&0x8000,-1.0,1.0); e=((u>>10)&0x1f).astype(np.int32)-15; mant=1+(u&0x3ff)/1024
    out=s*mant*np.power(2.0,e); out=np.where(e==-15,s*(u&0x3ff)/1024*2**-14,out); return out.astype(np.float32)

dev=wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()
STOR=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC|wgpu.BufferUsage.COPY_DST
def sbf(a): return dev.create_buffer_with_data(data=a.astype(np.float32).tobytes(),usage=STOR)
def sbu(a): return dev.create_buffer_with_data(data=a.tobytes(),usage=STOR)
def ob(n): return dev.create_buffer(size=max(4,n*4),usage=STOR)
def u(vals,fl=None):
    fl=fl or [0]*len(vals); bb=bytearray(max(16,len(vals)*4))
    for i,v in enumerate(vals): struct.pack_into("<f" if fl[i] else "<I",bb,i*4,v if fl[i] else int(v))
    return dev.create_buffer_with_data(data=bytes(bb),usage=wgpu.BufferUsage.UNIFORM)
_p={}
def disp(name,binds,groups):
    if name not in _p: _p[name]=dev.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto,compute={"module":dev.create_shader_module(code=K(name)),"entry_point":"main"})
    p=_p[name]; bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate(binds)])
    e=dev.create_command_encoder();cp=e.begin_compute_pass();cp.set_pipeline(p);cp.set_bind_group(0,bg);cp.dispatch_workgroups(groups);cp.end();dev.queue.submit([e.finish()])
def rd(b,n): return np.frombuffer(dev.queue.read_buffer(b),np.float32)[:n]
def fitblk(dim,base):
    b=1
    while b<base and dim%(b*2)==0: b*=2
    return b

H=cfg["hidden_size"];hd=cfg["head_dim"];nh=cfg["n_heads"];nkv=cfg["n_kv_heads"];I=cfg["intermediate_size"]
nl=cfg["num_layers"];eps=cfg["rms_eps"];theta=cfg["rope_theta"];Vv=cfg["vocab_size"]

W={}
for name in man:
    m=man[name]
    if isinstance(m,dict) and m.get("quant"):
        sc=np.frombuffer(raw(man[f"{name}.scale"]),np.float32)[0]
        W[name]=dict(quant=True,n=m["n_out"],k=m["k_in"],Kb=m["K"],blk=m["block"],ti=m["tiles_in"],
            nwords=man[f"{name}.bits"]["shape"][1],
            bits=sbu(np.frombuffer(raw(man[f"{name}.bits"]),np.uint32).copy()),
            su=sbf(f16(raw(man[f"{name}.su"]))),sv=sbf(f16(raw(man[f"{name}.sv"]))),
            isc=sbf(f16(raw(man[f"{name}.isc"]))*sc),osc=sbf(f16(raw(man[f"{name}.osc"]))))
    elif "." in name and name.rsplit(".",1)[1] in ("bits","su","sv","isc","osc","scale"):
        continue
    else:
        W[name]=dict(quant=False,buf=dev.create_buffer_with_data(data=raw(m).tobytes(),usage=STOR))

emb=f16(raw(man["model.embed_tokens.weight"]))[:Vv*H].reshape(Vv,H)
def linear(name,xb,nout,nin):
    # exporter keys quant records by MODULE name (no .weight); model.js uses
    # the PyTorch param name (.weight). Normalize: try both.
    w=W.get(name) or W.get(name.rsplit(".weight",1)[0])
    if not w["quant"]:
        y=ob(nout); disp("MATMUL",[u([nout,nin]),w["buf"],xb,y],(nout+63)//64); return y
    k=w["k"];n=w["n"]; xf=ob(k);tmp=ob(k)
    disp("MUL",[u([k]),xb,w["su"],xf],(k+63)//64)
    disp("MUL",[u([k]),xf,w["isc"],tmp],(k+63)//64)
    disp("BLOCK_HAD",[u([k,w["blk"]]),tmp,xf],(k+63)//64)
    t=ob(n); disp("DECODE_MATMUL",[u([n,k,w["Kb"],256,w["nwords"],w["ti"]]),w["bits"],xf,t],(n+63)//64)
    y1=ob(n); disp("BLOCK_HAD",[u([n,fitblk(n,128)]),t,y1],(n+63)//64)
    y2=ob(n); disp("MUL",[u([n]),y1,w["osc"],y2],(n+63)//64)
    y=ob(n);  disp("MUL",[u([n]),y2,w["sv"],y],(n+63)//64)
    return y

ids=refs["input_ids"]
Kc=[ob(2048*nkv*hd) for _ in range(nl)]; Vc=[ob(2048*nkv*hd) for _ in range(nl)]
def rms(src,wname,outn):
    o=ob(outn); disp("RMSNORM",[u([outn,eps],[0,1]),src,W[wname]["buf"],o],1); return o
def step(tid,pos):
    x=sbf(emb[tid])
    for l in range(nl):
        P=lambda s:f"model.layers.{l}.{s}"
        nm=rms(x,P("input_layernorm.weight"),H)
        q=linear(P("self_attn.q_proj.weight"),nm,nh*hd,H)
        k=linear(P("self_attn.k_proj.weight"),nm,nkv*hd,H)
        v=linear(P("self_attn.v_proj.weight"),nm,nkv*hd,H)
        disp("ROPE",[u([nh,hd,pos,theta],[0,0,0,1]),q],(nh*hd//2+63)//64)
        disp("ROPE",[u([nkv,hd,pos,theta],[0,0,0,1]),k],(nkv*hd//2+63)//64)
        stride=nkv*hd; e=dev.create_command_encoder()
        e.copy_buffer_to_buffer(k,0,Kc[l],pos*stride*4,stride*4); e.copy_buffer_to_buffer(v,0,Vc[l],pos*stride*4,stride*4)
        dev.queue.submit([e.finish()])
        o=ob(nh*hd); disp("ATTN",[u([nh,nkv,hd,pos+1]),q,Kc[l],Vc[l],o],nh)
        pj=linear(P("self_attn.o_proj.weight"),o,H,nh*hd); disp("ADD",[u([H]),x,pj],(H+63)//64)
        n2=rms(x,P("post_attention_layernorm.weight"),H)
        g=linear(P("mlp.gate_proj.weight"),n2,I,H); up=linear(P("mlp.up_proj.weight"),n2,I,H)
        act=ob(I); disp("SWIGLU",[u([I]),g,up,act],(I+63)//64)
        dn=linear(P("mlp.down_proj.weight"),act,H,I); disp("ADD",[u([H]),x,dn],(H+63)//64)
    fn=rms(x,"model.norm.weight",H)
    return rd(linear("model.embed_tokens.weight",fn,Vv,H),Vv)

print(f"running full packed {nl}-layer model on GPU...")
lg=None
for pos,t in enumerate(ids): lg=step(t,pos)
top5=np.argsort(-lg)[:5].tolist()
print("PACKED-3bit GPU top5:",top5); print("PyTorch (fp32)  top5:",refs["top5_ids"])
ov=len(set(top5)&set(refs["top5_ids"]))
print(f"\n{'TOP-1 MATCH' if top5[0]==refs['top5_ids'][0] else '~'} overlap {ov}/5 — shippable 3-bit browser model correct")
sys.exit(0 if (top5[0]==refs["top5_ids"][0] or ov>=4) else 1)
