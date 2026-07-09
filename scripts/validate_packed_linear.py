#!/usr/bin/env python
"""Verify the FULL packed linear() path (decode-matmul + IP fold) on GPU, exactly
as web/packed.js chains it, vs true W_hat @ x for a real layer. Uses the WGSL
straight from web/kernels.js.
"""
import os, re, sys, zlib, struct
import numpy as np, torch, wgpu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis.qtip import codebook_lut, viterbi_tb, IPContext, T_SEQ, _fit_block
from transformers import AutoModelForCausalLM

KJS = open(os.path.join(os.path.dirname(__file__), "..", "web", "kernels.js")).read()
def K(n): return re.search(rf"export const {n} = /\* wgsl \*/`(.*?)`;", KJS, re.S).group(1)

name = "model.layers.5.mlp.down_proj.weight"
mo = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32).eval()
w = dict(mo.named_parameters())[name].data.float(); n_out, k_in = w.shape
lut = codebook_lut("mcg"); Kbits = 3
ip = IPContext(w, lut.square().mean().sqrt().item(), seed=zlib.crc32(name.encode()))
z = ip.z; tiles = z.view(n_out//16,16,k_in//16,16).permute(0,2,1,3).reshape(-1,T_SEQ)
bits, dq = viterbi_tb(tiles, Kbits, lut, dp_dtype=torch.float32)
zq = dq.reshape(n_out//16,k_in//16,16,16).permute(0,2,1,3).reshape(n_out,k_in)
W_hat = ip.restore(zq).numpy()
x = np.random.randn(k_in).astype(np.float32)
y_ref = W_hat @ x

# pack bits reversed
nt, T = bits.shape; total = T*Kbits; nwords = (total+31)//32
packed = np.zeros((nt, nwords), np.uint32); b = bits.numpy().astype(np.int64)
for t in range(T):
    for kb in range(Kbits):
        pos = (T-1-t)*Kbits+kb
        packed[:, pos>>5] |= ((b[:,t]>>kb)&1).astype(np.uint32) << (pos&31)

bk = _fit_block(k_in,128); bn = _fit_block(n_out,128)
su = ip.su.numpy().astype(np.float32); sv = ip.sv.numpy().astype(np.float32)
isc = (ip.in_s.numpy()*ip.scale).astype(np.float32); osc = ip.out_s.numpy().astype(np.float32)

dev = wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()
STOR = wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC|wgpu.BufferUsage.COPY_DST
def sb(a): return dev.create_buffer_with_data(data=a.astype(np.float32).tobytes() if a.dtype!=np.uint32 else a.tobytes(), usage=STOR)
def ob(n): return dev.create_buffer(size=n*4, usage=STOR)
def u(vals):
    bb=bytearray(max(16,len(vals)*4))
    for i,v in enumerate(vals): struct.pack_into("<I",bb,i*4,int(v))
    return dev.create_buffer_with_data(data=bytes(bb),usage=wgpu.BufferUsage.UNIFORM)
_pipes={}
def disp(name,binds,groups):
    if name not in _pipes:
        _pipes[name]=dev.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto,compute={"module":dev.create_shader_module(code=K(name)),"entry_point":"main"})
    p=_pipes[name]; bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate(binds)])
    e=dev.create_command_encoder();cp=e.begin_compute_pass();cp.set_pipeline(p);cp.set_bind_group(0,bg);cp.dispatch_workgroups(groups);cp.end();dev.queue.submit([e.finish()])
def rd(b,n): return np.frombuffer(dev.queue.read_buffer(b),np.float32)[:n]

xb=sb(x); suB=sb(su); iscB=sb(isc); svB=sb(sv); oscB=sb(osc); bitsB=sb(packed.reshape(-1))
xf=ob(k_in); tmp=ob(k_in)
disp("MUL",[u([k_in]),xb,suB,xf],(k_in+63)//64)
disp("MUL",[u([k_in]),xf,iscB,tmp],(k_in+63)//64)
disp("BLOCK_HAD",[u([k_in,bk]),tmp,xf],(k_in+63)//64)
t=ob(n_out)
disp("DECODE_MATMUL",[u([n_out,k_in,Kbits,256,nwords,k_in//16]),bitsB,xf,t],(n_out+63)//64)
y1=ob(n_out); disp("BLOCK_HAD",[u([n_out,bn]),t,y1],(n_out+63)//64)
y2=ob(n_out); disp("MUL",[u([n_out]),y1,oscB,y2],(n_out+63)//64)
y=ob(n_out);  disp("MUL",[u([n_out]),y2,svB,y],(n_out+63)//64)
got=rd(y,n_out)
rel=np.abs(got-y_ref).max()/(np.abs(y_ref).max()+1e-6)
print(f"PACKED linear() full path on GPU vs W_hat@x: rel_err={rel:.3e}  {'OK' if rel<1e-3 else 'FAIL'}")
print(f"  {name}  W={n_out}x{k_in}")
sys.exit(0 if rel<1e-3 else 1)
