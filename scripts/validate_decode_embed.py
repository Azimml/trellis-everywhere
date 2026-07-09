#!/usr/bin/env python
"""Verify DECODE_EMBED on GPU: embedding-row lookup from packed 3-bit trellis
weights with the FULL IP restore fused in the kernel, vs ip.restore(zq)[r,:].
Builds a small synthetic 'embedding' (vocab x H) with IPContext + viterbi_tb
(fp32 dp) exactly like validate_packed_linear.py builds its layer.
Host prefolds isc = in_s * scale (scale folded into the in-scale buffer).
"""
import os, re, sys, struct
import numpy as np, torch, wgpu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trellis.qtip import codebook_lut, viterbi_tb, IPContext, T_SEQ, _fit_block

KJS = open(os.path.join(os.path.dirname(__file__), "..", "web", "kernels.js")).read()
def K(n): return re.search(rf"export const {n} = /\* wgsl \*/`(.*?)`;", KJS, re.S).group(1)

torch.manual_seed(0); np.random.seed(0)
vocab, H, Kbits = 256, 256, 3
w = torch.randn(vocab, H)
lut = codebook_lut("mcg")
ip = IPContext(w, lut.square().mean().sqrt().item(), seed=1234)
tiles = ip.z.view(vocab//16, 16, H//16, 16).permute(0, 2, 1, 3).reshape(-1, T_SEQ)
bits, dq = viterbi_tb(tiles, Kbits, lut, dp_dtype=torch.float32)
zq = dq.reshape(vocab//16, H//16, 16, 16).permute(0, 2, 1, 3).reshape(vocab, H)
W_hat = ip.restore(zq).numpy()

# pack bits reversed (tail-biting window layout, same as validate_packed_linear.py)
nt, T = bits.shape; total = T*Kbits; nwords = (total+31)//32
packed = np.zeros((nt, nwords), np.uint32); b = bits.numpy().astype(np.int64)
for t in range(T):
    for kb in range(Kbits):
        pos = (T-1-t)*Kbits+kb
        packed[:, pos>>5] |= ((b[:, t]>>kb)&1).astype(np.uint32) << (pos&31)

blkH = _fit_block(H, 128); blkV = _fit_block(vocab, 128)
su = ip.su.numpy().astype(np.float32); sv = ip.sv.numpy().astype(np.float32)
isc = (ip.in_s.numpy()*ip.scale).astype(np.float32)   # in_s * scale prefolded
osc = ip.out_s.numpy().astype(np.float32)

dev = wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()
STOR = wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC|wgpu.BufferUsage.COPY_DST
def sb(a): return dev.create_buffer_with_data(data=a.tobytes(), usage=STOR)
def u(vals):
    bb = bytearray(max(16, ((len(vals)*4+15)//16)*16))
    for i, v in enumerate(vals): struct.pack_into("<I", bb, i*4, int(v))
    return dev.create_buffer_with_data(data=bytes(bb), usage=wgpu.BufferUsage.UNIFORM)

pipe = dev.create_compute_pipeline(layout=wgpu.AutoLayoutMode.auto,
    compute={"module": dev.create_shader_module(code=K("DECODE_EMBED")), "entry_point": "main"})
packedB = sb(packed.reshape(-1)); suB = sb(su); iscB = sb(isc); oscB = sb(osc); svB = sb(sv)
xB = dev.create_buffer(size=H*4, usage=STOR)

def lookup(r):
    ub = u([r, H, vocab, Kbits, 256, nwords, H//16, blkH, blkV])
    binds = [ub, packedB, suB, iscB, oscB, svB, xB]
    bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0),
        entries=[{"binding": i, "resource": {"buffer": bf, "offset": 0, "size": bf.size}}
                 for i, bf in enumerate(binds)])
    e = dev.create_command_encoder(); cp = e.begin_compute_pass()
    cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(H//blkH); cp.end(); dev.queue.submit([e.finish()])
    return np.frombuffer(dev.queue.read_buffer(xB), np.float32)[:H].copy()

ok = True
for r in [0, 1, 7, 63, 127, 128, 129, 200, 255]:
    got = lookup(r); ref = W_hat[r, :]
    rel = np.abs(got-ref).max()/(np.abs(ref).max()+1e-6)
    print(f"row {r:4d}: rel_err={rel:.3e}  {'OK' if rel < 1e-3 else 'FAIL'}")
    ok &= rel < 1e-3
print(f"DECODE_EMBED vs ip.restore(zq)[r,:] ({vocab}x{H}, K={Kbits}, blkH={blkH}, blkV={blkV}):",
      "ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
