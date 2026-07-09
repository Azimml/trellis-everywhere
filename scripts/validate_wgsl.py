#!/usr/bin/env python
"""Actually COMPILE + RUN each WGSL kernel on a real GPU via wgpu-py, and check
outputs against numpy references. Catches any shader error before the browser.

Extracts the WGSL source strings straight from web/kernels.js so we validate the
exact shaders the demo ships.
"""
import os
import re
import sys

import numpy as np
import wgpu

ROOT = os.path.join(os.path.dirname(__file__), "..")
KJS = open(os.path.join(ROOT, "web", "kernels.js")).read()


def extract(name):
    # export const NAME = /* wgsl */`...`;
    m = re.search(rf"export const {name} = /\* wgsl \*/`(.*?)`;", KJS, re.S)
    if not m:
        raise SystemExit(f"could not find WGSL {name}")
    return m.group(1)


dev = wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()


def run(wgsl, uniforms, storages, out_idx, out_len, groups, rw=()):
    """uniforms: bytes; storages: list of (np.float32 array or None-for-output).
    rw: set of 1-based binding indices that are read_write in the shader (must
    get a writable storage binding even though we seed them with data).
    Returns the out_idx storage buffer contents as float32[out_len]."""
    shader = dev.create_shader_module(code=wgsl)  # <-- compiles WGSL; raises on error
    bufs = []
    entries = []
    layout_entries = []
    # binding 0 = uniform
    ubuf = dev.create_buffer_with_data(data=uniforms, usage=wgpu.BufferUsage.UNIFORM)
    entries.append({"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}})
    layout_entries.append({"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                           "buffer": {"type": wgpu.BufferBindingType.uniform}})
    for i, arr in enumerate(storages):
        b = i + 1
        if arr is None:
            buf = dev.create_buffer(size=out_len * 4,
                                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        else:
            buf = dev.create_buffer_with_data(data=arr.astype(np.float32).tobytes(),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        bufs.append(buf)
        entries.append({"binding": b, "resource": {"buffer": buf, "offset": 0, "size": buf.size}})
        ro = (arr is not None) and (b not in rw)
        layout_entries.append({"binding": b, "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": wgpu.BufferBindingType.read_only_storage if ro
                       else wgpu.BufferBindingType.storage}})
    bgl = dev.create_bind_group_layout(entries=layout_entries)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = dev.create_compute_pipeline(layout=pl, compute={"module": shader, "entry_point": "main"})
    bg = dev.create_bind_group(layout=bgl, entries=entries)
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(groups); cp.end()
    dev.queue.submit([enc.finish()])
    out = dev.queue.read_buffer(bufs[out_idx - 1])
    return np.frombuffer(out, dtype=np.float32)[:out_len]


def u(vals, floats):
    import struct
    b = bytearray(max(16, len(vals) * 4))
    for i, (v, f) in enumerate(zip(vals, floats)):
        struct.pack_into("<f" if f else "<I", b, i * 4, v if f else int(v))
    return bytes(b)


def f16pack(arr):  # emulate storage of f16 pairs as u32 -> but our test uses f32 x; for
    return arr      # weight-free kernels (swiglu/add/rope) we pass f32 directly


ok = True
np.random.seed(0)

# SWIGLU
n = 256
g = np.random.randn(n).astype(np.float32); up = np.random.randn(n).astype(np.float32)
ref = (g / (1 + np.exp(-g))) * up
got = run(extract("SWIGLU"), u([n], [0]), [g, up, None], 3, n, (n + 63) // 64)
e = np.abs(got - ref).max()
print(f"SWIGLU   max_err={e:.2e}  {'OK' if e < 1e-4 else 'FAIL'}"); ok &= e < 1e-4

# ADD
a = np.random.randn(n).astype(np.float32); b = np.random.randn(n).astype(np.float32)
got = run(extract("ADD"), u([n], [0]), [a.copy(), b], 1, n, (n + 63) // 64, rw={1})
e = np.abs(got - (a + b)).max()
print(f"ADD      max_err={e:.2e}  {'OK' if e < 1e-5 else 'FAIL'}"); ok &= e < 1e-5

# ROPE (non-interleaved), 2 heads, hd=8, pos=3, theta=10000
nh, hd, pos, theta = 2, 8, 3, 10000.0
x = np.random.randn(nh * hd).astype(np.float32)
xr = x.reshape(nh, hd).copy(); h2 = hd // 2
for i in range(h2):
    fr = 1.0 / theta ** (2 * i / hd); ang = pos * fr; c, s = np.cos(ang), np.sin(ang)
    x1 = xr[:, i].copy(); x2 = xr[:, i + h2].copy()
    xr[:, i] = x1 * c - x2 * s; xr[:, i + h2] = x2 * c + x1 * s
ref = xr.reshape(-1)
got = run(extract("ROPE"), u([nh, hd, pos, theta], [0, 0, 0, 1]), [x.copy()], 1, nh * hd, (nh * h2 + 63) // 64, rw={1})
e = np.abs(got - ref).max()
print(f"ROPE     max_err={e:.2e}  {'OK' if e < 1e-4 else 'FAIL'}"); ok &= e < 1e-4

print("\nWGSL kernels compiled + ran on",
      "GPU. ALL PASS" if ok else "GPU. SOME FAILED")

# ---- kernels that read f16-packed weights (MATMUL, RMSNORM) + ATTN ----
def f16_to_u32pairs(arr_f32):
    """pack fp32 array as fp16, two per u32, as the shaders expect."""
    h = arr_f32.astype(np.float16).view(np.uint16)
    if h.size % 2: h = np.concatenate([h, np.zeros(1, np.uint16)])
    return h.view(np.uint32)

def run_raw(wgsl, uniforms, raw_storages, out_idx, out_len, groups, rw=()):
    """raw_storages: list of (np.uint32/np.float32 bytes-ready array or None)."""
    shader = dev.create_shader_module(code=wgsl)
    bufs=[]; entries=[]; le=[]
    ubuf=dev.create_buffer_with_data(data=uniforms,usage=wgpu.BufferUsage.UNIFORM)
    entries.append({"binding":0,"resource":{"buffer":ubuf,"offset":0,"size":ubuf.size}})
    le.append({"binding":0,"visibility":wgpu.ShaderStage.COMPUTE,"buffer":{"type":wgpu.BufferBindingType.uniform}})
    for i,arr in enumerate(raw_storages):
        b=i+1
        if arr is None:
            buf=dev.create_buffer(size=out_len*4,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
        else:
            buf=dev.create_buffer_with_data(data=arr.tobytes(),
                usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC|wgpu.BufferUsage.COPY_DST)
        bufs.append(buf)
        entries.append({"binding":b,"resource":{"buffer":buf,"offset":0,"size":buf.size}})
        ro=(arr is not None) and (b not in rw)
        le.append({"binding":b,"visibility":wgpu.ShaderStage.COMPUTE,
            "buffer":{"type":wgpu.BufferBindingType.read_only_storage if ro else wgpu.BufferBindingType.storage}})
    bgl=dev.create_bind_group_layout(entries=le); pl=dev.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe=dev.create_compute_pipeline(layout=pl,compute={"module":shader,"entry_point":"main"})
    bg=dev.create_bind_group(layout=bgl,entries=entries)
    enc=dev.create_command_encoder(); cp=enc.begin_compute_pass()
    cp.set_pipeline(pipe); cp.set_bind_group(0,bg); cp.dispatch_workgroups(groups); cp.end()
    dev.queue.submit([enc.finish()])
    return np.frombuffer(dev.queue.read_buffer(bufs[out_idx-1]),dtype=np.float32)[:out_len]

ok2=True
# MATMUL: y = W(f16)[nout,nin] @ x(f32)
nout,nin=48,64
W=np.random.randn(nout,nin).astype(np.float32)*0.1
x=np.random.randn(nin).astype(np.float32)
got=run_raw(extract("MATMUL"),u([nout,nin],[0,0]),
            [f16_to_u32pairs(W.reshape(-1)), x.astype(np.float32), None],3,nout,(nout+63)//64)
ref=(W.astype(np.float16).astype(np.float32))@x
e=np.abs(got-ref).max()/ (np.abs(ref).max()+1e-6)
print(f"MATMUL   rel_err={e:.2e}  {'OK' if e<1e-2 else 'FAIL'}"); ok2&=e<1e-2

# RMSNORM: binding 2 = f16 weight
H=64; xr=np.random.randn(H).astype(np.float32); wn=np.random.randn(H).astype(np.float32)
import struct
uni=bytearray(16); struct.pack_into("<I",uni,0,H); struct.pack_into("<f",uni,4,1e-5)
got=run_raw(extract("RMSNORM"),bytes(uni),
            [xr.astype(np.float32), f16_to_u32pairs(wn), None],3,H,1)
ref=xr/np.sqrt((xr*xr).mean()+1e-5)*wn.astype(np.float16).astype(np.float32)
e=np.abs(got-ref).max()/(np.abs(ref).max()+1e-6)
print(f"RMSNORM  rel_err={e:.2e}  {'OK' if e<1e-2 else 'FAIL'}"); ok2&=e<1e-2

print("f16-weight kernels:", "ALL PASS" if ok2 else "SOME FAILED")
sys.exit(0 if ok2 else 1)
