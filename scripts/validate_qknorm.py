#!/usr/bin/env python
"""Compile + run the QKNORM WGSL kernel (Qwen3 per-head RMSNorm for q/k) on the
GPU via wgpu-py and compare against a torch reference.

reference (per head): x / sqrt((x*x).mean(-1,keepdim=True)+eps) * w
w is f16, length head_dim, shared across heads.
"""
import os
import re
import struct
import sys

import numpy as np
import torch
import wgpu

ROOT = os.path.join(os.path.dirname(__file__), "..")
KJS = open(os.path.join(ROOT, "web", "kernels.js")).read()

m = re.search(r"export const QKNORM = /\* wgsl \*/`(.*?)`;", KJS, re.S)
if not m:
    raise SystemExit("could not find WGSL QKNORM in web/kernels.js")
WGSL = m.group(1)

dev = wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()

# ---- test config ----
n_heads, head_dim, eps = 32, 128, 1e-6
torch.manual_seed(0)
x = torch.randn(n_heads, head_dim, dtype=torch.float32)
w = torch.randn(head_dim, dtype=torch.float16)

# torch reference (f16 weight upcast to f32, same as kernel's unpack2x16float)
wf = w.to(torch.float32)
ref = (x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * wf).numpy().reshape(-1)

# ---- pack buffers ----
x_np = x.numpy().reshape(-1).astype(np.float32)
wh = w.numpy().view(np.uint16)
if wh.size % 2:
    wh = np.concatenate([wh, np.zeros(1, np.uint16)])
w_u32 = wh.view(np.uint32)

uni = bytearray(16)
struct.pack_into("<I", uni, 0, n_heads)
struct.pack_into("<I", uni, 4, head_dim)
struct.pack_into("<f", uni, 8, eps)

shader = dev.create_shader_module(code=WGSL)  # raises on any WGSL error

xbuf = dev.create_buffer_with_data(
    data=x_np.tobytes(),
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
wbuf = dev.create_buffer_with_data(data=w_u32.tobytes(), usage=wgpu.BufferUsage.STORAGE)
ubuf = dev.create_buffer_with_data(data=bytes(uni), usage=wgpu.BufferUsage.UNIFORM)

bgl = dev.create_bind_group_layout(entries=[
    {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
     "buffer": {"type": wgpu.BufferBindingType.uniform}},
    {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
     "buffer": {"type": wgpu.BufferBindingType.storage}},
    {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
     "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
])
pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
pipe = dev.create_compute_pipeline(layout=pl, compute={"module": shader, "entry_point": "main"})
bg = dev.create_bind_group(layout=bgl, entries=[
    {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}},
    {"binding": 1, "resource": {"buffer": xbuf, "offset": 0, "size": xbuf.size}},
    {"binding": 2, "resource": {"buffer": wbuf, "offset": 0, "size": wbuf.size}},
])
enc = dev.create_command_encoder()
cp = enc.begin_compute_pass()
cp.set_pipeline(pipe)
cp.set_bind_group(0, bg)
cp.dispatch_workgroups(n_heads)  # one workgroup per head
cp.end()
dev.queue.submit([enc.finish()])
got = np.frombuffer(dev.queue.read_buffer(xbuf), dtype=np.float32)[: n_heads * head_dim]

rel_err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
print(f"QKNORM  n_heads={n_heads} head_dim={head_dim} eps={eps}  "
      f"rel_err={rel_err:.2e}  {'OK' if rel_err < 1e-4 else 'FAIL'}")
sys.exit(0 if rel_err < 1e-4 else 1)
