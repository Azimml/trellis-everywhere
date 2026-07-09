// Packed 3-bit weight loading + IP-folded decode-matmul.
// linearPacked(x, W) computes W_hat @ x where W_hat = restore(decode(bits)),
// using the GPU decode-matmul (z-space) + folded IP transforms:
//   x-side (k):  xf = Hb_k( in_s * scale * su * x )
//   gpu:         t  = zq @ xf                (DECODE_MATMUL)
//   y-side (n):  y  = sv * out_s * Hb_n( t )
// Verified in Python: rel_err 4.9e-7 vs true W_hat@x.
import { makeKernel, f32buf, readF32 } from "./runtime.js";
import { DECODE_MATMUL, BLOCK_HAD, MUL, DECODE_EMBED, QKNORM } from "./kernels.js";

// Sharded loader: manifest entries may carry a "shard" filename. Shards are
// fetched sequentially, tensors uploaded to GPU, CPU copy freed — peak JS
// memory ~ largest shard (~100MB) instead of the whole model (3+GB for 8B).
export async function loadPackedSharded(device, dir, onProgress = null) {
  const cfg = await (await fetch(`${dir}/config.json`)).json();
  const man = await (await fetch(`${dir}/manifest.json`)).json();
  const T = {};
  // group manifest entries by shard
  const byShard = {};
  for (const [name, m] of Object.entries(man)) {
    const s = m.shard || "weights.bin";
    (byShard[s] = byShard[s] || {})[name] = m;
  }
  const shardNames = Object.keys(byShard).sort();
  let done = 0;
  for (const s of shardNames) {
    const blob = new Uint8Array(await (await fetch(`${dir}/${s}`)).arrayBuffer());
    ingestShard(device, T, byShard[s], blob);
    done++;
    if (onProgress) onProgress(done, shardNames.length, s);
  }
  return { cfg, tensors: T };
}

function ingestShard(device, T, entries, blob) {
  const byteLen = (m) => m.shape.reduce((a, b) => a * b, 1) *
    (m.dtype === "u32" || m.dtype === "f32" ? 4 : 2);
  const raw = (m) => blob.subarray(m.offset, m.offset + byteLen(m));
  const up = (bytes) => {
    const b = device.createBuffer({ size: (bytes.byteLength + 3) & ~3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(b, 0, bytes); return b;
  };
  for (const name of Object.keys(entries)) {
    const m = entries[name];
    if (m.quant) {
      const scRaw = raw(entries[`${name}.scale`] || m); // scale lives in same shard
      const scMeta = entries[`${name}.scale`];
      const sc = new Float32Array(raw(scMeta).buffer.slice(
        raw(scMeta).byteOffset, raw(scMeta).byteOffset + 4))[0];
      const iscF32 = new Float32Array(f16bytesToF32(raw(entries[`${name}.isc`])).buffer);
      for (let i = 0; i < iscF32.length; i++) iscF32[i] *= sc;
      T[name] = {
        quant: true, n_out: m.n_out, k_in: m.k_in, K: m.K, block: m.block,
        tiles_out: m.tiles_out, tiles_in: m.tiles_in,
        bits: up(raw(entries[`${name}.bits`])),
        su: up(f16bytesToF32(raw(entries[`${name}.su`]))),
        sv: up(f16bytesToF32(raw(entries[`${name}.sv`]))),
        isc: up(new Uint8Array(iscF32.buffer)),
        osc: up(f16bytesToF32(raw(entries[`${name}.osc`]))),
        scale: sc, nwords: entries[`${name}.bits`].shape[1],
      };
    } else if (name.match(/\.(bits|su|sv|isc|osc|scale)$/)) {
      continue;
    } else {
      T[name] = { quant: false, buffer: up(raw(m)), shape: m.shape };
    }
  }
}

export async function loadPacked(device, dir) {
  const cfg = await (await fetch(`${dir}/config.json`)).json();
  const man = await (await fetch(`${dir}/manifest.json`)).json();
  const blob = new Uint8Array(await (await fetch(`${dir}/weights.bin`)).arrayBuffer());
  const T = {};
  const raw = (meta) => blob.subarray(meta.offset, meta.offset + byteLen(meta));
  function byteLen(m) {
    const n = m.shape.reduce((a, b) => a * b, 1);
    return n * (m.dtype === "u32" || m.dtype === "f32" ? 4 : 2);
  }
  function up(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    const b = device.createBuffer({ size: (bytes.byteLength + 3) & ~3, usage });
    device.queue.writeBuffer(b, 0, bytes); return b;
  }
  for (const name of Object.keys(man)) {
    const m = man[name];
    if (m.quant) {
      // fold the scalar `scale` into isc at load so the kernel path needs no scalar-mul
      const scRaw = raw(man[`${name}.scale`]);
      const scale = new Float32Array(scRaw.buffer.slice(scRaw.byteOffset, scRaw.byteOffset + 4))[0];
      const iscF32 = new Float32Array(f16bytesToF32(raw(man[`${name}.isc`])).buffer);
      for (let i = 0; i < iscF32.length; i++) iscF32[i] *= scale;   // isc <- isc*scale
      T[name] = {
        quant: true, n_out: m.n_out, k_in: m.k_in, K: m.K, block: m.block,
        tiles_out: m.tiles_out, tiles_in: m.tiles_in,
        bits: up(raw(man[`${name}.bits`])),
        su: up(f16bytesToF32(raw(man[`${name}.su`]))),
        sv: up(f16bytesToF32(raw(man[`${name}.sv`]))),
        isc: up(new Uint8Array(iscF32.buffer)),   // already ×scale
        osc: up(f16bytesToF32(raw(man[`${name}.osc`]))),
        scale,
        nwords: man[`${name}.bits`].shape[1],
      };
    } else if (!name.includes(".")) {
      continue;  // sub-tensors handled above
    } else if (name.endsWith(".bits") || name.endsWith(".su") || name.endsWith(".sv")
               || name.endsWith(".isc") || name.endsWith(".osc") || name.endsWith(".scale")) {
      continue;
    } else {
      T[name] = { quant: false, buffer: up(raw(m)), shape: m.shape };
    }
  }
  return { cfg, tensors: T };
}

// f16 stored bytes -> f32 array bytes (JS lacks native f16 in storage buffers as f32)
function f16bytesToF32(bytes) {
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16(u16[i]);
  return new Uint8Array(out.buffer);
}
function f16(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

export class PackedOps {
  constructor(device) {
    this.d = device;
    this.k = { dm: makeKernel(device, DECODE_MATMUL), had: makeKernel(device, BLOCK_HAD),
               mul: makeKernel(device, MUL), emb: makeKernel(device, DECODE_EMBED),
               qknorm: makeKernel(device, QKNORM) };
  }
  // decode embedding row `tokenId` from a quantized embed weight W -> x[H]
  embedRow(enc, W, tokenId) {
    const d = this.d, H = W.k_in;
    const blkH = this._blk(H, 128), blkV = this._blk(W.n_out, 128);
    const x = f32buf(d, H);
    this._dispEmb(enc, this.k.emb,
      [this._u([tokenId, H, W.n_out, W.K, 256, W.nwords, W.tiles_in, blkH, blkV]),
       W.bits, W.su, W.isc, W.osc, W.sv, x], H / blkH);
    return x;
  }
  _dispEmb(enc, pipe, binds, ngroups) {
    const bg = this.d.createBindGroup({ layout: pipe.getBindGroupLayout(0),
      entries: binds.map((buffer, i) => ({ binding: i, resource: { buffer } })) });
    const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg);
    p.dispatchWorkgroups(ngroups); p.end();
  }
  _u(vals) { const b = new Uint32Array(vals.length < 4 ? 4 : vals.length); b.set(vals);
    const buf = this.d.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.d.queue.writeBuffer(buf, 0, b); return buf; }
  _disp(enc, pipe, binds, nthreads) {
    const bg = this.d.createBindGroup({ layout: pipe.getBindGroupLayout(0),
      entries: binds.map((buffer, i) => ({ binding: i, resource: { buffer } })) });
    const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg);
    p.dispatchWorkgroups(Math.ceil(nthreads / 64)); p.end();
  }
  // y = W_hat @ x  (x: f32 buffer len k_in) -> f32 buffer len n_out
  linear(enc, W, xBuf) {
    const d = this.d, k = W.k_in, n = W.n_out;
    const xf = f32buf(d, k), tmp = f32buf(d, k);
    // x-side fold: xf = Hb_k( (isc*scale) * su * x )   [isc already ×scale at load]
    this._disp(enc, this.k.mul, [this._u([k]), xBuf, W.su, xf], k);          // xf = x*su
    this._disp(enc, this.k.mul, [this._u([k]), xf, W.isc, tmp], k);           // tmp = xf*isc(*scale)
    this._disp(enc, this.k.had, [this._u([k, W.block]), tmp, xf], k);          // xf = Hb_k(tmp)
    const t = f32buf(d, n);
    this._disp(enc, this.k.dm, [this._u([n, k, W.K, 256, W.nwords, W.tiles_in]), W.bits, xf, t], n);
    const y1 = f32buf(d, n);
    // out-dim Hadamard block = fit(n_out, 128) — NOT capped by the in-dim's
    // W.block. (Bug: capping at W.block scrambled gate/up outputs when
    // n_out's block (128) exceeded k_in's (64) — the browser-garbage cause.)
    this._disp(enc, this.k.had, [this._u([n, this._blk(n, 128)]), t, y1], n); // Hb_n(t)
    const y2 = f32buf(d, n);
    this._disp(enc, this.k.mul, [this._u([n]), y1, W.osc, y2], n);             // *out_s
    const y = f32buf(d, n);
    this._disp(enc, this.k.mul, [this._u([n]), y2, W.sv, y], n);               // *sv
    return { y, scale: W.scale };
  }
  _blk(dim, base) { let b = 1; while (b < base && dim % (b * 2) === 0) b *= 2; return b; }
}
