// Trellis Everywhere — in-browser Llama-style transformer runtime (WebGPU).
// v1: fp16 weights + WGSL ops (matmul, rmsnorm, rope, attention, swiglu).
// Every op is verified against PyTorch golden refs (refs.json) before we trust
// the full forward pass. Same architecture as the 8B, so ops scale up.

export async function initGPU() {
  if (!navigator.gpu) throw new Error("WebGPU not available");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  return { adapter, device };
}

// ---- fp16 helpers (JS has no native f16; store as u16, convert for checks) ----
export function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

// ---- model loader: weights.bin + manifest.json -> {name: {buffer, shape}} ----
export async function loadModel(device, dir) {
  const cfg = await (await fetch(`${dir}/config.json`)).json();
  const manifest = await (await fetch(`${dir}/manifest.json`)).json();
  const blob = new Uint8Array(await (await fetch(`${dir}/weights.bin`)).arrayBuffer());
  const tensors = {};
  for (const [name, meta] of Object.entries(manifest)) {
    const n = meta.shape.reduce((a, b) => a * b, 1);
    const bytes = blob.subarray(meta.offset, meta.offset + n * 2); // f16 = 2 bytes
    // upload as u32-padded storage buffer of f16 pairs; shaders read via unpack2x16float
    const padded = (n + 1) & ~1;
    const buf = device.createBuffer({
      size: padded * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, bytes);
    tensors[name] = { buffer: buf, shape: meta.shape, n };
  }
  return { cfg, tensors };
}

// ---- generic dispatch helper ----
export function makeKernel(device, code, entry = "main") {
  const mod = device.createShaderModule({ code });
  return device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: entry } });
}

export function f32buf(device, dataOrLen, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
  const len = typeof dataOrLen === "number" ? dataOrLen : dataOrLen.length;
  const buf = device.createBuffer({ size: len * 4, usage, mappedAtCreation: typeof dataOrLen !== "number" });
  if (typeof dataOrLen !== "number") { new Float32Array(buf.getMappedRange()).set(dataOrLen); buf.unmap(); }
  return buf;
}

export async function readF32(device, buf, len) {
  const rb = device.createBuffer({ size: len * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, len * 4);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  return new Float32Array(rb.getMappedRange()).slice();
}
