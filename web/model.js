// Full Llama-style forward pass in WebGPU, chaining the verified kernels.
// Verified by matching PyTorch top-5 next-token logits
// (scripts/run_packed_headless.py runs these exact kernels via wgpu-py).
import { makeKernel, f32buf, readF32 } from "./runtime.js";
import { MATMUL, RMSNORM, ROPE, ATTN, SWIGLU, ADD } from "./kernels.js";
import { PackedOps } from "./packed.js";

export class TrellisModel {
  constructor(device, cfg, tensors, { packed = false, maxSeq = 512 } = {}) {
    this.device = device; this.cfg = cfg; this.T = tensors;
    this.packedOps = packed ? new PackedOps(device) : null;
    this.maxSeq = maxSeq;
    this.k = {
      matmul: makeKernel(device, MATMUL), rmsnorm: makeKernel(device, RMSNORM),
      rope: makeKernel(device, ROPE), attn: makeKernel(device, ATTN),
      swiglu: makeKernel(device, SWIGLU), add: makeKernel(device, ADD),
    };
    // KV cache: [max_seq, n_kv*head_dim] for K and V, per layer. maxSeq is bounded
    // to keep VRAM low — 512 tokens is plenty for short prompts and fits a 4 GB GPU.
    const c = cfg;
    this.kv = [];
    for (let l = 0; l < c.num_layers; l++) {
      this.kv.push({
        K: f32buf(device, maxSeq * c.n_kv_heads * c.head_dim),
        V: f32buf(device, maxSeq * c.n_kv_heads * c.head_dim),
      });
    }
    this.pos = 0;
  }

  _uni(vals, floatMask = []) {
    const dv = new DataView(new ArrayBuffer(Math.max(16, vals.length * 4)));
    vals.forEach((v, i) => floatMask[i] ? dv.setFloat32(i * 4, v, true) : dv.setUint32(i * 4, v >>> 0, true));
    const b = this.device.createBuffer({ size: dv.buffer.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, dv.buffer); return b;
  }

  _dispatch(enc, pipe, binds, nthreads) {
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
      entries: binds.map((buffer, i) => ({ binding: i, resource: { buffer } })) });
    const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(nthreads / 64))); pass.end();
  }
  _dispatchWG(enc, pipe, binds, ngroups) {  // one workgroup per group (attn)
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
      entries: binds.map((buffer, i) => ({ binding: i, resource: { buffer } })) });
    const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(ngroups); pass.end();
  }

  // unified linear appended to encoder `enc` (no submit): y = W @ x.
  _linearE(enc, name, nOut, nIn, xBuf) {
    const W = this.T[name] || this.T[name.replace(/\.weight$/, "")];
    if (W && W.quant) return this.packedOps.linear(enc, W, xBuf).y;
    const y = f32buf(this.device, nOut);
    this._dispatch(enc, this.k.matmul, [this._uni([nOut, nIn]), W.buffer, xBuf, y], nOut);
    return y;
  }
  _normE(enc, src, wname, outN) {
    const o = f32buf(this.device, outN);
    this._dispatch(enc, this.k.rmsnorm, [this._uni([outN, this.cfg.rms_eps], [0, 1]), src, this.T[wname].buffer, o], 1);
    return o;
  }

  // one decode step for a single token id at position pos -> logits [vocab]
  async forward(tokenId) {
    const c = this.cfg, H = c.hidden_size, hd = c.head_dim, pos = this.pos;
    const dev = this.device, T = this.T;
    // embed: gather row tokenId from embed_tokens (f16) into an f32 buffer
    const nh_hd = c.n_heads * hd, nkv_hd = c.n_kv_heads * hd, I = c.intermediate_size;
    const stride = c.n_kv_heads * hd;
    // ONE encoder for the whole token (KV writes are buffer-copies in the same
    // encoder; only the final logits readback breaks the chain). ~1 submit/token
    // instead of ~450 — the difference between usable and unusable at 8B.
    const enc = dev.createCommandEncoder();
    let x = this._embed(tokenId, enc);   // embedding row (quant embed appends to enc)
    for (let l = 0; l < c.num_layers; l++) {
      const P = (s) => `model.layers.${l}.${s}`;
      const normed = this._normE(enc, x, P("input_layernorm.weight"), H);
      const q = this._linearE(enc, P("self_attn.q_proj.weight"), nh_hd, H, normed);
      const kk = this._linearE(enc, P("self_attn.k_proj.weight"), nkv_hd, H, normed);
      const vv = this._linearE(enc, P("self_attn.v_proj.weight"), nkv_hd, H, normed);
      if (c.qk_norm) {
        this._dispatchWG(enc, this.packedOps.k.qknorm, [this._uni([c.n_heads, hd, c.rms_eps], [0, 0, 1]), q, T[P("self_attn.q_norm.weight")].buffer], c.n_heads);
        this._dispatchWG(enc, this.packedOps.k.qknorm, [this._uni([c.n_kv_heads, hd, c.rms_eps], [0, 0, 1]), kk, T[P("self_attn.k_norm.weight")].buffer], c.n_kv_heads);
      }
      this._dispatch(enc, this.k.rope, [this._uni([c.n_heads, hd, pos, c.rope_theta], [0, 0, 0, 1]), q], nh_hd / 2);
      this._dispatch(enc, this.k.rope, [this._uni([c.n_kv_heads, hd, pos, c.rope_theta], [0, 0, 0, 1]), kk], nkv_hd / 2);
      enc.copyBufferToBuffer(kk, 0, this.kv[l].K, pos * stride * 4, stride * 4);
      enc.copyBufferToBuffer(vv, 0, this.kv[l].V, pos * stride * 4, stride * 4);
      const attnOut = f32buf(dev, nh_hd);
      this._dispatchWG(enc, this.k.attn, [this._uni([c.n_heads, c.n_kv_heads, hd, pos + 1]), q, this.kv[l].K, this.kv[l].V, attnOut], c.n_heads);
      const attnProj = this._linearE(enc, P("self_attn.o_proj.weight"), H, nh_hd, attnOut);
      this._dispatch(enc, this.k.add, [this._uni([H]), x, attnProj], H);   // x += attnProj
      const normed2 = this._normE(enc, x, P("post_attention_layernorm.weight"), H);
      const gate = this._linearE(enc, P("mlp.gate_proj.weight"), I, H, normed2);
      const up = this._linearE(enc, P("mlp.up_proj.weight"), I, H, normed2);
      const act = f32buf(dev, I);
      this._dispatch(enc, this.k.swiglu, [this._uni([I]), gate, up, act], I);
      const down = this._linearE(enc, P("mlp.down_proj.weight"), H, I, act);
      this._dispatch(enc, this.k.add, [this._uni([H]), x, down], H);       // x += down
    }
    const fn = this._normE(enc, x, "model.norm.weight", H);
    const headName = T["lm_head.weight"] ? "lm_head.weight" : "model.embed_tokens.weight";
    const logits = this._linearE(enc, headName, c.vocab_size, H, fn);
    dev.queue.submit([enc.finish()]);
    this.pos++;
    return readF32(dev, logits, c.vocab_size);
  }

  _embed(tokenId, enc) {   // append embedding-row op to enc; return f32 buf [H]
    const H = this.cfg.hidden_size, dev = this.device;
    const emb = this.T["model.embed_tokens.weight"];
    if (emb.quant) return this.packedOps.embedRow(enc, emb, tokenId);  // 8B: quant embed
    const pipe = this.k._embed || (this.k._embed = makeKernel(dev, /* wgsl */`
      struct P{ row:u32, H:u32 };
      @group(0) @binding(0) var<uniform> p:P;
      @group(0) @binding(1) var<storage,read> W:array<u32>;
      @group(0) @binding(2) var<storage,read_write> y:array<f32>;
      fn wf16(i:u32)->f32{let q=unpack2x16float(W[i>>1u]);return select(q.x,q.y,(i&1u)==1u);}
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
        let i=g.x; if(i>=p.H){return;} y[i]=wf16(p.row*p.H+i);
      }`));
    const x = f32buf(dev, H);
    this._dispatch(enc, pipe, [this._uni([tokenId, H]), emb.buffer, x], H);
    return x;
  }

  reset() { this.pos = 0; }

  // Stream generation: feed prompt ids, then greedily (or temp-sampled) decode,
  // calling onToken(id) as each token is produced. Returns the id list.
  async generate(promptIds, { maxTokens = 40, temperature = 0.0, topk = 40,
                              eosIds = [], onToken = null } = {}) {
    this.reset();
    const out = [...promptIds];
    let logits;
    for (const id of promptIds) logits = await this.forward(id);   // prime KV cache
    for (let n = 0; n < maxTokens; n++) {
      let next;
      if (temperature <= 0) {
        next = argmax(logits);
      } else {
        next = sampleTopK(logits, temperature, topk);
      }
      if (eosIds.includes(next)) break;
      out.push(next);
      if (onToken) await onToken(next);
      logits = await this.forward(next);
    }
    return out;
  }
}

function argmax(a) { let m = -Infinity, mi = 0; for (let i = 0; i < a.length; i++) if (a[i] > m) { m = a[i]; mi = i; } return mi; }
function sampleTopK(logits, temp, k) {
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, k);
  let mx = -Infinity; for (const i of idx) mx = Math.max(mx, logits[i]);
  const probs = idx.map(i => Math.exp((logits[i] - mx) / temp));
  const s = probs.reduce((a, b) => a + b, 0);
  let r = Math.random() * s;
  for (let j = 0; j < idx.length; j++) { r -= probs[j]; if (r <= 0) return idx[j]; }
  return idx[0];
}
