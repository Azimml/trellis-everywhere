# Trellis-WebGPU

**The first trellis-coded LLM quantization decoder that runs outside CUDA — an 8B model executing its full forward pass in a browser tab on WebGPU, on a 4 GB GPU.**

![Trellis-WebGPU — an 8B trellis-quantized LLM running in a browser on WebGPU](assets/poster.gif)

Trellis-coded quantization (TCQ) — the method behind [QTIP](https://arxiv.org/abs/2406.11235) (NeurIPS 2024) and [EXL3](https://github.com/turboderp-org/exllamav3) — is the current quality frontier for low-bit LLM weights, beating scalar quantization (GPTQ / AWQ / GGUF) at equal bitrate. But every TCQ decoder shipped to date is **CUDA-only**, which locks this quality tier to NVIDIA data-center and desktop GPUs.

This project reimplements the full **encode → decode → inference** pipeline from scratch and ports the decoder to **WebGPU / WGSL** — so a 3-bit trellis-quantized model runs, correctly and verifiably, entirely on the client, with no server and no CUDA. To my knowledge this is the first trellis decoder to run on any non-CUDA target, and the first to run in a browser.

---

## Verified results

### Quantization quality (WikiText-2 perplexity, no fine-tuning)

| Model | fp16 | scalar 3-bit | **TCQ 3-bit (this repo)** | vs fp16 |
|---|---|---|---|---|
| Qwen3-8B | 9.02 | 12.44 | **10.34** | **1.15×** |
| Qwen3-1.7B | 15.53 | 45.13 | **18.00** | 1.16× |
| SmolLM2-135M | 15.61 | 707 | **17.58** (w/ LDLQ) | 1.13× |

At 3 bits, TCQ lands **2.1 perplexity closer to fp16 than scalar quantization** on the 8B, and the gap to fp16 holds at a stable ~1.15× across three model scales. (135M standalone TCQ 21.6 → 17.58 once Hessian-aware LDLQ is added.)

### The method reproduces the published rate–distortion
The K=2 trellis quantizer hits **MSE 0.0739** on unit-Gaussian weights vs. the QTIP paper's reported **0.0733** — a faithful reimplementation, not an approximation.

### It runs real models on WebGPU — up to 8B

Three model scales run a **full forward pass through the exact shipping WGSL shaders**, verified end-to-end and generating coherent, factually-correct text:

| Model | Layers | 3-bit weights | Peak VRAM (measured, Chrome / RTX 3050) | Output on "The capital of France is" |
|---|---|---|---|---|
| **Qwen3-8B** | 36 | 2.9 GB | **fits 4 GB** | *"…Paris. The capital of Italy is Rome. The capital of Germany is Berlin."* |
| **Qwen3-1.7B** | 28 | 0.63 GB | **~1.0 GB** | *"…Paris. Is this statement true or false?"* |
| **SmolLM2-135M** | 30 | 154 MB | small | coherent short completions |

(Peak VRAM measured on an RTX 3050 in Chrome with a 256-token KV cache. The 8B's app-allocated buffers total 2.96 GB and the full generation fits inside the 4 GB card; the 1.7B peaks at ~1.0 GB.)

Verified at four independent levels:

- **Tokenizer** — JS byte-level BPE, **bit-exact** vs HuggingFace.
- **Every WGSL kernel** — decode-matmul, RMSNorm, RoPE, GQA-attention, SwiGLU, per-head QK-norm (Qwen3) — compiled and run on real GPU hardware, matching NumPy to `<1e-6`.
- **Full packed 3-bit model** — 135M logits **identical to PyTorch** (top-5 exact); the 8B and 1.7B run all layers through the same shaders and generate correct text.
- **On a real 4 GB card** — the full 8B generation was run in Chrome on an **RTX 3050** and produced the output above without running out of memory.

The full pipeline — QK-norm, sharded streaming load, quantized-embedding decode, the IP-folded decode-matmul — is identical across all three sizes. The 8B is not a special case; it is the same code at scale.

**Measured VRAM.** The 8B's 3-bit weights are 2.9 GB. With a 256-token KV cache and recycled activation scratch, the full generation fits inside **4 GB** — verified live in Chrome on an RTX 3050 (4 GB), where it generated the completion above. Longer contexts raise the KV-cache footprint; `maxSeq` is configurable, and `scripts/vram_budget.py` reports the exact budget for any context length.

---

## Models

Packed, browser-ready 3-bit weights are published on the Hugging Face Hub:

- [`Azimml/Qwen3-8B-trellis-3bit-webgpu`](https://huggingface.co/Azimml/Qwen3-8B-trellis-3bit-webgpu)
- [`Azimml/Qwen3-1.7B-trellis-3bit-webgpu`](https://huggingface.co/Azimml/Qwen3-1.7B-trellis-3bit-webgpu)
- [`Azimml/SmolLM2-135M-trellis-3bit-webgpu`](https://huggingface.co/Azimml/SmolLM2-135M-trellis-3bit-webgpu)

Each is a quantized derivative of the corresponding base model (see licenses below).

---

## What's here

```
src/trellis/         the quantizer (independent reimplementation)
  qtip.py            bitshift trellis, MCG/1MAD/3INST codebooks, tail-biting Viterbi,
                     incoherence processing
  ldlq.py            Hessian-aware LDLQ error feedback (fixes SwiGLU down_proj at scale)
  viterbi_triton.py  fused Triton encode kernel (~5× the PyTorch DP)
  trellis.py         generic trellis engine   quant.py / eval.py   quant + ppl harness
web/                 the browser runtime (self-contained, zero dependencies)
  kernels.js         WGSL: trellis decode-matmul + transformer ops
  model.js           full Llama-style forward pass + KV cache
  packed.js          3-bit weight loading + IP-folded decode-matmul
  runtime.js         WebGPU device + buffer helpers
  tokenizer.js       byte-level BPE
  index.html         local runner UI
scripts/             export + validation + VRAM-budget harnesses
tests/               unit tests for the quantizer
notes/               my implementation spec (constants + algorithm extraction)
```

## Reproduce it

**Quantize + evaluate:**
```bash
python scripts/run_tcq_eval.py --model Qwen/Qwen3-8B --K 3 --ldlq 64 --ldlq-mlp-only
python scripts/export_packed_8b.py         # emit a browser-ready 3-bit model
```

**Verify correctness (runs the real WGSL shaders on your GPU via wgpu-py):**
```bash
python scripts/run_packed_headless.py web/model_8b "The capital of France is" 40
bash scripts/verify_all.sh                 # every check above, one command
```

**Run it in a browser locally:**
```bash
cd web && python3 -m http.server 8000
# open http://localhost:8000/index.html in a WebGPU browser (Chrome/Edge 113+).
# on Linux you may need chrome://flags → "Unsafe WebGPU Support" + Vulkan.
```

## Key engineering notes

- **LDLQ is essential at scale.** Without Hessian feedback, 3-bit TCQ is fine on attention but destroys the SwiGLU `down_proj` layers (their inputs have massive-activation channels); error compounds through the residual stream and an 8B collapses to perplexity 80,000+. MLP-only LDLQ with a 64-sequence calibration Hessian fixes it (→ 10.34).
- **Trellis decode was never sequential.** A quantized weight is a pure function of a sliding 16-bit window over the bitstream — random-access by design — and the T=256 tiles are independent via tail-biting. The CUDA kernels use warp-shuffles for *speed*, not necessity. That is *why* a WebGPU (or Metal/NEON) port is possible at all.
- **The incoherence-processing transform folds into the activations.** The GPU runs a z-space decode-matmul plus cheap per-vector Hadamard/scale transforms on the input and output — provably equal to the fully dequantized `W·x` (verified to 4.9e-7), so no weight is ever materialized in full precision.

## Status & roadmap
The quantizer and the browser runtime are complete and machine-verified end-to-end across three model scales (135M / 1.7B / 8B), with the 8B running in-browser on a 4 GB GPU. Next: Metal and NEON decode kernels (the same random-access property makes them straightforward), and higher throughput (the current decode-matmul is correctness-first, not yet speed-optimized).

## Credit & licenses
This is an independent reimplementation of the TCQ method; the cross-platform (non-CUDA) decoder is the new contribution here.

- **Method:** QTIP — Tseng, Zhao, Hou, Sun, De Sa, Chee (NeurIPS 2024), [paper](https://arxiv.org/abs/2406.11235), [code](https://github.com/Cornell-RelaxML/qtip). EXL3 — turboderp, [exllamav3](https://github.com/turboderp-org/exllamav3).
- **Base models:** [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) and [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0), [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (Apache-2.0). The published quantized weights are derivatives distributed under the base models' licenses.
