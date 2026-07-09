# TCQ Implementation Spec (QTIP + EXL3) — extracted 2026-07-04

Sources: QTIP arXiv:2406.11235 (NeurIPS'24) + Cornell-RelaxML/qtip; turboderp-org/exllamav3.
Verbatim reference sources archived in `reference-src/`.

## A. Bitshift trellis
- 2^L states; transition `s' = ((s << kV) | fresh_bits) & (2^L - 1)`; fresh bits enter at LSB.
- State at step t == the sliding L-bit window over the packed bitstream. **Decode is a pure
  function of the bitstream window — random-access, no recursion.** (QTIP §3.1)
- QTIP released models: L=16, K∈{2,3,4}, V=2 (kV=2K bits/step). EXL3: L=16 hardcoded, V=1, K=1..8.
- Sequence T=256 = one 16×16 tile. QTIP intra-tile row-major (+`_PERMUTE` for kernel);
  EXL3 uses tensor-core lane order (`tensor_core_perm`).

## B. Codebooks (state -> value; all ≈ N(0,1) after IP normalization)
- **1MAD**: `x = (x*34038481 + 76625530) & 0xFFFFFFFF`; y = sum of 4 bytes; `(y-510)/147.800537109375`.
- **3INST**: `x = (x*89226354 + 64248484) & 0xFFFFFFFF`; `res = (x & 0x8FFF8FFF) ^ 0x3B603B60`;
  value = fp16(res>>16) + fp16(res&0xFFFF).
- **MCG (EXL3 default)**: `x *= 0xCBAC1FED` (no add), then same lop3 + fp16-halves-sum as 3INST.
- **MUL1**: `x *= 0x83DCD12D`; byte-sum+1024 as fp16; `hfma(h, 1/147.7, -(1024+510)/147.7)`.
- **HYB/quantlut_sym (QTIP released)**: `h = x*(x+1)`; sign from bit15; idx = bits (16-Q-1)..;
  Q=9 → 2^9×2 fp16 LUT (k-means of 2D normals, scaled by 0.9682458365518543), learnable.
- EXL3 codebook RMS constant: `codebook_scale = 1.24371088`.

## C. Encode (Viterbi, exact DP + approximate tail-biting)
- DP over **edges** = (L−kV)-bit overlap values (EXL3: edges = 65536>>K). For output state
  s'=(q<<(L−K))|e: predecessor edge = s'>>K. Cost = fp16/fp32 MSE vs target.
- **Tail-biting (QTIP Alg. 4, EXL3 identical with roll=128):**
  1. roll sequence by T/2; unconstrained Viterbi.
  2. overlap = (state at rolled position T/2) >> kV  — i.e. original position 0.
  3. re-run Viterbi on unrolled sequence with masks: at t=0 allowed initial states
     = (overlap<<kV)+c; at t=T−1 allowed final states = overlap + c·2^(L−kV).
  Cost of approximation ≈ 0 (paper Table 2). Gives exactly K bpw (no seed-state overhead).
- Both nest the trellis inside **BlockLDLQ** (block 16): quantize input-channel blocks in
  reverse order, target = W_blk + L_blkᵀ(W_later − Q_later). EXL3 computes H on the fly
  (250×2048 bundled corpus, sigma_reg=0.025·mean diag); falls back to pure MSE if no H.
  QTIP additionally fine-tunes (headline numbers include FT).

## D. Incoherence processing
- QTIP = QuIP# RHT: full-dim Hadamard × random ±1 signs both sides; one scalar Wscale/tensor
  (`Wr.square().mean().sqrt() / (cb.lut.rms() * 0.9)`).
- EXL3: block-diagonal Hadamard 128 both dims; ±1 sign vectors su/sv; per-in-channel RMS
  scales (+optional per-out-channel, default on); global scale via golden-section search
  [0.1,1.9] minimizing actual Viterbi MSE on sampled tiles; H gets signs+Hadamard only.

## E. Numbers to beat (wikitext2)
- QTIP with FT (Table 5): L2-7B(5.12): 2b **5.86**, 3b 5.28, 4b 5.17. L3-8B(5.54, ctx8192):
  2b **7.33**, 3b 6.01, 4b 5.67.
- **No-finetune computed codes (Table 3, our honest target)**: L2-7B 2b: 1MAD **6.82**,
  3INST 7.05 (QuIP# no-FT 8.22); 3b: 5.38/5.40; 4b: 5.12/5.17.
- Scalar baselines (QuIP# Table 2, ctx2048): AWQ 2b = 24.0, OmniQuant 2b = 37.4.
- GGUF IQ2 comparison: only PNG graphs in exllamav3/doc; no numeric table exists → producing
  one is itself a contribution.

## F. Parallelism — the key finding
- **Decode was never sequential.** Each V weights depend only on a contiguous L-bit window of
  the stream (windows overlap by L−kV bits). Any chunk can decode with L−kV bits of left
  context, which the format already carries. CUDA kernels use warp shuffles only to splice
  windows across register lanes — an efficiency trick, not a dependency fix.
- Tiles (T=256) are fully independent (tail-biting). 8B model ⇒ ~27M independent tiles.
- **⇒ Project reframe: no "block-parallel reformulation" research is needed. The genuinely
  unshipped thing is the cross-platform engineering: no non-CUDA trellis decoder exists
  anywhere (WebGPU/Metal/NEON), no encoder toolchain outside CUDA+Python, and no published
  numeric quality table for the quant tiers. Research risk → engineering risk. The
  mini-wedge block-restart experiments remain useful only as design-space data.**
