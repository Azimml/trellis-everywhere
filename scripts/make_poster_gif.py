#!/usr/bin/env python
"""Trellis-WebGPU animated technical poster (light mode, ByteByteGo style).

One dense system-design poster, everything animating simultaneously in a fast
~5s seamless loop:
  - numbered flow badges (1)-(8) across QUANTIZE -> DECODE -> TRANSFORMER
  - a real bitshift-trellis lattice with a Viterbi path drawing through it
  - a sliding 16-bit decoder-state window over the packed bitstream
  - flow dots on inter-panel arrows, 36 layer-ticks filling, highlight sweep
Pure Pillow, no browser. Output: web/poster.gif (1240x880).
"""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1240, 880
OUT = os.path.join(os.path.dirname(__file__), "..", "web", "poster.gif")
FD = "/usr/share/fonts/truetype/dejavu"

# ---- light palette (ByteByteGo-ish) ----
BG      = (250, 250, 247)
INK     = (17, 24, 39)
SUB     = (107, 114, 128)
BORDER  = (31, 41, 55)
PANEL_B = (156, 163, 175)      # dashed panel border
EDGE    = (209, 213, 219)      # lattice idle edges
PURPLE  = (233, 216, 253)
BLUE    = (214, 232, 255)
ORANGE  = (255, 225, 196)
YELLOW  = (255, 243, 191)
GREEN   = (211, 249, 216)
REDF    = (255, 214, 214)
ACCENT  = (232, 89, 12)        # orange flow/highlight
REDT    = (201, 42, 42)        # red flex text

def F(n, s): return ImageFont.truetype(os.path.join(FD, n), s)
SANS  = lambda s: F("DejaVuSans.ttf", s)
BOLD  = lambda s: F("DejaVuSans-Bold.ttf", s)
MONO  = lambda s: F("DejaVuSansMono.ttf", s)
MONOB = lambda s: F("DejaVuSansMono-Bold.ttf", s)

# ---------------- helpers ----------------
def dashed_rect(d, x0, y0, x1, y1, col=PANEL_B, dash=9, gap=6, width=2):
    def dline(a, b, vert=False):
        if not vert:
            x = a[0]
            while x < b[0]:
                d.line([x, a[1], min(x + dash, b[0]), a[1]], fill=col, width=width)
                x += dash + gap
        else:
            y = a[1]
            while y < b[1]:
                d.line([a[0], y, a[0], min(y + dash, b[1])], fill=col, width=width)
                y += dash + gap
    dline((x0, y0), (x1, y0)); dline((x0, y1), (x1, y1))
    dline((x0, y0), (x0, y1), True); dline((x1, y0), (x1, y1), True)

def badge(d, cx, cy, n, r=12):
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 255, 255), outline=BORDER, width=2)
    d.text((cx, cy), str(n), font=BOLD(13), fill=INK, anchor="mm")

def sbox(d, x, y, w, h, title, sub, fill):
    d.rounded_rectangle([x, y, x+w, y+h], 9, fill=fill, outline=BORDER, width=2)
    d.text((x+14, y+h//2), title, font=BOLD(13), fill=INK, anchor="lm")
    if sub:
        d.text((x+w-14, y+h//2), sub, font=SANS(11), fill=SUB, anchor="rm")

def varrow(d, x, y0, y1, col=BORDER, w=2):
    d.line([x, y0, x, y1-4], fill=col, width=w)
    d.polygon([(x, y1), (x-5, y1-8), (x+5, y1-8)], fill=col)

def harrow(d, x0, x1, y, col=BORDER, w=2):
    d.line([x0, y, x1-4, y], fill=col, width=w)
    d.polygon([(x1, y), (x1-8, y-5), (x1-8, y+5)], fill=col)

# ---------------- geometry ----------------
# Panel A (quantize, left top), Panel B (trellis, right top),
# Panel C (decode, left bottom), Panel D (transformer, right bottom)
PA = (40, 100, 560, 368)
PB = (600, 100, 1200, 368)
PC = (40, 424, 560, 664)
PD = (600, 424, 1200, 664)

ABOX_X, ABOX_W, ABOX_H = 70, 460, 38
A_YS = [142, 198, 254, 310]
C_YS = [466, 520, 574]

# bit row (panel B)
BITS = "011010011101011000101101"      # 24 cells
BX, BY, CW, CH = 640, 138, 21, 28
NB = len(BITS)
# lattice
LX0, LY0, LDX, LDY, LT, LS = 660, 215, 60, 38, 9, 4

# transformer flow mini boxes (panel D)
DFLOW = [("RMSNorm", 96, BLUE), ("GQA attn · RoPE · QK-norm", 190, PURPLE),
         ("SwiGLU", 96, ORANGE), ("+ residual", 110, GREEN)]
DF_Y, DF_H = 470, 36
TICK_Y0, TICK_Y1 = 540, 554
TOK_Y, TOK_H = 584, 38

# highlight sweep targets (10 x 9 frames = 90-frame loop)
HL = []
for y in A_YS:  HL.append((ABOX_X, y, ABOX_X+ABOX_W, y+ABOX_H))
for y in C_YS:  HL.append((ABOX_X, y, ABOX_X+ABOX_W, y+ABOX_H))
HL.append((630, DF_Y, 1158, DF_Y+DF_H))
HL.append((630, TICK_Y0-6, 1170, TICK_Y1+6))
HL.append((630, TOK_Y, 1158, TOK_Y+TOK_H))

FRAMES, DUR = 90, 60   # 5.4 s seamless loop

def path_states(bits):
    s, out = 0, [0]
    for b in bits:
        s = (2*s + b) % LS
        out.append(s)
    return out
PATHS = [path_states(p) for p in
         ([0,1,0,1,1,0,1,0], [1,1,0,0,1,0,0,1], [0,0,1,1,0,1,1,0])]

def node(t, s): return (LX0 + t*LDX, LY0 + s*LDY)

# ---------------- static base ----------------
def build_base():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # title
    d.text((W//2, 38), "How a 3-bit 8B LLM Runs in a Browser — Without CUDA",
           font=BOLD(30), fill=INK, anchor="mm")
    d.text((W//2, 74), "Trellis-WebGPU · QTIP/EXL3-quality trellis quantization · "
           "hand-written WebGPU/WGSL kernels", font=SANS(14), fill=SUB, anchor="mm")

    # ---- Panel A: QUANTIZE ----
    dashed_rect(d, *PA)
    d.text((PA[0]+16, PA[1]+22), "QUANTIZE — once, offline", font=BOLD(15), fill=INK, anchor="lm")
    d.text((PA[2]-16, PA[1]+22), "PyTorch + Triton", font=SANS(11), fill=SUB, anchor="rm")
    A = [("fp16 weights", "Qwen3-8B · 16 GB", PURPLE),
         ("incoherence processing", "±signs · Hadamard-128 · RMS scales", BLUE),
         ("Viterbi encode", "tail-biting · fused Triton kernel", ORANGE),
         ("3-bit trellis codes", "2.9 GB · 5.4× smaller", YELLOW)]
    for k, ((t, s, f_), y) in enumerate(zip(A, A_YS)):
        sbox(d, ABOX_X, y, ABOX_W, ABOX_H, t, s, f_)
        if k < 3:
            varrow(d, ABOX_X+ABOX_W//2, y+ABOX_H+2, A_YS[k+1]-2)
            badge(d, ABOX_X+ABOX_W//2+34, y+ABOX_H+9, k+1)

    # ship arrow A -> C
    varrow(d, 300, PA[3]+4, PC[1]-4, col=BORDER, w=3)
    badge(d, 328, (PA[3]+PC[1])//2, 4)
    d.text((348, (PA[3]+PC[1])//2), "ship sharded .bin — 2.9 GB", font=SANS(11), fill=SUB, anchor="lm")

    # ---- Panel B: THE TRELLIS ----
    dashed_rect(d, *PB)
    d.text((PB[0]+16, PB[1]+22), "THE TRELLIS — the key insight", font=BOLD(15), fill=INK, anchor="lm")
    d.text((PB[2]-16, PB[1]+22), "bitshift trellis · L=16", font=SANS(11), fill=SUB, anchor="rm")
    # bit row
    for j, b in enumerate(BITS):
        x = BX + j*CW
        d.rectangle([x, BY, x+CW-3, BY+CH], fill=(255, 255, 255), outline=EDGE, width=1)
        d.text((x+(CW-3)//2, BY+CH//2), b, font=MONOB(13), fill=INK, anchor="mm")
    d.text((BX, BY+CH+16), "decoder state = a sliding 16-bit window → random access, no sequential scan",
           font=SANS(11.5), fill=SUB, anchor="lm")
    # lattice edges + nodes
    for t in range(LT-1):
        for s in range(LS):
            for b in (0, 1):
                d.line([node(t, s), node(t+1, (2*s+b) % LS)], fill=EDGE, width=1)
    for t in range(LT):
        for s in range(LS):
            x, y = node(t, s)
            d.ellipse([x-4, y-4, x+4, y+4], fill=(255, 255, 255), outline=BORDER, width=1)
    d.text((PB[0]+16, PB[3]-18), "Viterbi picks the best path once (offline) — the browser only reads windows",
           font=SANS(11.5), fill=SUB, anchor="lm")

    # ---- Panel C: DECODE ----
    dashed_rect(d, *PC)
    d.text((PC[0]+16, PC[1]+22), "DECODE-MATMUL — in the browser", font=BOLD(15), fill=INK, anchor="lm")
    d.text((PC[2]-16, PC[1]+22), "WebGPU / WGSL", font=SANS(11), fill=SUB, anchor="rm")
    C = [("WGSL trellis decode", "random-access · in-register codebook", GREEN),
         ("z-space matmul", "decode fused into the dot product", GREEN),
         ("IP fold on activations", "= exact W·x (err 4.9e-7)", BLUE)]
    for k, ((t, s, f_), y) in enumerate(zip(C, C_YS)):
        sbox(d, ABOX_X, y, ABOX_W, ABOX_H, t, s, f_)
        if k < 2:
            varrow(d, ABOX_X+ABOX_W//2, y+ABOX_H+2, C_YS[k+1]-2)
            badge(d, ABOX_X+ABOX_W//2+34, y+ABOX_H+9, k+5)
    d.text((PC[0]+16, PC[3]-22), "one command encoder per token — ~450 GPU dispatches fused into 1 submit",
           font=SANS(11), fill=SUB, anchor="lm")

    # C -> D arrow
    harrow(d, PC[2]+4, PD[0]-4, 544, col=BORDER, w=3)
    badge(d, (PC[2]+PD[0])//2, 528, 7)

    # ---- Panel D: TRANSFORMER ----
    dashed_rect(d, *PD)
    d.text((PD[0]+16, PD[1]+22), "TRANSFORMER ×36 → tokens", font=BOLD(15), fill=INK, anchor="lm")
    d.text((PD[2]-16, PD[1]+22), "per token", font=SANS(11), fill=SUB, anchor="rm")
    x = 630
    for name, w_, f_ in DFLOW:
        d.rounded_rectangle([x, DF_Y, x+w_, DF_Y+DF_H], 8, fill=f_, outline=BORDER, width=2)
        d.text((x+w_//2, DF_Y+DF_H//2), name, font=BOLD(11.5), fill=INK, anchor="mm")
        if name != DFLOW[-1][0]:
            harrow(d, x+w_+1, x+w_+11, DF_Y+DF_H//2, w=2)
        x += w_ + 12
    d.text((630, 522), "every matmul above is a trellis decode — no weight is ever materialized in fp16",
           font=SANS(11), fill=SUB, anchor="lm")
    # tick outlines
    for i in range(36):
        tx = 640 + i*(520/35)
        d.line([tx, TICK_Y0, tx, TICK_Y1], fill=EDGE, width=3)
    d.text((1184, (TICK_Y0+TICK_Y1)//2), "×36", font=BOLD(12), fill=SUB, anchor="lm")
    # token row
    T = [("logits · 152k vocab", 180, BLUE), ("sample (greedy/top-k)", 180, PURPLE),
         ("next token", 120, REDF)]
    x = 630
    for k, (name, w_, f_) in enumerate(T):
        d.rounded_rectangle([x, TOK_Y, x+w_, TOK_Y+TOK_H], 8, fill=f_, outline=BORDER, width=2)
        d.text((x+w_//2, TOK_Y+TOK_H//2), name, font=BOLD(11.5), fill=INK, anchor="mm")
        if k < 2:
            harrow(d, x+w_+2, x+w_+22, TOK_Y+TOK_H//2, w=2)
            if k == 0: badge(d, x+w_+12, TOK_Y-8, 8)
        x += w_ + 24
    # loop-back arrow (token -> layers)
    d.line([1118, TOK_Y-2, 1118, DF_Y+DF_H+8], fill=BORDER, width=2)
    d.polygon([(1118, DF_Y+DF_H+4), (1113, DF_Y+DF_H+12), (1123, DF_Y+DF_H+12)], fill=BORDER)
    d.text((1126, TOK_Y-14), "repeat", font=SANS(10), fill=SUB, anchor="lm")

    # ---- stats chips ----
    chips = [("1.15× fp16 — 10.34 vs 9.02 ppl", YELLOW, INK),
             ("2.9 GB weights · fits a 4 GB GPU", GREEN, INK),
             ("WGSL kernels <1e-6 vs PyTorch", BLUE, INK),
             ("first trellis decoder outside CUDA", (255, 255, 255), REDT)]
    x = 48
    for txt, f_, tc in chips:
        oc = REDT if tc == REDT else BORDER
        d.rounded_rectangle([x, 700, x+273, 746], 10, fill=f_, outline=oc, width=2)
        d.text((x+136, 723), txt, font=BOLD(12.5), fill=tc, anchor="mm")
        x += 289

    # footer
    d.text((48, 800), "github.com/Azimml/trellis-webgpu", font=BOLD(14), fill=INK, anchor="lm")
    d.text((1192, 800), "Qwen3-8B · measured on a 4 GB RTX 3050 · numbers, not estimates",
           font=SANS(12), fill=SUB, anchor="rm")
    return img

# ---------------- dynamic overlays ----------------
ARROWS = [  # polylines for flow dots
    [(300, PA[3]+6), (300, PC[1]-6)],                 # ship A->C
    [(PC[2]+6, 544), (PD[0]-8, 544)],                 # C->D
]

def lerp(a, b, t): return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t)

def frame(i, base):
    img = base.copy(); d = ImageDraw.Draw(img)

    # 1) highlight sweep (10 targets x 9 frames)
    k = (i // 9) % len(HL)
    x0, y0, x1, y1 = HL[k]
    d.rounded_rectangle([x0-3, y0-3, x1+3, y1+3], 11, outline=ACCENT, width=3)

    # 2) flow dots (period 18 -> 5 cycles per loop)
    t = (i % 18) / 18
    for pl in ARROWS:
        p = lerp(pl[0], pl[1], t)
        d.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=ACCENT)
        q = lerp(pl[0], pl[1], max(0, t-0.12))
        d.ellipse([q[0]-2, q[1]-2, q[0]+2, q[1]+2], fill=ACCENT)

    # 3) sliding 16-bit window (period 30 -> 3 cycles)
    p = ((i % 30) / 30) * (NB - 8)
    pi = int(p)
    d.rounded_rectangle([BX+pi*CW-2, BY-3, BX+(pi+8)*CW-3+2, BY+CH+3], 6,
                        outline=ACCENT, width=3)

    # 4) Viterbi path through the lattice (period 30, 3 different paths/loop)
    cyc, idx = i % 30, (i // 30) % len(PATHS)
    prog = min(1.0, cyc / 22)
    nseg = int(prog * (LT-1))
    st = PATHS[idx]
    for tt in range(nseg):
        d.line([node(tt, st[tt]), node(tt+1, st[tt+1])], fill=ACCENT, width=3)
    for tt in range(nseg+1):
        x, y = node(tt, st[tt])
        d.ellipse([x-5, y-5, x+5, y+5], fill=ACCENT)

    # 5) layer ticks filling (2 frames per tick, resets each loop)
    n = min(36, (i % FRAMES) // 2)
    for j in range(n):
        tx = 640 + j*(520/35)
        d.line([tx, TICK_Y0, tx, TICK_Y1], fill=ACCENT, width=3)
    return img

def main():
    base = build_base()
    frames = [frame(i, base).convert("P", palette=Image.ADAPTIVE, colors=128)
              for i in range(FRAMES)]
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=DUR,
                   loop=0, optimize=True, disposal=2)
    print(f"wrote {OUT} ({FRAMES} frames, {FRAMES*DUR/1000:.1f}s loop, "
          f"{os.path.getsize(OUT)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
