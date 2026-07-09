#!/usr/bin/env python
"""Render the Trellis Everywhere hero animation directly to an animated GIF with
Pillow — no browser. 3 scenes, looping:
  1) CUDA -> browser contrast (the achievement)
  2) pipeline flow (bitstream -> trellis decode -> matmul -> tokens)
  3) 8B generating in a browser tab + VRAM meter filling to "fits 4 GB"
Output: web/hero.gif  (1280x720, ~15s loop)
"""
import os, math
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
OUT = os.path.join(os.path.dirname(__file__), "..", "web", "hero.gif")
FD = "/usr/share/fonts/truetype/dejavu"

# palette
BG_TOP=(14,22,34); BG_BOT=(7,10,15); PANEL=(17,24,35); LINE=(36,50,71)
INK=(230,237,243); DIM=(139,155,176); ACC=(63,185,80); ACC2=(126,231,135)
BLUE=(88,166,255); RED=(255,157,151); GRAD_A=(126,231,135); GRAD_B=(88,166,255)

def font(name, sz): return ImageFont.truetype(os.path.join(FD, name), sz)
SANS   = lambda s: font("DejaVuSans.ttf", s)
BOLD   = lambda s: font("DejaVuSans-Bold.ttf", s)
MONO   = lambda s: font("DejaVuSansMono.ttf", s)
MONOB  = lambda s: font("DejaVuSansMono-Bold.ttf", s)

def bg():
    img = Image.new("RGB", (W, H), BG_BOT)
    px = img.load()
    for y in range(H):
        t = y / H
        r = int(BG_TOP[0]*(1-t) + BG_BOT[0]*t)
        g = int(BG_TOP[1]*(1-t) + BG_BOT[1]*t)
        b = int(BG_TOP[2]*(1-t) + BG_BOT[2]*t)
        for x in range(0, W, 1):
            px[x, y] = (r, g, b)
    return img

def rr(d, box, rad, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=rad, fill=fill, outline=outline, width=width)

def ctext(d, cx, y, text, fnt, fill, anchor="mm"):
    d.text((cx, y), text, font=fnt, fill=fill, anchor=anchor)

def grad_text(img, cx, y, text, fnt):
    # draw text in a horizontal gradient by masking
    tmp = Image.new("L", (W, H), 0)
    td = ImageDraw.Draw(tmp)
    td.text((cx, y), text, font=fnt, fill=255, anchor="mm")
    grad = Image.new("RGB", (W, H))
    gp = grad.load()
    bb = tmp.getbbox()
    if not bb: return
    x0, x1 = bb[0], bb[2]
    for x in range(W):
        t = 0 if x1==x0 else max(0,min(1,(x-x0)/(x1-x0)))
        gp[x, 0] = tuple(int(GRAD_A[i]*(1-t)+GRAD_B[i]*t) for i in range(3))
    for x in range(W):
        c = gp[x, 0]
        for yy in range(H): gp[x, yy] = c
    img.paste(grad, (0,0), tmp)

def icon(d, cx, cy, kind, col=INK, s=30):
    """Crisp vector icons (no emoji fonts). s = half-size."""
    w = 3
    if kind == "gpu":       # chip: square + pins + inner die
        d.rounded_rectangle([cx-s, cy-s*0.8, cx+s, cy+s*0.8], 6, outline=col, width=w)
        d.rounded_rectangle([cx-s*0.5, cy-s*0.4, cx+s*0.5, cy+s*0.4], 3, outline=col, width=2)
        for i in range(-2,3):
            x=cx+i*s*0.4
            d.line([x, cy-s*0.8, x, cy-s*1.1], fill=col, width=2)   # top pins
            d.line([x, cy+s*0.8, x, cy+s*1.1], fill=col, width=2)   # bottom pins
        for i in range(-1,2):
            y=cy+i*s*0.45
            d.line([cx-s, y, cx-s*1.25, y], fill=col, width=2)      # left pins
            d.line([cx+s, y, cx+s*1.25, y], fill=col, width=2)      # right pins
    elif kind == "globe":
        d.ellipse([cx-s, cy-s, cx+s, cy+s], outline=col, width=w)
        d.ellipse([cx-s*0.45, cy-s, cx+s*0.45, cy+s], outline=col, width=2)
        d.line([cx-s, cy, cx+s, cy], fill=col, width=2)
        d.arc([cx-s, cy-s*0.5, cx+s, cy+s*1.6], 200, 340, fill=col, width=2)
        d.arc([cx-s, cy-s*1.6, cx+s, cy+s*0.5], 20, 160, fill=col, width=2)
    elif kind == "lock":
        d.rounded_rectangle([cx-s*0.7, cy-s*0.1, cx+s*0.7, cy+s*0.8], 4, outline=col, width=w)
        d.arc([cx-s*0.45, cy-s*0.8, cx+s*0.45, cy+s*0.2], 180, 360, fill=col, width=w)
    elif kind == "bits":    # stacked binary rows
        f=MONOB(15)
        rows=["1011","0110","1101"]
        for i,r in enumerate(rows):
            d.text((cx, cy-16+i*16), r, font=f, fill=col, anchor="mm")
    elif kind == "lattice": # trellis: nodes + diagonal edges
        pts=[(-s,-s*0.6),(-s,s*0.6),(0,-s*0.6),(0,s*0.6),(s,-s*0.6),(s,s*0.6)]
        P=[(cx+x,cy+y) for x,y in pts]
        edges=[(0,2),(1,2),(2,4),(3,4),(0,3),(1,3)]
        for a,b in edges: d.line([P[a],P[b]], fill=col, width=2)
        for p in P: d.ellipse([p[0]-3,p[1]-3,p[0]+3,p[1]+3], fill=col)
    elif kind == "bolt":
        pts=[(cx+4,cy-s),(cx-s*0.6,cy+3),(cx-2,cy+3),(cx-4,cy+s),(cx+s*0.6,cy-3),(cx+2,cy-3)]
        d.polygon(pts, outline=col, width=2, fill=col)
    elif kind == "chat":    # speech bubble with dots
        d.rounded_rectangle([cx-s, cy-s*0.7, cx+s, cy+s*0.4], 6, outline=col, width=w)
        d.polygon([(cx-s*0.3,cy+s*0.4),(cx-s*0.05,cy+s*0.8),(cx+s*0.2,cy+s*0.4)], fill=col)
        for i in range(-1,2):
            d.ellipse([cx+i*s*0.4-2.5, cy-s*0.2-2.5, cx+i*s*0.4+2.5, cy-s*0.2+2.5], fill=col)

def brand(d):
    d.text((W-28, H-30), "Trellis Everywhere", font=BOLD(15), fill=INK, anchor="rs")
    d.text((W-28, H-12), "github.com/Azimml/trellis-everywhere", font=SANS(13), fill=DIM, anchor="rs")

def fade(img, alpha):
    if alpha >= 1.0: return img
    black = Image.new("RGB", (W, H), (0,0,0))
    return Image.blend(black, img, max(0.0, alpha))

# ---------------- SCENE 1 ----------------
def scene1(prog, alpha):
    img = bg(); d = ImageDraw.Draw(img)
    ctext(d, W//2, 150, "SOTA 3-bit LLM quantization", BOLD(50), INK)
    d.text((W//2, 210), "was ", font=BOLD(50), fill=INK, anchor="rm")
    d.text((W//2, 210), "locked to CUDA.", font=BOLD(50), fill=RED, anchor="lm")
    ctext(d, W//2, 268, "Trellis-coded quantization (QTIP / EXL3) — best low-bit quality, NVIDIA-only.",
          SANS(19), DIM)
    # split panel
    bx0, bx1, by0, by1 = 140, 1140, 330, 630
    rr(d, [bx0, by0, bx1, by1], 16, fill=PANEL, outline=LINE, width=1)
    mid = (bx0+bx1)//2
    # before half tint
    rr(d, [bx0+1, by0+1, mid, by1-1], 0, fill=(26,15,16))
    rr(d, [mid, by0+1, bx1-1, by1-1], 0, fill=(14,28,20))
    d.line([mid, by0, mid, by1], fill=LINE, width=1)
    # before
    d.rounded_rectangle([mid-260, by0+40, mid-140, by0+70], 15, fill=(50,20,20), outline=(120,40,40))
    ctext(d, mid-200, by0+55, "BEFORE", BOLD(14), (255,157,151))
    icon(d, mid-200, by0+150, "gpu", RED, s=34)
    ctext(d, mid-200, by0+225, "NVIDIA GPU + CUDA", SANS(19), INK)
    ctext(d, mid-200, by0+253, "data-center / desktop only", SANS(15), DIM)
    icon(d, mid-30, by0+32, "lock", (200,120,120), s=13)
    # after
    d.rounded_rectangle([mid+150, by0+40, mid+230, by0+70], 15, fill=(20,50,28), outline=(50,140,70))
    ctext(d, mid+190, by0+55, "NOW", BOLD(14), ACC2)
    icon(d, mid+200, by0+150, "globe", ACC2, s=34)
    ctext(d, mid+200, by0+225, "A browser tab", SANS(19), INK)
    ctext(d, mid+200, by0+253, "on a 4 GB laptop GPU · no server", SANS(15), DIM)
    brand(d)
    return fade(img, alpha)

# ---------------- SCENE 2 ----------------
STAGES = [
    ("bitstream",  "3-bit bitstream",  "packed trellis codes"),
    ("decode",     "trellis decode",   "random-access, not sequential"),
    ("matmul",     "z-space matmul",   "+ folded incoherence transform"),
    ("tokens",     "tokens",           "on your GPU, offline"),
]
def scene2(prog, alpha):
    # prog 0..1 across the whole scene; light boxes sequentially
    img = bg(); d = ImageDraw.Draw(img)
    ctext(d, W//2, 120, "The first trellis decoder that runs", BOLD(30), INK)
    grad_text(img, W//2, 162, "outside CUDA", BOLD(34))
    d = ImageDraw.Draw(img)
    ctext(d, W//2, 210, "Full 8B forward pass, hand-written in WebGPU / WGSL", SANS(17), DIM)
    # 4 boxes
    n=4; bw=200; gap=(W-160-n*bw)//(n-1); x0=80; cy=400; bh=160
    lit = int(prog * (n+1)) - 1   # which box is lit
    centers=[]
    for i,(k,name,det) in enumerate(STAGES):
        x = x0 + i*(bw+gap)
        centers.append((x+bw//2, cy))
        islit = (i == lit)
        oc = ACC if islit else LINE
        rr(d, [x, cy-bh//2, x+bw, cy+bh//2], 14, fill=PANEL, outline=oc, width=2 if islit else 1)
        if islit:
            rr(d, [x-1, cy-bh//2-1, x+bw+1, cy+bh//2+1], 15, outline=ACC, width=1)
        kinds=["bits","lattice","bolt","chat"]
        icol = ACC2 if islit else INK
        icon(d, x+bw//2, cy-38, kinds[i], icol, s=22)
        ctext(d, x+bw//2, cy+12, name, BOLD(17), INK)
        ctext(d, x+bw//2, cy+44, det, SANS(12), DIM)
    # arrows between
    for i in range(n-1):
        ax0 = centers[i][0]+bw//2+6; ax1=centers[i+1][0]-bw//2-6
        d.line([ax0, cy, ax1, cy], fill=LINE, width=2)
        d.polygon([(ax1,cy),(ax1-8,cy-5),(ax1-8,cy+5)], fill=LINE)
        # spark travels on the arrow that's currently active
        seg = prog*(n+1)-1
        if i < seg < i+1:
            fr = seg - i
            sx = ax0 + (ax1-ax0)*fr
            d.line([sx-14, cy, sx, cy], fill=ACC2, width=3)
            d.ellipse([sx-3,cy-3,sx+3,cy+3], fill=ACC2)
    ctext(d, W//2, 560,
          "verified end-to-end · WGSL kernels match PyTorch to <1e-6 · 1.15× fp16 perplexity",
          SANS(14), DIM)
    brand(d)
    return fade(img, alpha)

# ---------------- SCENE 3 ----------------
GEN = " Paris. The capital of Italy is Rome. The capital of Germany is Berlin."
def scene3(nchars, alpha, blink):
    img = bg(); d = ImageDraw.Draw(img)
    ctext(d, W//2, 96, "Qwen3-8B, 3-bit, running in a browser — on 4 GB.", BOLD(26), INK)
    # tab
    tx0, tx1, ty0 = 230, 1050, 150
    th = 300
    rr(d, [tx0, ty0, tx1, ty0+th], 14, fill=PANEL, outline=LINE, width=1)
    # tab bar
    rr(d, [tx0, ty0, tx1, ty0+46], 14, fill=(13,20,32))
    d.rectangle([tx0, ty0+30, tx1, ty0+46], fill=(13,20,32))
    for i,c in enumerate([(255,95,87),(254,188,46),(40,200,64)]):
        d.ellipse([tx0+18+i*20, ty0+17, tx0+29+i*20, ty0+28], fill=c)
    rr(d, [tx0+90, ty0+12, tx0+430, ty0+34], 7, fill=(10,14,20), outline=LINE)
    d.text((tx0+104, ty0+23), "trellis-everywhere · 100% offline", font=SANS(13), fill=DIM, anchor="lm")
    d.line([tx0, ty0+46, tx1, ty0+46], fill=LINE, width=1)
    # body
    d.text((tx0+30, ty0+90), "▸ The capital of France is", font=MONO(19), fill=BLUE, anchor="lm")
    # generated text, word-wrapped
    shown = GEN[:nchars]
    words = shown
    fnt = MONO(19); maxw = tx1-tx0-60; x=tx0+30; y=ty0+130
    line=""
    def tw(s): return d.textlength(s, font=fnt)
    for ch in shown:
        if tw(line+ch) > maxw and ch==" ":
            d.text((tx0+30, y), line, font=fnt, fill=INK, anchor="lm"); y+=30; line=""
        else:
            line+=ch
    d.text((tx0+30, y), line, font=fnt, fill=INK, anchor="lm")
    # cursor
    if blink:
        cx = tx0+30+tw(line)+3
        d.rectangle([cx, y-10, cx+8, y+10], fill=ACC)
    # VRAM meter
    my = ty0+th-40
    mx0=tx0+30; mx1=tx1-190
    rr(d, [mx0, my, mx1, my+16], 8, fill=(10,14,20), outline=LINE)
    frac = 0.15 + 0.70*(nchars/len(GEN))    # 0.6GB -> 3.4GB feel
    vram = 0.6 + 2.8*(nchars/len(GEN))
    fw = int((mx1-mx0)*min(1,vram/4.0))
    if fw>4: rr(d, [mx0, my, mx0+fw, my+16], 8, fill=ACC)
    d.text((mx1+16, my+8), f"VRAM {vram:.1f} / 4.0 GB · fits", font=SANS(14), fill=ACC2, anchor="lm")
    ctext(d, W//2, 500, "Qwen3-8B · 36 layers · ~3 bits/weight · verified on an RTX 3050",
          SANS(15), DIM)
    brand(d)
    return fade(img, alpha)

# ---------------- assemble ----------------
def main():
    frames=[]; durs=[]
    FPS_MS=66  # ~15fps
    def add(img, ms=FPS_MS): frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=256)); durs.append(ms)

    # S1: fade in (6), hold (26), fade out (6)
    for i in range(6): add(scene1(0, (i+1)/6))
    for i in range(30): add(scene1(0, 1.0))
    for i in range(6): add(scene1(0, 1-(i+1)/6))
    # S2: fade in, animate pipeline, fade out
    for i in range(6): add(scene2(0.02, (i+1)/6))
    for i in range(46):
        add(scene2(0.02 + 0.96*i/45, 1.0))
    for i in range(10): add(scene2(1.0, 1.0))
    for i in range(6): add(scene2(1.0, 1-(i+1)/6))
    # S3: fade in, type tokens, hold, fade out
    for i in range(6): add(scene3(0, (i+1)/6, True))
    N=len(GEN)
    for i in range(N):
        blink = (i//3)%2==0
        add(scene3(i+1, 1.0, True))
    for i in range(24): add(scene3(N, 1.0, (i//5)%2==0))
    for i in range(6): add(scene3(N, 1-(i+1)/6, False))

    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durs,
                   loop=0, optimize=True, disposal=2)
    print(f"wrote {OUT}  ({len(frames)} frames, ~{sum(durs)/1000:.1f}s)")

if __name__=="__main__":
    main()
