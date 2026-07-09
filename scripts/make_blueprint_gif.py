#!/usr/bin/env python
"""Animated 'system-design blueprint' of Trellis-WebGPU, rendered directly to a GIF
with Pillow (no browser). ByteByteGo / 'How LLM works end-to-end' style: titled
sections, labeled sub-boxes, arrows — with a glow that flows through the pipeline
stage by stage, and the final tokens + VRAM meter animating.

Layout (top -> bottom):
  Title
  [1] WEIGHTS OFFLINE (once):  fp16 weights -> incoherence proc -> Viterbi encode -> 3-bit trellis codes
  [2] IN THE BROWSER (per token): 3-bit codes -> trellis decode (random access) -> z-space matmul
                                   -> fold IP transform -> transformer layer x36 -> logits -> token
  Footer strip: model card (Qwen3-8B, 1.15x fp16, fits 4GB RTX 3050) + VRAM meter

Output: web/blueprint.gif  (1280x900)
"""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 900
OUT = os.path.join(os.path.dirname(__file__), "..", "web", "blueprint.gif")
FD = "/usr/share/fonts/truetype/dejavu"

BG=(11,16,24); BG2=(8,11,17); CARD=(18,26,38); CARD2=(15,22,33); LINE=(38,52,74)
INK=(232,238,245); DIM=(140,156,178); MUT=(96,110,130)
ACC=(63,185,80); ACC2=(126,231,135); BLUE=(88,166,255); AMBER=(227,179,65)
VIOLET=(188,140,255); RED=(248,110,102)

def f(name,s): return ImageFont.truetype(os.path.join(FD,name), s)
SANS=lambda s: f("DejaVuSans.ttf",s); BOLD=lambda s: f("DejaVuSans-Bold.ttf",s)
MONO=lambda s: f("DejaVuSansMono.ttf",s); MONOB=lambda s: f("DejaVuSansMono-Bold.ttf",s)

def bg():
    img=Image.new("RGB",(W,H),BG2); px=img.load()
    for y in range(H):
        t=y/H
        c=tuple(int(BG[i]*(1-t)+BG2[i]*t) for i in range(3))
        for x in range(W): px[x,y]=c
    return img

def rr(d,box,rad,fill=None,outline=None,width=1):
    d.rounded_rectangle(box,radius=rad,fill=fill,outline=outline,width=width)
def ct(d,cx,y,t,fnt,fill,anchor="mm"): d.text((cx,y),t,font=fnt,fill=fill,anchor=anchor)

def mix(a,b,t): return tuple(int(a[i]*(1-t)+b[i]*t) for i in range(3))

def box(d, x, y, w, h, title, sub, glow=0.0, accent=BLUE, icon=None):
    """A labeled sub-box. glow 0..1 lights the border/accent."""
    oc = mix(LINE, accent, glow)
    fillc = mix(CARD, mix(CARD, accent, 0.10), glow)
    rr(d,[x,y,x+w,y+h],10, fill=fillc, outline=oc, width=1+int(2*glow))
    if glow>0.05:
        rr(d,[x-1,y-1,x+w+1,y+h+1],11, outline=mix(BG,accent,glow*0.8), width=1)
    tx=x+16; cy=y+h//2
    if icon:
        draw_icon(d, x+26, cy, icon, mix(DIM,accent,glow), 12)
        tx=x+52
    d.text((tx, cy-11), title, font=BOLD(15), fill=mix(INK,accent,glow*0.5), anchor="lm")
    if sub: d.text((tx, cy+11), sub, font=SANS(11.5), fill=DIM, anchor="lm")

def draw_icon(d,cx,cy,kind,col,s):
    if kind=="bits":
        fo=MONOB(9)
        for i,r in enumerate(["101","011"]): d.text((cx,cy-6+i*11),r,font=fo,fill=col,anchor="mm")
    elif kind=="lattice":
        pts=[(-s,-s*.6),(-s,s*.6),(0,-s*.6),(0,s*.6),(s,-s*.6),(s,s*.6)]
        P=[(cx+x,cy+y) for x,y in pts]
        for a,b in [(0,2),(1,2),(2,4),(3,4),(0,3),(1,3)]: d.line([P[a],P[b]],fill=col,width=2)
        for p in P: d.ellipse([p[0]-2,p[1]-2,p[0]+2,p[1]+2],fill=col)
    elif kind=="bolt":
        d.polygon([(cx+3,cy-s),(cx-s*.6,cy+2),(cx-1,cy+2),(cx-3,cy+s),(cx+s*.6,cy-2),(cx+1,cy-2)],fill=col)
    elif kind=="fold":
        d.arc([cx-s,cy-s,cx+s,cy+s],30,330,fill=col,width=2)
        d.line([cx,cy-s,cx+s*.7,cy-s*.4],fill=col,width=2)
    elif kind=="layers":
        for i in range(3):
            yy=cy-6+i*6; d.line([cx-s,yy,cx+s,yy],fill=col,width=2)
    elif kind=="chat":
        rr(d,[cx-s,cy-s*.7,cx+s,cy+s*.3],4,outline=col,width=2)
        for i in range(-1,2): d.ellipse([cx+i*6-2,cy-s*.2-2,cx+i*6+2,cy-s*.2+2],fill=col)
    elif kind=="hash":
        d.rounded_rectangle([cx-s,cy-s*.7,cx+s,cy+s*.7],3,outline=col,width=2)
    elif kind=="wave":
        d.arc([cx-s,cy-s*.4,cx,cy+s*.6],180,360,fill=col,width=2)
        d.arc([cx,cy-s*.6,cx+s,cy+s*.4],0,180,fill=col,width=2)

def arrow(d, x0, y0, x1, y1, glow=0.0, accent=ACC2):
    c = mix(LINE, accent, glow)
    d.line([x0,y0,x1,y1], fill=c, width=2+int(glow))
    # arrowhead (vertical down assumed)
    if abs(y1-y0)>abs(x1-x0):
        d.polygon([(x1,y1),(x1-5,y1-8),(x1+5,y1-8)],fill=c)
    else:
        d.polygon([(x1,y1),(x1-8,y1-5),(x1-8,y1+5)],fill=c)
    if glow>0.05:  # traveling pulse
        t=glow
        px=x0+(x1-x0)*t; py=y0+(y1-y0)*t
        d.ellipse([px-4,py-4,px+4,py+4], fill=accent)

def section_label(d, x, y, n, text, col):
    rr(d,[x,y-13,x+26,y+13],6, fill=mix(BG,col,0.25), outline=col, width=1)
    ct(d, x+13, y, str(n), BOLD(15), col)
    d.text((x+38, y), text, font=BOLD(17), fill=INK, anchor="lm")

GEN = " Paris. The capital of Italy is Rome. The capital of Germany is Berlin."

def frame(step, sub, nchars):
    """step: which stage is glowing (float, for the browser row). sub: 0..1 within.
    nchars: chars of generated text shown."""
    img=bg(); d=ImageDraw.Draw(img)
    # ---- title ----
    ct(d, W//2, 46, "Trellis-WebGPU — a 3-bit LLM, end-to-end, outside CUDA", BOLD(28), INK)
    ct(d, W//2, 78, "SOTA trellis-coded quantization (QTIP/EXL3 quality) running its full forward pass in a browser on WebGPU",
       SANS(14), DIM)

    # ============ SECTION 1: OFFLINE ============
    sx=60; sw=W-120
    section_label(d, sx, 128, 1, "QUANTIZE — once, offline", AMBER)
    d.text((sx+sw, 128), "PyTorch + Triton", font=SANS(12), fill=MUT, anchor="rm")
    y1=155; bh=64
    n1=4; gap=24; bw=(sw-(n1-1)*gap)//n1
    S1=[("fp16 weights","the original model","hash",VIOLET),
        ("incoherence proc","±sign · Hadamard · RMS scales","wave",VIOLET),
        ("Viterbi encode","tail-biting trellis, Triton","lattice",AMBER),
        ("3-bit codes","~3 bits / weight","bits",AMBER)]
    off_glow = 1.0 if step<0 else 0.0   # section 1 lights only in the intro sweep
    for i,(t,s,ic,acc) in enumerate(S1):
        x=sx+i*(bw+gap)
        g = 1.0 if (step<0 and (-step)>i) else (0.25 if step>=0 else 0.0)
        box(d,x,y1,bw,bh,t,s, glow=g, accent=acc, icon=ic)
        if i<n1-1:
            arrow(d, x+bw+2, y1+bh//2, x+bw+gap-2, y1+bh//2,
                  glow=(1.0 if (step<0 and (-step)>i+0.5) else 0.0), accent=AMBER)

    # ============ SECTION 2: IN THE BROWSER ============
    section_label(d, sx, 268, 2, "RUN — in the browser, per token", ACC)
    d.text((sx+sw, 268), "WebGPU / WGSL · no server · offline", font=SANS(12), fill=MUT, anchor="rm")
    # row A: decode path (3 boxes)
    y2=298
    S2a=[("trellis decode","random-access, not sequential","lattice",ACC),
         ("z-space matmul","packed codes × activations","bolt",ACC),
         ("fold IP transform","= full W·x, verified 4.9e-7","fold",ACC)]
    n2=3; bw2=(sw-(n2-1)*gap)//n2
    for i,(t,s,ic,acc) in enumerate(S2a):
        x=sx+i*(bw2+gap)
        g = clamp(step - i)
        box(d,x,y2,bw2,bh,t,s, glow=g, accent=acc, icon=ic)
        if i<n2-1:
            arrow(d, x+bw2+2, y2+bh//2, x+bw2+gap-2, y2+bh//2, glow=clamp(step-i-0.5), accent=ACC2)
    # down arrow into transformer stack
    arrow(d, sx+sw//2, y2+bh+2, sx+sw//2, y2+bh+26, glow=clamp(step-2.5), accent=ACC2)

    # row B: transformer stack (one wide box representing x36 layers)
    y3=y2+bh+30; bh3=70
    g3=clamp(step-3)
    box(d, sx, y3, sw, bh3, "transformer layer  ×36",
        "RMSNorm · GQA-attention + RoPE + QK-norm · SwiGLU · residual — every matmul is a trellis decode",
        glow=g3, accent=ACC, icon="layers")
    # tiny layer ticks inside
    for i in range(36):
        xx=sx+40 + i*((sw-80)/35)
        on = g3>0.1 and (i/35) <= (sub if step>=3 and step<4 else 1.0)
        d.line([xx, y3+bh3-12, xx, y3+bh3-6], fill=(mix(LINE,ACC,0.9) if on else LINE), width=2)
    arrow(d, sx+sw//2, y3+bh3+2, sx+sw//2, y3+bh3+24, glow=clamp(step-4), accent=ACC2)

    # row C: logits -> token
    y4=y3+bh3+28
    S2c=[("logits","152k-way softmax","wave",BLUE),
         ("sample","greedy / top-k","hash",BLUE),
         ("token","appended, repeat","chat",ACC)]
    for i,(t,s,ic,acc) in enumerate(S2c):
        x=sx+i*(bw2+gap)
        g=clamp(step-4-i*0.5)
        box(d,x,y4,bw2,bh,t,s, glow=g, accent=acc, icon=ic)
        if i<2: arrow(d, x+bw2+2, y4+bh//2, x+bw2+gap-2, y4+bh//2, glow=clamp(step-4-i*0.5-0.25), accent=ACC2)

    # ============ OUTPUT STRIP ============
    oy=y4+bh+26
    rr(d,[sx,oy,sx+sw,oy+120],12, fill=CARD2, outline=LINE, width=1)
    # left: the running output
    d.text((sx+22, oy+26), "▸ The capital of France is", font=MONO(15), fill=BLUE, anchor="lm")
    shown=GEN[:nchars]
    # wrap
    fx=MONO(15); maxw=sw-260; x=sx+22; yy=oy+54; line=""
    def tw(s): return d.textlength(s,font=fx)
    for ch in shown:
        if tw(line+ch)>maxw and ch==" ":
            d.text((sx+22,yy),line,font=fx,fill=INK,anchor="lm"); yy+=24; line=""
        else: line+=ch
    d.text((sx+22,yy),line,font=fx,fill=INK,anchor="lm")
    if nchars<len(GEN) and step>=4:
        cx=sx+22+tw(line)+3; d.rectangle([cx,yy-8,cx+7,yy+8],fill=ACC)
    # right: stat chips + VRAM meter
    rx=sx+sw-230
    chips=[("Qwen3-8B · 3-bit",ACC2),("1.15× fp16 ppl",AMBER),("fits 4 GB · RTX 3050",BLUE)]
    for i,(txt,c) in enumerate(chips):
        cyy=oy+22+i*26
        rr(d,[rx,cyy-11,sx+sw-20,cyy+11],10, fill=mix(CARD2,c,0.10), outline=mix(LINE,c,0.5), width=1)
        d.text((rx+12,cyy),txt,font=BOLD(12),fill=mix(INK,c,0.4),anchor="lm")
    # vram meter
    my=oy+100; mx0=rx; mx1=sx+sw-20
    rr(d,[mx0,my-6,mx1,my+6],7, fill=BG, outline=LINE)
    vram=0.6+2.8*(nchars/len(GEN)) if step>=4 else (0.6 if step>=0 else 0.0)
    fw=int((mx1-mx0)*min(1,vram/4.0))
    if fw>4: rr(d,[mx0,my-6,mx0+fw,my+6],7, fill=ACC)
    d.text(((mx0+mx1)//2,my+20),f"VRAM {vram:.1f} / 4.0 GB",font=SANS(11),fill=DIM,anchor="mm")

    # footer brand
    d.text((sx, H-24), "github.com/Azimml/trellis-webgpu", font=SANS(13), fill=MUT, anchor="lm")
    d.text((sx+sw, H-24), "verified end-to-end · WGSL kernels match PyTorch to <1e-6", font=SANS(12), fill=MUT, anchor="rm")
    return img

def clamp(v): return max(0.0,min(1.0,v))

def main():
    frames=[]; durs=[]
    def add(img,ms=70): frames.append(img.convert("P",palette=Image.ADAPTIVE,colors=256)); durs.append(ms)
    # intro: sweep section 1 (offline) lighting boxes  (step encoded negative)
    for k in range(1,6): add(frame(-(k*0.9), 0, 0), 120)
    add(frame(-10, 0, 0), 500)  # section1 fully lit, hold
    # main: sweep the browser pipeline; step 0..5 glides
    STEPS=60
    for i in range(STEPS):
        step = i/STEPS*5.2
        sub = (step-3) if 3<=step<4 else 0
        add(frame(step, clamp(sub), 0), 70)
    # generation: type tokens while step held at 4.5
    N=len(GEN)
    for i in range(N):
        add(frame(4.8, 1, i+1), 90)
    # hold final
    for i in range(22): add(frame(4.8,1,N), 90)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durs,
                   loop=0, optimize=True, disposal=2)
    print(f"wrote {OUT} ({len(frames)} frames, ~{sum(durs)/1000:.1f}s)")

if __name__=="__main__":
    main()
