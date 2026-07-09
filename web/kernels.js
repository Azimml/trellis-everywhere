// WGSL kernels for a Llama-style forward pass. f16 weights are read as packed
// u32 pairs via unpack2x16float. Each kernel is verified against refs.json.
// All activations are f32 buffers; weights are f16 (packed u32).

// matmul: y[row] = sum_col W[row,col]*x[col] + (bias). W is [n_out,n_in] f16.
// x: [n_in] f32. y: [n_out] f32. One invocation per output row.
export const MATMUL = /* wgsl */`
struct Dims { n_out:u32, n_in:u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage,read> W: array<u32>;   // f16 pairs
@group(0) @binding(2) var<storage,read> x: array<f32>;
@group(0) @binding(3) var<storage,read_write> y: array<f32>;
fn wf16(i:u32)->f32{ let p=unpack2x16float(W[i>>1u]); return select(p.x,p.y,(i&1u)==1u); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let row=gid.x; if(row>=d.n_out){return;}
  var acc=0.0; let base=row*d.n_in;
  for(var c=0u;c<d.n_in;c=c+1u){ acc=acc+wf16(base+c)*x[c]; }
  y[row]=acc;
}`;

// rmsnorm: y = x / sqrt(mean(x^2)+eps) * weight   (per row of length H)
export const RMSNORM = /* wgsl */`
struct P { H:u32, eps:f32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read> w:array<u32>;    // f16 weight
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
fn wf16(i:u32)->f32{ let q=unpack2x16float(w[i>>1u]); return select(q.x,q.y,(i&1u)==1u); }
@compute @workgroup_size(1)
fn main(){
  var ss=0.0; for(var i=0u;i<p.H;i=i+1u){ ss=ss+x[i]*x[i]; }
  let inv=1.0/sqrt(ss/f32(p.H)+p.eps);
  for(var i=0u;i<p.H;i=i+1u){ y[i]=x[i]*inv*wf16(i); }
}`;

// rope: apply rotary embedding to a [n_heads, head_dim] q or k at position pos.
// non-interleaved (Llama "default"): pairs are (i, i+hd/2).
export const ROPE = /* wgsl */`
struct P { n_heads:u32, head_dim:u32, pos:u32, theta:f32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read_write> x:array<f32>;   // [n_heads*head_dim]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let hd2=p.head_dim/2u;
  let idx=gid.x; if(idx>=p.n_heads*hd2){return;}
  let h=idx/hd2; let i=idx%hd2;
  let base=h*p.head_dim;
  let freq=1.0/pow(p.theta, f32(2u*i)/f32(p.head_dim));
  let ang=f32(p.pos)*freq; let c=cos(ang); let s=sin(ang);
  let a=x[base+i]; let b=x[base+i+hd2];
  x[base+i]=a*c-b*s;
  x[base+i+hd2]=b*c+a*s;
}`;

// attention (single new token vs KV cache), GQA. Writes attn output [n_heads*head_dim].
// q:[n_heads*hd], Kc/Vc: [seqlen, n_kv*hd] cache (row-major), one wg per query head.
export const ATTN = /* wgsl */`
struct P { n_heads:u32, n_kv:u32, head_dim:u32, seqlen:u32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> q:array<f32>;
@group(0) @binding(2) var<storage,read> Kc:array<f32>;
@group(0) @binding(3) var<storage,read> Vc:array<f32>;
@group(0) @binding(4) var<storage,read_write> o:array<f32>;
var<workgroup> scores:array<f32,4096>;
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid:vec3<u32>){
  let h=wid.x;                      // query head
  let g=p.n_heads/p.n_kv;           // heads per kv group
  let kvh=h/g;                      // which kv head
  let hd=p.head_dim;
  let scale=1.0/sqrt(f32(hd));
  var mx=-1e30;
  for(var t=0u;t<p.seqlen;t=t+1u){
    var dot=0.0;
    for(var i=0u;i<hd;i=i+1u){ dot=dot+q[h*hd+i]*Kc[t*p.n_kv*hd+kvh*hd+i]; }
    dot=dot*scale; scores[t]=dot; mx=max(mx,dot);
  }
  var sum=0.0;
  for(var t=0u;t<p.seqlen;t=t+1u){ scores[t]=exp(scores[t]-mx); sum=sum+scores[t]; }
  for(var i=0u;i<hd;i=i+1u){
    var acc=0.0;
    for(var t=0u;t<p.seqlen;t=t+1u){ acc=acc+scores[t]*Vc[t*p.n_kv*hd+kvh*hd+i]; }
    o[h*hd+i]=acc/sum;
  }
}`;

// swiglu: out = (silu(gate) * up)   elementwise, length = intermediate
export const SWIGLU = /* wgsl */`
struct P { n:u32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> gate:array<f32>;
@group(0) @binding(2) var<storage,read> up:array<f32>;
@group(0) @binding(3) var<storage,read_write> o:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x; if(i>=p.n){return;}
  let g=gate[i]; let silu=g/(1.0+exp(-g));
  o[i]=silu*up[i];
}`;

// decode-matmul: y[row] = sum_col decode(packed, row, col) * x[col].
// Decodes 3-bit trellis weights in z-space (verified rel_err 3.3e-7 on GPU).
// The IP restore is folded into x (before) and y (after) on the JS/host side.
export const DECODE_MATMUL = /* wgsl */`
struct D { n_out:u32, k_in:u32, K:u32, T:u32, nwords:u32, tiles_in:u32 };
@group(0) @binding(0) var<uniform> d:D;
@group(0) @binding(1) var<storage,read> packed:array<u32>;
@group(0) @binding(2) var<storage,read> x:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
fn f16b(h:u32)->f32{ let s=select(1.0,-1.0,(h&0x8000u)!=0u); let e=i32((h>>10u)&0x1Fu)-15;
  let m=1.0+f32(h&0x3FFu)/1024.0; return s*m*exp2(f32(e)); }
fn mcg(st:u32)->f32{ let v=st*0xCBAC1FEDu; let r=(v&0x8FFF8FFFu)^0x3B603B60u;
  return f16b(r>>16u)+f16b(r&0xFFFFu); }
fn wz(row:u32,col:u32)->f32{
  let tile=(row/16u)*d.tiles_in+(col/16u);
  let t=(row%16u)*16u+(col%16u); let total=d.T*d.K;
  let base=tile*d.nwords; let bp=((d.T-1u-t)*d.K)%total;
  let w0=packed[base+(bp>>5u)]; let w1=packed[base+(((bp>>5u)+1u)%d.nwords)];
  let sh=bp&31u; let win=select((w0>>sh)|(w1<<(32u-sh)),w0,sh==0u);
  return mcg(win&0xFFFFu);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
  let row=g.x; if(row>=d.n_out){return;}
  var acc=0.0; for(var c=0u;c<d.k_in;c=c+1u){ acc=acc+wz(row,c)*x[c]; }
  y[row]=acc;
}`;

// block-diagonal Hadamard (normalized) over vector of length n, block B (pow2).
// One invocation per element; recomputes its block via the fast WHT sign pattern.
export const BLOCK_HAD = /* wgsl */`
struct P { n:u32, B:u32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
fn popcnt(v:u32)->u32{ var c=v; c=c-((c>>1u)&0x55555555u); c=(c&0x33333333u)+((c>>2u)&0x33333333u);
  c=(c+(c>>4u))&0x0F0F0F0Fu; return (c*0x01010101u)>>24u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
  let i=g.x; if(i>=p.n){return;}
  let blk=i/p.B; let r=i%p.B; let base=blk*p.B;
  var acc=0.0;
  for(var c=0u;c<p.B;c=c+1u){
    let sign=select(1.0,-1.0,(popcnt(r&c)&1u)==1u);
    acc=acc+sign*x[base+c];
  }
  y[i]=acc/sqrt(f32(p.B));
}`;

// elementwise multiply: y = a * b   (for per-channel scales / sign vectors)
export const MUL = /* wgsl */`
struct P { n:u32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> a:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3<u32>){ let i=g.x; if(i>=p.n){return;} y[i]=a[i]*b[i]; }`;

// qknorm: Qwen3 per-head RMSNorm for q/k, applied after projection, before RoPE.
// x: [n_heads*head_dim] f32, normalized in place per head over head_dim.
// w: f16 (packed u32 pairs) of length head_dim, SHARED across heads.
// One workgroup per head: dispatch_workgroups(n_heads).
export const QKNORM = /* wgsl */`
struct P { n_heads:u32, head_dim:u32, eps:f32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read_write> x:array<f32>;
@group(0) @binding(2) var<storage,read> w:array<u32>;    // f16 weight pairs
fn wf16(i:u32)->f32{ let q=unpack2x16float(w[i>>1u]); return select(q.x,q.y,(i&1u)==1u); }
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid:vec3<u32>){
  let h=wid.x; if(h>=p.n_heads){return;}
  let base=h*p.head_dim;
  var ss=0.0;
  for(var i=0u;i<p.head_dim;i=i+1u){ let v=x[base+i]; ss=ss+v*v; }
  let inv=1.0/sqrt(ss/f32(p.head_dim)+p.eps);
  for(var i=0u;i<p.head_dim;i=i+1u){ x[base+i]=x[base+i]*inv*wf16(i); }
}`;

// add (residual): a += b
export const ADD = /* wgsl */`
struct P { n:u32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read_write> a:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x; if(i>=p.n){return;} a[i]=a[i]+b[i];
}`;

// decode-embed: embedding-row lookup from packed K-bit trellis weights WITH the
// full IP restore. Returns x[j] = W_hat[r,j] for token row r, where
// W_hat = restore(zq): zq*scale -> block-Had(in) -> *in_s -> block-Had(out)
// -> *out_s -> *sv(row)*su(col). The out-dim Hadamard mixes the blkV rows of
// the row-block containing r, so we decode all blkV rows but only emit row r's mix.
// Host prefolds isc[j] = in_s[j]*scale (like packed.js folds scale into in-scales).
// Math (per column j, jb = block start of j, rb = blkV*(r/blkV), rr = r%blkV):
//   x[j] = su[j]*sv[r]*osc[r]*isc[j] / sqrt(blkV*blkH)
//          * sum_l signIn(j%blkH,l) * [ sum_c signOut(rr,c) * zq[rb+c, jb+l] ]
// (in/out Hadamard sums commute, so each zq is decoded exactly ONCE: blkV*H decodes.)
// Dispatch H/blkH workgroups of 128 threads; each workgroup owns one in-dim block.
export const DECODE_EMBED = /* wgsl */`
struct D { r:u32, H:u32, vocab:u32, K:u32, T:u32, nwords:u32, tiles_in:u32, blkH:u32, blkV:u32 };
@group(0) @binding(0) var<uniform> d:D;
@group(0) @binding(1) var<storage,read> packed:array<u32>;
@group(0) @binding(2) var<storage,read> su:array<f32>;    // [H] in-dim signs
@group(0) @binding(3) var<storage,read> isc:array<f32>;   // [H] in_s*scale prefolded
@group(0) @binding(4) var<storage,read> osc:array<f32>;   // [vocab] out_s
@group(0) @binding(5) var<storage,read> sv:array<f32>;    // [vocab] out-dim signs
@group(0) @binding(6) var<storage,read_write> x:array<f32>; // [H] output
fn f16b(h:u32)->f32{ let s=select(1.0,-1.0,(h&0x8000u)!=0u); let e=i32((h>>10u)&0x1Fu)-15;
  let m=1.0+f32(h&0x3FFu)/1024.0; return s*m*exp2(f32(e)); }
fn mcg(st:u32)->f32{ let v=st*0xCBAC1FEDu; let r=(v&0x8FFF8FFFu)^0x3B603B60u;
  return f16b(r>>16u)+f16b(r&0xFFFFu); }
fn wz(row:u32,col:u32)->f32{
  let tile=(row/16u)*d.tiles_in+(col/16u);
  let t=(row%16u)*16u+(col%16u); let total=d.T*d.K;
  let base=tile*d.nwords; let bp=((d.T-1u-t)*d.K)%total;
  let w0=packed[base+(bp>>5u)]; let w1=packed[base+(((bp>>5u)+1u)%d.nwords)];
  let sh=bp&31u; let win=select((w0>>sh)|(w1<<(32u-sh)),w0,sh==0u);
  return mcg(win&0xFFFFu);
}
fn popcnt(v:u32)->u32{ var c=v; c=c-((c>>1u)&0x55555555u); c=(c&0x33333333u)+((c>>2u)&0x33333333u);
  c=(c+(c>>4u))&0x0F0F0F0Fu; return (c*0x01010101u)>>24u; }
var<workgroup> A:array<f32,128>;   // out-mixed zq for this in-block (blkH<=128)
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let l=lid.x; let jb=wid.x*d.blkH;
  let rb=(d.r/d.blkV)*d.blkV; let rr=d.r%d.blkV;
  if(l<d.blkH){
    var a=0.0;
    for(var c=0u;c<d.blkV;c=c+1u){
      let s=select(1.0,-1.0,(popcnt(rr&c)&1u)==1u);
      a=a+s*wz(rb+c, jb+l);
    }
    A[l]=a;
  }
  workgroupBarrier();
  if(l<d.blkH){
    var acc=0.0;
    for(var m=0u;m<d.blkH;m=m+1u){
      let s=select(1.0,-1.0,(popcnt(l&m)&1u)==1u);
      acc=acc+s*A[m];
    }
    let j=jb+l;
    x[j]=acc*isc[j]*su[j]*sv[d.r]*osc[d.r]/sqrt(f32(d.blkV)*f32(d.blkH));
  }
}`;
