"""Convert RAVU-Zoom (bjin/mpv-prescalers, weights LGPL-3.0) to ONNX for --vpp-onnx.

RAVU-Zoom is the arbitrary-ratio RAVU. Per output pixel it maps back to a
continuous source position, snaps to the integer source pixel, gathers an n x n
window (n = radius*2), classifies it (structure tensor -> angle/strength/coherence)
and applies a filter whose coefficients are BILINEARLY interpolated from a
lut_size x lut_size grid of trained filters indexed by the continuous sub-pixel
offset. ONNX needs a fixed scale, so we bake a fixed INTEGER ratio p: at ratio p
the sub-pixel offsets become a constant set of p*p phases, so the lut_size grid is
pre-interpolated at those phases offline and the model reduces to "per phase:
classify the (phase-shifted) window, apply the per-class per-phase filter", then a
pixel-shuffle. Self-verifies an independent numpy reference vs the exported ONNX.

Builds both PLAIN and ANTI-RINGING variants.
"""
import argparse, sys, math, numpy as np, torch, torch.nn as nn
import onnxruntime as ort
from onnx_export_common import export_onnx

EPS = 1.192092896e-7

def load_weights(radius, upstream):
    ns = {}
    exec(open(f"{upstream}/weights/ravu-zoom_weights-r{radius}.py").read(), ns)
    p = {k: ns[k] for k in ['radius','lut_size','gradient_radius','quant_angle','quant_strength',
                            'quant_coherence','min_strength','min_coherence','gaussian']}
    p['model_weights'] = np.array(ns['model_weights'], dtype=np.float64)      # (qa,qs,qc, L*L*ksz)
    p['model_weights_ar'] = np.array(ns['model_weights_ar'], dtype=np.float64)  # (qa,qs,qc, L*L*8)
    return p

AR_STRENGTH = 0.8

def phase_params(m, p):
    t = (m + 0.5)/p - 0.5
    c = math.floor(t)
    return c, t - c

def grid_bilinear(grid, gx, gy, L):
    x0 = int(math.floor(gx)); x1 = min(x0+1, L-1); fx = gx - x0
    y0 = int(math.floor(gy)); y1 = min(y0+1, L-1); fy = gy - y0
    x0 = max(0,min(x0,L-1)); x1 = max(0,min(x1,L-1)); y0 = max(0,min(y0,L-1)); y1 = max(0,min(y1,L-1))
    g0 = grid[:, y0, x0]*(1-fx) + grid[:, y0, x1]*fx
    g1 = grid[:, y1, x0]*(1-fx) + grid[:, y1, x1]*fx
    return g0*(1-fy) + g1*fy

def phase_filters(p_w, ratio):
    r = p_w['radius']; L = p_w['lut_size']; n = 2*r; ksz = r*r*2
    qa,qs,qc = p_w['quant_angle'],p_w['quant_strength'],p_w['quant_coherence']
    ncls = qa*qs*qc
    grid = p_w['model_weights'].reshape(ncls, L*L, ksz).reshape(ncls, L, L, ksz)
    out = {}
    for mr in range(ratio):
        cr, sy = phase_params(mr, ratio)
        for mc in range(ratio):
            cc, sx = phase_params(mc, ratio)
            W0 = grid_bilinear(grid, sx*(L-1), sy*(L-1), L)
            W1 = grid_bilinear(grid, (1-sx)*(L-1), (1-sy)*(L-1), L)
            full = np.zeros((ncls, n*n))
            half = n*n//2
            for k in range(half):       full[:, k]          = W0[:, k]
            for k in range(half, n*n):  full[:, k]          = W1[:, n*n-1-k]
            out[(mr,mc)] = (full, cr, cc)
    return out

def ar_indices(r):
    n = 2*r
    idx = []
    for x in range(r-2, r+2):
        for y in range(r-2, r+2):
            idx.append(x*n + y)
    return idx

def phase_filters_ar(p_w, ratio):
    L = p_w['lut_size']; ksz = 8
    qa,qs,qc = p_w['quant_angle'],p_w['quant_strength'],p_w['quant_coherence']
    ncls = qa*qs*qc
    grid = p_w['model_weights_ar'].reshape(ncls, L*L, ksz).reshape(ncls, L, L, ksz)
    out = {}
    for mr in range(ratio):
        _, sy = phase_params(mr, ratio)
        for mc in range(ratio):
            _, sx = phase_params(mc, ratio)
            W0 = grid_bilinear(grid, sx*(L-1), sy*(L-1), L)
            W1 = grid_bilinear(grid, (1-sx)*(L-1), (1-sy)*(L-1), L)
            full = np.zeros((ncls, 16))
            for k in range(8):      full[:, k]      = W0[:, k]
            for k in range(8, 16):  full[:, k]      = W1[:, 15-k]
            out[(mr,mc)] = full
    return out

def ar_envelope_np(ar, Wa, res_raw, strength):
    def p32(z):
        z2=z*z; z4=z2*z2; z8=z4*z4; z16=z8*z8; return z16*z16
    pa = 0.1 + ar; Ma = np.maximum(pa.max(-1, keepdims=True), 1e-12); ua = p32(pa/Ma)
    hi = (ua*pa*Wa).sum(-1) / (np.where(np.abs((ua*Wa).sum(-1))<1e-20, 1e-20, (ua*Wa).sum(-1))) - 0.1
    pb = 1.1 - ar; Mb = np.maximum(pb.max(-1, keepdims=True), 1e-12); ub = p32(pb/Mb)
    lo = 1.1 - (ub*pb*Wa).sum(-1) / (np.where(np.abs((ub*Wa).sum(-1))<1e-20, 1e-20, (ub*Wa).sum(-1)))
    clamped = np.minimum(np.maximum(res_raw, lo), hi)
    return res_raw*(1-strength) + clamped*strength

# ---------------- classifier (numpy), 4th-order interior diff, n=2r ----------------
def classify(win, p_w):
    n = win.shape[0]
    qa,qs,qc = p_w['quant_angle'],p_w['quant_strength'],p_w['quant_coherence']
    gl = p_w['radius'] - p_w['gradient_radius']; gr = n - gl
    gauss = np.array(p_w['gaussian'])
    def S(i,j): return win[i, j]
    def ndiff(get, t):
        if t == 0:   return get(1) - get(0)
        if t == n-1: return get(n-1) - get(n-2)
        if t == 1 or t == n-2: return (get(t+1) - get(t-1)) / 2.0
        return (-get(t+2) + 8.0*get(t+1) - 8.0*get(t-1) + get(t-2)) / 12.0
    H,W = win.shape[2], win.shape[3]
    a=np.zeros((H,W)); b=np.zeros((H,W)); d=np.zeros((H,W))
    for i in range(gl,gr):
        for j in range(gl,gr):
            gx = ndiff(lambda i2: S(i2,j), i); gy = ndiff(lambda j2: S(i,j2), j)
            gw = gauss[i-gl][j-gl]
            a += gx*gx*gw; b += gx*gy*gw; d += gy*gy*gw
    T=a+d; D=a*d-b*b
    delta=np.sqrt(np.maximum(T*T/4.0-D,0.0)); L1=T/2.0+delta; L2=T/2.0-delta
    sL1=np.sqrt(np.maximum(L1,0)); sL2=np.sqrt(np.maximum(L2,0))
    theta=np.where(np.abs(b)<EPS,0.0,np.mod(np.arctan2(L1-a,b)+math.pi,math.pi))
    lam=sL1; mu=np.where(sL1+sL2<EPS,0.0,(sL1-sL2)/(sL1+sL2+1e-30))
    angle=np.clip(np.floor(theta*qa/math.pi),0,qa-1).astype(np.int64)
    strength=np.zeros((H,W),np.int64)
    for s in p_w['min_strength']: strength += (lam>=s).astype(np.int64)
    coherence=np.zeros((H,W),np.int64)
    for s in p_w['min_coherence']: coherence += (mu>=s).astype(np.int64)
    return (angle*qs+strength)*qc+coherence

def win_phase_np(src, n, r, cr, cc):
    H,W = src.shape; pad = n + 4
    sp = np.pad(src, pad, mode='edge')
    win = np.zeros((n,n,H,W))
    for x in range(n):
        for y in range(n):
            roff = cr + (y-(r-1)); coff = cc + (x-(r-1))
            win[x,y] = sp[pad+roff:pad+roff+H, pad+coff:pad+coff+W]
    return win

def ravu_zoom_ref(img, p_w, ratio, ar=False):
    r = p_w['radius']; n = 2*r; H,W = img.shape
    phf = phase_filters(p_w, ratio)
    phf_ar = phase_filters_ar(p_w, ratio) if ar else None
    ar_idx = ar_indices(r)
    out = np.zeros((ratio*H, ratio*W))
    for (mr,mc),(full,cr,cc) in phf.items():
        win = win_phase_np(img, n, r, cr, cc)
        cls = classify(win, p_w)
        winp = win.reshape(n*n,H,W).transpose(1,2,0)
        res_raw = (full[cls]*winp).sum(-1)
        if ar:
            arw = winp[:,:,ar_idx]
            Wa = phf_ar[(mr,mc)][cls]
            plane = ar_envelope_np(arw, Wa, res_raw, AR_STRENGTH)
        else:
            plane = np.clip(res_raw, 0.0, 1.0)
        out[mr::ratio, mc::ratio] = plane
    return out

# ---------------- torch module ----------------
class RavuZoom(nn.Module):
    def __init__(self, p_w, ratio, ar=False):
        super().__init__()
        self.r = p_w['radius']; self.n = 2*self.r; self.ratio = ratio
        self.ar = ar; self.ar_strength = AR_STRENGTH; self.ar_idx = ar_indices(self.r)
        self.qa,self.qs,self.qc = p_w['quant_angle'],p_w['quant_strength'],p_w['quant_coherence']
        self.gl = self.r - p_w['gradient_radius']; self.gr = self.n - self.gl
        self.register_buffer('gauss', torch.tensor(np.array(p_w['gaussian']), dtype=torch.float32))
        self.register_buffer('min_s', torch.tensor(p_w['min_strength'], dtype=torch.float32))
        self.register_buffer('min_c', torch.tensor(p_w['min_coherence'], dtype=torch.float32))
        phf = phase_filters(p_w, ratio)
        phf_ar = phase_filters_ar(p_w, ratio) if ar else None
        self.phases = []
        for idx,(mr,mc) in enumerate([(a,b) for a in range(ratio) for b in range(ratio)]):
            full,cr,cc = phf[(mr,mc)]
            self.register_buffer(f'ew{idx}', torch.tensor(full, dtype=torch.float32))
            if ar:
                self.register_buffer(f'ewar{idx}', torch.tensor(phf_ar[(mr,mc)], dtype=torch.float32))
            self.phases.append((mr,mc,cr,cc))

    def win_phase(self, x, cr, cc):
        n=self.n; r=self.r; pad=n+4
        xp = torch.nn.functional.pad(x,(pad,pad,pad,pad),mode='replicate')
        _,_,Hp,Wp = xp.shape; H=Hp-2*pad; W=Wp-2*pad
        taps=[]
        for xc in range(n):
            for yr in range(n):
                roff = cr+(yr-(r-1)); coff = cc+(xc-(r-1))
                taps.append(xp[:,:,pad+roff:pad+roff+H, pad+coff:pad+coff+W])
        return torch.cat(taps,dim=1)

    def classify_apply(self, win, ew, ewar=None):
        n=self.n
        def S(i,j): return win[:, (i*n+j):(i*n+j)+1]
        def ndiff(get,t):
            if t==0: return get(1)-get(0)
            if t==n-1: return get(n-1)-get(n-2)
            if t==1 or t==n-2: return (get(t+1)-get(t-1))*0.5
            return (-get(t+2)+8.0*get(t+1)-8.0*get(t-1)+get(t-2))/12.0
        a=b=d=None
        for i in range(self.gl,self.gr):
            for j in range(self.gl,self.gr):
                gx=ndiff(lambda i2:S(i2,j),i); gy=ndiff(lambda j2:S(i,j2),j)
                gw=self.gauss[i-self.gl,j-self.gl]
                a=gx*gx*gw if a is None else a+gx*gx*gw
                b=gx*gy*gw if b is None else b+gx*gy*gw
                d=gy*gy*gw if d is None else d+gy*gy*gw
        T=a+d; D=a*d-b*b
        delta=torch.sqrt(torch.clamp(T*T/4.0-D,min=0.0)); L1=T/2.0+delta; L2=T/2.0-delta
        sL1=torch.sqrt(torch.clamp(L1,min=0)); sL2=torch.sqrt(torch.clamp(L2,min=0))
        theta=torch.where(torch.abs(b)<EPS,torch.zeros_like(b),torch.remainder(torch.atan2(L1-a,b)+math.pi,math.pi))
        lam=sL1; mu=torch.where(sL1+sL2<EPS,torch.zeros_like(b),(sL1-sL2)/(sL1+sL2+1e-30))
        angle=torch.clamp(torch.floor(theta*self.qa/math.pi),0,self.qa-1)
        strength=torch.zeros_like(b)
        for s in self.min_s: strength=strength+(lam>=s).float()
        coherence=torch.zeros_like(b)
        for s in self.min_c: coherence=coherence+(mu>=s).float()
        cls=((angle*self.qs+strength)*self.qc+coherence).long().squeeze(1)
        _,_,H,W=win.shape
        clsf=cls.reshape(-1)
        filt=ew[clsf].reshape(H,W,n*n)
        winp=win.squeeze(0).permute(1,2,0)
        res_raw=(filt*winp).sum(-1)
        if ewar is None:
            return torch.clamp(res_raw,0.0,1.0)[None,None]
        arw=winp[:,:,self.ar_idx]
        Wa=ewar[clsf].reshape(H,W,16)
        def p32(z):
            z2=z*z; z4=z2*z2; z8=z4*z4; z16=z8*z8; return z16*z16
        pa=0.1+arw; Ma=torch.clamp(pa.max(-1,keepdim=True).values,min=1e-12); ua=p32(pa/Ma)
        den_h=(ua*Wa).sum(-1); den_h=torch.where(den_h.abs()<1e-20, torch.full_like(den_h,1e-20), den_h)
        hi=(ua*pa*Wa).sum(-1)/den_h - 0.1
        pb=1.1-arw; Mb=torch.clamp(pb.max(-1,keepdim=True).values,min=1e-12); ub=p32(pb/Mb)
        den_l=(ub*Wa).sum(-1); den_l=torch.where(den_l.abs()<1e-20, torch.full_like(den_l,1e-20), den_l)
        lo=1.1-(ub*pb*Wa).sum(-1)/den_l
        clamped=torch.minimum(torch.maximum(res_raw,lo),hi)
        return (res_raw*(1-self.ar_strength)+clamped*self.ar_strength)[None,None]

    def forward(self, x):
        ratio=self.ratio
        chans=[None]*(ratio*ratio)
        for idx,(mr,mc,cr,cc) in enumerate(self.phases):
            ew=getattr(self,f'ew{idx}')
            ewar=getattr(self,f'ewar{idx}') if self.ar else None
            chans[mr*ratio+mc]=self.classify_apply(self.win_phase(x,cr,cc), ew, ewar)
        stack=torch.cat(chans,dim=1)
        return torch.nn.functional.pixel_shuffle(stack, ratio)

def convert(radius, ratio, ar, upstream, output_dir):
    p_w = load_weights(radius, upstream)
    tag = f"r{radius} {ratio}x{'-ar' if ar else ''}"
    mod = RavuZoom(p_w, ratio, ar=ar).eval()
    rng = np.random.default_rng(0); img = rng.random((36,44)).astype(np.float32)
    with torch.no_grad(): t = mod(torch.tensor(img)[None,None]).numpy()[0,0]
    ref = ravu_zoom_ref(img.astype(np.float64), p_w, ratio, ar=ar)
    print(f"[zoom {tag}] torch-vs-numpyref max diff: {np.abs(t-ref).max():.2e}")
    const = np.full((24,24),0.42,np.float32)
    with torch.no_grad(): co = mod(torch.tensor(const)[None,None]).numpy()[0,0]
    print(f"[zoom {tag}] constant-image max dev from 0.42: {np.abs(co-0.42).max():.2e}")
    suffix = "_ar" if ar else ""
    path = f"{output_dir}/ravu_zoom_{ratio}x_r{radius}{suffix}.onnx"
    export_onnx(mod, torch.tensor(img)[None,None], path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{2:'h',3:'w'}, 'output':{2:'hk',3:'wk'}},
                      do_constant_folding=True)
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    o = s.run(None, {'input': img[None,None]})[0][0,0]
    print(f"[zoom {tag}] ONNX-vs-numpyref max diff: {np.abs(o-ref).max():.2e} | out {s.get_outputs()[0].shape} -> {path}")
    return np.abs(o-ref).max()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("radius", type=int, nargs="?", default=2)
    ap.add_argument("ratio", type=int, nargs="?", default=2)
    ap.add_argument("ar", nargs="?", default="")
    ap.add_argument("--upstream", default="upstream")
    ap.add_argument("--output-dir", default=".")
    a = ap.parse_args()
    convert(a.radius, a.ratio, a.ar in ('ar','1','true'), a.upstream, a.output_dir)
