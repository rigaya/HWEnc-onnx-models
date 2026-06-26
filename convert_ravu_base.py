"""Convert base RAVU (bjin/mpv-prescalers, weights LGPL-3.0) to ONNX for --vpp-onnx.

Base RAVU is a 2x prescaler that fills a doubled grid in three passes (the GLSL
shader's step1/2/3; step4 is pure routing):
  O[2I  ][2J  ] = source
  O[2I+1][2J+1] = "center"  (step1) : from an n x n window of the source
  O[2I  ][2J+1] = "int10"   (step2) : from a 45-deg window of source + centers
  O[2I+1][2J  ] = "int01"   (step3) : from a 45-deg window of source + centers
with n = radius*2. All three passes share one classifier+LUT; only the window
geometry differs. Each pass: gaussian-weighted structure tensor (4th-order
interior differential) -> closed-form 2x2 eigen-analysis -> quantize to
(angle, strength, coherence) -> gather the per-class symmetric linear filter ->
apply -> clamp. Self-verifies an independent numpy reference vs the exported ONNX,
plus constant/ramp sanity checks.
"""
import argparse, sys, math, numpy as np, torch, torch.nn as nn
import onnxruntime as ort

EPS = 1.192092896e-7

def load_weights(radius, upstream):
    ns = {}
    exec(open(f"{upstream}/weights/ravu_weights-r{radius}.py").read(), ns)
    p = {k: ns[k] for k in ['radius','gradient_radius','quant_angle','quant_strength',
                            'quant_coherence','min_strength','min_coherence','gaussian']}
    p['model_weights'] = np.array(ns['model_weights'], dtype=np.float64)   # (qa,qs,qc,n*n)
    return p

def eff_weights(p):
    """symmetric, sum-normalised filter per class: ew[k] = (mw[k]+mw[~k])/(2*sum)."""
    qa,qs,qc = p['quant_angle'],p['quant_strength'],p['quant_coherence']
    n2 = p['model_weights'].shape[-1]
    mw = p['model_weights'].reshape(qa*qs*qc, n2)
    ksum = mw.sum(1, keepdims=True)
    return (mw + mw[:, ::-1]) / (2.0*ksum)        # [cls, n*n]

# ---------------- shared classifier+apply (numpy) ----------------
def classify_apply(win, ew, p):
    """win: [n,n,H,W] indexed [x=col][y=row]. returns predicted plane [H,W]."""
    n = win.shape[0]; c = n//2
    qa,qs,qc = p['quant_angle'],p['quant_strength'],p['quant_coherence']
    gl = p['radius'] - p['gradient_radius']; gr = n - gl
    gauss = np.array(p['gaussian'])
    def S(i,j): return win[i, j]
    def ndiff(get, t):
        if t == 0:   return get(1) - get(0)
        if t == n-1: return get(n-1) - get(n-2)
        if t == 1 or t == n-2: return (get(t+1) - get(t-1)) / 2.0
        return (-get(t+2) + 8.0*get(t+1) - 8.0*get(t-1) + get(t-2)) / 12.0
    H,W = win.shape[2], win.shape[3]
    a = np.zeros((H,W)); b = np.zeros((H,W)); d = np.zeros((H,W))
    for i in range(gl,gr):
        for j in range(gl,gr):
            gx = ndiff(lambda i2: S(i2, j), i)
            gy = ndiff(lambda j2: S(i, j2), j)
            gw = gauss[i-gl][j-gl]
            a += gx*gx*gw; b += gx*gy*gw; d += gy*gy*gw
    T = a + d; D = a*d - b*b
    delta = np.sqrt(np.maximum(T*T/4.0 - D, 0.0))
    L1 = T/2.0 + delta; L2 = T/2.0 - delta
    sL1 = np.sqrt(np.maximum(L1,0)); sL2 = np.sqrt(np.maximum(L2,0))
    theta = np.where(np.abs(b) < EPS, 0.0, np.mod(np.arctan2(L1 - a, b) + math.pi, math.pi))
    lam = sL1
    mu = np.where(sL1 + sL2 < EPS, 0.0, (sL1 - sL2)/(sL1 + sL2 + 1e-30))
    angle = np.clip(np.floor(theta * qa / math.pi), 0, qa-1).astype(np.int64)
    strength = np.zeros((H,W), np.int64)
    for s in p['min_strength']: strength += (lam >= s).astype(np.int64)
    coherence = np.zeros((H,W), np.int64)
    for s in p['min_coherence']: coherence += (mu >= s).astype(np.int64)
    cls = (angle*qs + strength)*qc + coherence
    filt = ew[cls]                                  # [H,W,n*n]
    winp = win.reshape(n*n, H, W).transpose(1,2,0)  # [H,W,n*n], k = x*n+y
    res = (filt*winp).sum(-1)
    return np.clip(res, 0.0, 1.0)

# ---------------- numpy reference ----------------
def win_step1_np(src, n):
    c = n//2 - 1; H,W = src.shape; pad = n
    sp = np.pad(src, pad, mode='edge')
    win = np.zeros((n,n,H,W))
    for x in range(n):
        for y in range(n):
            rs = y - c; cs = x - c
            win[x,y] = sp[pad+rs:pad+rs+H, pad+cs:pad+cs+W]
    return win

def win_axial_np(src, C, n, ta, tb):
    H,W = src.shape; pad = n+2
    sp = np.pad(src, pad, mode='edge'); cp = np.pad(C, pad, mode='edge')
    win = np.zeros((n,n,H,W))
    for x in range(n):
        for y in range(n):
            drow = y - x; dcol = x + y - n + 1
            ar = tb + drow; ac = ta + dcol
            assert ar % 2 == ac % 2
            if ar % 2 == 0:
                rs = ar//2; cs = ac//2; plane = sp
            else:
                rs = (ar-1)//2; cs = (ac-1)//2; plane = cp
            win[x,y] = plane[pad+rs:pad+rs+H, pad+cs:pad+cs+W]
    return win

def ravu_base_ref(img, p):
    n = p['radius']*2; ew = eff_weights(p); H,W = img.shape
    C    = classify_apply(win_step1_np(img, n), ew, p)               # O[2I+1][2J+1]
    int10 = classify_apply(win_axial_np(img, C, n, ta=1, tb=0), ew, p)  # O[2I][2J+1]
    int01 = classify_apply(win_axial_np(img, C, n, ta=0, tb=1), ew, p)  # O[2I+1][2J]
    out = np.zeros((2*H, 2*W))
    out[0::2,0::2] = img; out[0::2,1::2] = int10
    out[1::2,0::2] = int01; out[1::2,1::2] = C
    return out

# ---------------- torch module (for ONNX export) ----------------
class RavuBase(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p; self.n = p['radius']*2
        self.qa,self.qs,self.qc = p['quant_angle'],p['quant_strength'],p['quant_coherence']
        self.gl = p['radius'] - p['gradient_radius']; self.gr = self.n - self.gl
        self.register_buffer('gauss', torch.tensor(np.array(p['gaussian']), dtype=torch.float32))
        self.register_buffer('ew', torch.tensor(eff_weights(p), dtype=torch.float32))  # [cls,n*n]
        self.register_buffer('min_s', torch.tensor(p['min_strength'], dtype=torch.float32))
        self.register_buffer('min_c', torch.tensor(p['min_coherence'], dtype=torch.float32))

    def classify_apply(self, win):
        n = self.n
        def S(i,j): return win[:, (i*n+j):(i*n+j)+1]
        def ndiff(get, t):
            if t == 0:   return get(1) - get(0)
            if t == n-1: return get(n-1) - get(n-2)
            if t == 1 or t == n-2: return (get(t+1) - get(t-1)) * 0.5
            return (-get(t+2) + 8.0*get(t+1) - 8.0*get(t-1) + get(t-2)) / 12.0
        a=b=d=None
        for i in range(self.gl,self.gr):
            for j in range(self.gl,self.gr):
                gx = ndiff(lambda i2: S(i2,j), i)
                gy = ndiff(lambda j2: S(i,j2), j)
                gw = self.gauss[i-self.gl, j-self.gl]
                a = gx*gx*gw if a is None else a+gx*gx*gw
                b = gx*gy*gw if b is None else b+gx*gy*gw
                d = gy*gy*gw if d is None else d+gy*gy*gw
        T=a+d; D=a*d-b*b
        delta = torch.sqrt(torch.clamp(T*T/4.0 - D, min=0.0))
        L1=T/2.0+delta; L2=T/2.0-delta
        sL1=torch.sqrt(torch.clamp(L1,min=0)); sL2=torch.sqrt(torch.clamp(L2,min=0))
        theta = torch.where(torch.abs(b) < EPS, torch.zeros_like(b),
                            torch.remainder(torch.atan2(L1-a, b) + math.pi, math.pi))
        lam = sL1
        mu = torch.where(sL1+sL2 < EPS, torch.zeros_like(b), (sL1-sL2)/(sL1+sL2+1e-30))
        angle = torch.clamp(torch.floor(theta * self.qa / math.pi), 0, self.qa-1)
        strength = torch.zeros_like(b)
        for s in self.min_s: strength = strength + (lam >= s).float()
        coherence = torch.zeros_like(b)
        for s in self.min_c: coherence = coherence + (mu >= s).float()
        cls = ((angle*self.qs + strength)*self.qc + coherence).long().squeeze(1)  # [1,H,W]
        B,_,H,W = win.shape
        filt = self.ew[cls.reshape(-1)].reshape(H, W, n*n)        # [H,W,n*n]
        winp = win.squeeze(0).permute(1,2,0)                      # [H,W,n*n], k=x*n+y
        res = (filt*winp).sum(-1)                                 # [H,W]
        return torch.clamp(res, 0.0, 1.0)[None,None]              # [1,1,H,W]

    def win_step1(self, x):
        n = self.n; c = n//2 - 1
        xp = torch.nn.functional.pad(x, (n,n,n,n), mode='replicate')
        _,_,Hp,Wp = xp.shape; H=Hp-2*n; W=Wp-2*n
        taps=[]
        for xc in range(n):
            for yr in range(n):
                rs = yr - c; cs = xc - c
                taps.append(xp[:, :, n+rs:n+rs+H, n+cs:n+cs+W])
        return torch.cat(taps, dim=1)                            # [1,n*n,H,W]

    def win_axial(self, src, C, ta, tb):
        n = self.n; pad = n+2
        sp = torch.nn.functional.pad(src, (pad,pad,pad,pad), mode='replicate')
        cp = torch.nn.functional.pad(C,   (pad,pad,pad,pad), mode='replicate')
        _,_,Hp,Wp = sp.shape; H=Hp-2*pad; W=Wp-2*pad
        taps=[]
        for xc in range(n):
            for yr in range(n):
                drow = yr - xc; dcol = xc + yr - n + 1
                ar = tb + drow; ac = ta + dcol
                if ar % 2 == 0:
                    rs = ar//2; cs = ac//2; plane = sp
                else:
                    rs = (ar-1)//2; cs = (ac-1)//2; plane = cp
                taps.append(plane[:, :, pad+rs:pad+rs+H, pad+cs:pad+cs+W])
        return torch.cat(taps, dim=1)

    def forward(self, x):                                        # [1,1,H,W] -> [1,1,2H,2W]
        C     = self.classify_apply(self.win_step1(x))
        int10 = self.classify_apply(self.win_axial(x, C, ta=1, tb=0))
        int01 = self.classify_apply(self.win_axial(x, C, ta=0, tb=1))
        chans = torch.cat([x, int10, int01, C], dim=1)           # [src,int10,int01,C]
        return torch.nn.functional.pixel_shuffle(chans, 2)

def convert(radius, upstream, output_dir):
    p = load_weights(radius, upstream)
    mod = RavuBase(p).eval()
    rng = np.random.default_rng(0)
    img = rng.random((40,52)).astype(np.float32)
    with torch.no_grad():
        t = mod(torch.tensor(img)[None,None]).numpy()[0,0]
    ref = ravu_base_ref(img.astype(np.float64), p)
    print(f"[base r{radius}] torch-vs-numpyref max diff: {np.abs(t-ref).max():.2e}")
    const = np.full((24,24), 0.37, np.float32)
    with torch.no_grad(): co = mod(torch.tensor(const)[None,None]).numpy()[0,0]
    print(f"[base r{radius}] constant-image max dev from 0.37: {np.abs(co-0.37).max():.2e}")
    path = f"{output_dir}/ravu_r{radius}.onnx"
    torch.onnx.export(mod, torch.tensor(img)[None,None], path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{2:'h',3:'w'}, 'output':{2:'h2',3:'w2'}},
                      do_constant_folding=True)
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    o = s.run(None, {'input': img[None,None]})[0][0,0]
    print(f"[base r{radius}] ONNX-vs-numpyref max diff: {np.abs(o-ref).max():.2e} | out {s.get_outputs()[0].shape} -> {path}")
    return np.abs(o-ref).max()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("radius", type=int, nargs="?", default=2)
    ap.add_argument("--upstream", default="upstream")
    ap.add_argument("--output-dir", default=".")
    a = ap.parse_args()
    convert(a.radius, a.upstream, a.output_dir)
