"""Convert RAVU-3x (bjin/mpv-prescalers, weights LGPL-3.0) to ONNX for --vpp-onnx.

RAVU-3x is a single-pass 3x prescaler. For each source pixel it gathers an n x n
window (n = radius*2-1, centred), classifies it exactly like RAVU/RAVU-Lite
(gaussian-weighted structure tensor -> 2x2 eigen-analysis -> quantize to
angle/strength/coherence) and applies 8 trained linear sub-pixel filters to fill
the 3x3 output block; the block centre is the source pixel (passthrough).

model_weights[a][s][c] has shape [8, n*n]: the 8 non-centre sub-pixel filters
(symmetrised the same way the GLSL LUT packing does). The sub-pixel -> output
position order is taken from the shader's imageStore loop:
  (row,col) in the 3x3 block:
    (0,0)=z0 (0,1)=z3 (0,2)=z5
    (1,0)=z1 (1,1)=src (1,2)=z6
    (2,0)=z2 (2,1)=z4 (2,2)=z7
Self-verifies an independent numpy reference vs the exported ONNX.
"""
import argparse, sys, math, numpy as np, torch, torch.nn as nn
import onnxruntime as ort
from onnx_export_common import export_onnx

EPS = 1.192092896e-7

def load_weights(radius, upstream):
    ns = {}
    exec(open(f"{upstream}/weights/ravu-3x_weights-r{radius}.py").read(), ns)
    p = {k: ns[k] for k in ['radius','gradient_radius','quant_angle','quant_strength',
                            'quant_coherence','min_strength','min_coherence','gaussian']}
    p['model_weights'] = np.array(ns['model_weights'], dtype=np.float64)   # (qa,qs,qc,8,n*n)
    return p

def sym_filters(p):
    """symmetrise like the GLSL LUT: sw[z][k] = (mw[z][k] + mw[7-z][n2-1-k])/2."""
    qa,qs,qc = p['quant_angle'],p['quant_strength'],p['quant_coherence']
    n2 = p['model_weights'].shape[-1]
    mw = p['model_weights'].reshape(qa*qs*qc, 8, n2)
    sw = (mw + mw[:, ::-1, ::-1]) / 2.0           # flip z (axis1) and k (axis2)
    return sw                                      # [cls,8,n*n]

# channel stack order for pixel_shuffle(scale=3): index = orow*3 + ocol
# values are sub-pixel id z, or -1 for the centre (source passthrough)
STACK = [0, 3, 5, 1, -1, 6, 2, 4, 7]

def classify(win, p):
    """win: [n,n,H,W] indexed [x=col][y=row]. returns class index [H,W] (int)."""
    n = win.shape[0]
    qa,qs,qc = p['quant_angle'],p['quant_strength'],p['quant_coherence']
    gl = p['radius'] - p['gradient_radius']; gr = n - gl
    gauss = np.array(p['gaussian'])
    def S(i,j): return win[i, j]
    def ndiff(get, t):
        if t == 0:   return get(1) - get(0)
        if t == n-1: return get(n-1) - get(n-2)
        return (get(t+1) - get(t-1)) / 2.0
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
    return (angle*qs + strength)*qc + coherence

def win_center_np(src, n):
    c = n//2; H,W = src.shape; pad = n
    sp = np.pad(src, pad, mode='edge')
    win = np.zeros((n,n,H,W))
    for x in range(n):
        for y in range(n):
            rs = y - c; cs = x - c
            win[x,y] = sp[pad+rs:pad+rs+H, pad+cs:pad+cs+W]
    return win

def ravu_3x_ref(img, p):
    n = p['radius']*2 - 1; sw = sym_filters(p); H,W = img.shape
    win = win_center_np(img, n)
    cls = classify(win, p)
    filt = sw[cls]                                   # [H,W,8,n*n]
    winp = win.reshape(n*n, H, W).transpose(1,2,0)   # [H,W,n*n], k=x*n+y
    outz = np.einsum('hwzk,hwk->hwz', filt, winp)    # [H,W,8]
    outz = np.clip(outz, 0.0, 1.0)
    out = np.zeros((3*H, 3*W))
    for ch, z in enumerate(STACK):
        orow, ocol = ch // 3, ch % 3
        plane = img if z < 0 else outz[:,:,z]
        out[orow::3, ocol::3] = plane
    return out

class Ravu3x(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p; self.n = p['radius']*2 - 1
        self.qa,self.qs,self.qc = p['quant_angle'],p['quant_strength'],p['quant_coherence']
        self.gl = p['radius'] - p['gradient_radius']; self.gr = self.n - self.gl
        self.register_buffer('gauss', torch.tensor(np.array(p['gaussian']), dtype=torch.float32))
        self.register_buffer('sw', torch.tensor(sym_filters(p), dtype=torch.float32))  # [cls,8,n*n]
        self.register_buffer('min_s', torch.tensor(p['min_strength'], dtype=torch.float32))
        self.register_buffer('min_c', torch.tensor(p['min_coherence'], dtype=torch.float32))

    def win_center(self, x):
        n = self.n; c = n//2
        xp = torch.nn.functional.pad(x, (n,n,n,n), mode='replicate')
        _,_,Hp,Wp = xp.shape; H=Hp-2*n; W=Wp-2*n
        taps=[]
        for xc in range(n):
            for yr in range(n):
                rs = yr - c; cs = xc - c
                taps.append(xp[:, :, n+rs:n+rs+H, n+cs:n+cs+W])
        return torch.cat(taps, dim=1)                # [1,n*n,H,W]

    def forward(self, x):                            # [1,1,H,W] -> [1,1,3H,3W]
        n = self.n
        win = self.win_center(x)
        def S(i,j): return win[:, (i*n+j):(i*n+j)+1]
        def ndiff(get, t):
            if t == 0:   return get(1) - get(0)
            if t == n-1: return get(n-1) - get(n-2)
            return (get(t+1) - get(t-1)) * 0.5
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
        _,_,H,W = win.shape
        filt = self.sw[cls.reshape(-1)].reshape(H, W, 8, n*n)         # [H,W,8,n*n]
        winp = win.squeeze(0).permute(1,2,0)                          # [H,W,n*n]
        outz = torch.clamp(torch.einsum('hwzk,hwk->hwz', filt, winp), 0.0, 1.0)  # [H,W,8]
        chans = []
        for z in STACK:
            plane = x[0,0] if z < 0 else outz[:,:,z]
            chans.append(plane[None,None])
        stack = torch.cat(chans, dim=1)                               # [1,9,H,W]
        return torch.nn.functional.pixel_shuffle(stack, 3)

def convert(radius, upstream, output_dir):
    p = load_weights(radius, upstream)
    sw = sym_filters(p)
    print(f"[3x r{radius}] sub-filter sums: min {sw.sum(-1).min():.4f} max {sw.sum(-1).max():.4f} (==1 -> DC preserving)")
    mod = Ravu3x(p).eval()
    rng = np.random.default_rng(0)
    img = rng.random((40,52)).astype(np.float32)
    with torch.no_grad():
        t = mod(torch.tensor(img)[None,None]).numpy()[0,0]
    ref = ravu_3x_ref(img.astype(np.float64), p)
    print(f"[3x r{radius}] torch-vs-numpyref max diff: {np.abs(t-ref).max():.2e}")
    const = np.full((24,24), 0.37, np.float32)
    with torch.no_grad(): co = mod(torch.tensor(const)[None,None]).numpy()[0,0]
    print(f"[3x r{radius}] constant-image max dev from 0.37: {np.abs(co-0.37).max():.2e}")
    path = f"{output_dir}/ravu_3x_r{radius}.onnx"
    export_onnx(mod, torch.tensor(img)[None,None], path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{2:'h',3:'w'}, 'output':{2:'h3',3:'w3'}},
                      do_constant_folding=True)
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    o = s.run(None, {'input': img[None,None]})[0][0,0]
    print(f"[3x r{radius}] ONNX-vs-numpyref max diff: {np.abs(o-ref).max():.2e} | out {s.get_outputs()[0].shape} -> {path}")
    return np.abs(o-ref).max()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("radius", type=int, nargs="?", default=2)
    ap.add_argument("--upstream", default="upstream")
    ap.add_argument("--output-dir", default=".")
    a = ap.parse_args()
    convert(a.radius, a.upstream, a.output_dir)
