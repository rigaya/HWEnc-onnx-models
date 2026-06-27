"""Convert RAVU-Lite (bjin/mpv-prescalers, LGPL-3.0) to ONNX for --vpp-onnx.
Faithful port of weights/ravu-lite_weights-r*.py + ravu-lite.py forward.
Self-verifies: an independent numpy reference vs the exported ONNX.
"""
import argparse, sys, math, numpy as np, torch, torch.nn as nn
import onnxruntime as ort
from onnx_export_common import export_onnx

EPS = 1.192092896e-7

def load_weights(radius, upstream):
    ns = {}
    exec(open(f"{upstream}/weights/ravu-lite_weights-r{radius}.py").read(), ns)
    p = {k: ns[k] for k in ['gradient_radius','quant_angle','quant_strength',
                            'quant_coherence','min_strength','min_coherence','gaussian']}
    p['model_weights'] = np.array(ns['model_weights'], dtype=np.float64)  # (qa,qs,qc,4,n*n)
    return p

# ---------------- independent numpy reference (literal port) ----------------
def ravu_ref(img, p, radius):
    n = radius*2 - 1; c = n//2
    qa,qs,qc = p['quant_angle'],p['quant_strength'],p['quant_coherence']
    gl = radius - p['gradient_radius']; gr = n - gl
    gauss = np.array(p['gaussian'])
    mw = p['model_weights'].reshape(qa*qs*qc, 4, n*n)   # [class,4,n*n]
    H,W = img.shape
    pad = np.pad(img, c, mode='edge')
    # window[k] = sample at offset dx=k//n - c, dy=k%n - c  (dx=col, dy=row)
    win = np.zeros((n*n, H, W))
    for k in range(n*n):
        dx = k//n - c; dy = k%n - c
        win[k] = pad[c+dy:c+dy+H, c+dx:c+dx+W]
    def S(i,j): return win[i*n + j]
    def ndiff(get, x):
        if x == 0:   return get(1) - get(0)
        if x == n-1: return get(n-1) - get(n-2)
        return (get(x+1) - get(x-1)) / 2.0
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
    angle = np.floor(theta * qa / math.pi).astype(np.int64)
    angle = np.clip(angle, 0, qa-1)
    strength = np.zeros((H,W), np.int64)
    for s in p['min_strength']: strength += (lam >= s).astype(np.int64)
    coherence = np.zeros((H,W), np.int64)
    for s in p['min_coherence']: coherence += (mu >= s).astype(np.int64)
    cls = (angle*qs + strength)*qc + coherence
    filt = mw[cls]                         # [H,W,4,n*n]
    res = np.einsum('hwzk,khw->zhw', filt, win)   # [4,H,W]
    res = np.clip(res, 0.0, 1.0)
    # pixelshuffle: out[2r+dr,2c+dc] = res[dc*2+dr]
    out = np.zeros((2*H, 2*W))
    for z in range(4):
        dr = z % 2; dc = z // 2
        out[dr::2, dc::2] = res[z]
    return out

# ---------------- torch module (for ONNX export) ----------------
class RavuLite(nn.Module):
    def __init__(self, p, radius):
        super().__init__()
        self.r = radius; self.n = radius*2-1; self.c = self.n//2
        self.qa,self.qs,self.qc = p['quant_angle'],p['quant_strength'],p['quant_coherence']
        self.gl = radius - p['gradient_radius']; self.gr = self.n - self.gl
        self.register_buffer('gauss', torch.tensor(np.array(p['gaussian']), dtype=torch.float32))
        mw = p['model_weights'].reshape(self.qa*self.qs*self.qc, 4, self.n*self.n)
        self.register_buffer('filt', torch.tensor(mw, dtype=torch.float32))   # [cls,4,nn]
        self.register_buffer('min_s', torch.tensor(p['min_strength'], dtype=torch.float32))
        self.register_buffer('min_c', torch.tensor(p['min_coherence'], dtype=torch.float32))

    def forward(self, x):                       # x: [1,1,H,W]
        n,c = self.n, self.c
        xp = torch.nn.functional.pad(x, (c,c,c,c), mode='replicate')
        B,_,Hp,Wp = xp.shape; H = Hp-2*c; W = Wp-2*c
        win = []
        for k in range(n*n):
            dx = k//n - c; dy = k%n - c
            win.append(xp[:, :, c+dy:c+dy+H, c+dx:c+dx+W])
        win = torch.cat(win, dim=1)             # [1,nn,H,W]
        def S(i,j): return win[:, i*n+j:i*n+j+1]
        def ndiff(get, xi):
            if xi == 0:   return get(1) - get(0)
            if xi == n-1: return get(n-1) - get(n-2)
            return (get(xi+1) - get(xi-1)) * 0.5
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
        cls = ((angle*self.qs + strength)*self.qc + coherence).long().squeeze(1)   # [1,H,W]
        filt = self.filt[cls.reshape(-1)].reshape(H, W, 4, n*n)                      # [H,W,4,nn]
        winp = win.squeeze(0).permute(1,2,0)                                         # [H,W,nn]
        res = torch.einsum('hwzk,hwk->zhw', filt, winp)                             # [4,H,W]
        res = torch.clamp(res, 0.0, 1.0)                 # [4,H,W], z = dc*2+dr
        # clean pixel-shuffle: pixel_shuffle wants in-channel = dr*2+dc -> perm [0,2,1,3]
        resb = res[[0,2,1,3]].unsqueeze(0)               # [1,4,H,W]
        out = torch.nn.functional.pixel_shuffle(resb, 2) # [1,1,2H,2W]
        return out

def convert(radius, upstream, output_dir):
    p = load_weights(radius, upstream)
    mod = RavuLite(p, radius).eval()
    rng = np.random.default_rng(0)
    img = rng.random((48,64)).astype(np.float32)
    with torch.no_grad():
        t_out = mod(torch.tensor(img)[None,None]).numpy()[0,0]
    ref = ravu_ref(img.astype(np.float64), p, radius)
    print(f"[r{radius}] torch-vs-numpyref max abs diff: {np.abs(t_out-ref).max():.2e}")
    onnx_path = f"{output_dir}/ravu_lite_r{radius}.onnx"
    export_onnx(mod, torch.tensor(img)[None,None], onnx_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{2:'h',3:'w'}, 'output':{2:'h2',3:'w2'}},
                      do_constant_folding=True)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    o_out = sess.run(None, {'input': img[None,None]})[0][0,0]
    print(f"[r{radius}] ONNX-vs-numpyref max abs diff: {np.abs(o_out-ref).max():.2e}  -> {onnx_path}")
    return np.abs(o_out-ref).max()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("radius", type=int, nargs="?", default=2)
    ap.add_argument("--upstream", default="upstream")
    ap.add_argument("--output-dir", default=".")
    a = ap.parse_args()
    convert(a.radius, a.upstream, a.output_dir)
