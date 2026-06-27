"""Export Anime4KCPP FSRCNNX models to ONNX from the ORIGINAL FSRCNNX.p header.

FSRCNNX is a luma (1ch -> 1ch, 2x) net:
  head    : conv 1->F 5x5 (identity)
  body x4 : conv F->F 3x3 + PReLU
  fusion  : conv F->F 1x1 + PReLU, + head (long skip)
  features: conv F->4 3x3 (identity)
  upscale : + nearest-luma residual, PixelShuffle(r=2) -> 1ch at 2x, clamp

Four variants: F=8 / 16 (s / m tiers) x {Normal, DistortPlus} (the *_dp strongly
denoise). Weights are stored NHWC (= OHWI) and transposed to OIHW here; PReLU
alphas are stored alongside (body x4 + fusion = 5 PReLU layers).

Anime4KCPP code is MIT (TianZerL), but the FSRCNNX weights originate from
igv/FSRCNN-TensorFlow (GPL-3.0). See licenses/fsrcnnx.txt for details.
"""
import os, re, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from onnx_export_common import export_onnx

VARIANTS = {  # out suffix -> (array prefix, F)
    "s":    ("FSRCNNX_F8_NHWC",               8),
    "s_dp": ("FSRCNNX_F8_DistortPlus_NHWC",   8),
    "m":    ("FSRCNNX_F16_NHWC",              16),
    "m_dp": ("FSRCNNX_F16_DistortPlus_NHWC",  16),
}

def load_array(text, name):
    m = re.search(re.escape(name) + r"\[\]\s*=\s*\{([^}]*)\}", text)
    return np.array([float(x.rstrip('f')) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?", m.group(1))], np.float32)

def ohwi(flat, o, h, w, i):
    return np.ascontiguousarray(flat[:o*h*w*i].reshape(o, h, w, i).transpose(0, 3, 1, 2))

class FSRCNNX(nn.Module):
    def __init__(self, F, K, B, A):
        super().__init__()
        # split kernels in forward order: head, body0..3, fusion, features
        off = 0
        def take(o, h, w, i):
            nonlocal off
            n = o*h*w*i; k = ohwi(K[off:off+n], o, h, w, i); off += n; return k
        hk = take(F, 5, 5, 1)
        bk = [take(F, 3, 3, F) for _ in range(4)]
        fk = take(F, 1, 1, F)
        xk = take(4, 3, 3, F)
        bo = 0
        def tb(n):
            nonlocal bo
            v = B[bo:bo+n]; bo += n; return v
        hb = tb(F); bb = [tb(F) for _ in range(4)]; fb = tb(F); xb = tb(4)
        ao = 0
        def ta():
            nonlocal ao
            v = A[ao:ao+F]; ao += F; return v
        ba = [ta() for _ in range(4)]; fa = ta()
        self.head = self._c(1, F, 5, hk, hb)
        self.body = nn.ModuleList([self._c(F, F, 3, bk[i], bb[i]) for i in range(4)])
        self.bp = nn.ModuleList([self._p(F, ba[i]) for i in range(4)])
        self.fus = self._c(F, F, 1, fk, fb); self.fp = self._p(F, fa)
        self.feat = self._c(F, 4, 3, xk, xb)
    @staticmethod
    def _c(i, o, k, w, b):
        c = nn.Conv2d(i, o, k, padding=0); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b); return c
    @staticmethod
    def _p(n, a):
        p = nn.PReLU(n); p.weight.data = torch.tensor(a); return p
    @staticmethod
    def pad(x, k): return F.pad(x, ((k-1)//2,)*4, mode='replicate')
    def forward(self, x):
        h = self.head(self.pad(x, 5))
        f = h
        for c, p in zip(self.body, self.bp):
            f = p(c(self.pad(f, 3)))
        fused = self.fp(self.fus(f)) + h          # 1x1 + PReLU + long skip
        feat = self.feat(self.pad(fused, 3))      # F->4 (absolute luma, NO residual)
        return torch.clamp(torch.nn.functional.pixel_shuffle(feat, 2), 0.0, 1.0)

def export_one(suffix, prefix, Fw, text, out_dir):
    K = load_array(text, prefix + "_kernels")
    B = load_array(text, prefix + "_biases")
    A = load_array(text, prefix + "_alphas")
    net = FSRCNNX(Fw, K, B, A).eval()
    dummy = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        o = net(dummy)
    assert tuple(o.shape[1:]) == (1, 128, 128), f"{suffix}: out {o.shape}"
    out = os.path.join(out_dir, f"fsrcnnx_{suffix}.onnx")
    export_onnx(net, dummy, out, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   fsrcnnx_{suffix:5} F={Fw}  kernels={len(K)} biases={len(B)} alphas={len(A)}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Anime4KCPP FSRCNNX models to ONNX")
    parser.add_argument("--param-header", required=True, help="Path to FSRCNNX.p header file")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    text = open(args.param_header, encoding='utf-8', errors='ignore').read()
    print("=== Anime4KCPP FSRCNNX (from FSRCNNX.p) -> ONNX ===")
    n = sum(export_one(s, p, Fw, text, out_dir) for s, (p, Fw) in VARIANTS.items())
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
