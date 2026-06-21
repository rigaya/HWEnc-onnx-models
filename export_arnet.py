"""Export Anime4KCPP ARNet models (MIT, TianZer) to ONNX from the original GLSL.

ARNet is ACNet's ResNet-style sibling (luma 1ch -> 1ch, 2x):
  head    : conv 1->8 (vec4 * LUMA.x, two parts) + PReLU                 -> F0
  block xN: conv0 8->8 + PReLU; conv1 8->8; F = 0.2*conv1(PReLU(conv0(F))) + F
            N = 8 / 16 / 32 / 64 for the s / m / l / xl tiers
  fusion  : conv 8->8 1x1 + PReLU, + F0 (long skip)
  upscale : conv 8->4 + nearest-luma residual, PixelShuffle(r=2) -> 1ch at 2x

Same GLSL conventions as ACNet (see export_acnet.py): head reads are vec4*luma.x,
body/fusion/upscale are mat4 (weight[oc,ic]=mat4[ic*4+oc]); features are
[TMP/FEAT _0 (0..3), _1 (4..7)]. The 0.2 residual scale is folded into conv1.
Like ACNet, this is from the public ACNetGLSL shaders (a slightly different weight
snapshot than kaizen's ARNet.p; visually equivalent). Anime4KCPP is MIT.
See ../ACKNOWLEDGMENTS.md and ../VERIFICATION.md.
"""
import os, re, glob, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

TIER = {"f8b8": "s", "f8b16": "m", "f8b32": "l", "f8b64": "xl"}

SAVE_RE   = re.compile(r"//!SAVE\s+(\S+)")
BIAS_RE   = re.compile(r"vec4 result = vec4\(\s*([-\d\.\,\seE+]+?)\s*\)")
SCALAR_RE = re.compile(r"vec4\(\s*([-\d\.\,\seE+]+?)\s*\)\s*\*\s*(\w+)_texOff\(vec2\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\)\.x")
MAT4_RE   = re.compile(r"mat4\(\s*([-\d\.\,\seE+]+?)\s*\)\s*\*\s*(\w+)_texOff\(vec2\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\)")
SLOPE_RE  = re.compile(r"\+\s*vec4\(\s*([-\d\.\,\seE+]+?)\s*\)\s*\*\s*min\(result")

def fl(s, n):
    v = [float(x) for x in s.split(",") if x.strip()]; assert len(v) == n, f"{len(v)}!={n}"; return v

def passes(text):
    out = []
    for s in text.split("//!DESC")[1:]:
        sv = SAVE_RE.search(s)
        if sv: out.append((s.split("\n")[0].strip(), s))
    return out

def fill(body, in_c, k, out_off, w, bias, prelu):
    bias[out_off:out_off+4] = np.asarray(fl(BIAS_RE.search(body).group(1), 4), np.float32)
    if in_c == 1:
        for m in SCALAR_RE.finditer(body):
            vec = fl(m.group(1), 4); kx, ky = int(float(m.group(3))), int(float(m.group(4)))
            for oc in range(4):
                w[out_off + oc, 0, ky+1, kx+1] = vec[oc]
    else:
        for m in MAT4_RE.finditer(body):
            mat = fl(m.group(1), 16); tex = m.group(2); kx, ky = int(float(m.group(3))), int(float(m.group(4)))
            ioff = 0 if tex.endswith("_0") else 4
            hh, ww = (ky+1, kx+1) if k == 3 else (0, 0)
            for oc in range(4):
                for ic in range(4):
                    w[out_off + oc, ioff + ic, hh, ww] = mat[ic*4 + oc]
    sl = SLOPE_RE.search(body)
    if sl:
        prelu[out_off:out_off+4] = np.asarray(fl(sl.group(1), 4), np.float32)
    return sl is not None

class ARNet(nn.Module):
    def __init__(self, head, blocks, fus, up):
        super().__init__()
        self.head = self._c(1, 8, 3, *head[:2]); self.hp = self._p(head[2])
        self.c0 = nn.ModuleList(); self.p0 = nn.ModuleList(); self.c1 = nn.ModuleList()
        for (c0w, c0b, c0p, c1w, c1b) in blocks:
            self.c0.append(self._c(8, 8, 3, c0w, c0b)); self.p0.append(self._p(c0p))
            self.c1.append(self._c(8, 8, 3, c1w * 0.2, c1b * 0.2))     # fold 0.2 residual scale
        self.fus = self._c(8, 8, 1, *fus[:2]); self.fp = self._p(fus[2])
        self.up = self._c(8, 4, 3, *up)
    @staticmethod
    def _c(i, o, k, w, b):
        c = nn.Conv2d(i, o, k, padding=0); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b); return c
    @staticmethod
    def _p(s):
        p = nn.PReLU(8); p.weight.data = torch.tensor(s); return p
    @staticmethod
    def cr(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    def forward(self, x):
        F0 = self.hp(self.cr(self.head, x))
        f = F0
        for c0, p0, c1 in zip(self.c0, self.p0, self.c1):
            f = self.cr(c1, p0(self.cr(c0, f))) + f
        fused = self.fp(self.fus(f)) + F0          # 1x1, then long skip
        u = self.cr(self.up, fused) + x
        return torch.clamp(F.pixel_shuffle(u, 2), 0.0, 1.0)

def export_one(glsl_path, out_dir):
    name = os.path.basename(glsl_path).replace(".glsl", "")
    tk = re.search(r"(f8b\d+)", name).group(1)
    out_name = "arnet_" + name.replace(tk, TIER[tk]).replace("arnet_", "")
    ps = [p[1] for p in passes(open(glsl_path, encoding='utf-8', errors='ignore').read())
          if SCALAR_RE.search(p[1]) or MAT4_RE.search(p[1])]
    nblk = (len(ps) - 5) // 4
    hw = np.zeros((8, 1, 3, 3), np.float32); hb = np.zeros(8, np.float32); hp = np.zeros(8, np.float32)
    fill(ps[0], 1, 3, 0, hw, hb, hp); fill(ps[1], 1, 3, 4, hw, hb, hp)
    blocks = []
    for k in range(nblk):
        base = 2 + 4*k
        c0w = np.zeros((8, 8, 3, 3), np.float32); c0b = np.zeros(8, np.float32); c0p = np.zeros(8, np.float32)
        fill(ps[base+0], 8, 3, 0, c0w, c0b, c0p); fill(ps[base+1], 8, 3, 4, c0w, c0b, c0p)
        c1w = np.zeros((8, 8, 3, 3), np.float32); c1b = np.zeros(8, np.float32); _d = np.zeros(8, np.float32)
        fill(ps[base+2], 8, 3, 0, c1w, c1b, _d); fill(ps[base+3], 8, 3, 4, c1w, c1b, _d)
        blocks.append((c0w, c0b, c0p, c1w, c1b))
    fb = 2 + 4*nblk
    fw = np.zeros((8, 8, 1, 1), np.float32); fbi = np.zeros(8, np.float32); fp = np.zeros(8, np.float32)
    fill(ps[fb+0], 8, 1, 0, fw, fbi, fp); fill(ps[fb+1], 8, 1, 4, fw, fbi, fp)
    uw = np.zeros((4, 8, 3, 3), np.float32); ub = np.zeros(4, np.float32); _u = np.zeros(4, np.float32)
    fill(ps[fb+2], 8, 3, 0, uw, ub, _u)
    net = ARNet((hw, hb, hp), blocks, (fw, fbi, fp), (uw, ub)).eval()
    dummy = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        o = net(dummy)
    assert tuple(o.shape[1:]) == (1, 128, 128), f"{name}: out {o.shape}"
    out = os.path.join(out_dir, out_name + ".onnx")
    torch.onnx.export(net, dummy, out, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {name:22} -> {out_name:18} 1ch x2  ({nblk} blocks)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Anime4KCPP ARNet GLSL models to ONNX")
    parser.add_argument("--glsl-dir", required=True, help="Directory containing ARNet GLSL shaders")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print("=== Anime4KCPP ARNet (from GLSL) -> ONNX ===")
    n = sum(export_one(g, out_dir) for g in sorted(glob.glob(os.path.join(args.glsl_dir, "arnet_f8b*.glsl"))))
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
