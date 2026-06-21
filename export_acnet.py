"""Export Anime4KCPP ACNet models (MIT, TianZerL) to ONNX from the original GLSL.

ACNet is a luma (1-channel) 2x upscaler:
  head    : conv 1->8 (two parts, each `vec4 * LUMA_texOff(kx,ky).x`) + PReLU
  body xN : conv 8->8 (two parts, each `mat4 * TMP_TEX_{0,1}_texOff(kx,ky)`) + PReLU
            N = 4 / 8 / 18 blocks for the s / m / l tiers (f8b4 / f8b8 / f8b18)
  upscale : conv 8->4 + nearest-luma residual, then PixelShuffle(r=2) -> 1ch at 2x

mpv samples a single-channel LUMA as (luma, 0, 0, 1), so the head reads use the
scalar `.x`; weight[oc] = vec4[oc]. Body convs use full mat4 (col-major):
weight[oc,ic] = mat4[ic*4+oc]. PReLU slope is the per-channel vec4 in
`max(r,0) + vec4(slope) * min(r,0)`. Features are laid out [TMP_0 (0..3), TMP_1
(4..7)]; the two BIND textures map to in-channel offsets 0 and 4.

So the ONNX is 1ch in -> 1ch out, scale 2 (the filter's LumaSR fast path).
Anime4KCPP is MIT. See ../ACKNOWLEDGMENTS.md.
"""
import os, re, glob, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

TIER = {"f8b4": "s", "f8b8": "m", "f8b18": "l"}

DESC_RE   = re.compile(r"//!DESC([^\n]*)")
SAVE_RE   = re.compile(r"//!SAVE\s+(\S+)")
BIND_RE   = re.compile(r"//!BIND\s+(\S+)")
BIAS_RE   = re.compile(r"vec4 result = vec4\(\s*([-\d\.\,\seE+]+?)\s*\)")
SCALAR_RE = re.compile(r"vec4\(\s*([-\d\.\,\seE+]+?)\s*\)\s*\*\s*(\w+)_texOff\(vec2\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\)\.x")
MAT4_RE   = re.compile(r"mat4\(\s*([-\d\.\,\seE+]+?)\s*\)\s*\*\s*(\w+)_texOff\(vec2\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\)")
PRELU_RE  = re.compile(r"min\(result, vec4\(0\.0\)\);")
SLOPE_RE  = re.compile(r"\+\s*vec4\(\s*([-\d\.\,\seE+]+?)\s*\)\s*\*\s*min\(result")
RESID_RE  = re.compile(r"result\s*\+\s*(\w+)_texOff\(vec2\(\s*0\.0\s*,\s*0\.0\s*\)\)\.x")

def fl(s, n):
    v = [float(x) for x in s.split(",") if x.strip()]; assert len(v) == n, f"{len(v)}!={n}"; return v

def parse_passes(text):
    blocks = []
    parts = text.split("//!DESC")
    for p in parts[1:]:
        body = "//!DESC" + p
        save = SAVE_RE.search(body)
        if not save:  # not a conv pass
            continue
        binds = BIND_RE.findall(body)
        blocks.append((save.group(1), binds, body))
    return blocks

def conv_pass(body, in_c, out_off, w, bias, prelu):
    """Fill conv weights for one pass into (w,bias,prelu); returns (has_prelu, resid_tex)."""
    b = np.asarray(fl(BIAS_RE.search(body).group(1), 4), np.float32)
    for oc in range(4):
        bias[out_off + oc] = b[oc]
    if in_c == 1:   # head: vec4 * LUMA.x
        for m in SCALAR_RE.finditer(body):
            vec = fl(m.group(1), 4); kx, ky = int(float(m.group(3))), int(float(m.group(4)))
            for oc in range(4):
                w[out_off + oc, 0, ky+1, kx+1] = vec[oc]
    else:           # body/upscale: mat4 * TMP_TEX_{0,1}
        for m in MAT4_RE.finditer(body):
            mat = fl(m.group(1), 16); tex = m.group(2); kx, ky = int(float(m.group(3))), int(float(m.group(4)))
            ioff = 0 if tex.endswith("_0") else 4
            for oc in range(4):
                for ic in range(4):
                    w[out_off + oc, ioff + ic, ky+1, kx+1] = mat[ic*4 + oc]
    sl = SLOPE_RE.search(body)
    if sl:
        s = fl(sl.group(1), 4)
        for oc in range(4):
            prelu[out_off + oc] = s[oc]
    resid = RESID_RE.search(body)
    return (sl is not None), (resid.group(1) if resid else None)

class ACNet(nn.Module):
    def __init__(self, hw, hb, hp, bodies, uw, ub):
        super().__init__()
        self.head = self._c(1, 8, hw, hb); self.hp = nn.PReLU(8); self.hp.weight.data = torch.tensor(hp)
        self.body = nn.ModuleList(); self.bp = nn.ModuleList()
        for (bw, bb, bpr) in bodies:
            self.body.append(self._c(8, 8, bw, bb))
            p = nn.PReLU(8); p.weight.data = torch.tensor(bpr); self.bp.append(p)
        self.up = self._c(8, 4, uw, ub)
    @staticmethod
    def _c(i, o, w, b):
        c = nn.Conv2d(i, o, 3, padding=0); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b); return c
    @staticmethod
    def cr(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    def forward(self, x):
        f = self.hp(self.cr(self.head, x))
        for c, p in zip(self.body, self.bp):
            f = p(self.cr(c, f))
        u = self.cr(self.up, f) + x          # 4ch + nearest luma (broadcast)
        return torch.clamp(F.pixel_shuffle(u, 2), 0.0, 1.0)   # -> 1ch at 2x, clamped (matches GLSL)

def export_one(glsl_path, out_dir):
    name = os.path.basename(glsl_path).replace(".glsl", "")
    tierkey = re.search(r"(f8b\d+)", name).group(1)
    out_name = "acnet_" + name.replace(tierkey, TIER[tierkey]).replace("acnet_", "")
    passes = parse_passes(open(glsl_path, encoding='utf-8', errors='ignore').read())
    convs = [p for p in passes if SCALAR_RE.search(p[2]) or MAT4_RE.search(p[2])]
    # head = first 2 (bind LUMA, scalar); body pairs; upscale = last conv (has LUMA residual)
    nblk = (len(convs) - 3) // 2
    hw = np.zeros((8, 1, 3, 3), np.float32); hb = np.zeros(8, np.float32); hp = np.zeros(8, np.float32)
    conv_pass(convs[0][2], 1, 0, hw, hb, hp); conv_pass(convs[1][2], 1, 4, hw, hb, hp)
    bodies = []
    for k in range(nblk):
        bw = np.zeros((8, 8, 3, 3), np.float32); bb = np.zeros(8, np.float32); bp = np.zeros(8, np.float32)
        conv_pass(convs[2 + 2*k][2], 8, 0, bw, bb, bp); conv_pass(convs[3 + 2*k][2], 8, 4, bw, bb, bp)
        bodies.append((bw, bb, bp))
    uw = np.zeros((4, 8, 3, 3), np.float32); ub = np.zeros(4, np.float32); up = np.zeros(4, np.float32)
    has_p, resid = conv_pass(convs[-1][2], 8, 0, uw, ub, up)
    assert not has_p and resid is not None, f"{name}: upscale should have no PReLU + a LUMA residual"
    net = ACNet(hw, hb, hp, bodies, uw, ub).eval()
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
    parser = argparse.ArgumentParser(description="Export Anime4KCPP ACNet GLSL models to ONNX")
    parser.add_argument("--glsl-dir", required=True, help="Directory containing acnet_f8b*.glsl shaders")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print("=== Anime4KCPP ACNet (from GLSL) -> ONNX ===")
    n = sum(export_one(g, out_dir) for g in sorted(glob.glob(os.path.join(args.glsl_dir, "acnet_f8b*.glsl"))))
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
