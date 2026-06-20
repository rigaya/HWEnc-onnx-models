"""Export bloc97 Anime4K 3DGraphics Upscale x2 models (MIT) to ONNX from GLSL.

Tiny 3-conv 2x upscaler for 3D/CG content. From the original GLSL:
  L1 head : Conv 3->4 3x3, no activation (alpha column zero)
  L2 body : Conv 8->4 3x3 + ReLU; input = CReLU(L1)  (go_0 pos -> ic 0..3, go_1 neg -> 4..7)
  L3      : Conv 4->4 3x3, no activation; input = ReLU(L2)  (go_0 only)
  tail    : PixelShuffle(r=2) of L3 (4->1) -> luma residual broadcast onto bilinear-2x RGB
Convs replicate-pad (GLSL clamp). Anime4K is MIT (bloc97). See ../ACKNOWLEDGMENTS.md.
"""
import os, re, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
VARIANTS = {
    "x2":    "Anime4K_3DGraphics_Upscale_x2_US.glsl",
    "aa_x2": "Anime4K_3DGraphics_AA_Upscale_x2_US.glsl",
}
MAT4_RE = re.compile(r"mat4\(\s*([\d\.\-e\+\,\s]+?)\s*\)\s*\*\s*go_(\d)\(\s*([+-]?[\d\.]+)\s*,\s*([+-]?[\d\.]+)\s*\)")
BIAS_RE = re.compile(r"result\s*\+=\s*vec4\(\s*([\d\.\-e\+\,\s]+?)\s*\)\s*;")
PASS_RE = re.compile(r"//!DESC\s+([^\n]+)")

def f16(s):
    p = [float(x) for x in s.split(",") if x.strip()]; assert len(p) == 16; return p
def f4(s):
    p = [float(x) for x in s.split(",") if x.strip()]; assert len(p) == 4; return p
def passes(t):
    ms = list(PASS_RE.finditer(t))
    return [(ms[i].group(1).strip(), t[ms[i].end():(ms[i+1].start() if i+1 < len(ms) else len(t))]) for i in range(len(ms))]

def conv_single(body, in_c, drop_alpha):
    """9 mat4 * go_0 single-source -> (4, in_c, 3,3) + bias."""
    byoff = {}
    for m in MAT4_RE.finditer(body):
        assert int(m.group(2)) == 0
        byoff[(int(float(m.group(3))), int(float(m.group(4))))] = f16(m.group(1))
    assert len(byoff) == 9
    bias = np.asarray(f4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, in_c, 3, 3), np.float32)
    for h in range(3):
        for ww in range(3):
            fl = byoff[(ww-1, h-1)]
            for oc in range(4):
                for ic in range(in_c):
                    w[oc, ic, h, ww] = fl[ic*4 + oc]
            if drop_alpha:
                for oc in range(4):
                    assert fl[3*4 + oc] == 0.0
    return w, bias

def conv_body(body):
    """18 mat4 * go_0/1 -> (4,8,3,3) + bias."""
    bylo = {}
    for m in MAT4_RE.finditer(body):
        bylo[(int(m.group(2)), int(float(m.group(3))), int(float(m.group(4))))] = f16(m.group(1))
    assert len(bylo) == 18
    bias = np.asarray(f4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, 8, 3, 3), np.float32)
    for h in range(3):
        for ww in range(3):
            for lobe in (0, 1):
                fl = bylo[(lobe, ww-1, h-1)]
                for oc in range(4):
                    for ic in range(4):
                        w[oc, lobe*4 + ic, h, ww] = fl[ic*4 + oc]
    return w, bias

class Anime3D(nn.Module):
    def __init__(self, w1, b1, w2, b2, w3, b3):
        super().__init__()
        self.l1 = nn.Conv2d(3, 4, 3, padding=0); self.l1.weight.data = torch.tensor(w1); self.l1.bias.data = torch.tensor(b1)
        self.l2 = nn.Conv2d(8, 4, 3, padding=0); self.l2.weight.data = torch.tensor(w2); self.l2.bias.data = torch.tensor(b2)
        self.l3 = nn.Conv2d(4, 4, 3, padding=0); self.l3.weight.data = torch.tensor(w3); self.l3.bias.data = torch.tensor(b3)
    @staticmethod
    def cr(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    @staticmethod
    def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    def forward(self, x):
        f1 = self.cr(self.l1, x)
        f2 = F.relu(self.cr(self.l2, self.crelu(f1)))
        f3 = self.cr(self.l3, f2)
        res = F.pixel_shuffle(f3, 2)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + res

def export_one(suffix, glsl_name, glsl_dir, out_dir):
    path = os.path.join(glsl_dir, glsl_name)
    if not os.path.isfile(path):
        print(f"  SKIP {glsl_name}: not found"); return False
    convs = [(d, b) for d, b in passes(open(path, encoding='utf-8', errors='ignore').read()) if MAT4_RE.search(b)]
    assert len(convs) == 3, f"Anime3D: {len(convs)} convs"
    w1, b1 = conv_single(convs[0][1], 3, True)
    w2, b2 = conv_body(convs[1][1])
    w3, b3 = conv_single(convs[2][1], 4, False)
    net = Anime3D(w1, b1, w2, b2, w3, b3).eval()
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        o = net(dummy)
    out = os.path.join(out_dir, f"anime3d_{suffix}.onnx")
    torch.onnx.export(net, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {glsl_name:42} -> {os.path.basename(out):22} x2 out{tuple(o.shape[1:])}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Anime4K 3DGraphics Upscale x2 GLSL models to ONNX")
    parser.add_argument("--glsl-dir", required=True, help="Directory containing Anime4K 3DGraphics GLSL shaders")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print("=== bloc97 Anime4K 3DGraphics Upscale x2 (from GLSL) -> ONNX ===")
    n = sum(export_one(s, g, args.glsl_dir, out_dir) for s, g in VARIANTS.items())
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
