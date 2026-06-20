"""Export bloc97 Anime4K Upscale_GAN models (MIT) to ONNX from the original GLSL.

GAN is a DenseNet RGB upscaler: a head (MAIN->4), then dense blocks whose 1x1
"aggregate" convs read CReLU of several prior layers, then conv0ups sibling
aggregates, then a final 3x3 conv that upsamples + adds the bilinear MAIN.

This is a generic GLSL-pass graph replay: each pass declares its source layers
via `#define go_N/g_N (max(+-NAME_tex...))`, a conv (mat4 * reader), and a SAVE
name. We parse every pass into (name, [sources], weight, kind) and replay the
graph in PyTorch. CReLU = concat(relu, -relu) per source; weight ic_base =
src*8 + polarity*4. The final pass runs at 2x: the GLSL samples the src-res
aggregates at offset*0.5, which equals bilinear-upsample-to-2x then a 3x3 conv;
the bilinear MAIN residual is added at the end.

Anime4K is MIT (bloc97). See ../ACKNOWLEDGMENTS.md and ../VERIFICATION.md.
"""
import os, re, sys, glob, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_anime4k_upscale_gan_glsl import (collect_terms, parse_g_mapping,
    parse_bias, parse_glsl_passes, GO_DEFINE_RE)
FILES = {  # out suffix -> glsl
    "s_x2":  "Anime4K_Upscale_GAN_x2_S.glsl",  "m_x2":  "Anime4K_Upscale_GAN_x2_M.glsl",
    "l_x3":  "Anime4K_Upscale_GAN_x3_L.glsl",  "vl_x3": "Anime4K_Upscale_GAN_x3_VL.glsl",
    "ul_x4": "Anime4K_Upscale_GAN_x4_UL.glsl", "uul_x4":"Anime4K_Upscale_GAN_x4_UUL.glsl",
}
DESC_RE  = re.compile(r"//!DESC\s+([^\n]+)")
SAVE_RE  = re.compile(r"//!SAVE\s+(\S+)")
BIND_RE  = re.compile(r"//!BIND\s+(\S+)")
W2_RE    = re.compile(r"//!WIDTH\s+\S+\s+(\d+)\s*\*")
WIDTH_RE = re.compile(r"//!WIDTH\s+(\w+)\.w(?:\s+(\d+)\s*\*)?")
# #define go_N(x,y) (max(+-(NAME_texOff/tex(...)),0.0))  -> N, sign, NAME
DEF_RE   = re.compile(r"#define\s+g[o]?_(\d+)\([^)]*\)\s*\(max\(\s*(-?)\(?\s*(\w+?)_tex")
# conv terms: mat4(16) * reader(args)   OR   vec4(4) * MAIN_texOff(..).x  (head)
MAT4_RD  = re.compile(r"mat4\(\s*([-\d\.\,\seE+]+?)\s*\)\s*\*\s*(\w+)\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)")
MAINHEAD = re.compile(r"mat4\(\s*([-\d\.\,\seE+]+?)\s*\)\s*\*\s*MAIN_texOff\(vec2\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)\)")
MAINRES  = re.compile(r"\+\s*MAIN_tex(?:Off)?\(\s*MAIN_pos\s*\)")

def fl(s, n=None):
    v = [float(x.rstrip('f')) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?", s)]
    return v if n is None else v[:n]

HEAD_GO0 = re.compile(r"#define\s+go_0\([^)]*\)\s*\(\s*MAIN_texOff")   # head: MAIN via go_0, NO max

def parse(text):
    secs = []
    parts = re.split(r"(//!DESC)", text)
    blocks = ["".join(parts[i:i+2]) for i in range(1, len(parts), 2)]
    for b in blocks:
        if SAVE_RE.search(b) and "mat4(" in b:
            secs.append(b)
    return secs

def src_names(body):
    """unique source layer names in first-appearance order across the defines."""
    order = []
    for m in GO_DEFINE_RE.finditer(body):
        nm = m.group(3)
        if nm not in order:
            order.append(nm)
    return order

def width_spec(body):
    m = WIDTH_RE.search(body)
    if not m: return (None, 1)
    return (m.group(1), int(m.group(2)) if m.group(2) else 1)

def conv_weight(body, oc):
    is2x = width_spec(body); main_resid = bool(MAINRES.search(body))
    if HEAD_GO0.search(body):                       # head: mat4 * go_0(=MAIN), 4-in alpha-fold -> 3
        w = np.zeros((4, 3, 3, 3), np.float32); boff = np.zeros(4, np.float32)
        for floats, gname, kx, ky in collect_terms(body):
            for o in range(4):
                for ic in range(3):
                    w[o, ic, ky+1, kx+1] = floats[ic*4 + o]
                boff[o] += floats[3*4 + o]           # alpha column (MAIN.a = 1) -> bias
        bias = parse_bias(body, 4) + boff
        return ("head", w, bias, ["MAIN"], is2x, main_resid)
    mapping = parse_g_mapping(body); names = src_names(body)
    has_sp = any(kx is not None for _, _, kx, _ in collect_terms(body))
    k = 3 if has_sp else 1
    w = np.zeros((oc, 8*len(names), k, k), np.float32)
    for floats, gname, kx, ky in collect_terms(body):
        src_i, pol = mapping[int(gname.split("_")[1])]
        base = src_i*8 + pol*4
        h, ww = (ky+1, kx+1) if kx is not None else (0, 0)
        for o in range(oc):
            for ic in range(4):
                w[o, base + ic, h, ww] = floats[ic*4 + o]
    return ("conv", w, parse_bias(body, oc), names, is2x, main_resid)

class GAN(nn.Module):
    def __init__(self, layers, scale):
        super().__init__()
        self.scale = scale; self.specs = []
        self.mods = nn.ModuleDict()
        for i, (name, kind, w, b, srcs, wspec, mres) in enumerate(layers):
            oc, ic, k, _ = w.shape
            c = nn.Conv2d(ic, oc, k, padding=0); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b)
            key = f"c{i}"; self.mods[key] = c
            self.specs.append((name, key, kind, srcs, wspec, mres, k))
    @staticmethod
    def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    @staticmethod
    def cr(c, x, k): return c(F.pad(x, ((k-1)//2,)*4, mode='replicate')) if k > 1 else c(x)
    @staticmethod
    def up(x, f): return x if f == 1 else F.interpolate(x, scale_factor=f, mode='bilinear', align_corners=False)
    def forward(self, x):
        t = {"MAIN": x}; sc = {"MAIN": 1}; out = None
        for name, key, kind, srcs, wspec, mres, k in self.specs:
            wref, wmul = wspec
            tgt = (sc.get(wref, 1) * wmul) if wref else (1 if kind == "head" else sc[srcs[0]])
            c = self.mods[key]
            if kind == "head":
                y = self.cr(c, x, k)
            else:
                inp = torch.cat([self.crelu(self.up(t[s], tgt // sc[s])) for s in srcs], dim=1)
                y = self.cr(c, inp, k)
            if mres:
                y = y + self.up(x, tgt)
            t[name] = y; sc[name] = tgt; out = y
        return torch.clamp(out, 0.0, 1.0)

def export_one(suffix, glsl, scale, glsl_dir, out_dir):
    path = os.path.join(glsl_dir, glsl)
    if not os.path.isfile(path):
        print(f"  SKIP {glsl}: not found"); return False
    secs = parse(open(path, encoding='utf-8', errors='ignore').read())
    layers = []
    for b in secs:
        name = SAVE_RE.search(b).group(1)
        oc = 3 if MAINRES.search(b) else 4          # any pass adding bilinear MAIN outputs RGB (3ch)
        kind, w, bias, srcs, is2x, mres = conv_weight(b, oc)
        layers.append((name, kind, w, bias, srcs, is2x, mres))
    try:
        net = GAN(layers, scale).eval()
        dummy = torch.rand(1, 3, 48, 48)
        with torch.no_grad():
            o = net(dummy)
    except Exception as e:
        print(f"  FAIL {glsl}: {str(e)[:90]}"); return False
    out = os.path.join(out_dir, f"anime4k_gan_{suffix}.onnx")
    torch.onnx.export(net, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {glsl:38} -> anime4k_gan_{suffix}.onnx  out{tuple(o.shape[1:])} ({len(layers)} layers)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Anime4K Upscale_GAN GLSL models to ONNX")
    parser.add_argument("--glsl-dir", required=True, help="Directory containing Anime4K Upscale GAN GLSL shaders")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    glsl_dir = args.glsl_dir
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print("=== bloc97 Anime4K Upscale_GAN (from GLSL) -> ONNX ===")
    n = 0
    for s, g in FILES.items():
        n += export_one(s, g, int(s[-1]), glsl_dir, out_dir)
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
