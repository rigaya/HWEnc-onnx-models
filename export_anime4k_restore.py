"""Export bloc97 Anime4K Restore_CNN models (MIT) to ONNX from the original GLSL.

Restore is same-resolution (scale=1) restoration: a dual/triple-branch CNN (same
body topology as the Upscale_CNN family) whose final conv produces a 3-channel
RGB residual that is ADDED to the source (no PixelShuffle, no bilinear-upscale).

  L  : 2 heads + 3 mid pairs(16->4) + 1 out conv(16->3)               -> x + residual
  VL : 2 heads + 6 mid pairs(16->4) + 1 out conv(16->3)
  UL : 3 heads + 6 mid trios(24->4) + 1 out conv(120->3)  (reads mid_2..mid_6)

CReLU channel layout + go_N conventions are identical to the Upscale_CNN family;
this reuses those GLSL parsers. Anime4K is MIT (bloc97). See ../ACKNOWLEDGMENTS.md.
"""
import os, re, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_anime4k_upscale_cnn import (parse_passes, head_OIHW, mid_dual_OIHW,
    mid_triple_OIHW, MAT4_RE, MAT4_BARE_RE, GO_UL, floats16)
from onnx_export_common import export_onnx
OUTBIAS_RE = re.compile(r"result(?:\.\w+)?\s*\+=\s*vec[34]\(\s*([\d\.\-e\+\,\s]+?)\s*\)\s*;")

L_VARIANTS  = {"l": "Anime4K_Restore_CNN_L.glsl", "soft_l": "Anime4K_Restore_CNN_Soft_L.glsl"}
VL_VARIANTS = {"vl": "Anime4K_Restore_CNN_VL.glsl", "soft_vl": "Anime4K_Restore_CNN_Soft_VL.glsl"}
UL_VARIANTS = {"ul": "Anime4K_Restore_CNN_UL.glsl", "soft_ul": "Anime4K_Restore_CNN_Soft_UL.glsl"}

def out_dual_OIHW(body, fan):
    """final 3x3 dual-branch conv -> (3, fan, 3, 3) RGB residual (out_c=3)."""
    w4, _ = (mid_dual_OIHW(body) if fan == 16 else (None, None))
    if fan != 16:
        raise ValueError("use out_triple for fan!=16")
    bias = np.asarray([float(x) for x in OUTBIAS_RE.search(body).group(1).split(",") if x.strip()], np.float32)
    return w4[:3].copy(), bias[:3]

CH_VL = {0: 0, 1: 8, 2: 4, 3: 12}   # VL dense tail j -> offset (a_pos/b_pos/a_neg/b_neg)

def out_dense_OIHW(body):
    """28 bare g_N (1x1) -> (3,112,1,1) reading 7 mid pairs; g=4i+j, ic=i*16+CH_VL[j]."""
    reads = {int(m.group(2)): floats16(m.group(1)) for m in MAT4_BARE_RE.finditer(body)}
    assert len(reads) == 28, f"VL out: {len(reads)} reads"
    bias = np.asarray([float(x) for x in OUTBIAS_RE.search(body).group(1).split(",") if x.strip()], np.float32)[:3]
    w = np.zeros((3, 112, 1, 1), np.float32)
    for g, fl in reads.items():
        base = (g // 4) * 16 + CH_VL[g % 4]
        for oc in range(3):
            for ic in range(4):
                w[oc, base + ic, 0, 0] = fl[ic*4 + oc]
    return w, bias

class RestoreVLDense(nn.Module):
    """VL: 2 heads + 7 mid pairs(16->4) + dense 1x1 out(112->3) reading all mids."""
    def __init__(self, ha, hb, mids, outw, outb):
        super().__init__()
        self.ha = _c(3, 4, 3, *ha); self.hb = _c(3, 4, 3, *hb)
        self.ma = nn.ModuleList(); self.mb = nn.ModuleList()
        for (aw, ab, bw, bb) in mids:
            self.ma.append(_c(16, 4, 3, aw, ab)); self.mb.append(_c(16, 4, 3, bw, bb))
        self.out = _c(112, 3, 1, outw, outb)
    def forward(self, x):
        a = cr(self.ha, x); b = cr(self.hb, x); mids = []
        for ca, cb in zip(self.ma, self.mb):
            mi = torch.cat([crelu(a), crelu(b)], dim=1); a = cr(ca, mi); b = cr(cb, mi); mids.append((a, b))
        dense = torch.cat([torch.cat([crelu(ta), crelu(tb)], dim=1) for ta, tb in mids], dim=1)  # 112
        return x + self.out(dense)

def export_vl(suffix, glsl, glsl_dir, out_dir):
    path = os.path.join(glsl_dir, glsl)
    if not os.path.isfile(path): print(f"  SKIP {glsl}"); return False
    c = _convs(path); assert len(c) == 2 + 14 + 1, f"{glsl}: {len(c)} convs"
    ha = head_OIHW(c[0][1]); hb = head_OIHW(c[1][1])
    mids = [(*mid_dual_OIHW(c[2+2*k][1]), *mid_dual_OIHW(c[3+2*k][1])) for k in range(7)]
    outw, outb = out_dense_OIHW(c[-1][1])
    return _save(RestoreVLDense(ha, hb, mids, outw, outb).eval(), suffix, glsl, out_dir)

def out_triple_OIHW(body):
    """final 1x1 tail (120-fanin) -> (3,120,1,1) RGB residual."""
    reads = {int(m.group(2)): floats16(m.group(1)) for m in MAT4_BARE_RE.finditer(body)}
    assert len(reads) == 30, f"UL out: {len(reads)} reads"
    bias = np.asarray([float(x) for x in OUTBIAS_RE.search(body).group(1).split(",") if x.strip()], np.float32)[:3]
    w = np.zeros((3, 120, 1, 1), np.float32)
    for g, fl in reads.items():
        base = (g // 6) * 24 + GO_UL[g % 6]
        for oc in range(3):
            for ic in range(4):
                w[oc, base + ic, 0, 0] = fl[ic*4 + oc]
    return w, bias

def _c(ic, oc, k, w, b):
    c = nn.Conv2d(ic, oc, k, padding=0); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b); return c
def cr(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)

class RestoreDual(nn.Module):
    """L/VL: 2 heads + N mid pairs(16->4) + out(16->3); out = x + residual."""
    def __init__(self, ha, hb, mids, outw, outb):
        super().__init__()
        self.ha = _c(3, 4, 3, *ha); self.hb = _c(3, 4, 3, *hb)
        self.ma = nn.ModuleList(); self.mb = nn.ModuleList()
        for (aw, ab, bw, bb) in mids:
            self.ma.append(_c(16, 4, 3, aw, ab)); self.mb.append(_c(16, 4, 3, bw, bb))
        self.out = _c(16, 3, 3, outw, outb)
    def forward(self, x):
        a = cr(self.ha, x); b = cr(self.hb, x)
        for ca, cb in zip(self.ma, self.mb):
            mi = torch.cat([crelu(a), crelu(b)], dim=1); a = cr(ca, mi); b = cr(cb, mi)
        res = cr(self.out, torch.cat([crelu(a), crelu(b)], dim=1))
        return x + res

class RestoreTriple(nn.Module):
    """UL: 3 heads + 6 mid trios(24->4) + out(120->3); out = x + residual."""
    def __init__(self, heads, mids, outw, outb):
        super().__init__()
        self.head = nn.ModuleList([_c(3, 4, 3, *h) for h in heads])
        self.mid = nn.ModuleList([_c(24, 4, 3, *m) for m in mids])
        self.out = _c(120, 3, 1, outw, outb)
    def tri(self, abc): return torch.cat([crelu(t) for t in abc], dim=1)
    def forward(self, x):
        abc = [cr(h, x) for h in self.head]; stages = [abc]
        for s in range(7):
            mi = self.tri(abc); abc = [cr(self.mid[3*s+k], mi) for k in range(3)]; stages.append(abc)
        dense = torch.cat([self.tri(stages[3+s]) for s in range(5)], dim=1)   # mid3..mid7
        return x + self.out(dense)

def _convs(path):
    return [(d, b) for d, b in parse_passes(open(path, encoding='utf-8', errors='ignore').read()) if MAT4_RE.search(b) or MAT4_BARE_RE.search(b)]

def export_dual(suffix, glsl, glsl_dir, out_dir, npairs):
    path = os.path.join(glsl_dir, glsl)
    if not os.path.isfile(path): print(f"  SKIP {glsl}"); return False
    c = _convs(path); assert len(c) == 2 + 2*npairs + 1, f"{glsl}: {len(c)} convs (npairs={npairs})"
    ha = head_OIHW(c[0][1]); hb = head_OIHW(c[1][1])
    mids = [(*mid_dual_OIHW(c[2+2*k][1]), *mid_dual_OIHW(c[3+2*k][1])) for k in range(npairs)]
    outw, outb = out_dual_OIHW(c[-1][1], 16)
    return _save(RestoreDual(ha, hb, mids, outw, outb).eval(), suffix, glsl, out_dir)

def export_triple(suffix, glsl, glsl_dir, out_dir):
    path = os.path.join(glsl_dir, glsl)
    if not os.path.isfile(path): print(f"  SKIP {glsl}"); return False
    c = _convs(path); assert len(c) == 3 + 21 + 1, f"{glsl}: {len(c)} convs"
    heads = [head_OIHW(c[k][1]) for k in range(3)]
    mids = [mid_triple_OIHW(c[3+k][1]) for k in range(21)]
    outw, outb = out_triple_OIHW(c[-1][1])
    return _save(RestoreTriple(heads, mids, outw, outb).eval(), suffix, glsl, out_dir)

def _save(net, suffix, glsl, out_dir):
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        o = net(dummy)
    assert tuple(o.shape[1:]) == (3, 64, 64), f"{suffix}: scale!=1 ({o.shape})"
    out = os.path.join(out_dir, f"anime4k_restore_cnn_{suffix}.onnx")
    export_onnx(net, dummy, out, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {glsl:36} -> {os.path.basename(out):32} scale1")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Anime4K Restore_CNN GLSL models to ONNX")
    parser.add_argument("--glsl-dir", required=True, help="Directory containing Anime4K Restore GLSL shaders")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    glsl_dir = args.glsl_dir
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print("=== bloc97 Anime4K Restore_CNN (from GLSL) -> ONNX ===")
    n = 0
    for s, g in L_VARIANTS.items():  n += export_dual(s, g, glsl_dir, out_dir, 3)
    for s, g in VL_VARIANTS.items(): n += export_vl(s, g, glsl_dir, out_dir)
    for s, g in UL_VARIANTS.items(): n += export_triple(s, g, glsl_dir, out_dir)
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
