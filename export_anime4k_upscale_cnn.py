"""Export bloc97 Anime4K Upscale_CNN x2 models (MIT) to ONNX from the ORIGINAL
GLSL shaders (not the kaizen .kaizenw blobs, which are OHWI-fp16 baked for oneDNN).

The S tier shares websr-S's topology: a 3->4 head conv (alpha column hard-zeroed
in bloc97's GLSL), three 8->4 body convs fed CReLU-expanded input (go_0 = max(x,0)
positive lobe -> in_ch 0..3, go_1 = max(-x,0) negative -> in_ch 4..7), then
PixelShuffle(r=2) giving a luma residual broadcast onto bilinear-2x RGB. The GLSL
encodes each conv as `mat4(16 floats) * go_N(kx, ky)`; weight[oc,ic,h,w] =
mat4_at(kx=w-1, ky=h-1)[ic*4 + oc]. Convs use replicate padding (GLSL clamp).

Anime4K is MIT (bloc97). See ../ACKNOWLEDGMENTS.md.
"""
import os, re, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from onnx_export_common import export_onnx


# tier suffix -> (glsl file, scale, tier)
S_VARIANTS = {
    "s":    ("Anime4K_Upscale_CNN_x2_S.glsl",         2),
    "s_dn": ("Anime4K_Upscale_Denoise_CNN_x2_S.glsl", 2),
}

MAT4_RE = re.compile(r"mat4\(\s*([\d\.\-e\+\,\s]+?)\s*\)\s*\*\s*go_(\d)\(\s*([+-]?[\d\.]+)\s*,\s*([+-]?[\d\.]+)\s*\)")
MAT4_BARE_RE = re.compile(r"mat4\(\s*([\d\.\-e\+\,\s]+?)\s*\)\s*\*\s*g_(\d+)")     # tail 1x1 (no spatial)
BIAS_RE = re.compile(r"result\s*\+=\s*vec4\(\s*([\d\.\-e\+\,\s]+?)\s*\)\s*;")
PASS_RE = re.compile(r"//!DESC\s+([^\n]+)")
GO_MID  = {0: 0, 1: 8, 2: 4, 3: 12}   # go_N (tf+/tf1+/tf-/tf1-) -> ic_base in the 16-ch chunk
CH_TAIL = {0: 0, 1: 8, 2: 4, 3: 12}   # tail j -> offset within source's 16-ch chunk

def parse_passes(text):
    ms = list(PASS_RE.finditer(text))
    return [(ms[i].group(1).strip(), text[ms[i].end(): (ms[i+1].start() if i+1 < len(ms) else len(text))]) for i in range(len(ms))]

def floats16(s):
    p = [float(x) for x in s.split(",") if x.strip()]; assert len(p) == 16; return p
def floats4(s):
    p = [float(x) for x in s.split(",") if x.strip()]; assert len(p) == 4; return p

def head_OIHW(body):
    byoff = {}
    for m in MAT4_RE.finditer(body):
        assert int(m.group(2)) == 0, "head uses only go_0"
        byoff[(int(float(m.group(3))), int(float(m.group(4))))] = floats16(m.group(1))
    assert len(byoff) == 9, f"head: {len(byoff)} offsets"
    bias = np.asarray(floats4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, 3, 3, 3), np.float32)
    for h in range(3):
        for ww in range(3):
            fl = byoff[(ww - 1, h - 1)]
            for oc in range(4):
                for ic in range(3):
                    w[oc, ic, h, ww] = fl[ic*4 + oc]
                assert fl[3*4 + oc] == 0.0, "head alpha col not zero"
    return w, bias

def body_OIHW(body):
    bylo = {}
    for m in MAT4_RE.finditer(body):
        g = int(m.group(2)); assert g in (0, 1)
        bylo[(g, int(float(m.group(3))), int(float(m.group(4))))] = floats16(m.group(1))
    assert len(bylo) == 18, f"body: {len(bylo)} entries"
    bias = np.asarray(floats4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, 8, 3, 3), np.float32)
    for h in range(3):
        for ww in range(3):
            for lobe in (0, 1):
                fl = bylo[(lobe, ww - 1, h - 1)]
                for oc in range(4):
                    for ic in range(4):
                        w[oc, lobe*4 + ic, h, ww] = fl[ic*4 + oc]
    return w, bias

def mid_dual_OIHW(body):
    """36 mat4 * go_N(kx,ky) (N=0..3 dual-branch CReLU) -> (4,16,3,3)."""
    byoff = {}
    for m in MAT4_RE.finditer(body):
        byoff[(int(m.group(2)), int(float(m.group(3))), int(float(m.group(4))))] = floats16(m.group(1))
    assert len(byoff) == 36, f"mid: {len(byoff)} reads"
    bias = np.asarray(floats4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, 16, 3, 3), np.float32)
    for (g, kx, ky), fl in byoff.items():
        base = GO_MID[g]
        for oc in range(4):
            for ic in range(4):
                w[oc, base + ic, ky + 1, kx + 1] = fl[ic*4 + oc]
    return w, bias

def tail_dense_OIHW(body):
    """28 mat4 * g_N bare (1x1) -> (4,112,1,1). ic_base = (N//4)*16 + CH_TAIL[N%4]."""
    reads = {}
    for m in MAT4_BARE_RE.finditer(body):
        reads[int(m.group(2))] = floats16(m.group(1))
    assert len(reads) == 28, f"tail: {len(reads)} reads"
    bias = np.asarray(floats4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, 112, 1, 1), np.float32)
    for g, fl in reads.items():
        base = (g // 4) * 16 + CH_TAIL[g % 4]
        for oc in range(4):
            for ic in range(4):
                w[oc, base + ic, 0, 0] = fl[ic*4 + oc]
    return w, bias

class UpscaleCnnS(nn.Module):
    def __init__(self, hw, hb, bws, bbs):
        super().__init__()
        self.head = nn.Conv2d(3, 4, 3, padding=0)
        self.head.weight.data = torch.tensor(hw); self.head.bias.data = torch.tensor(hb)
        self.body = nn.ModuleList()
        for w, b in zip(bws, bbs):
            c = nn.Conv2d(8, 4, 3, padding=0); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b)
            self.body.append(c)
    @staticmethod
    def cr(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    @staticmethod
    def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    def forward(self, x):
        f = self.cr(self.head, x)
        for c in self.body:
            f = self.cr(c, self.crelu(f))
        res = F.pixel_shuffle(f, 2)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + res

def export_S(suffix, glsl_name, glsl_dir, out_dir):
    path = os.path.join(glsl_dir, glsl_name)
    if not os.path.isfile(path):
        print(f"  SKIP {glsl_name}: not found"); return False
    passes = parse_passes(open(path, encoding='utf-8', errors='ignore').read())
    convs = [(d, b) for d, b in passes if MAT4_RE.search(b)]
    # head = first conv pass with only go_0 + 3-input; bodies = the rest with go_1.
    hw, hb = head_OIHW(convs[0][1])
    bws, bbs = [], []
    for d, b in convs[1:4]:
        w, bias = body_OIHW(b); bws.append(w); bbs.append(bias)
    assert len(bws) == 3, f"expected 3 body convs, got {len(bws)}"
    net = UpscaleCnnS(hw, hb, bws, bbs).eval()
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        o = net(dummy)
    out = os.path.join(out_dir, f"anime4k_upscale_cnn_{suffix}.onnx")
    export_onnx(net, dummy, out, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {glsl_name:38} -> {os.path.basename(out):30} x2 out{tuple(o.shape[1:])}")
    return True

VL_VARIANTS = {
    "vl":    "Anime4K_Upscale_CNN_x2_VL.glsl",
    "vl_dn": "Anime4K_Upscale_Denoise_CNN_x2_VL.glsl",
}
M_VARIANTS = {"m": "Anime4K_Upscale_CNN_x2_M.glsl", "m_dn": "Anime4K_Upscale_Denoise_CNN_x2_M.glsl"}
L_VARIANTS = {"l": "Anime4K_Upscale_CNN_x2_L.glsl", "l_dn": "Anime4K_Upscale_Denoise_CNN_x2_L.glsl"}

def tail_M_OIHW(body):
    """14 bare g_N (1x1) -> (4,56,1,1) single-chain: g=2s+pol, ic=s*8+pol*4."""
    reads = {int(m.group(2)): floats16(m.group(1)) for m in MAT4_BARE_RE.finditer(body)}
    assert len(reads) == 14, f"M tail: {len(reads)} reads"
    bias = np.asarray(floats4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, 56, 1, 1), np.float32)
    for g, fl in reads.items():
        base = (g // 2) * 8 + (g % 2) * 4
        for oc in range(4):
            for ic in range(4):
                w[oc, base + ic, 0, 0] = fl[ic*4 + oc]
    return w, bias

class UpscaleCnnM(nn.Module):
    """head(3->4) + 6 body(8->4) + 1 dense 1x1 tail(56->4) -> luma-broadcast residual."""
    def __init__(self, hw, hb, bws, bbs, tw, tb):
        super().__init__()
        self.head = nn.Conv2d(3, 4, 3, padding=0); self.head.weight.data = torch.tensor(hw); self.head.bias.data = torch.tensor(hb)
        self.body = nn.ModuleList()
        for w, b in zip(bws, bbs):
            c = nn.Conv2d(8, 4, 3, padding=0); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b); self.body.append(c)
        self.tail = nn.Conv2d(56, 4, 1); self.tail.weight.data = torch.tensor(tw); self.tail.bias.data = torch.tensor(tb)
    @staticmethod
    def cr(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    @staticmethod
    def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    def forward(self, x):
        f = self.cr(self.head, x); feats = [f]
        for c in self.body:
            f = self.cr(c, self.crelu(f)); feats.append(f)
        dense = torch.cat([self.crelu(ff) for ff in feats], dim=1)         # 56
        res = F.pixel_shuffle(self.tail(dense), 2)                         # 4->1 luma
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + res

class UpscaleCnnL(nn.Module):
    """dual-branch shallow: 2 heads, 2 mid pairs(16->4), 3 per-colour 3x3 tails(16->4
    reading dual-CReLU(mid_2))."""
    def __init__(self, ha, hb, m1, m2, tails):
        super().__init__()
        self.head_a = self._c(3, 4, 3, ha); self.head_b = self._c(3, 4, 3, hb)
        self.m1a = self._c(16, 4, 3, m1[0]); self.m1b = self._c(16, 4, 3, m1[1])
        self.m2a = self._c(16, 4, 3, m2[0]); self.m2b = self._c(16, 4, 3, m2[1])
        self.tail = nn.ModuleList([self._c(16, 4, 3, t) for t in tails])
    @staticmethod
    def _c(ic, oc, k, wb):
        c = nn.Conv2d(ic, oc, k, padding=0); c.weight.data = torch.tensor(wb[0]); c.bias.data = torch.tensor(wb[1]); return c
    @staticmethod
    def cr(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    @staticmethod
    def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    def forward(self, x):
        a = self.cr(self.head_a, x); b = self.cr(self.head_b, x)
        i1 = torch.cat([self.crelu(a), self.crelu(b)], dim=1); a = self.cr(self.m1a, i1); b = self.cr(self.m1b, i1)
        i2 = torch.cat([self.crelu(a), self.crelu(b)], dim=1); a = self.cr(self.m2a, i2); b = self.cr(self.m2b, i2)
        ti = torch.cat([self.crelu(a), self.crelu(b)], dim=1)
        res = torch.cat([F.pixel_shuffle(self.cr(t, ti), 2) for t in self.tail], dim=1)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + res

GO_UL = {0: 0, 1: 8, 2: 16, 3: 4, 4: 12, 5: 20}   # triple-branch go_N -> ic_base (24-ch chunk)
UL_VARIANTS = {"ul": "Anime4K_Upscale_CNN_x2_UL.glsl", "ul_dn": "Anime4K_Upscale_Denoise_CNN_x2_UL.glsl"}

def mid_triple_OIHW(body):
    """54 mat4 * go_N(kx,ky) (N=0..5 triple-branch CReLU) -> (4,24,3,3)."""
    byoff = {}
    for m in MAT4_RE.finditer(body):
        byoff[(int(m.group(2)), int(float(m.group(3))), int(float(m.group(4))))] = floats16(m.group(1))
    assert len(byoff) == 54, f"UL mid: {len(byoff)} reads"
    bias = np.asarray(floats4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, 24, 3, 3), np.float32)
    for (g, kx, ky), fl in byoff.items():
        base = GO_UL[g]
        for oc in range(4):
            for ic in range(4):
                w[oc, base + ic, ky + 1, kx + 1] = fl[ic*4 + oc]
    return w, bias

def tail_UL_OIHW(body):
    """30 bare g_N (1x1) -> (4,120,1,1): g=6s+j, ic_base = s*24 + GO_UL[j]."""
    reads = {int(m.group(2)): floats16(m.group(1)) for m in MAT4_BARE_RE.finditer(body)}
    assert len(reads) == 30, f"UL tail: {len(reads)} reads"
    bias = np.asarray(floats4(BIAS_RE.search(body).group(1)), np.float32)
    w = np.zeros((4, 120, 1, 1), np.float32)
    for g, fl in reads.items():
        base = (g // 6) * 24 + GO_UL[g % 6]
        for oc in range(4):
            for ic in range(4):
                w[oc, base + ic, 0, 0] = fl[ic*4 + oc]
    return w, bias

class UpscaleCnnUL(nn.Module):
    """triple-branch deepest: 3 heads, 6 mid stages x 3 branches(24->4), 3 dense 1x1
    tails(120->4) reading mid_2..mid_6 trios; per-colour residual."""
    def __init__(self, heads, mids, tails):
        super().__init__()
        self.head = nn.ModuleList([self._c(3, 4, 3, h) for h in heads])
        self.mid = nn.ModuleList([self._c(24, 4, 3, m) for m in mids])   # 18, grouped by stage of 3
        self.tail = nn.ModuleList([self._c(120, 4, 1, t) for t in tails])
    @staticmethod
    def _c(ic, oc, k, wb):
        c = nn.Conv2d(ic, oc, k, padding=0); c.weight.data = torch.tensor(wb[0]); c.bias.data = torch.tensor(wb[1]); return c
    @staticmethod
    def cr(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    @staticmethod
    def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    def tri(self, abc): return torch.cat([self.crelu(t) for t in abc], dim=1)   # 24ch
    def forward(self, x):
        abc = [self.cr(h, x) for h in self.head]; stages = [abc]
        for s in range(6):
            mi = self.tri(abc)
            abc = [self.cr(self.mid[3*s + k], mi) for k in range(3)]
            stages.append(abc)
        dense = torch.cat([self.tri(stages[2 + s]) for s in range(5)], dim=1)   # 5*24 = 120
        res = torch.cat([F.pixel_shuffle(t(dense), 2) for t in self.tail], dim=1)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + res

def export_UL(suffix, glsl_name, glsl_dir, out_dir):
    path = os.path.join(glsl_dir, glsl_name)
    if not os.path.isfile(path):
        print(f"  SKIP {glsl_name}: not found"); return False
    convs = [(d, b) for d, b in parse_passes(open(path, encoding='utf-8', errors='ignore').read()) if MAT4_RE.search(b) or MAT4_BARE_RE.search(b)]
    assert len(convs) == 24, f"UL: {len(convs)} convs"
    heads = [head_OIHW(convs[k][1]) for k in range(3)]
    mids  = [mid_triple_OIHW(convs[3 + k][1]) for k in range(18)]
    tails = [tail_UL_OIHW(convs[21 + k][1]) for k in range(3)]
    return _save(UpscaleCnnUL(heads, mids, tails).eval(), suffix, glsl_name, out_dir)

def export_M(suffix, glsl_name, glsl_dir, out_dir):
    path = os.path.join(glsl_dir, glsl_name)
    if not os.path.isfile(path):
        print(f"  SKIP {glsl_name}: not found"); return False
    convs = [(d, b) for d, b in parse_passes(open(path, encoding='utf-8', errors='ignore').read()) if MAT4_RE.search(b) or MAT4_BARE_RE.search(b)]
    assert len(convs) == 8, f"M: {len(convs)} convs"
    hw, hb = head_OIHW(convs[0][1])
    bws, bbs = zip(*[body_OIHW(convs[1+k][1]) for k in range(6)])
    tw, tb = tail_M_OIHW(convs[7][1])
    return _save(UpscaleCnnM(hw, hb, list(bws), list(bbs), tw, tb).eval(), suffix, glsl_name, out_dir)

def export_L(suffix, glsl_name, glsl_dir, out_dir):
    path = os.path.join(glsl_dir, glsl_name)
    if not os.path.isfile(path):
        print(f"  SKIP {glsl_name}: not found"); return False
    convs = [(d, b) for d, b in parse_passes(open(path, encoding='utf-8', errors='ignore').read()) if MAT4_RE.search(b) or MAT4_BARE_RE.search(b)]
    assert len(convs) == 9, f"L: {len(convs)} convs"
    ha = head_OIHW(convs[0][1]); hb = head_OIHW(convs[1][1])
    m1 = (mid_dual_OIHW(convs[2][1]), mid_dual_OIHW(convs[3][1]))
    m2 = (mid_dual_OIHW(convs[4][1]), mid_dual_OIHW(convs[5][1]))
    tails = [mid_dual_OIHW(convs[6+k][1]) for k in range(3)]
    return _save(UpscaleCnnL(ha, hb, m1, m2, tails).eval(), suffix, glsl_name, out_dir)

def _save(net, suffix, glsl_name, out_dir):
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        o = net(dummy)
    out = os.path.join(out_dir, f"anime4k_upscale_cnn_{suffix}.onnx")
    export_onnx(net, dummy, out, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {glsl_name:40} -> {os.path.basename(out):30} x2 out{tuple(o.shape[1:])}")
    return True

def export_VL(suffix, glsl_name, glsl_dir, out_dir):
    from export_websr import WebsrCnn2xL    # VL == websr-L topology (dual-branch)
    path = os.path.join(glsl_dir, glsl_name)
    if not os.path.isfile(path):
        print(f"  SKIP {glsl_name}: not found"); return False
    convs = [(d, b) for d, b in parse_passes(open(path, encoding='utf-8', errors='ignore').read()) if MAT4_RE.search(b) or MAT4_BARE_RE.search(b)]
    assert len(convs) == 17, f"VL: expected 17 conv passes, got {len(convs)}"
    ha = head_OIHW(convs[0][1]); hb = head_OIHW(convs[1][1])           # 2 parallel heads
    mids = []
    for k in range(6):
        aw, ab = mid_dual_OIHW(convs[2 + 2*k][1])
        bw, bb = mid_dual_OIHW(convs[3 + 2*k][1])
        mids.append((aw, ab, bw, bb))
    tails = [tail_dense_OIHW(convs[14 + k][1]) for k in range(3)]
    net = WebsrCnn2xL(ha, hb, mids, tails).eval()
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        o = net(dummy)
    out = os.path.join(out_dir, f"anime4k_upscale_cnn_{suffix}.onnx")
    export_onnx(net, dummy, out, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {glsl_name:40} -> {os.path.basename(out):30} x2 out{tuple(o.shape[1:])}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Anime4K Upscale_CNN x2 GLSL models to ONNX")
    parser.add_argument("--glsl-dir", required=True, help="Directory containing Anime4K Upscale GLSL shaders")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    glsl_dir = args.glsl_dir
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print("=== bloc97 Anime4K Upscale_CNN x2 S (from GLSL) -> ONNX ===")
    n = sum(export_S(s, g, glsl_dir, out_dir) for s, (g, sc) in S_VARIANTS.items())
    print("=== bloc97 Anime4K Upscale_CNN x2 M (from GLSL) -> ONNX ===")
    n += sum(export_M(s, g, glsl_dir, out_dir) for s, g in M_VARIANTS.items())
    print("=== bloc97 Anime4K Upscale_CNN x2 L (from GLSL) -> ONNX ===")
    n += sum(export_L(s, g, glsl_dir, out_dir) for s, g in L_VARIANTS.items())
    print("=== bloc97 Anime4K Upscale_CNN x2 VL (from GLSL) -> ONNX ===")
    n += sum(export_VL(s, g, glsl_dir, out_dir) for s, g in VL_VARIANTS.items())
    print("=== bloc97 Anime4K Upscale_CNN x2 UL (from GLSL) -> ONNX ===")
    n += sum(export_UL(s, g, glsl_dir, out_dir) for s, g in UL_VARIANTS.items())
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
