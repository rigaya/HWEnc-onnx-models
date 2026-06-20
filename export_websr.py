"""Export websr CNN-2x-S models (bloc97 / Bhattacharyya, MIT) to ONNX.

Reconstructs the websr cnn-2x-s topology from the original JSON weights. The net
is a tiny RGB 2x upscaler: a 3->4 head conv (the 4th "alpha" input column is a
constant 1, folded into the bias), three 8->4 body convs each fed CReLU-expanded
input (concat(relu(x), relu(-x))), then a PixelShuffle(r=2) producing a single
luma residual that is broadcast-added onto the bilinear-2x-upsampled RGB. This
matches the kaizen forward exactly (kernel_kaizen_websr_pixshuffle_bilinear_rgb):
PyTorch pixel_shuffle uses the same sub-pixel index, and F.interpolate(bilinear,
align_corners=False) matches the (dx-0.5)*0.5 sample convention. Convs use
replicate padding to match the GLSL clamp-to-edge sampling.

websr is MIT (bloc97 anime corpus; _rl = sb2702 Xiph retrain, _3d = derf retrain).
See ../ACKNOWLEDGMENTS.md.
"""
import os, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

VARIANTS_S = {  # output name suffix -> json
    "s_an": "cnn-2x-s-an.json", "s_rl": "cnn-2x-s-rl.json", "s_3d": "cnn-2x-s-3d.json",
}
VARIANTS_M = {
    "m_an": "cnn-2x-m-an.json", "m_rl": "cnn-2x-m-rl.json", "m_3d": "cnn-2x-m-3d.json",
}
VARIANTS_L = {
    "l_an": "cnn-2x-l-an.json", "l_rl": "cnn-2x-l-rl.json", "l_3d": "cnn-2x-l-3d.json",
}
# L channel layouts (per the verified extract_websr_cnn2xl.py)
CH_START_MID  = {(0, 0): 0, (0, 1): 4, (1, 0): 8, (1, 1): 12}   # 16ch = [tf_pos,tf_neg,tf1_pos,tf1_neg]
CH_START_TAIL = {0: 0, 1: 8, 2: 4, 3: 12}                       # per-source 16ch chunk

def mid_dual_weight_OIHW(flat):
    """576 floats -> (4,16,3,3). block = i_spatial + g*9 + p*18; spatial [i%3, i//3]."""
    arr = np.asarray(flat, dtype=np.float32)
    w = np.zeros((4, 16, 3, 3), np.float32)
    for i in range(9):
        kh, kw = i % 3, i // 3
        for p in (0, 1):
            for g in (0, 1):
                base = (i + g*9 + p*18) * 16
                cs = CH_START_MID[(g, p)]
                for col in range(4):
                    for row in range(4):
                        w[row, cs + col, kh, kw] = arr[base + col*4 + row]
    return w

def tail_dual_weight_OIHW(flat):
    """448 floats -> (4,112,1,1). block = 4*i + j; in_ch = i*16 + CH_START_TAIL[j] + col."""
    arr = np.asarray(flat, dtype=np.float32)
    w = np.zeros((4, 112, 1, 1), np.float32)
    for i in range(7):
        for j in range(4):
            base = (4*i + j) * 16
            for col in range(4):
                in_ch = i*16 + CH_START_TAIL[j] + col
                for row in range(4):
                    w[row, in_ch, 0, 0] = arr[base + col*4 + row]
    return w

def tail_weight_OIHW(flat):
    """224 floats -> (4,56,1,1). 14 mat4 blocks; block b: tensor i=b//2 (0=head,
    1..6=mid), polarity b%2 (0=pos,1=neg); global in_ch = i*8 + pol*4 + in_col.
    Matches the dense concat [CReLU(head), CReLU(mid1)..CReLU(mid6)] (56ch)."""
    arr3 = np.asarray(flat, dtype=np.float32).reshape(14, 4, 4)   # [block][in_col][out_row]
    w = np.zeros((4, 56, 1, 1), np.float32)
    for b in range(14):
        base = (b // 2) * 8 + (b % 2) * 4
        for col in range(4):
            for row in range(4):
                w[row, base + col, 0, 0] = arr3[b, col, row]
    return w

def head_weight_OIHW(flat):
    """144 floats -> (4,3,3,3) conv weight (alpha column folded) + (4,) bias offset."""
    arr3 = np.asarray(flat, dtype=np.float32).reshape(9, 4, 4)   # [spatial][ic][oc]
    full = np.zeros((4, 4, 3, 3), np.float32)
    for h in range(3):
        for w in range(3):
            for oc in range(4):
                for ic in range(4):
                    full[oc, ic, h, w] = arr3[h*3 + w, ic, oc]
    bias_off = full[:, 3, :, :].sum(axis=(1, 2))                 # alpha=1 contribution
    return full[:, 0:3, :, :].copy(), bias_off

def body_weight_OIHW(flat):
    """288 floats -> (4,8,3,3): ic 0-3 = relu(x) kernels, ic 4-7 = relu(-x) kernels."""
    arr3 = np.asarray(flat, dtype=np.float32).reshape(18, 4, 4)
    w = np.zeros((4, 8, 3, 3), np.float32)
    for h in range(3):
        for ww in range(3):
            for oc in range(4):
                for ic in range(4):
                    w[oc, ic,     h, ww] = arr3[h*3 + ww,       ic, oc]
                    w[oc, ic + 4, h, ww] = arr3[(h*3 + ww) + 9, ic, oc]
    return w

class WebsrCnn2xS(nn.Module):
    def __init__(self, hw, hb, bws, bbs):
        super().__init__()
        self.head = nn.Conv2d(3, 4, 3, padding=0)
        self.head.weight.data = torch.tensor(hw); self.head.bias.data = torch.tensor(hb)
        self.body = nn.ModuleList()
        for w, b in zip(bws, bbs):
            c = nn.Conv2d(8, 4, 3, padding=0)
            c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b)
            self.body.append(c)
    @staticmethod
    def conv_rep(c, x):                    # replicate-pad 1px (GLSL clamp-to-edge), then 3x3
        return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    @staticmethod
    def crelu(x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    def forward(self, x):
        f = self.conv_rep(self.head, x)
        for c in self.body:
            f = self.conv_rep(c, self.crelu(f))
        res  = F.pixel_shuffle(f, 2)                                   # 4ch -> 1ch at 2x
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return base + res                                             # broadcast luma residual

class WebsrCnn2xM(nn.Module):
    """head(3->4) + 6 mid(8->4 CReLU) + 3 dense 1x1 tails(56->4) reading the
    CReLU concat of [head, mid1..mid6]; per-colour residual onto bilinear-2x."""
    def __init__(self, hw, hb, mws, mbs, tws, tbs):
        super().__init__()
        self.head = nn.Conv2d(3, 4, 3, padding=0)
        self.head.weight.data = torch.tensor(hw); self.head.bias.data = torch.tensor(hb)
        self.mid = nn.ModuleList()
        for w, b in zip(mws, mbs):
            c = nn.Conv2d(8, 4, 3, padding=0); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b)
            self.mid.append(c)
        self.tail = nn.ModuleList()
        for w, b in zip(tws, tbs):
            c = nn.Conv2d(56, 4, 1); c.weight.data = torch.tensor(w); c.bias.data = torch.tensor(b)
            self.tail.append(c)
    @staticmethod
    def conv_rep(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    @staticmethod
    def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    def forward(self, x):
        h = self.conv_rep(self.head, x)
        feats = [h]; m = h
        for c in self.mid:
            m = self.conv_rep(c, self.crelu(m)); feats.append(m)
        dense = torch.cat([self.crelu(f) for f in feats], dim=1)        # 7*8 = 56ch
        res = torch.cat([F.pixel_shuffle(t(dense), 2) for t in self.tail], dim=1)  # 3ch at 2x
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return base + res

def export_S(suffix, json_name, json_dir, out_dir):
    path = os.path.join(json_dir, json_name)
    if not os.path.isfile(path):
        print(f"  SKIP {json_name}: not found"); return False
    layers = json.load(open(path))["layers"]
    order = ["conv2d_tf", "conv2d_1_tf", "conv2d_2_tf", "conv2d_last_tf", "pixel_shuffle"]
    assert list(layers.keys()) == order, f"unexpected layer order {list(layers.keys())}"
    hw, hoff = head_weight_OIHW(layers["conv2d_tf"]["weights"])
    hb = np.asarray(layers["conv2d_tf"]["bias"], np.float32) + hoff
    bws = [body_weight_OIHW(layers[ln]["weights"]) for ln in ["conv2d_1_tf", "conv2d_2_tf", "conv2d_last_tf"]]
    bbs = [np.asarray(layers[ln]["bias"], np.float32) for ln in ["conv2d_1_tf", "conv2d_2_tf", "conv2d_last_tf"]]
    net = WebsrCnn2xS(hw, hb, bws, bbs).eval()
    return _export(net, suffix, json_name, out_dir)

def export_M(suffix, json_name, json_dir, out_dir):
    path = os.path.join(json_dir, json_name)
    if not os.path.isfile(path):
        print(f"  SKIP {json_name}: not found"); return False
    layers = json.load(open(path))["layers"]
    mids = [f"conv2d_{i}_tf" for i in range(1, 7)]
    tails = ["conv2d_7_tf", "conv2d_7_tf1", "conv2d_7_tf2"]
    order = ["conv2d_tf"] + mids + tails + ["pixel_shuffle"]
    assert list(layers.keys()) == order, f"unexpected layer order {list(layers.keys())}"
    hw, hoff = head_weight_OIHW(layers["conv2d_tf"]["weights"])
    hb = np.asarray(layers["conv2d_tf"]["bias"], np.float32) + hoff
    mws = [body_weight_OIHW(layers[ln]["weights"]) for ln in mids]
    mbs = [np.asarray(layers[ln]["bias"], np.float32) for ln in mids]
    tws = [tail_weight_OIHW(layers[ln]["weights"]) for ln in tails]
    tbs = [np.asarray(layers[ln]["bias"], np.float32) for ln in tails]
    net = WebsrCnn2xM(hw, hb, mws, mbs, tws, tbs).eval()
    return _export(net, suffix, json_name, out_dir)

class WebsrCnn2xL(nn.Module):
    """dual-branch: 2 parallel heads(3->4); 6 mid PAIRS (each 16->4 reading
    CReLU of both prev branches); 3 dense 1x1 tails(112->4) reading the 14-tensor
    CReLU concat; per-colour residual onto bilinear-2x."""
    def __init__(self, ha, hb, mids, tails):
        super().__init__()
        self.head_a = self._c(3, 4, 3, ha); self.head_b = self._c(3, 4, 3, hb)
        self.mid_a = nn.ModuleList(); self.mid_b = nn.ModuleList()
        for (aw, ab, bw, bb) in mids:
            self.mid_a.append(self._c(16, 4, 3, (aw, ab)))
            self.mid_b.append(self._c(16, 4, 3, (bw, bb)))
        self.tail = nn.ModuleList([self._c(112, 4, 1, t) for t in tails])
    @staticmethod
    def _c(ic, oc, k, wb):
        c = nn.Conv2d(ic, oc, k, padding=0)
        c.weight.data = torch.tensor(wb[0]); c.bias.data = torch.tensor(wb[1]); return c
    @staticmethod
    def conv_rep(c, x): return c(F.pad(x, (1, 1, 1, 1), mode='replicate'))
    @staticmethod
    def crelu(x): return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    def forward(self, x):
        a = self.conv_rep(self.head_a, x); b = self.conv_rep(self.head_b, x)
        srcs = [(a, b)]
        for ca, cb in zip(self.mid_a, self.mid_b):
            mi = torch.cat([self.crelu(a), self.crelu(b)], dim=1)     # 16ch
            a = self.conv_rep(ca, mi); b = self.conv_rep(cb, mi); srcs.append((a, b))
        dense = torch.cat([torch.cat([self.crelu(sa), self.crelu(sb)], dim=1) for sa, sb in srcs], dim=1)  # 112ch
        res = torch.cat([F.pixel_shuffle(t(dense), 2) for t in self.tail], dim=1)
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return base + res

def export_L(suffix, json_name, json_dir, out_dir):
    path = os.path.join(json_dir, json_name)
    if not os.path.isfile(path):
        print(f"  SKIP {json_name}: not found"); return False
    layers = json.load(open(path))["layers"]
    def head(ln):
        w, off = head_weight_OIHW(layers[ln]["weights"])
        return w, np.asarray(layers[ln]["bias"], np.float32) + off
    ha = head("conv2d_tf"); hb = head("conv2d_tf1")
    mids = []
    for i in range(1, 7):
        aw = mid_dual_weight_OIHW(layers[f"conv2d_{i}_tf"]["weights"])
        ab = np.asarray(layers[f"conv2d_{i}_tf"]["bias"], np.float32)
        bw = mid_dual_weight_OIHW(layers[f"conv2d_{i}_tf1"]["weights"])
        bb = np.asarray(layers[f"conv2d_{i}_tf1"]["bias"], np.float32)
        mids.append((aw, ab, bw, bb))
    tails = [(tail_dual_weight_OIHW(layers[ln]["weights"]), np.asarray(layers[ln]["bias"], np.float32))
             for ln in ["conv2d_last_tf", "conv2d_last_tf1", "conv2d_last_tf2"]]
    net = WebsrCnn2xL(ha, hb, mids, tails).eval()
    return _export(net, suffix, json_name, out_dir)

def _export(net, suffix, json_name, out_dir):
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        o = net(dummy)
    out = os.path.join(out_dir, f"websr_cnn2x_{suffix}.onnx")
    torch.onnx.export(net, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {json_name:18} -> {os.path.basename(out):24} x2  out{tuple(o.shape[1:])}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export websr CNN-2x models to ONNX")
    parser.add_argument("--json-dir", required=True, help="Directory containing websr JSON weight files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    json_dir = args.json_dir
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print("=== websr cnn-2x-s -> ONNX ===")
    n = sum(export_S(s, j, json_dir, out_dir) for s, j in VARIANTS_S.items())
    print("=== websr cnn-2x-m -> ONNX ===")
    n += sum(export_M(s, j, json_dir, out_dir) for s, j in VARIANTS_M.items())
    print("=== websr cnn-2x-l -> ONNX ===")
    n += sum(export_L(s, j, json_dir, out_dir) for s, j in VARIANTS_L.items())
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
