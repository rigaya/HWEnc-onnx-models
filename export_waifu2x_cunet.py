"""Export waifu2x CUNet models to ONNX for --vpp-onnx.

CUNet is a valid-convolution U-net with squeeze-excite blocks; its skip
connections mean the flat JSON weight list alone is not enough to rebuild it.
So this script takes the TOPOLOGY from the ncnn .param file (which encodes the
graph: Split/Pooling/InnerProduct/Scale/Deconvolution/Crop/Eltwise) and the
WEIGHTS from the original waifu2x JSON (fp32, same layer order as the .param,
cross-checked layer by layer). The valid convolutions shrink the image, so the
input is reflect-padded by the network offset and the output is cropped to the
exact target size, giving a clean 1x (denoise) or 2x (scale) model.

waifu2x is MIT (nagadomi). See ../ACKNOWLEDGMENTS.md.
"""
import os, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ACT = {0: None, 1: 'relu', 2: 'leaky', 4: 'sigmoid'}

def fp(x): return int(float(x))

def parse_param(path):
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    assert lines[0].strip() == "7767517"
    layers = []
    for ln in lines[2:]:
        t = ln.split(); typ, name = t[0], t[1]; nin, nout = int(t[2]), int(t[3]); i = 4
        ins = t[i:i+nin]; i += nin; outs = t[i:i+nout]; i += nout
        params = {int(k): v for k, v in (tok.split("=") for tok in t[i:])}
        layers.append((typ, name, ins, outs, params))
    return layers

class CUNet(nn.Module):
    def __init__(self, param_path, json_path, scale):
        super().__init__()
        self.layers = parse_param(param_path)
        self.scale = scale
        self.pad = 0
        jl = json.load(open(json_path))          # weights, same order as learnable layers
        self.mods = nn.ModuleDict(); self.meta = {}; ji = 0
        for typ, name, ins, outs, pr in self.layers:
            if typ in ("Convolution", "Deconvolution", "InnerProduct"):
                L = jl[ji]; ji += 1
                w = np.array(L['weight'], dtype=np.float32); b = np.array(L['bias'], dtype=np.float32)
                act = ACT.get(fp(pr.get(9, "0")))
                if typ == "Convolution":
                    outc, inc, kh, kw = w.shape
                    m = nn.Conv2d(inc, outc, (kh, kw), stride=fp(pr.get(3, "1")), padding=fp(pr.get(4, "0")))
                    m.weight.data = torch.tensor(w)
                elif typ == "Deconvolution":
                    inc, outc, kh, kw = w.shape       # JSON deconv = [in,out,kh,kw] = ConvTranspose2d layout
                    m = nn.ConvTranspose2d(inc, outc, (kh, kw), stride=fp(pr.get(3, "1")), padding=fp(pr.get(4, "0")))
                    m.weight.data = torch.tensor(w)
                else:  # InnerProduct: JSON [out,in,1,1] -> Linear [out,in]
                    outc = w.shape[0]; inc = w.shape[1]
                    m = nn.Linear(inc, outc)
                    m.weight.data = torch.tensor(w.reshape(outc, inc))
                m.bias.data = torch.tensor(b)
                self.mods[name] = m; self.meta[name] = act
        self.eval()

    @staticmethod
    def _act(x, a):
        return F.relu(x) if a == 'relu' else F.leaky_relu(x, 0.1) if a == 'leaky' else torch.sigmoid(x) if a == 'sigmoid' else x

    def _run(self, x):
        blob = {}
        for typ, name, ins, outs, pr in self.layers:
            if typ == "Input": blob[outs[0]] = x
            elif typ in ("Convolution", "Deconvolution"):
                blob[outs[0]] = self._act(self.mods[name](blob[ins[0]]), self.meta[name])
            elif typ == "InnerProduct":
                v = self.mods[name](blob[ins[0]].flatten(1))
                blob[outs[0]] = self._act(v, self.meta[name]).view(v.shape[0], v.shape[1], 1, 1)
            elif typ == "Split":
                for o in outs: blob[o] = blob[ins[0]]
            elif typ == "Pooling": blob[outs[0]] = F.adaptive_avg_pool2d(blob[ins[0]], 1)
            elif typ == "Scale": blob[outs[0]] = blob[ins[0]] * blob[ins[1]]
            elif typ == "Crop":
                ref = blob[ins[1]]; wo = fp(pr.get(0, "0")); ho = fp(pr.get(1, "0"))
                blob[outs[0]] = blob[ins[0]][:, :, ho:ho + ref.shape[2], wo:wo + ref.shape[3]]
            elif typ == "Eltwise": blob[outs[0]] = blob[ins[0]] + blob[ins[1]]
            else: raise NotImplementedError(typ)
        return blob[self.layers[-1][3][0]]

    def set_pad(self):
        # find the reflect pad that makes the padded output cover scale*input
        S = 64
        with torch.no_grad():
            o = self._run(torch.zeros(1, 3, S, S)).shape[2]
        deficit = self.scale * S - o
        self.pad = max(0, -(-deficit // (2 * self.scale)))  # ceil

    def forward(self, x):
        n, c, h, w = x.shape
        xp = F.pad(x, (self.pad, self.pad, self.pad, self.pad), 'reflect')
        out = self._run(xp)
        th, tw = h * self.scale, w * self.scale
        top = (out.shape[2] - th) // 2; left = (out.shape[3] - tw) // 2
        return out[:, :, top:top + th, left:left + tw]

def export_one(param_path, json_dir, out_dir):
    base = os.path.basename(param_path).replace(".param", "")
    json_path = os.path.join(json_dir, base + ".json")
    if not os.path.isfile(json_path):
        print(f"  SKIP {base}: no matching json"); return False
    scale = 2 if "scale2.0x" in base else 1
    net = CUNet(param_path, json_path, scale); net.set_pad()
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = net(dummy)
    if out.shape[2] != 64 * scale or out.shape[3] != 64 * scale:
        print(f"  ?? {base}: output {tuple(out.shape[2:])} != {64*scale}x{64*scale}");
    name = base.replace("_model", "").replace("scale2.0x", "scale2x")
    out_path = os.path.join(out_dir, f"waifu2x_cunet_{name}.onnx" if name else "waifu2x_cunet.onnx")
    torch.onnx.export(net, dummy, out_path, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out_path))
    print(f"  OK   {os.path.basename(out_path):42} x{scale} pad{net.pad}  in64 -> out{tuple(out.shape[2:])}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export waifu2x CUNet models to ONNX")
    parser.add_argument("--param-dir", required=True, help="Directory containing ncnn .param files")
    parser.add_argument("--json-dir", required=True, help="Directory containing waifu2x JSON weight files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print(f"=== waifu2x CUNet (.param topology + JSON fp32 weights) -> ONNX ===")
    n = sum(export_one(p, args.json_dir, out_dir) for p in sorted(glob.glob(os.path.join(args.param_dir, "*.param"))))
    print(f"done: {n} cunet models exported")

if __name__ == "__main__":
    main()
