"""Export waifu2x models to ONNX for --vpp-onnx.

waifu2x has no .pth; the weights live in the original JSON model files (a flat
list of convolution layers with weights and biases). This script reconstructs
the network straight from that JSON, so no waifu2x install is needed.

Covered (flat architectures, reconstructed exactly from JSON):
  - vgg_7   : 7 valid 3x3 convs, 1x (denoise). Input is reflect-padded by 7 so
              the output is the same size as the input.
  - upconv_7: 6 valid 3x3 convs + a 4x4 stride-2 transposed conv, 2x upscale.

Not covered here: the cunet architecture (a U-net with skip connections); its
JSON does not lay out cleanly as a flat stack. The Real-CUGAN UpCunet models
(../weights/realcugan) are the same cunet lineage and are already exported.

waifu2x is MIT (nagadomi). See ../ACKNOWLEDGMENTS.md.
"""
import os, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F

PAD = 7  # waifu2x offset: makes vgg_7 output 1x and upconv_7 output 2x exactly

class Waifu2xNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for L in layers:
            nIn, nOut, kW, kH = L['nInputPlane'], L['nOutputPlane'], L['kW'], L['kH']
            w = torch.tensor(L['weight'], dtype=torch.float32)
            b = torch.tensor(L['bias'], dtype=torch.float32)
            if L.get('dW', 1) == 2:  # transposed conv = the 2x upscale layer
                c = nn.ConvTranspose2d(nIn, nOut, (kH, kW), stride=2, padding=L.get('padW', 0))
                c.weight.data = w           # [in, out, kH, kW] matches ConvTranspose2d
            else:
                c = nn.Conv2d(nIn, nOut, (kH, kW), stride=1, padding=0)
                c.weight.data = w           # [out, in, kH, kW] matches Conv2d
            c.bias.data = b
            self.convs.append(c)
    def forward(self, x):
        x = F.pad(x, (PAD, PAD, PAD, PAD), 'reflect')
        for i, c in enumerate(self.convs):
            x = c(x)
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x, 0.1)
        return x

def clean_name(arch, path):
    sub = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path).replace('_model.json', '').replace('.json', '').replace('scale2.0x', 'scale2x')
    return f"waifu2x_{arch.replace('_','')}_{sub}_{name}.onnx".replace('__', '_')

def export_arch(arch, models_root, out_dir):
    files = sorted(glob.glob(os.path.join(models_root, arch, "**", "*.json"), recursive=True))
    n = 0
    for f in files:
        try:
            layers = json.load(open(f))
            if not isinstance(layers, list) or 'nInputPlane' not in layers[0]:
                raise ValueError("not a waifu2x model json")
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {type(e).__name__}")
            continue
        ch = layers[0]['nInputPlane']
        net = Waifu2xNet(layers).eval()
        dummy = torch.randn(1, ch, 64, 64)
        with torch.no_grad():
            out = net(dummy)
        out_path = os.path.join(out_dir, clean_name(arch, f))
        torch.onnx.export(net, dummy, out_path, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
        import onnx; onnx.checker.check_model(onnx.load(out_path))
        sc = out.shape[2] // 64
        print(f"  OK   {os.path.basename(out_path):46} {ch}ch x{sc}")
        n += 1
    return n

def main():
    parser = argparse.ArgumentParser(description="Export waifu2x vgg7/upconv7 JSON models to ONNX")
    parser.add_argument("--models-dir", required=True, help="Root directory containing waifu2x model JSONs (e.g. repos/waifu2x/models)")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print(f"=== waifu2x -> ONNX  (out: {out_dir}) ===")
    total = 0
    for arch in ["vgg_7", "upconv_7"]:
        print(f"-- {arch} --")
        total += export_arch(arch, args.models_dir, out_dir)
    print(f"done: {total} models exported (vgg_7 + upconv_7; cunet skipped, see header)")

if __name__ == "__main__":
    main()
