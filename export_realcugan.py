"""Export Real-CUGAN UpCunet models (.pth) to ONNX for --vpp-onnx.

Real-CUGAN has three networks (UpCunet2x / 3x / 4x); each weight file is one of
them at a given denoise level (no-denoise / conservative / denoise1x..3x). The
original forward() has tile / cache / byte-conversion branches that ONNX cannot
trace; we wrap each network with the plain tile_mode=0 path and return the
float 0..1 result (skipping the *255 byte step), which is what onnx wants.

The network definitions are imported from the upstream Real-CUGAN repo (set
CUGAN_ROOT). Real-CUGAN is MIT (bilibili). See ../ACKNOWLEDGMENTS.md.
"""

import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from onnx_export_common import export_onnx

# ONNX-traceable wrappers: tile_mode=0 path, fixed symmetric reflect pad
# (caller feeds aligned dims), float 0..1 output.
class Export2x(nn.Module):
    def __init__(self, b): super().__init__(); self.unet1, self.unet2 = b.unet1, b.unet2
    def forward(self, x):
        x = F.pad(x, (18, 18, 18, 18), 'reflect')
        x = self.unet1(x); x0 = self.unet2(x, 1.0)
        x = F.pad(x, (-20, -20, -20, -20))
        return torch.add(x0, x)

class Export3x(nn.Module):
    def __init__(self, b): super().__init__(); self.unet1, self.unet2 = b.unet1, b.unet2
    def forward(self, x):
        x = F.pad(x, (14, 14, 14, 14), 'reflect')
        x = self.unet1(x); x0 = self.unet2(x, 1.0)
        x = F.pad(x, (-20, -20, -20, -20))
        return torch.add(x0, x)

class Export4x(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.unet1, self.unet2, self.ps, self.conv_final = b.unet1, b.unet2, b.ps, b.conv_final
    def forward(self, x):
        x00 = x
        x = F.pad(x, (19, 19, 19, 19), 'reflect')
        x = self.unet1(x); x0 = self.unet2(x, 1.0)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        x = self.conv_final(x)
        x = F.pad(x, (-1, -1, -1, -1))
        x = self.ps(x)
        return x + F.interpolate(x00, scale_factor=4, mode='nearest')

SCALES = {}

def export_one(fname, weights_dir, out_dir):
    prefix = fname.split("-")[0]
    if prefix not in SCALES:
        print(f"  SKIP {fname}: unknown scale prefix"); return False
    BaseCls, WrapCls = SCALES[prefix]
    base = BaseCls(in_channels=3, out_channels=3)
    state = torch.load(os.path.join(weights_dir, fname), map_location='cpu', weights_only=True)
    try:
        base.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"  SKIP {fname}: state_dict mismatch ({e})"); return False
    net = WrapCls(base).eval()
    dummy = torch.randn(1, 3, 64, 64)  # 64 is divisible by 4 and even -> valid for all scales
    with torch.no_grad():
        out = net(dummy)
    out_name = os.path.splitext(fname)[0].replace('-', '_') + ".onnx"
    out_path = os.path.join(out_dir, out_name)
    export_onnx(net, dummy, out_path, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out_path))
    print(f"  OK   {fname} -> {out_name}  in 64x64 -> out {tuple(out.shape)[2:]} ({os.path.getsize(out_path):,} b)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Real-CUGAN models (.pth) to ONNX")
    parser.add_argument("--repo-root", required=True, help="Path to Real-CUGAN repo root")
    parser.add_argument("--weights-dir", required=True, help="Directory containing .pth files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    from upcunet_v3 import UpCunet2x, UpCunet3x, UpCunet4x
    global SCALES
    SCALES = {"up2x": (UpCunet2x, Export2x), "up3x": (UpCunet3x, Export3x), "up4x": (UpCunet4x, Export4x)}
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print(f"=== Real-CUGAN -> ONNX  (out: {out_dir}) ===")
    files = sorted(f for f in os.listdir(args.weights_dir) if f.endswith(".pth"))
    n = sum(export_one(f, args.weights_dir, out_dir) for f in files)
    print(f"done: {n}/{len(files)} models exported")

if __name__ == "__main__":
    main()
