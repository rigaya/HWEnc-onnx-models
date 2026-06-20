"""Export BSRGAN models (.pth) to ONNX for --vpp-onnx.

BSRGAN uses the older ESRGAN-style RRDBNet (key names RRDB_trunk / trunk_conv),
which differs from Real-ESRGAN's, so we import BSRGAN's own self-contained
network definition (set BSRGAN_ROOT). 3-channel RGB in/out.
BSRGAN is Apache-2.0 / its repo licence (Kai Zhang). See ../ACKNOWLEDGMENTS.md.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import torch

# filename -> scale factor
MODELS = {"BSRGAN.pth": 4, "BSRGANx2.pth": 2, "BSRNet.pth": 4}

def export_one(fname, sf, weights_dir, out_dir):
    pth = os.path.join(weights_dir, fname)
    if not os.path.isfile(pth):
        print(f"  SKIP {fname}: not found"); return False
    net = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
    state = torch.load(pth, map_location='cpu', weights_only=True)
    try:
        net.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"  SKIP {fname}: state_dict mismatch ({e})"); return False
    net.eval()
    dummy = torch.randn(1, 3, 128, 128)
    out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".onnx")
    torch.onnx.export(net, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {fname} -> {os.path.basename(out)} (x{sf}) ({os.path.getsize(out):,} b)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export BSRGAN models (.pth) to ONNX")
    parser.add_argument("--repo-root", required=True, help="Path to KAIR/BSRGAN repo root")
    parser.add_argument("--weights-dir", required=True, help="Directory containing .pth files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    global RRDBNet
    from models.network_rrdbnet import RRDBNet
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print(f"=== BSRGAN -> ONNX  (out: {out_dir}) ===")
    n = sum(export_one(f, sf, args.weights_dir, out_dir) for f, sf in MODELS.items())
    print(f"done: {n}/{len(MODELS)} models exported")

if __name__ == "__main__":
    main()
