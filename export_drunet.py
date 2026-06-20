"""Export KAIR DRUNet models (.pth) to ONNX for --vpp-onnx.

DRUNet is UNetRes (denoise/restore at the same resolution, scale=1). Input is
the image plus one noise-level map: color = 4 channels in / 3 out, grayscale =
2 channels in / 1 out. The network code is imported from the KAIR repo (set
KAIR_ROOT). KAIR is MIT (Kai Zhang). See ../ACKNOWLEDGMENTS.md.

These are noise-conditioned (and scale=1) models; onnx runs them once its
restore/RGB+noise tiers land (see ../PLAN.md). Input height/width must be a
multiple of 8 (four downsample levels).
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import torch

MODELS = ["drunet_color.pth", "drunet_deblocking_color.pth",
          "drunet_gray.pth", "drunet_deblocking_grayscale.pth"]

def export_one(fname, weights_dir, out_dir):
    pth = os.path.join(weights_dir, fname)
    if not os.path.isfile(pth):
        print(f"  SKIP {fname}: not found"); return False
    gray = ("gray" in fname or "grayscale" in fname)
    in_nc, out_nc = (2, 1) if gray else (4, 3)
    net = UNetRes(in_nc=in_nc, out_nc=out_nc, nc=[64, 128, 256, 512], nb=4,
                  act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False)
    state = torch.load(pth, map_location='cpu', weights_only=True)
    net.load_state_dict(state, strict=True); net.eval()
    dummy = torch.randn(1, in_nc, 64, 64)
    out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".onnx")
    torch.onnx.export(net, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {fname} -> {os.path.basename(out)} ({in_nc}ch in / {out_nc}ch out, scale=1) ({os.path.getsize(out):,} b)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export KAIR DRUNet models (.pth) to ONNX")
    parser.add_argument("--repo-root", required=True, help="Path to KAIR repo root")
    parser.add_argument("--weights-dir", required=True, help="Directory containing .pth files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    global UNetRes
    from models.network_unet import UNetRes
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print(f"=== DRUNet -> ONNX  (out: {out_dir}) ===")
    n = sum(export_one(f, args.weights_dir, out_dir) for f in MODELS)
    print(f"done: {n}/{len(MODELS)} models exported")

if __name__ == "__main__":
    main()
