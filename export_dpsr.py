"""Export KAIR DPSR models (.pth) to ONNX for --vpp-onnx.

DPSR is MSRResNet_prior: input is 4 channels (RGB + one noise-level map),
output is 3-channel RGB at the model's upscale (x2/x3/x4). The network code is
imported from the KAIR repo (set KAIR_ROOT). KAIR is MIT (Kai Zhang).
See ../ACKNOWLEDGMENTS.md.

Note: these are 4-channel (noise-conditioned) models; onnx runs them once
its RGB+noise tier lands (see ../PLAN.md). The ONNX is valid and exported now.
"""
import os, sys, re, argparse, warnings
warnings.filterwarnings("ignore")
import torch

MODELS = ["dpsr_x2.pth", "dpsr_x3.pth", "dpsr_x4.pth", "dpsr_x4_gan.pth"]

def export_one(fname, weights_dir, out_dir):
    pth = os.path.join(weights_dir, fname)
    if not os.path.isfile(pth):
        print(f"  SKIP {fname}: not found"); return False
    scale = int(re.search(r"x(\d)", fname).group(1))
    net = MSRResNet_prior(in_nc=4, out_nc=3, nc=96, nb=16, upscale=scale,
                          act_mode='R', upsample_mode='pixelshuffle')
    state = torch.load(pth, map_location='cpu', weights_only=True)
    net.load_state_dict(state, strict=False); net.eval()
    dummy = torch.randn(1, 4, 64, 64)
    out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".onnx")
    torch.onnx.export(net, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {fname} -> {os.path.basename(out)} (x{scale}, 4ch in) ({os.path.getsize(out):,} b)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export KAIR DPSR models (.pth) to ONNX")
    parser.add_argument("--repo-root", required=True, help="Path to KAIR repo root")
    parser.add_argument("--weights-dir", required=True, help="Directory containing .pth files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    global MSRResNet_prior
    from models.network_dpsr import MSRResNet_prior
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print(f"=== DPSR -> ONNX  (out: {out_dir}) ===")
    n = sum(export_one(f, args.weights_dir, out_dir) for f in MODELS)
    print(f"done: {n}/{len(MODELS)} models exported")

if __name__ == "__main__":
    main()
