"""Export the KAIR-dialect RRDBNet photo models (ESRGAN / RealSR / FSSR) to ONNX.

These are the same older ESRGAN-style RRDBNet (key names conv_first / RRDB_trunk
/ trunk_conv) used by BSRGAN, so we reuse BSRGAN's self-contained network
definition (set BSRGAN_ROOT). 3-channel RGB in/out, 4x upscale, nb=23.
kaizen already ships these as modes (EsrganPhotoX4 / RealsrSmartphoneX4 /
RealsrJpegX4 / FssrSmartphoneX4 / FssrJpegX4), so including the ONNX is
consistent. Per-model upstream attribution is in ../ACKNOWLEDGMENTS.md.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import torch
from onnx_export_common import export_onnx

# filename -> scale factor (all 23-block 4x RRDBNet)
MODELS = {
    "ESRGAN.pth":     4,  # xinntao ESRGAN (DF2K general), Apache-2.0
    "RealSR_DPED.pth":  4,  # RealSR, smartphone-camera-trained
    "RealSR_JPEG.pth":  4,  # RealSR, JPEG-degradation-trained
    "FSSR_DPED.pth":    4,  # Frequency-Separation SR, DPED
    "FSSR_JPEG.pth":    4,  # Frequency-Separation SR, JPEG
}

def export_one(fname, sf, weights_dir, out_dir):
    pth = os.path.join(weights_dir, fname)
    if not os.path.isfile(pth):
        print(f"  SKIP {fname}: not found"); return False
    sd = torch.load(pth, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'params' in sd:
        sd = sd['params']
    net = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
    try:
        net.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"  SKIP {fname}: state_dict mismatch ({str(e)[:80]})"); return False
    net.eval()
    dummy = torch.rand(1, 3, 96, 96)
    out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".onnx")
    export_onnx(net, dummy, out, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {fname:18} -> {os.path.basename(out):26} (x{sf}, {os.path.getsize(out):,} b)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export KAIR ESRGAN/RealSR/FSSR models (.pth) to ONNX")
    parser.add_argument("--repo-root", required=True, help="Path to KAIR/BSRGAN repo root")
    parser.add_argument("--weights-dir", required=True, help="Directory containing .pth files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    global RRDBNet
    from models.network_rrdbnet import RRDBNet
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print(f"=== KAIR RRDBNet photo (ESRGAN / RealSR / FSSR) -> ONNX ===")
    n = sum(export_one(f, sf, args.weights_dir, out_dir) for f, sf in MODELS.items())
    print(f"done: {n}/{len(MODELS)} models exported")

if __name__ == "__main__":
    main()
