"""Export KAIR SRMD / SRMDNF models (.pth) to ONNX for --vpp-onnx.

SRMD's head takes the RGB image plus a 15-d bicubic-kernel Degradation Map (and,
for the noise variants, a 16th noise-level channel). The onnx filter only
feeds 1-4 channel tensors, so this exporter BAKES the fixed bicubic DMap (PCA
basis @ bicubic kernel, computed exactly as the kaizen team's extract script
does) as a constant inside a thin wrapper. The ONNX input then becomes:

  - srmdnf_xN : 3ch RGB           -> onnx RGB mode
  - srmd_xN   : 4ch RGB + noise   -> onnx RGBNoise mode (the filter's sigma
                                     plane becomes SRMD's noise channel; the
                                     wrapper inserts the 15 DMap planes before it)

The KAIR SRMD class already includes the pixelshuffle upsampler, so the ONNX
output is the finished 3ch RGB at scale. KAIR is MIT (Kai Zhang). See
../ACKNOWLEDGMENTS.md.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np, torch, torch.nn as nn
import scipy.io as sio
from onnx_export_common import export_onnx

PCA_MAT     = "srmd_pca_matlab.mat"
KERNELS_MAT = "kernels_bicubicx234.mat"

# (filename, scale, noise_flag)
TARGETS = [
    ("srmd_x2.pth",   2, 1), ("srmd_x3.pth",   3, 1), ("srmd_x4.pth",   4, 1),
    ("srmdnf_x2.pth", 2, 0), ("srmdnf_x3.pth", 3, 0), ("srmdnf_x4.pth", 4, 0),
]

def compute_bicubic_dmaps(folder):
    """{2,3,4} -> (15,) float32, bicubic kernel projected through KAIR's PCA basis."""
    P  = sio.loadmat(os.path.join(folder, PCA_MAT))["P"]          # (15, 225)
    ks = sio.loadmat(os.path.join(folder, KERNELS_MAT))["kernels"] # (1, 3) object
    dmaps = {}
    for idx, scale in enumerate((2, 3, 4)):
        k15 = ks[0, idx][5:20, 5:20]                              # center-crop 25x25 -> 15x15
        dmaps[scale] = (P @ k15.flatten()).astype(np.float32)     # (15,)
    return dmaps

class SRMDWrap(nn.Module):
    """Bake the constant DMap; present a single RGB(+noise) tensor."""
    def __init__(self, net, dmap15, noise_flag):
        super().__init__()
        self.net = net
        self.noise_flag = noise_flag
        self.register_buffer("dmap", torch.tensor(dmap15, dtype=torch.float32).view(1, 15, 1, 1))
    def forward(self, x):
        rgb = x[:, :3]
        h, w = x.shape[-2], x.shape[-1]
        dmap = self.dmap.repeat(1, 1, h, w)
        if self.noise_flag:
            full = torch.cat([rgb, dmap, x[:, 3:4]], dim=1)   # 19ch: RGB + DMap(15) + noise
        else:
            full = torch.cat([rgb, dmap], dim=1)              # 18ch: RGB + DMap(15)
        return self.net(full)

def export_one(fname, scale, noise_flag, dmap15, weights_dir, out_dir):
    pth = os.path.join(weights_dir, fname)
    if not os.path.isfile(pth):
        print(f"  SKIP {fname}: not found"); return False
    in_nc = 3 + 15 + (1 if noise_flag else 0)
    net = SRMD(in_nc=in_nc, out_nc=3, nc=128, nb=12, upscale=scale, act_mode='R', upsample_mode='pixelshuffle')
    blob = torch.load(pth, map_location='cpu', weights_only=True)
    state = blob.get('params', blob) if isinstance(blob, dict) else blob
    try:
        net.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"  SKIP {fname}: state_dict mismatch ({str(e)[:80]})"); return False
    net.eval()
    wrap = SRMDWrap(net, dmap15, noise_flag).eval()
    in_ch = 4 if noise_flag else 3
    dummy = torch.rand(1, in_ch, 48, 48)
    with torch.no_grad():
        o = wrap(dummy)
    out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".onnx")
    export_onnx(wrap, dummy, out, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {fname:14} -> {os.path.basename(out):18} in{in_ch} x{scale}  out{tuple(o.shape[1:])}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export KAIR SRMD/SRMDNF models (.pth) to ONNX")
    parser.add_argument("--repo-root", required=True, help="Path to KAIR repo root")
    parser.add_argument("--weights-dir", required=True, help="Directory containing .pth files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    global SRMD
    from models.network_srmd import SRMD
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print("=== bicubic DMaps (KAIR PCA basis @ bicubic kernels) ===")
    dmaps = compute_bicubic_dmaps(args.weights_dir)
    for s in (2, 3, 4):
        print(f"  x{s}: " + " ".join(f"{v:+.2f}" for v in dmaps[s]))
    print("=== SRMD / SRMDNF -> ONNX ===")
    n = sum(export_one(f, s, nf, dmaps[s], args.weights_dir, out_dir) for f, s, nf in TARGETS)
    print(f"done: {n}/{len(TARGETS)} models exported")

if __name__ == "__main__":
    main()
