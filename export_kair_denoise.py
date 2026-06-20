"""Export KAIR denoise/deblock models (.pth) to ONNX for --vpp-onnx.

Covers three KAIR families (all MIT, Kai Zhang). The network definitions are
imported from the KAIR repo (set KAIR_ROOT); the architecture (in/out channels,
depth, BatchNorm presence) is inferred from each weight file so one loader
serves every variant.

  - DnCNN   : plain Conv+ReLU stack, residual forward (output = x - net(x), so
              the ONNX already emits the clean image). gray (1ch) or color (3ch),
              scale 1.  -> onnx LumaSR (1->1) or RGB (3->3).
  - FDnCNN  : DnCNN with a noise-sigma map concatenated to the image at the head;
              the input tensor is [image | sigma] so in_nc = image_ch + 1, output
              is the absolute denoised image. gray (2->1) or color (4->1/3).
              -> onnx GrayNoise (2->1) or RGBNoise (4->3).
  - FFDNet  : PixelUnshuffle + sigma-map concat + PixelShuffle. forward(x, sigma)
              takes the image and a scalar sigma separately, so a thin wrapper
              presents a single [image | sigma-plane] tensor (the filter fills the
              sigma plane with a constant) and extracts the scalar internally.
              gray (2->1) or color (4->3), scale 1.

KAIR is MIT (Kai Zhang). See ../ACKNOWLEDGMENTS.md.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn

def load_sd(fname, weights_dir):
    sd = torch.load(os.path.join(weights_dir, fname), map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'params' in sd:
        sd = sd['params']
    return sd

def conv_weights(sd):
    return [v for k, v in sd.items() if k.endswith('.weight') and v.dim() == 4]

def has_bn(sd):
    return any('running_mean' in k for k in sd)

# ---- DnCNN / FDnCNN ----------------------------------------------------------
def export_dncnn_like(fname, family, weights_dir, out_dir):
    sd = load_sd(fname, weights_dir)
    cw = conv_weights(sd)
    in_nc  = cw[0].shape[1]
    out_nc = cw[-1].shape[0]
    nb     = len(cw)
    nc     = cw[0].shape[0]
    act    = 'BR' if has_bn(sd) else 'R'
    Net    = DnCNN if family == 'dncnn' else FDnCNN
    net = Net(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act)
    try:
        net.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"  SKIP {fname}: state_dict mismatch ({str(e)[:80]})"); return False
    net.eval()
    dummy = torch.rand(1, in_nc, 64, 64)
    out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".onnx")
    torch.onnx.export(net, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {fname:24} -> {os.path.basename(out):28} in{in_nc} out{out_nc} nb{nb} act={act}")
    return True

# ---- FFDNet (single-input wrapper) -------------------------------------------
class FFDNetWrap(nn.Module):
    """Present FFDNet as a single [image | sigma-plane] tensor: split off the
    image and a scalar sigma (the filter fills the sigma plane with a constant),
    then run the two-input FFDNet."""
    def __init__(self, net, image_ch):
        super().__init__()
        self.net = net
        self.image_ch = image_ch
    def forward(self, x):
        img   = x[:, :self.image_ch]
        sigma = x[:, self.image_ch:self.image_ch + 1, :1, :1]  # constant plane -> scalar
        return self.net(img, sigma)

def export_ffdnet(fname, weights_dir, out_dir):
    sd = load_sd(fname, weights_dir)
    cw = conv_weights(sd)
    head_in = cw[0].shape[1]            # = image_ch*4 + 1
    image_ch = (head_in - 1) // 4
    out_nc   = cw[-1].shape[0] // 4
    nc       = cw[0].shape[0]
    nb       = len(cw)
    net = FFDNet(in_nc=image_ch, out_nc=out_nc, nc=nc, nb=nb, act_mode='R')
    try:
        net.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"  SKIP {fname}: state_dict mismatch ({str(e)[:80]})"); return False
    net.eval()
    wrap = FFDNetWrap(net, image_ch).eval()
    dummy = torch.rand(1, image_ch + 1, 64, 64)
    with torch.no_grad():
        o = wrap(dummy)
    out = os.path.join(out_dir, os.path.splitext(fname)[0] + ".onnx")
    torch.onnx.export(wrap, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {fname:24} -> {os.path.basename(out):28} in{image_ch+1} out{out_nc} nb{nb}  ({tuple(o.shape[1:])})")
    return True

DNCNN  = ["dncnn_gray_blind.pth", "dncnn_color_blind.pth", "dncnn3.pth",
          "dncnn_15.pth", "dncnn_25.pth", "dncnn_50.pth"]
FDNCNN = ["fdncnn_gray.pth", "fdncnn_gray_clip.pth", "fdncnn_color.pth", "fdncnn_color_clip.pth"]
FFDNET = ["ffdnet_gray.pth", "ffdnet_gray_clip.pth", "ffdnet_color.pth", "ffdnet_color_clip.pth"]

def main():
    parser = argparse.ArgumentParser(description="Export KAIR DnCNN/FDnCNN/FFDNet models (.pth) to ONNX")
    parser.add_argument("--repo-root", required=True, help="Path to KAIR repo root")
    parser.add_argument("--weights-dir", required=True, help="Directory containing .pth files")
    parser.add_argument("--output", required=True, help="Output root directory (dncnn/fdncnn/ffdnet subdirs created)")
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)
    global DnCNN, FDnCNN, FFDNet
    from models.network_dncnn import DnCNN, FDnCNN
    from models.network_ffdnet import FFDNet
    out_root = os.path.abspath(args.output)
    n = 0
    d = os.path.join(out_root, "dncnn");  os.makedirs(d, exist_ok=True)
    print("=== DnCNN -> ONNX ===")
    n += sum(export_dncnn_like(f, 'dncnn', args.weights_dir, d) for f in DNCNN)
    d = os.path.join(out_root, "fdncnn"); os.makedirs(d, exist_ok=True)
    print("=== FDnCNN -> ONNX ===")
    n += sum(export_dncnn_like(f, 'fdncnn', args.weights_dir, d) for f in FDNCNN)
    d = os.path.join(out_root, "ffdnet"); os.makedirs(d, exist_ok=True)
    print("=== FFDNet -> ONNX ===")
    n += sum(export_ffdnet(f, args.weights_dir, d) for f in FFDNET)
    print(f"done: {n} models exported")

if __name__ == "__main__":
    main()
