"""Export Real-ESRGAN models (.pth) to ONNX for --vpp-onnx.

Real-ESRGAN ships two network shapes:
  - RRDBNet      : RealESRGAN_x4plus / _x2plus / _x4plus_anime_6B / RealESRNet
  - SRVGGNetCompact : realesr-animevideov3 / realesr-general-*-x4v3

Both network definitions are included below so this script is self-contained
(no need to install Real-ESRGAN or basicsr). Edit MODELS_DIR / OUT_DIR if your
weights live elsewhere.

Real-ESRGAN is BSD-3-Clause (Xintao Wang). See ../ACKNOWLEDGMENTS.md.
"""

import os, time, argparse, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F

# ---- edit these two paths if needed ----
# -----------------------------------------

# ---------------- RRDBNet (BasicSR / Real-ESRGAN key naming) ----------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = ResidualDenseBlock(nf, gc), ResidualDenseBlock(nf, gc), ResidualDenseBlock(nf, gc)
    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

def pixel_unshuffle(x, scale):
    b, c, h, w = x.shape
    return x.view(b, c, h // scale, scale, w // scale, scale).permute(0, 1, 3, 5, 2, 4).reshape(
        b, c * scale * scale, h // scale, w // scale)

class RRDBNet(nn.Module):
    """BasicSR / Real-ESRGAN RRDBNet. The body always upsamples 4x via two
    nearest+conv steps; for net scale 2 or 1 the input is pixel-unshuffled first
    (so x2 -> 12ch, x1 -> 48ch), which is why those .pth have a 12/48-channel
    conv_first."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        if scale == 2:   num_in_ch *= 4
        elif scale == 1: num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        feat = pixel_unshuffle(x, 2) if self.scale == 2 else (pixel_unshuffle(x, 4) if self.scale == 1 else x)
        fea = self.conv_first(feat)
        fea = fea + self.conv_body(self.body(fea))
        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(fea)))

# ---------------- SRVGGNetCompact (realesr-*v3) ----------------
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)
    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')

# ---------------- model registry (filename -> builder) ----------------
def rrdb(nb, sf): return lambda: RRDBNet(3, 3, 64, nb, 32, sf)
def srvgg(nc):    return lambda: SRVGGNetCompact(3, 3, 64, nc, 4)

REGISTRY = {
    "RealESRGAN_x4plus.pth":          rrdb(23, 4),
    "RealESRGAN_x4plus_anime_6B.pth": rrdb(6, 4),
    "RealESRGAN_x2plus.pth":          rrdb(23, 2),
    "RealESRNet_x4plus.pth":          rrdb(23, 4),
    "realesr-animevideov3.pth":       srvgg(16),
    "realesr-general-x4v3.pth":       srvgg(32),
    "realesr-general-wdn-x4v3.pth":   srvgg(32),
}

def export_one(fname, builder, models_dir, out_dir):
    pth = os.path.join(models_dir, fname)
    if not os.path.isfile(pth):
        print(f"  SKIP {fname}: not found"); return False
    net = builder(); net.eval()
    raw = torch.load(pth, map_location='cpu', weights_only=True)
    state = raw.get('params_ema', raw.get('params', raw))
    try:
        net.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"  SKIP {fname}: state_dict mismatch ({e})"); return False
    dummy = torch.randn(1, 3, 128, 128)
    out = os.path.join(out_dir, os.path.splitext(fname)[0].replace('-', '_') + ".onnx")
    torch.onnx.export(net, dummy, out, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch',2:'height',3:'width'}, 'output': {0:'batch',2:'height',3:'width'}})
    import onnx; onnx.checker.check_model(onnx.load(out))
    print(f"  OK   {fname} -> {os.path.basename(out)} ({os.path.getsize(out):,} bytes)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Real-ESRGAN models (.pth) to ONNX")
    parser.add_argument("--models-dir", required=True, help="Directory containing Real-ESRGAN .pth files")
    parser.add_argument("--output", required=True, help="Output directory for ONNX files")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.output); os.makedirs(out_dir, exist_ok=True)
    print(f"=== Real-ESRGAN -> ONNX  (out: {out_dir}) ===")
    n = sum(export_one(f, b, args.models_dir, out_dir) for f, b in REGISTRY.items())
    print(f"done: {n}/{len(REGISTRY)} models exported")

if __name__ == "__main__":
    main()
