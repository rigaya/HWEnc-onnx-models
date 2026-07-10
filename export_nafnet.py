"""Export NAFNet deblur models to ONNX for --vpp-onnx (3ch RGB, scale 1).

NAFNet: "Simple Baselines for Image Restoration", Chen et al., ECCV 2022.
Source + weights: https://github.com/megvii-research/NAFNet  (MIT, megvii-model).

This adds the missing *deblur* capability to the --vpp-onnx model collection (the
bundle has denoise + JPEG-deblock, but no motion-deblur). Two pretrained models:

  nafnet_gopro_width32  single-image motion deblur (GoPro), ~17M params
  nafnet_reds_width64   video deblur (REDS dataset),        ~68M params

NAFNet is fully convolutional (SimpleGate + simplified channel attention +
LayerNorm; no FFT, no transformer), restores at the same resolution (scale 1), and
operates on [0,1] RGB with an input residual, matching the --vpp-onnx host range,
so only a final clamp to [0,1] is baked in. H/W are dynamic (the network pads
internally to a multiple of 16 and crops back; this exports dynamically).

The network is loaded from the vendored nafnet.py (the upstream NAFNet arch with
its custom autograd LayerNorm replaced by a mathematically identical, export-safe
nn.Module of the same weight/bias), so no basicsr install is needed.

NAFNet is MIT (megvii-model). See licenses/nafnet.txt.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
from onnx_export_common import export_onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nafnet import NAFNet

# (weights file, width, enc_blk_nums, middle_blk_num, dec_blk_nums)
# GoPro/REDS = deblur (deep [1,1,1,28] bottleneck); SIDD = denoise ([2,2,4,8]).
REGISTRY = {
    "nafnet_gopro_width32": ("NAFNet-GoPro-width32.pth", 32, [1, 1, 1, 28], 1,  [1, 1, 1, 1]),
    "nafnet_reds_width64":  ("NAFNet-REDS-width64.pth",  64, [1, 1, 1, 28], 1,  [1, 1, 1, 1]),
    "nafnet_sidd_width32":  ("NAFNet-SIDD-width32.pth",  32, [2, 2, 4, 8],  12, [2, 2, 2, 2]),
    "nafnet_sidd_width64":  ("NAFNet-SIDD-width64.pth",  64, [2, 2, 4, 8],  12, [2, 2, 2, 2]),
}


class Clamp01(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)


def export_one(name, weights_dir, out_dir):
    wfile, width, enc, mid, dec = REGISTRY[name]
    net = NAFNet(img_channel=3, width=width, enc_blk_nums=enc, middle_blk_num=mid, dec_blk_nums=dec).eval()
    sd = torch.load(os.path.join(weights_dir, wfile), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["params"] if "params" in sd else sd, strict=True)
    model = Clamp01(net).eval()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".onnx")
    export_onnx(
        model, torch.rand(1, 3, 64, 64), out, do_constant_folding=True, opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}})
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(out))
    rng = np.random.default_rng(0)
    # 40x56 is NOT a multiple of 16 -> also checks the dynamic internal pad/crop
    xt = rng.random((1, 3, 40, 56), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xt)).numpy()
    got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(["output"], {"input": xt})[0]
    assert got.shape == (1, 3, 40, 56), got.shape
    diff = float(np.max(np.abs(ref - got)))
    mean = float(np.mean(np.abs(ref - got)))
    print(f"  OK   {name:22} width{width}  scale1  max|ref-onnx|={diff:.2e} mean={mean:.2e}  ({os.path.getsize(out):,} b)")
    # threshold scales with model depth/width (float32 accumulation); a wrong
    # architecture would diverge by ~0.1+, this catches that while allowing the
    # benign accumulation of the deep 68M/116M models.
    assert diff < 5e-3 and mean < 1e-4, f"{name} mismatch max={diff} mean={mean}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--output", default="onnx")
    a = ap.parse_args()
    for name in REGISTRY:
        export_one(name, a.weights_dir, a.output)


if __name__ == "__main__":
    main()
