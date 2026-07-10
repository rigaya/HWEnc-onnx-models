"""Export super-image lightweight SR models to ONNX for --vpp-onnx (3ch RGB path).

super-image (https://github.com/eugenesiow/super-image, Apache-2.0) is a PyTorch
super-resolution model zoo. This converts the lightweight / moderate families
(skipping the heavy DRLN/HAN/RCAN/full-EDSR, which are too slow for video and
overlap the existing Real-ESRGAN/BSRGAN tier):

  pan / pan-bam     Pixel Attention Network            (~0.26M)
  carn / carn-bam   Cascading Residual Network         (~1.6M)
  a2n               Attention in Attention Network     (~1.0M)
  awsrn-bam         Adaptive Weighted SR (BAM variant) (~1.4M)
  msrn / msrn-bam   Multi-Scale Residual Network       (~6M)

each at x2 / x3 / x4 (awsrn has only the BAM weights published). BAM = Balanced
Attention Module, a small quality-boosting variant. 24 models total.

All operate on [0,1] RGB (CARN mean-shifts internally with rgb_range=1; the others
add a bilinear/residual path; MSRN defines a 255 mean-shift but leaves it disabled
in forward), matching the --vpp-onnx host range, so the only thing baked in is a
final clamp to [0,1]. H/W are dynamic.

Networks/configs load from the vendored super_image/ mini-package (original
upstream modeling/config files + a dependency-free base stub), so no super-image
install is needed -- only the pretrained weights.

super-image is Apache-2.0 (Eugene Siow et al.). See licenses/superimage.txt.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
from onnx_export_common import export_onnx


def build_registry(repo_root):
    sys.path.insert(0, repo_root)
    from super_image.models.pan.modeling_pan import PanModel
    from super_image.models.pan.configuration_pan import PanConfig
    from super_image.models.carn.modeling_carn import CarnModel
    from super_image.models.carn.configuration_carn import CarnConfig
    from super_image.models.a2n.modeling_a2n import A2nModel
    from super_image.models.a2n.configuration_a2n import A2nConfig
    from super_image.models.awsrn.modeling_awsrn import AwsrnModel
    from super_image.models.awsrn.configuration_awsrn import AwsrnConfig
    from super_image.models.msrn.modeling_msrn import MsrnModel
    from super_image.models.msrn.configuration_msrn import MsrnConfig
    return {
        "pan":       (PanModel,   PanConfig,   {}),
        "pan_bam":   (PanModel,   PanConfig,   {"bam": True}),
        "carn":      (CarnModel,  CarnConfig,  {}),
        "carn_bam":  (CarnModel,  CarnConfig,  {"bam": True}),
        "a2n":       (A2nModel,   A2nConfig,   {}),
        "awsrn_bam": (AwsrnModel, AwsrnConfig, {"bam": True}),
        "msrn":      (MsrnModel,  MsrnConfig,  {}),
        "msrn_bam":  (MsrnModel,  MsrnConfig,  {"bam": True}),
    }


def load_sd(path):
    # weights are trusted HF releases; some are DataParallel-saved (module. prefix)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


class Clamp01(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)


def export_one(name, net, scale, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".onnx")
    model = Clamp01(net).eval()
    dummy = torch.rand(1, 3, 32, 48)
    export_onnx(
        model, dummy, out, do_constant_folding=True, opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}})
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(out))
    rng = np.random.default_rng(0)
    xt = rng.random((1, 3, 40, 56), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xt)).numpy()
    got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(["output"], {"input": xt})[0]
    assert got.shape == (1, 3, 40 * scale, 56 * scale), got.shape
    diff = float(np.max(np.abs(ref - got)))
    print(f"  OK   {name:14} x{scale}  out{got.shape[2]}x{got.shape[3]}  max|ref-onnx|={diff:.2e}  ({os.path.getsize(out):,} b)")
    assert diff < 1e-4, f"{name} mismatch {diff}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--output", default="onnx")
    a = ap.parse_args()

    reg = build_registry(a.repo_root)
    for key, (Model, Config, extra) in reg.items():
        for scale in (2, 3, 4):
            wpath = os.path.join(a.weights_dir, f"{key}_{scale}x.pt")
            if not os.path.isfile(wpath):
                print(f"  SKIP {key}_{scale}x (no weights)")
                continue
            net = Model(Config(scale=scale, **extra)).eval()
            net.load_state_dict(load_sd(wpath), strict=True)
            export_one(f"{key}_{scale}x", net, scale, a.output)


if __name__ == "__main__":
    main()
