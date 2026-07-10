"""Export FBCNN (Flexible Blind JPEG Artifacts Removal) to ONNX for --vpp-onnx.

FBCNN: "Towards Flexible Blind JPEG Artifacts Removal", Jiang et al., ICCV 2021.
Source + weights: https://github.com/jiaxi-jiang/FBCNN  (Apache-2.0).

FBCNN restores compressed (JPEG/blocking) images at the same resolution
(scale = 1) and has a controllable target quality factor (QF). Two presentations
are exported, mirroring how this repo already handles DnCNN (blind) and
FFDNet/FDnCNN (with a level input):

  - blind   : image -> restored image; the QF is auto-estimated by the network.
              gray = 1->1 (onnx LumaSR, scale 1), color = 3->3 (onnx RGB, scale 1).
  - flexible: the image plus a constant QF plane are presented as a single tensor
              [image | qf-plane]; a thin wrapper extracts the scalar QF (the
              --vpp-onnx 'noise' param / 255, 0 = strongest removal .. 1 = mildest)
              and runs the conditioned network.
              gray = 2->1 (onnx GrayNoise), color = 4->3 (onnx RGBNoise), scale 1.

FBCNN operates on [0,1] RGB / luma, matching the --vpp-onnx host normalisation,
so no value rescale is baked in. Input height/width must be a multiple of 8
(four downsample levels), the same constraint as DRUNet in this repo.

FBCNN is Apache-2.0 (Jiaxi Jiang, Kai Zhang, Radu Timofte). See licenses/.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
from onnx_export_common import export_onnx


def import_fbcnn(repo_root):
    sys.path.insert(0, repo_root)
    from models.network_fbcnn import FBCNN  # FBCNN repo: models/network_fbcnn.py
    return FBCNN


# FBCNN.forward pads to a multiple of 8 with np.ceil(h/8), which bakes a constant
# crop at trace time and breaks dynamic shapes. We replicate the inner forward
# without that pad/crop (valid for multiple-of-8 input, like DRUNet), so the ONNX
# is dynamic in H/W.
def fbcnn_core(net, x, qf_input):
    x1 = net.m_head(x)
    x2 = net.m_down1(x1)
    x3 = net.m_down2(x2)
    x4 = net.m_down3(x3)
    x = net.m_body_encoder(x4)
    qf = net.qf_pred(x)
    x = net.m_body_decoder(x)
    emb = net.qf_embed(qf_input if qf_input is not None else qf)
    g3, b3 = net.to_gamma_3(emb), net.to_beta_3(emb)
    g2, b2 = net.to_gamma_2(emb), net.to_beta_2(emb)
    g1, b1 = net.to_gamma_1(emb), net.to_beta_1(emb)
    x = x + x4
    x = net.m_up3[0](x)
    for i in range(net.nb):
        x = net.m_up3[i + 1](x, g3, b3)
    x = x + x3
    x = net.m_up2[0](x)
    for i in range(net.nb):
        x = net.m_up2[i + 1](x, g2, b2)
    x = x + x2
    x = net.m_up1[0](x)
    for i in range(net.nb):
        x = net.m_up1[i + 1](x, g1, b1)
    x = x + x1
    x = net.m_tail(x)
    return x


class FBCNNBlind(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return fbcnn_core(self.net, x, None)


class FBCNNFlex(nn.Module):
    """[image | qf-plane] -> restored image. The qf plane is constant (filled by
    the filter); its value is the target quality factor in [0,1]."""
    def __init__(self, net, image_ch):
        super().__init__()
        self.net = net
        self.image_ch = image_ch

    def forward(self, x):
        img = x[:, :self.image_ch]
        qf = x[:, self.image_ch:self.image_ch + 1, :1, :1].reshape(-1, 1)
        return fbcnn_core(self.net, img, qf)


def load_net(FBCNN, pth, nch):
    net = FBCNN(in_nc=nch, out_nc=nch, nc=[64, 128, 256, 512], nb=4, act_mode='R')
    net.load_state_dict(torch.load(pth, map_location='cpu', weights_only=True), strict=True)
    return net.eval()


def export_one(name, wrap, in_ch, out_ch, ref_fn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".onnx")
    dummy = torch.rand(1, in_ch, 64, 64)  # 64 is a multiple of 8 -> no padding baked
    export_onnx(
        wrap.eval(), dummy, out, do_constant_folding=True, opset_version=17,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},
                      'output': {0: 'batch', 2: 'height', 3: 'width'}})
    # self-verify vs the reference forward on a different multiple-of-8 size
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(out))
    rng = np.random.default_rng(0)
    xt = rng.random((1, in_ch, 40, 56), dtype=np.float32)
    with torch.no_grad():
        ref = ref_fn(torch.from_numpy(xt)).numpy()
    got = ort.InferenceSession(out, providers=['CPUExecutionProvider']).run(['output'], {'input': xt})[0]
    assert got.shape == (1, out_ch, 40, 56), got.shape
    diff = float(np.max(np.abs(ref - got)))
    print(f"  OK   {name:22} in{in_ch} out{out_ch} scale1  max|ref-onnx|={diff:.2e}  ({os.path.getsize(out):,} b)")
    assert diff < 1e-4, f"{name} mismatch {diff}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="FBCNN repo root (contains models/network_fbcnn.py)")
    ap.add_argument("--weights-dir", default="weights",
                    help="directory with fbcnn_color.pth / fbcnn_gray.pth")
    ap.add_argument("--output", default="onnx")
    a = ap.parse_args()

    FBCNN = import_fbcnn(a.repo_root)
    for tag, nch in [("color", 3), ("gray", 1)]:
        net = load_net(FBCNN, os.path.join(a.weights_dir, f"fbcnn_{tag}.pth"), nch)

        # blind: reference is the original FBCNN.forward (image only) -> take image output
        def ref_blind(x, _net=net):
            return _net(x, None)[0]
        export_one(f"fbcnn_{tag}_blind", FBCNNBlind(net), nch, nch, ref_blind, a.output)

        # flexible: reference splits [image|qf] and runs the two-arg forward
        def ref_flex(x, _net=net, _ch=nch):
            img = x[:, :_ch]
            qf = x[:, _ch:_ch + 1, :1, :1].reshape(-1, 1)
            return _net(img, qf)[0]
        export_one(f"fbcnn_{tag}_flex", FBCNNFlex(net, nch), nch + 1, nch, ref_flex, a.output)


if __name__ == "__main__":
    main()
