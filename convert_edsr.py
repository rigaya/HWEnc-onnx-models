#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from onnx_export_common import export_onnx


def default_conv(in_ch, out_ch, k, bias=True):
    return nn.Conv2d(in_ch, out_ch, k, padding=(k // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, k, bias=True, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        layers = []
        for i in range(2):
            layers.append(conv(n_feats, n_feats, k, bias=bias))
            if i == 0:
                layers.append(act)
        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bias=True):
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(conv(n_feats, 4 * n_feats, 3, bias))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(conv(n_feats, 9 * n_feats, 3, bias))
            layers.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        super().__init__(*layers)


class EDSR(nn.Module):
    def __init__(self, scale, n_resblocks=16, n_feats=64, res_scale=1,
                 n_colors=3, rgb_range=255, conv=default_conv):
        super().__init__()
        k = 3
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.head = nn.Sequential(conv(n_colors, n_feats, k))
        body = [ResBlock(conv, n_feats, k, act=act, res_scale=res_scale)
                for _ in range(n_resblocks)]
        body.append(conv(n_feats, n_feats, k))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(Upsampler(conv, scale, n_feats),
                                  conv(n_feats, n_colors, k))

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


class EDSRNorm(nn.Module):
    def __init__(self, edsr, rgb_range=255):
        super().__init__()
        self.edsr = edsr
        self.rgb_range = float(rgb_range)

    def forward(self, x):
        y = self.edsr(x * self.rgb_range)
        y = y / self.rgb_range
        return torch.clamp(y, 0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[2, 3, 4])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = EDSR(scale=args.scale, n_resblocks=16, n_feats=64, res_scale=1)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict, strict=True)
    model = EDSRNorm(net).eval()

    dummy = torch.rand(1, 3, 32, 48)
    export_onnx(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"},
                      "output": {0: "N", 2: "Hs", 3: "Ws"}},
        opset_version=args.opset, do_constant_folding=True)

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(out_path)))

    rng = np.random.default_rng(0)
    xt = rng.random((1, 3, 40, 56), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xt)).numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(["output"], {"input": xt})[0]
    expected_shape = (1, 3, 40 * args.scale, 56 * args.scale)
    if got.shape != expected_shape:
        raise RuntimeError(f"unexpected output shape: {got.shape}, expected {expected_shape}")
    diff = float(np.max(np.abs(ref - got)))
    if diff >= 1e-4:
        raise RuntimeError(f"torch vs onnx mismatch {diff}")

    flat = np.full((1, 3, 24, 24), 0.5, dtype=np.float32)
    of = sess.run(["output"], {"input": flat})[0]
    print(f"[x{args.scale}] out shape {got.shape}  max|torch-onnx|={diff:.2e}  "
          f"flat0.5 -> mean {of.mean():.4f} std {of.std():.4f}")
    print(f"OK  {out_path}")


if __name__ == "__main__":
    main()
