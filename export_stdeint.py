"""ST-DeInt を QSVEnc 用の ONNX へ出力する。"""

import argparse
import importlib
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from onnx_export_common import export_onnx


class GridSampleConvOffset2D(nn.Conv2d):
    """ST-DeInt の ConvOffset2D を ONNX GridSample で等価に実装する。"""

    def __init__(self, source):
        super().__init__(
            source.in_channels, source.out_channels, source.kernel_size,
            stride=source.stride, padding=source.padding, dilation=source.dilation,
            groups=source.groups, bias=False, padding_mode=source.padding_mode)
        self.load_state_dict(source.state_dict())

    def forward(self, x):
        batch, channels, height, width = x.shape
        # 元実装の contiguous().view(-1, H, W, 2) の要素順を変えない。
        offsets = super().forward(x).contiguous().view(-1, height, width, 2)
        data = x.contiguous().view(-1, 1, height, width)
        y, x_coord = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=x.dtype),
            torch.arange(width, device=x.device, dtype=x.dtype),
            indexing="ij")
        coordinates = offsets + torch.stack((y, x_coord), dim=-1)

        # 元実装の座標順は (y, x)、GridSample の座標順は (x, y)。
        grid = torch.stack((
            coordinates[..., 1] * (2.0 / (width - 1)) - 1.0,
            coordinates[..., 0] * (2.0 / (height - 1)) - 1.0,
        ), dim=-1)
        sampled = F.grid_sample(
            data, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.view(batch, channels, height, width)


class RestorationExportWrapper(nn.Module):
    """2フィールドの欠落行予測を半高の6chで返す。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_frame):
        restore_a, restore_b = self.model(input_frame)
        return torch.cat((restore_a, restore_b), dim=1)


def load_model(source_dir: Path, checkpoint: Path, no_deform: bool = False) -> RestorationExportWrapper:
    source_dir = source_dir.resolve()
    if not (source_dir / "model.py").is_file():
        raise FileNotFoundError(f"model.py が見つかりません: {source_dir}")
    sys.path.insert(0, str(source_dir))
    try:
        model_module = importlib.import_module("model")
        # 公式配布物は model.Net を pickle 化した重みである。信頼できる配布元の
        # チェックポイントだけを指定すること。
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        sys.path.pop(0)

    if isinstance(loaded, nn.Module):
        model = loaded
    else:
        model = model_module.Net()
        state_dict = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
        model.load_state_dict(state_dict)
    if not isinstance(model.alignment[0], nn.Conv2d):
        raise TypeError("ST-DeInt の alignment[0] が Conv2d 互換ではありません")
    model.alignment[0] = nn.Identity() if no_deform else GridSampleConvOffset2D(model.alignment[0])
    return RestorationExportWrapper(model).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="ST-DeInt を半高6ch復元出力 ONNX へ変換する")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="公式 ST-DeInt の model.py/deform_conv.py のあるディレクトリ")
    parser.add_argument("--checkpoint", type=Path,
                        help="公式モデル重み。省略時は --source-dir/model_w_atten.pt")
    parser.add_argument("--output", required=True, type=Path,
                        help="ONNX の出力ディレクトリ")
    parser.add_argument("--height", type=int, default=720, help="検証および export 用の入力高さ（偶数、4以上）")
    parser.add_argument("--width", type=int, default=1280, help="検証および export 用の入力幅（2以上）")
    parser.add_argument("--static-shape", action="store_true", help="高さ・幅を固定した ONNX を出力する")
    parser.add_argument("--no-deform", action="store_true", help="deform alignmentを省略した高速モデルを出力する")
    args = parser.parse_args()

    if args.height < 4 or args.height % 2 != 0:
        parser.error("--height は 4 以上の偶数で指定してください")
    if args.width < 2:
        parser.error("--width は 2 以上で指定してください")
    checkpoint = args.checkpoint or args.source_dir / "model_w_atten.pt"
    if not checkpoint.is_file():
        parser.error(f"チェックポイントが見つかりません: {checkpoint}")

    model = load_model(args.source_dir, checkpoint, args.no_deform)
    dummy = torch.rand(1, 3, args.height, args.width)
    with torch.no_grad():
        expected_shape = (1, 6, args.height // 2, args.width)
        if tuple(model(dummy).shape) != expected_shape:
            raise RuntimeError(f"予期しない出力形状: {tuple(model(dummy).shape)}")

    dynamic_axes = None if args.static_shape else {
        "input": {2: "height", 3: "width"},
        "restorations": {2: "half_height", 3: "width"},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = (args.output / ("stdeint_fast.onnx" if args.no_deform else "stdeint.onnx")).resolve()
    export_onnx(
        model, dummy, str(output), opset_version=16, do_constant_folding=True,
        input_names=["input"], output_names=["restorations"], dynamic_axes=dynamic_axes)
    exported = onnx.load(output)
    onnx.checker.check_model(exported)
    has_grid_sample = any(node.op_type == "GridSample" for node in exported.graph.node)
    if not args.no_deform and not has_grid_sample:
        raise RuntimeError("出力 ONNX に GridSample が含まれていません")
    if args.no_deform and has_grid_sample:
        raise RuntimeError("高速モデルの出力 ONNX にGridSampleが残っています")
    print(f"exported: {output}")
    print(f"input: (1, 3, H, W), output: (1, 6, H/2, W), dynamic: {not args.static_shape}, no_deform: {args.no_deform}")


if __name__ == "__main__":
    main()
