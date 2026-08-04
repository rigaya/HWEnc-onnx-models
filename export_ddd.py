#!/usr/bin/env python3
"""Disney Deep Deinterlacing (DDD) の学習済み重みを ONNX へ変換する。

ネットワーク定義は Disney-Deep-Deinterlacing の MIT 実装
（固定コミット ea3442a2a403b91f950a6f4688e872ed0fe60460）に由来する。
外部リポジトリを import せず、推論に必要な層だけを同じ構成・名前で再実装する。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from onnx_export_common import export_onnx


UPSTREAM_COMMIT = "ea3442a2a403b91f950a6f4688e872ed0fe60460"
UPSTREAM_URL = "https://github.com/vincentvdschaft/Disney-Deep-Deinterlacing"
CHECKPOINT_SHA256 = "4638eba617cce205ca91aacb71199bbffd98907ba0e688c432a7a5b8af64e16c"


class DenseBlock(nn.Module):
    """upstream src/models/deep_deinterlacing.py と同じ dense block。"""

    def __init__(self, in_channels: int, out_channels: int, activation) -> None:
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat((x, self.activation(self.conv(x))), dim=1)


class CompressionBlock(nn.Module):
    """upstream の圧縮畳み込み。"""

    def __init__(self, in_channels: int, out_channels: int, activation) -> None:
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class ResidualDenseBlock(nn.Module):
    """複数の DenseBlock と skip connection。"""

    def __init__(self, channels: int, num_blocks: int, filters_added: int, activation) -> None:
        super().__init__()
        # upstream は all_modules だけを ModuleList として登録している。
        # 同じ state_dict キーを再現するため、dense_blocks は通常の list とする。
        self.dense_blocks = []
        in_channels = channels
        out_channels = channels + filters_added
        for _ in range(num_blocks):
            self.dense_blocks.append(DenseBlock(in_channels, out_channels, activation))
            in_channels = out_channels
            out_channels += filters_added
        self.compression_block = CompressionBlock(in_channels, channels, activation)
        self.all_modules = nn.ModuleList([*self.dense_blocks, self.compression_block])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.dense_blocks:
            h = block(h)
        return self.compression_block(h) + x


class DisneyModel(nn.Module):
    """固定設定 (3 frames, RGB, 3 RDB x 5 dense) の DisneyModel。"""

    def __init__(self) -> None:
        super().__init__()
        activation = lambda x: torch.nn.functional.leaky_relu(x, 0.1)
        self.expand = nn.Conv2d(9, 64, 3, padding="same")
        self.residual_dense_blocks = []
        for _ in range(3):
            self.residual_dense_blocks.append(ResidualDenseBlock(64, 5, 32, activation))
        self.combine = nn.Conv2d(64, 3, 3, padding="same")
        # upstream の state_dict は all_modules.* にパラメータを保存する。
        self.all_modules = nn.ModuleList([self.expand, *self.residual_dense_blocks, self.combine])

    @staticmethod
    def _linear_row_interpolation(x: torch.Tensor) -> torch.Tensor:
        # 元実装の zeros_like + 末尾軸方向の代入を、ONNX exporter が安定して扱える
        # concat に置き換える。計算は先頭要素のコピーと隣接要素平均で同値である。
        # 元実装は末尾軸（転置フィールドの走査線方向）を補間する。
        first = x[..., 0:1]
        average = 0.5 * (x[..., 1:] + x[..., :-1])
        return torch.concat((first, average), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.leaky_relu(self.expand(x), 0.1)
        h0 = h
        for block in self.residual_dense_blocks:
            h = block(h)
        h = self.combine(h + h0)
        interpolated = self._linear_row_interpolation(x[:, 3:6])
        return interpolated + h


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint: Path) -> DisneyModel:
    actual_hash = checkpoint_sha256(checkpoint)
    if actual_hash != CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {actual_hash}"
        )
    model = DisneyModel()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint is not a state_dict")
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def verify_runtime(model: nn.Module, output: Path, shapes: list[tuple[int, int]]) -> None:
    """PyTorch と ONNX Runtime の数値一致および dynamic H/W を検証する。"""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - venv には通常存在する
        raise RuntimeError("onnxruntime is required for DDD export verification") from exc
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(20260730)
    max_error = 0.0
    with torch.no_grad():
        for height, width in shapes:
            array = rng.standard_normal((1, 9, height, width), dtype=np.float32)
            expected = model(torch.from_numpy(array)).numpy()
            actual = session.run(None, {input_name: array})[0]
            if actual.shape != (1, 3, height, width):
                raise RuntimeError(f"unexpected ONNX output shape: {actual.shape}")
            error = float(np.max(np.abs(expected - actual)))
            max_error = max(max_error, error)
            if error > 2e-4:
                raise RuntimeError(f"PyTorch/ONNX mismatch at {height}x{width}: {error}")
    print(f"runtime verification: shapes={shapes}, max_abs_error={max_error:.3g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="model_state_dict.pt")
    parser.add_argument("--output", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument("--height", type=int, default=32, help="export 検証用の高さ")
    parser.add_argument("--width", type=int, default=48, help="export 検証用の幅")
    parser.add_argument("--skip-runtime-check", action="store_true", help="ONNX Runtime 検証を省略")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.height < 2 or args.width < 1:
        raise SystemExit("height は2以上、width は1以上で指定してください")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    model = load_model(checkpoint)
    dummy = torch.randn(1, 9, args.height, args.width)
    with torch.no_grad():
        expected = model(dummy)
    if tuple(expected.shape) != (1, 3, args.height, args.width):
        raise RuntimeError(f"unexpected PyTorch output shape: {tuple(expected.shape)}")

    args.output.mkdir(parents=True, exist_ok=True)
    output = (args.output / "DDD.onnx").resolve()
    export_onnx(
        model,
        dummy,
        str(output),
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
    )
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    input_shape = exported.graph.input[0].type.tensor_type.shape.dim
    output_shape = exported.graph.output[0].type.tensor_type.shape.dim
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise RuntimeError("DDD.onnx must have NCHW input/output")
    if input_shape[0].dim_value != 1 or output_shape[0].dim_value != 1:
        raise RuntimeError("DDD.onnx batch contract must be fixed to 1")
    if input_shape[1].dim_value != 9 or output_shape[1].dim_value != 3:
        raise RuntimeError("DDD.onnx channel contract must be 9 -> 3")
    if not input_shape[2].dim_param or not input_shape[3].dim_param:
        raise RuntimeError("DDD.onnx height/width must be dynamic")
    if not output_shape[2].dim_param or not output_shape[3].dim_param:
        raise RuntimeError("DDD.onnx output height/width must be dynamic")
    if not args.skip_runtime_check:
        verify_runtime(model, output, [(args.height, args.width), (args.height + 8, args.width + 16)])
    print(f"exported: {output}")
    print("contract: input (1,9,H,W), output (1,3,H,W), dynamic H/W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
