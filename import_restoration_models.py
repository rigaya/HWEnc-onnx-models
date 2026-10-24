#!/usr/bin/env python3
"""作者が配布する実写・VHS修復用ONNXを検証して配置する。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import onnx


MODELS = (
    ("2xLiveActionV1_SPAN_490000.onnx", "liveaction", "bfa72f3c6347076aed140d0836cee30c27ea434c047beeaf9466469483836ecc"),
)


def import_model(source: Path, destination: Path, expected_hash: str) -> None:
    """元モデルを改変せず、ハッシュとONNX形式を検証してコピーする。"""
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256が一致しません: {source}: {actual_hash}")
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError(f"1入力・1出力のモデルではありません: {source}")
    for value in (*model.graph.input, *model.graph.output):
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT or len(tensor.shape.dim) != 4:
            raise ValueError(f"FP32のNCHWテンソルではありません: {value.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[OK] {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path, help="配布済みONNXの保存先")
    parser.add_argument("--output", required=True, type=Path, help="モデル出力ルート")
    args = parser.parse_args()
    for filename, family, expected_hash in MODELS:
        import_model(args.models_dir / filename, args.output / family / filename, expected_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
