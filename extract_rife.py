#!/usr/bin/env python3
"""公式配布アーカイブからRIFE v4.x ONNXモデルを取得・検証する。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import onnx
import py7zr


MODELS = {
    "rife_v4.0.onnx": "bd642212b1e6856e26670f8cad02d83a6fd429a69d3dc03f73b9178fcfd95d9e",
    "rife_v4.2.onnx": "c53b7cb68fc7897afbb6abcd9b1a9a8f8cff35d4c71b31c2c2dcb5c6e93384ed",
    "rife_v4.3.onnx": "d0b9fb2e1d7315a3ab58b2a773e72dcbdff640cdce8580c3d133ce6462a6dfb7",
    "rife_v4.4.onnx": "b4f8de5f4eba89fcf821cef099ab6d2a90d417221236193142791f31164aafc5",
    "rife_v4.5.onnx": "14f7acd826602e73a5a463a3c5d91364236fa4adc4ec17f612edb9fce766ab05",
    "rife_v4.6.onnx": "b210fbec2bdf94dd37eb255b96035c2d2a16dab3576c484822bbe72351b6cf4d",
    "rife_v4.7.onnx": "31ee257db95a9cf10a6d06be6724982e4a7464cd09dc3b9e8d057121a70ee013",
    "rife_v4.8.onnx": "0c97a567a57456abfd2ddd2d849437297ae374df5fa1979728eb4afe1a2eba6a",
    "rife_v4.9.onnx": "ff95b617482aeba18c6957661a0330ea4fcb30489f7ad321d1fca3aa88a7292f",
    "rife_v4.10.onnx": "397d5d3047066e5c4eacf0208bb9ec5c650e8c2f1086191327f57de2eb972204",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channels(value_info: onnx.ValueInfoProto) -> int:
    shape = value_info.type.tensor_type.shape.dim
    if len(shape) < 2 or not shape[1].HasField("dim_value"):
        return -1
    return shape[1].dim_value


def verify_model(path: Path, expected_hash: str) -> None:
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256不一致: {path.name}: {actual_hash}")
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError(f"入出力数が不正: {path.name}")
    if channels(model.graph.input[0]) != 11 or channels(model.graph.output[0]) != 3:
        raise RuntimeError(f"RIFEの入出力チャンネルではない: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(f"RIFEアーカイブが見つかりません: {args.archive}")

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hwenc-rife-") as temp_dir:
        temp_root = Path(temp_dir)
        with py7zr.SevenZipFile(args.archive, mode="r") as archive:
            names = set(archive.getnames())
            members = [f"rife/{name}" for name in MODELS]
            missing = sorted(set(members) - names)
            if missing:
                raise RuntimeError(f"RIFEアーカイブに必要モデルがありません: {', '.join(missing)}")
            archive.extract(path=temp_root, targets=members)

        for name, expected_hash in MODELS.items():
            source = temp_root / "rife" / name
            verify_model(source, expected_hash)
            target = args.output / name
            if target.is_file() and sha256(target) == expected_hash:
                print(f"  [SKIP] {name}")
                continue
            shutil.copy2(source, target)
            print(f"  [OK] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
