#!/usr/bin/env python3
"""PAL/SECAM用vhs-restoreのCompact重みを動的H/WのONNXへ変換する。

フィールドの上下配置と復元はエンコーダー側で行うため、ここではRGBの
修復ネットワークのみを出力する。ノイズ除去やデインターレースは含めない。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch

from export_realesrgan import SRVGGNetCompact
from onnx_export_common import export_onnx


UPSTREAM_COMMIT = "a8ed1a7fc0070d671b3ff2204e655a8c36c8e3fd"
MODELS = {
    "pal": "b7d23bf72354315440c45b80ab4a11c6669ee74163b60dda8b33af080a2a0a9b",
    "secam": "0432784264e7465fe463b4150bdfc28afe20322fb57db335352e8818f811132e",
}


def copy_source(source: Path, destination: Path) -> None:
    """同梱したソースから再生成する場合は同じファイルのコピーを避ける。"""
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def load_model(checkpoint: Path, expected_hash: str) -> SRVGGNetCompact:
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256が一致しません: {checkpoint}: {actual_hash}")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw.get("params_ema", raw.get("params", raw))
    # 既存のReal-ESRGAN変換器と同じCompact定義を、等倍修復として使用する。
    model = SRVGGNetCompact(3, 3, 64, 16, upscale=1)
    model.load_state_dict(state, strict=True)
    return model.eval()


def verify_runtime(model: SRVGGNetCompact, output: Path) -> None:
    """異なる2解像度でPyTorchとの一致と動的H/Wを確認する。"""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20261024)
    with torch.no_grad():
        for height, width in ((16, 24), (20, 28)):
            array = rng.random((1, 3, height, width), dtype=np.float32)
            expected = model(torch.from_numpy(array)).numpy()
            actual = session.run(None, {session.get_inputs()[0].name: array})[0]
            if actual.shape != array.shape:
                raise ValueError(f"出力形状が一致しません: {actual.shape}")
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
            print(f"[VERIFY] {output.name}: {height}x{width}, 誤差最大値={np.max(np.abs(actual-expected)):.3g}")


def export_one(system: str, models_dir: Path, output_dir: Path, skip_runtime_check: bool) -> Path:
    checkpoint = models_dir / f"1x_vhs-restore_{system}_compact_80k.pth"
    model = load_model(checkpoint, MODELS[system])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"vhs_restore_{system}.onnx"
    export_onnx(
        model, torch.zeros(1, 3, 16, 24), str(output),
        do_constant_folding=True, input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
    )
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    for value in (*exported.graph.input, *exported.graph.output):
        tensor = value.type.tensor_type
        dims = tensor.shape.dim
        if (tensor.elem_type != onnx.TensorProto.FLOAT or len(dims) != 4
                or dims[0].dim_value != 1 or dims[1].dim_value != 3
                or not dims[2].dim_param or not dims[3].dim_param):
            raise ValueError("vhs-restoreの契約はFP32の(1,3,H,W)入出力・動的H/Wです")
    if not skip_runtime_check:
        verify_runtime(model, output)
    # GPLモデルは改変に用いる元の重みと再現用スクリプトも同時に配布する。
    source_dir = output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    copy_source(checkpoint, source_dir / checkpoint.name)
    script_dir = Path(__file__).resolve().parent
    for filename in ("export_vhs_restore.py", "export_realesrgan.py", "onnx_export_common.py"):
        copy_source(script_dir / filename, source_dir / filename)
    copy_source(script_dir / "LICENSE", source_dir / "LICENSE")
    (source_dir / "licenses").mkdir(exist_ok=True)
    copy_source(script_dir / "licenses" / "realesrgan.txt", source_dir / "licenses" / "realesrgan.txt")
    (source_dir / "requirements.txt").write_text("numpy>=1.26\nonnx>=1.16\nonnxruntime>=1.17\ntorch>=2.0\n", encoding="utf-8")
    (source_dir / "README.txt").write_text(
        "元のPAL/SECAM重みと変換器。ライセンスと由来は../LICENSE.txtを参照。\n"
        "再生成: python export_vhs_restore.py --models-dir . --output ../\n"
        "依存パッケージ: python -m pip install -r requirements.txt\n",
        encoding="utf-8",
    )
    print(f"[OK] {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path, help="PAL/SECAM重みのディレクトリ")
    parser.add_argument("--output", required=True, type=Path, help="ONNX出力ディレクトリ")
    parser.add_argument("--skip-runtime-check", action="store_true", help="数値一致の確認を省略")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for system in MODELS:
        export_one(system, args.models_dir, args.output, args.skip_runtime_check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
