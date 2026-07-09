#!/usr/bin/env python3
"""HDRTVNet++ の AGCM / Ensemble_AGCM_LE を ONNX に変換する。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from onnx_export_common import export_onnx


class SingleInputWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model([x, x])
        if isinstance(output, tuple):
            return output[0]
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HDRTVNet++ SDR-to-HDR models to ONNX")
    parser.add_argument("--repo-root", required=True, type=Path, help="HDRTVNet-plus repository root")
    parser.add_argument("--weights-dir", required=True, type=Path, help="Directory containing AGCM.pth and Ensemble_AGCM_LE.pth")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for ONNX files")
    return parser.parse_args()


def load_weights(model: nn.Module, weight_path: Path) -> None:
    checkpoint = torch.load(weight_path, map_location="cpu")
    for key in ("params", "params_ema", "state_dict", "model", "net"):
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{weight_path}: unsupported checkpoint format")

    state = {}
    for key, value in checkpoint.items():
        name = key
        for prefix in ("module.", "netG.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        state[name] = value
    model.load_state_dict(state, strict=True)


def export_one(name: str, model: nn.Module, weight_path: Path, out_path: Path) -> None:
    load_weights(model, weight_path)
    wrapped = SingleInputWrapper(model).eval()
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = wrapped(dummy)
    assert tuple(out.shape) == (1, 3, 64, 64), f"{name}: unexpected output shape {tuple(out.shape)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(
        wrapped,
        dummy,
        str(out_path),
        do_constant_folding=True,
        input_names=["sdr"],
        output_names=["hdr"],
        dynamic_axes={
            "sdr": {0: "batch", 2: "height", 3: "width"},
            "hdr": {0: "batch", 2: "height", 3: "width"},
        },
    )
    onnx.checker.check_model(onnx.load(str(out_path)))
    print(f"  OK   {out_path.name} ({name})")


def bake_stable_model(dynamic_path: Path, stable_path: Path) -> None:
    """動的HDRTVNet++モデルから時間的に安定した単入力モデルを生成する。"""
    dynamic_model = onnx.load(str(dynamic_path))
    global_average_pools = [
        node for node in dynamic_model.graph.node if node.op_type == "GlobalAveragePool"
    ]
    if len(global_average_pools) != 1:
        raise ValueError(
            f"{dynamic_path.name}: GlobalAveragePool は1個である必要があります "
            f"(検出数: {len(global_average_pools)})"
        )

    frame_input = dynamic_model.graph.input[0].name
    frame_output = dynamic_model.graph.output[0].name
    condition_output = global_average_pools[0].output[0]
    with tempfile.TemporaryDirectory(prefix="hdrtvnetpp_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        condition_path = temp_dir / "condition.onnx"
        base_path = temp_dir / "base.onnx"
        onnx.utils.extract_model(
            str(dynamic_path), str(condition_path), [frame_input], [condition_output]
        )
        onnx.utils.extract_model(
            str(dynamic_path), str(base_path), [frame_input, condition_output], [frame_output]
        )

        condition_session = ort.InferenceSession(
            str(condition_path), providers=["CPUExecutionProvider"]
        )
        base_model = onnx.load(str(base_path))
        conditions = []
        for brightness in (0.20, 0.35, 0.50, 0.65, 0.80):
            frame = np.full((1, 3, 216, 384), brightness, dtype=np.float32)
            conditions.append(condition_session.run(None, {frame_input: frame})[0])
        fixed_condition = np.mean(conditions, axis=0, dtype=np.float32)

        base_model.graph.initializer.append(
            onnx.numpy_helper.from_array(fixed_condition, name=condition_output)
        )
        retained_inputs = [
            value_info
            for value_info in base_model.graph.input
            if value_info.name != condition_output
        ]
        del base_model.graph.input[:]
        base_model.graph.input.extend(retained_inputs)
        onnx.checker.check_model(base_model)
        onnx.save(base_model, str(stable_path))

    stable_session = ort.InferenceSession(
        str(stable_path), providers=["CPUExecutionProvider"]
    )
    if [item.name for item in stable_session.get_inputs()] != [frame_input]:
        raise ValueError(f"{stable_path.name}: 単入力モデルの生成に失敗しました")
    print(
        f"  OK   {stable_path.name} (固定condition: "
        f"{np.array2string(fixed_condition.reshape(-1), precision=4)})"
    )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    weights_dir = args.weights_dir.resolve()
    out_dir = args.output.resolve()
    codes_dir = repo_root / "codes"
    if not codes_dir.is_dir():
        raise FileNotFoundError(f"HDRTVNet++ codes directory not found: {codes_dir}")
    sys.path.insert(0, str(codes_dir))

    from models.modules.Condition_arch import ConditionNet
    from models.modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE

    print("=== HDRTVNet++ -> ONNX ===")
    agcm_dynamic_path = out_dir / "hdrtvnetpp_agcm_dynamic.onnx"
    export_one(
        "AGCM",
        ConditionNet(classifier="color_condition", cond_c=6),
        weights_dir / "AGCM.pth",
        agcm_dynamic_path,
    )
    bake_stable_model(
        agcm_dynamic_path,
        out_dir / "hdrtvnetpp_agcm_stable.onnx",
    )
    ensemble_dynamic_path = out_dir / "hdrtvnetpp_ensemble_dynamic.onnx"
    export_one(
        "Ensemble_AGCM_LE",
        Ensemble_AGCM_LE(
            classifier="color_condition",
            cond_c=6,
            in_nc=3,
            out_nc=3,
            nf=32,
            act_type="relu",
            weighting_network=False,
        ),
        weights_dir / "Ensemble_AGCM_LE.pth",
        ensemble_dynamic_path,
    )
    bake_stable_model(
        ensemble_dynamic_path,
        out_dir / "hdrtvnetpp_ensemble_stable.onnx",
    )
    print("done: 4 models exported")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
