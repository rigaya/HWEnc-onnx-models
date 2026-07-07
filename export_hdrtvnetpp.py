#!/usr/bin/env python3
"""HDRTVNet++ の AGCM / Ensemble_AGCM_LE を ONNX に変換する。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))
    print(f"  OK   {out_path.name} ({name})")


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
    export_one(
        "AGCM",
        ConditionNet(classifier="color_condition", cond_c=6),
        weights_dir / "AGCM.pth",
        out_dir / "hdrtvnetpp_agcm_dynamic.onnx",
    )
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
        out_dir / "hdrtvnetpp_ensemble_dynamic.onnx",
    )
    print("done: 2 models exported")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
