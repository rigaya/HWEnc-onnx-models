#!/usr/bin/env python3
"""EDVR-M REDS x4モデルをHWEncの時間窓入力用ONNXへ変換する。"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper, shape_inference
from torch import nn
from torch.nn.modules.utils import _pair, _single

from onnx_export_common import export_onnx


EDVR_FRAMES = 5
EDVR_CHANNELS = EDVR_FRAMES * 3
EDVR_SCALE = 4
EXPECTED_REDUCE_SUMS = 5


class ModulatedDeformConv(nn.Module):
    """BasicSRのDCNパラメータ配置だけを再現するCPU向け基底クラス。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deformable_groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for size in self.kernel_size:
            n *= size
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class ModulatedDeformConvPack(ModulatedDeformConv):
    """学習済みconv_offsetを保持し、実行本体はBasicSRのDCNv2Packへ委譲する。"""

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True,
        )
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, _x):
        raise NotImplementedError("BasicSRのDCNv2Pack.forward()で上書きされます")


class Registry:
    """EDVRクラスのデコレータ登録に必要な最小レジストリ。"""

    def __init__(self):
        self.entries = {}

    def register(self, obj=None):
        def decorator(item):
            self.entries[item.__name__] = item
            return item
        return decorator if obj is None else decorator(obj)


class PackedEDVR(nn.Module):
    """15ch入力を5フレームのRGB入力へ戻してEDVRへ渡す。"""

    def __init__(self, model: nn.Module, height: int, width: int):
        super().__init__()
        self.model = model
        self.height = height
        self.width = width

    def forward(self, packed):
        frames = packed.reshape(-1, EDVR_FRAMES, 3, self.height, self.width)
        return self.model(frames)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Pythonモジュールを読み込めません: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_edvr_class(repo_root: Path):
    """BasicSR全体をimportせず、EDVRに必要な2ファイルだけを読み込む。"""
    arch_dir = repo_root / "basicsr" / "archs"
    arch_util_path = arch_dir / "arch_util.py"
    edvr_path = arch_dir / "edvr_arch.py"
    if not arch_util_path.is_file() or not edvr_path.is_file():
        raise FileNotFoundError(f"BasicSRのEDVR実装がありません: {repo_root}")

    basicsr = types.ModuleType("basicsr")
    basicsr.__path__ = [str(repo_root / "basicsr")]
    archs = types.ModuleType("basicsr.archs")
    archs.__path__ = [str(arch_dir)]
    utils = types.ModuleType("basicsr.utils")
    utils.get_root_logger = lambda: logging.getLogger("edvr-export")
    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = Registry()
    ops = types.ModuleType("basicsr.ops")
    ops.__path__ = []
    dcn = types.ModuleType("basicsr.ops.dcn")
    dcn.ModulatedDeformConvPack = ModulatedDeformConvPack
    dcn.modulated_deform_conv = None

    sys.modules.update({
        "basicsr": basicsr,
        "basicsr.archs": archs,
        "basicsr.utils": utils,
        "basicsr.utils.registry": registry,
        "basicsr.ops": ops,
        "basicsr.ops.dcn": dcn,
    })
    load_module("basicsr.archs.arch_util", arch_util_path)
    return load_module("basicsr.archs.edvr_arch", edvr_path).EDVR


def remap_mmagic_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """MMagicのEDVRNetキーを同一構造のBasicSRキーへ変換する。"""
    remapped = {}
    for original_key, value in state_dict.items():
        if not original_key.startswith("generator."):
            continue
        key = original_key[len("generator."):]
        key = key.replace("pcd_alignment", "pcd_align")
        for source, dest in (
            ("feat_l2_conv1", "conv_l2_1"),
            ("feat_l2_conv2", "conv_l2_2"),
            ("feat_l3_conv1", "conv_l3_1"),
            ("feat_l3_conv2", "conv_l3_2"),
        ):
            key = key.replace(source, dest)
        key = key.replace("upsample1.upsample_conv", "upconv1")
        key = key.replace("upsample2.upsample_conv", "upconv2")
        if key.endswith(".conv.weight"):
            key = key[:-len(".conv.weight")] + ".weight"
        elif key.endswith(".conv.bias"):
            key = key[:-len(".conv.bias")] + ".bias"
        remapped[key] = value
    return remapped


def build_model(repo_root: Path, weights_path: Path) -> nn.Module:
    edvr_class = load_edvr_class(repo_root)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint:
        raise RuntimeError("EDVRチェックポイントにstate_dictがありません")
    weights = remap_mmagic_weights(checkpoint["state_dict"])
    model = edvr_class(
        num_feat=64,
        num_frame=EDVR_FRAMES,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=10,
        center_frame_idx=2,
        with_tsa=True,
    )
    expected = set(model.state_dict())
    actual = set(weights)
    if expected != actual:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise RuntimeError(f"EDVR重みが一致しません: missing={missing}, extra={extra}")
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def register_deform_conv_exporter():
    import deform_conv2d_onnx_exporter as exporter
    from torch.onnx._internal.torchscript_exporter._type_utils import JitScalarType

    # deform-conv2d-onnx-exporter 1.2.0が参照する旧配置を現行PyTorchへ接続する。
    exporter.JitScalarType = JitScalarType
    exporter.register_deform_conv2d_onnx_op(use_gathernd=True)


def tensor_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = shape_inference.infer_shapes(model)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    return {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in values
    }


def attribute_int(node: onnx.NodeProto, name: str, default: int) -> int:
    return next((attribute.i for attribute in node.attribute if attribute.name == name), default)


def rewrite_channel_reductions(model: onnx.ModelProto, height: int, width: int) -> int:
    """CUDA reductionの不正バッファ問題を避けるため、EDVR固有の和を1x1 Convへ置換する。"""
    graph = model.graph
    shapes = tensor_shapes(model)
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    constants = {}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                constants[node.output[0]] = numpy_helper.to_array(attribute.t)

    reductions = [node for node in graph.node if node.op_type == "ReduceSum"]
    if len(reductions) != EXPECTED_REDUCE_SUMS:
        raise RuntimeError(f"EDVRのReduceSum数が想定外です: {len(reductions)}")

    used_names = {value.name for value in graph.input}
    used_names.update(value.name for value in graph.output)
    for node in graph.node:
        used_names.update(node.input)
        used_names.update(node.output)

    weight_name = "edvr_channel_sum_weight"
    shape_name = "edvr_channel_sum_shape"
    if weight_name in used_names or shape_name in used_names:
        raise RuntimeError("EDVR変換用テンソル名が既存グラフと衝突しました")

    new_nodes = []
    replaced_axes = set()
    for index, node in enumerate(graph.node):
        if node.op_type != "ReduceSum":
            new_nodes.append(node)
            continue
        if len(node.input) != 2 or not node.name:
            raise RuntimeError(f"ReduceSumの形式が想定外です: {node.name or '<unnamed>'}")
        axes = initializers.get(node.input[1])
        if axes is None:
            axes = constants.get(node.input[1])
        input_shape = shapes.get(node.input[0])
        output_shape = shapes.get(node.output[0])
        if axes is None or axes.tolist() != [1] or attribute_int(node, "keepdims", 1) != 0:
            raise RuntimeError(f"ReduceSumの軸設定が想定外です: {node.name}")
        if input_shape != [1, 64, height, width] or output_shape != [1, height, width]:
            raise RuntimeError(
                f"ReduceSumの入出力形状が想定外です: {node.name}: {input_shape} -> {output_shape}"
            )
        conv_output = f"edvr_channel_sum_conv_{index}"
        new_nodes.append(helper.make_node(
            "Conv",
            [node.input[0], weight_name],
            [conv_output],
            name=f"{node.name}_conv",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        ))
        new_nodes.append(helper.make_node(
            "Reshape",
            [conv_output, shape_name],
            [node.output[0]],
            name=f"{node.name}_reshape",
        ))
        replaced_axes.add(node.input[1])

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend([
        numpy_helper.from_array(np.ones((1, 64, 1, 1), dtype=np.float32), weight_name),
        numpy_helper.from_array(np.array([1, height, width], dtype=np.int64), shape_name),
    ])

    # 置換で参照されなくなったaxesだけを除去する。その他の初期化子には触れない。
    referenced = {name for node in graph.node for name in node.input}
    kept_initializers = [
        item for item in graph.initializer
        if item.name not in replaced_axes or item.name in referenced
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    kept_nodes = [
        node for node in graph.node
        if node.op_type != "Constant" or any(output in referenced for output in node.output)
    ]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    return len(reductions)


def verify_model(path: Path, packed_model: nn.Module, sample: torch.Tensor,
                 height: int, width: int):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    shapes = tensor_shapes(model)
    input_info = model.graph.input[0]
    output_info = model.graph.output[0]
    if shapes[input_info.name] != [1, EDVR_CHANNELS, height, width]:
        raise RuntimeError(f"EDVR入力形状が不正です: {shapes[input_info.name]}")
    if shapes[output_info.name] != [1, 3, height * EDVR_SCALE, width * EDVR_SCALE]:
        raise RuntimeError(f"EDVR出力形状が不正です: {shapes[output_info.name]}")
    if any(node.op_type == "ReduceSum" for node in model.graph.node):
        raise RuntimeError("EDVR出力にReduceSumが残っています")

    with torch.no_grad():
        reference = packed_model(sample).cpu().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {input_info.name: sample.cpu().numpy()})[0]
    max_diff = float(np.max(np.abs(reference - actual)))
    if not np.isfinite(max_diff) or max_diff > 1.0e-3:
        raise RuntimeError(f"PyTorchとONNXの差が大きすぎます: {max_diff:.3e}")
    return max_diff


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(f"解像度はWIDTHxHEIGHT形式で指定してください: {value}") from error
    if width <= 0 or height <= 0 or width % 4 != 0 or height % 4 != 0:
        raise argparse.ArgumentTypeError(f"幅と高さには4の倍数を指定してください: {value}")
    return width, height


def main() -> int:
    parser = argparse.ArgumentParser(description="EDVR-M REDS x4をHWEnc用ONNXへ変換する")
    parser.add_argument("--repo-root", required=True, type=Path, help="BasicSR v1.4.2のルート")
    parser.add_argument("--weights", required=True, type=Path, help="MMagic EDVR-Mの学習済み重み")
    parser.add_argument("--output", required=True, type=Path, help="ONNX出力ディレクトリ")
    parser.add_argument(
        "--size",
        action="append",
        type=parse_size,
        help="固定入力解像度WIDTHxHEIGHT。複数回指定可能（既定: 128x128, 160x88）",
    )
    args = parser.parse_args()
    sizes = args.size or [(128, 128), (160, 88)]
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    register_deform_conv_exporter()
    model = build_model(args.repo_root.resolve(), args.weights.resolve())

    for width, height in sizes:
        packed_model = PackedEDVR(model, height, width).eval()
        sample = torch.rand(1, EDVR_CHANNELS, height, width, dtype=torch.float32)
        output_path = args.output / f"edvrm_x4_{width}x{height}_frames5.onnx"
        export_onnx(
            packed_model,
            sample,
            output_path,
            opset_version=16,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamo=False,
        )
        exported = onnx.load(output_path)
        replaced = rewrite_channel_reductions(exported, height, width)
        onnx.checker.check_model(exported)
        onnx.save(exported, output_path)
        max_diff = verify_model(output_path, packed_model, sample, height, width)
        print(
            f"OK {output_path.name}: [1,{EDVR_CHANNELS},{height},{width}] -> "
            f"[1,3,{height * EDVR_SCALE},{width * EDVR_SCALE}], "
            f"ReduceSum置換={replaced}, max|PyTorch-ONNX|={max_diff:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
