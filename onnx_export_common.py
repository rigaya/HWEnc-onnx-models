"""HWEnc --vpp-onnx 用ONNX出力ヘルパー。"""

import inspect
import torch


DEFAULT_ONNX_OPSET = 17


def export_onnx(model, args, f, **kwargs):
    """CUDA EPで余分なMemcpyが入らないよう、旧ONNX exporterで出力する。"""
    kwargs.setdefault("opset_version", DEFAULT_ONNX_OPSET)
    kwargs.setdefault("dynamo", False)
    kwargs.setdefault("external_data", False)
    # 古いPyTorchではdynamo/external_data引数がないので、対応済み引数だけ渡す。
    supported = inspect.signature(torch.onnx.export).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return torch.onnx.export(model, args, f, **kwargs)
