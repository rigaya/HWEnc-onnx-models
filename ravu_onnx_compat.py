"""RAVUのONNXグラフをCUDA Execution Provider互換へ変換する。"""

import numpy as np
import onnx
from onnx import helper, numpy_helper


_PI_OVER_2 = 1.5707963267948966
_PI_OVER_4 = 0.7853981633974483
_TAN_3PI_OVER_8 = 2.414213562373095
_TAN_PI_OVER_8 = 0.4142135623730950

# Cephesの単精度atan近似で使われる係数。最大誤差はfloat32の丸め誤差程度。
_ATAN_COEFFICIENTS = (
    8.05374449538e-2,
    -1.38776856032e-1,
    1.99777106478e-1,
    -3.33329491539e-1,
)


def _constant(name, value):
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name)


def _build_atan_subgraph(input_name, output_name, prefix):
    """CUDA EPで実行可能な演算だけでatanを計算する部分グラフを返す。"""
    nodes = []
    initializers = []

    def add_constant(tag, value):
        name = f"{prefix}_constant_{tag}"
        initializers.append(_constant(name, value))
        return name

    def add_node(op_type, inputs, outputs):
        nodes.append(helper.make_node(op_type, inputs, outputs,
                                      name=f"{prefix}_{outputs[0]}"))

    absolute = f"{prefix}_absolute"
    add_node("Abs", [input_name], [absolute])

    high_region = f"{prefix}_high_region"
    middle_region = f"{prefix}_middle_region"
    add_node("Greater", [absolute, add_constant("tan_3pi_over_8", _TAN_3PI_OVER_8)], [high_region])
    add_node("Greater", [absolute, add_constant("tan_pi_over_8", _TAN_PI_OVER_8)], [middle_region])

    high_z = f"{prefix}_high_z"
    add_node("Div", [add_constant("minus_one", -1.0), absolute], [high_z])
    middle_numerator = f"{prefix}_middle_numerator"
    middle_denominator = f"{prefix}_middle_denominator"
    middle_z = f"{prefix}_middle_z"
    one = add_constant("one", 1.0)
    add_node("Sub", [absolute, one], [middle_numerator])
    add_node("Add", [absolute, one], [middle_denominator])
    add_node("Div", [middle_numerator, middle_denominator], [middle_z])

    selected_z = f"{prefix}_selected_z"
    z = f"{prefix}_z"
    add_node("Where", [middle_region, middle_z, absolute], [selected_z])
    add_node("Where", [high_region, high_z, selected_z], [z])

    selected_offset = f"{prefix}_selected_offset"
    offset = f"{prefix}_offset"
    add_node("Where", [middle_region,
                       add_constant("pi_over_4", _PI_OVER_4),
                       add_constant("zero", 0.0)], [selected_offset])
    add_node("Where", [high_region,
                       add_constant("pi_over_2", _PI_OVER_2),
                       selected_offset], [offset])

    squared = f"{prefix}_squared"
    add_node("Mul", [z, z], [squared])
    polynomial = add_constant("coefficient_3", _ATAN_COEFFICIENTS[0])
    for index, coefficient in enumerate(_ATAN_COEFFICIENTS[1:], start=2):
        multiplied = f"{prefix}_polynomial_mul_{index}"
        added = f"{prefix}_polynomial_add_{index}"
        add_node("Mul", [polynomial, squared], [multiplied])
        add_node("Add", [multiplied, add_constant(f"coefficient_{4 - index}", coefficient)], [added])
        polynomial = added

    z_times_squared = f"{prefix}_z_times_squared"
    correction = f"{prefix}_correction"
    offset_plus_z = f"{prefix}_offset_plus_z"
    positive_result = f"{prefix}_positive_result"
    add_node("Mul", [z, squared], [z_times_squared])
    add_node("Mul", [z_times_squared, polynomial], [correction])
    add_node("Add", [offset, z], [offset_plus_z])
    add_node("Add", [offset_plus_z, correction], [positive_result])

    negative_result = f"{prefix}_negative_result"
    is_negative = f"{prefix}_is_negative"
    add_node("Neg", [positive_result], [negative_result])
    add_node("Less", [input_name, add_constant("sign_zero", 0.0)], [is_negative])
    nodes.append(helper.make_node("Where",
                                  [is_negative, negative_result, positive_result],
                                  [output_name], name=f"{prefix}_result"))
    return nodes, initializers


def rewrite_atan(model):
    """標準ドメインのAtanをCUDA EP互換の部分グラフへ置換する。"""
    graph = model.graph
    nodes = []
    initializers = []
    replaced = 0
    for node in graph.node:
        if node.op_type == "Atan" and node.domain in ("", "ai.onnx"):
            replacement, constants = _build_atan_subgraph(
                node.input[0], node.output[0], f"ravu_atan_{replaced}")
            nodes.extend(replacement)
            initializers.extend(constants)
            replaced += 1
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.initializer.extend(initializers)
    return replaced


def rewrite_atan_file(path):
    """ONNXファイル内のAtanを置換し、検証して同じパスへ保存する。"""
    model = onnx.load(path)
    replaced = rewrite_atan(model)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, path)
    return replaced
