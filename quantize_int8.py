#!/usr/bin/env python3
"""Post-training INT8 quantization of FP32 ONNX models using nncf.

Generates INT8 variants used by --vpp-onnx. Two presets are supported:
  - PERFORMANCE (default): aggressive quantization for speed (_int8_perf suffix)
  - MIXED: more conservative quantization (_int8 suffix)

Usage:
    python quantize_int8.py --onnx-dir /path/to/onnx --output /path/to/output
    python quantize_int8.py --onnx-dir /path/to/onnx --output /path/to/output --dry-run
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# (source_subdir, source_file, output_subdir, output_file, input_channels, preset)
# preset: "perf" = PERFORMANCE, "mixed" = MIXED
TARGETS = [
    # ArtCNN
    ("artcnn", "ArtCNN_R16F128.onnx", "artcnn", "ArtCNN_R16F128_int8.onnx", 1, "mixed"),
    ("artcnn", "ArtCNN_R16F128.onnx", "artcnn", "ArtCNN_R16F128_int8_perf.onnx", 1, "perf"),
    ("artcnn", "ArtCNN_R16F96.onnx", "artcnn", "ArtCNN_R16F96_int8_perf.onnx", 1, "perf"),
    ("artcnn", "ArtCNN_R8F64_Chroma_DN.onnx", "artcnn", "ArtCNN_R8F64_Chroma_DN_int8_perf.onnx", 3, "perf"),
    ("artcnn", "ArtCNN_R8F64_Chroma.onnx", "artcnn", "ArtCNN_R8F64_Chroma_int8_perf.onnx", 3, "perf"),
    ("artcnn", "ArtCNN_R8F64.onnx", "artcnn", "ArtCNN_R8F64_int8_perf.onnx", 1, "perf"),
    # dpsr
    ("dpsr", "dpsr_x2.onnx", "dpsr", "dpsr_x2_int8.onnx", 4, "mixed"),
    # drunet
    ("drunet", "drunet_color.onnx", "drunet", "drunet_color_int8.onnx", 4, "mixed"),
    # realcugan
    ("realcugan", "upcunet2x_no_denoise.onnx", "realcugan", "upcunet2x_no_denoise_int8.onnx", 3, "mixed"),
    ("realcugan", "upcunet2x_no_denoise.onnx", "realcugan", "upcunet2x_no_denoise_int8_v2.onnx", 3, "perf"),
    # realesrgan
    ("realesrgan", "realesrgan_anime_6b.onnx", "realesrgan", "realesrgan_anime_6b_int8.onnx", 3, "mixed"),
    ("realesrgan", "realesrgan_anime_6b.onnx", "realesrgan", "realesrgan_anime_6b_int8_v3.onnx", 3, "perf"),
]


def quantize_one(src_path, dst_path, input_channels, preset_name, dry_run):
    import nncf
    import onnx

    preset = (nncf.QuantizationPreset.PERFORMANCE if preset_name == "perf"
              else nncf.QuantizationPreset.MIXED)

    if not os.path.isfile(src_path):
        print(f"  SKIP {os.path.basename(src_path)}: source not found")
        return False

    if os.path.isfile(dst_path):
        print(f"  SKIP {os.path.basename(dst_path)}: already exists")
        return True

    if dry_run:
        print(f"  [DRY] {os.path.basename(src_path)} -> {os.path.basename(dst_path)} ({preset_name})")
        return True

    model = onnx.load(src_path)
    input_name = model.graph.input[0].name

    calibration_data = [
        {input_name: np.random.rand(1, input_channels, 64, 64).astype(np.float32)}
        for _ in range(300)
    ]

    quantized = nncf.quantize(
        model,
        nncf.Dataset(calibration_data),
        preset=preset,
    )

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    onnx.save(quantized, dst_path)
    sz = os.path.getsize(dst_path)
    print(f"  OK   {os.path.basename(dst_path):45} {sz:>10,} bytes ({preset_name})")
    return True


def main():
    parser = argparse.ArgumentParser(description="INT8 quantize FP32 ONNX models")
    parser.add_argument("--onnx-dir", required=True, help="Root ONNX directory (contains family subdirs)")
    parser.add_argument("--output", required=True, help="Output ONNX directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== INT8 Quantization (nncf) ===")
    print(f"  source: {args.onnx_dir}")
    print(f"  output: {args.output}")

    ok = 0
    fail = 0
    for src_sub, src_file, dst_sub, dst_file, in_ch, preset in TARGETS:
        src_path = os.path.join(args.onnx_dir, src_sub, src_file)
        dst_path = os.path.join(args.output, dst_sub, dst_file)
        try:
            if quantize_one(src_path, dst_path, in_ch, preset, args.dry_run):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  FAIL {dst_file}: {e}")
            fail += 1

    print(f"done: {ok}/{ok + fail} quantized")


if __name__ == "__main__":
    main()
