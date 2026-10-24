#!/usr/bin/env python3
"""Post-training INT8 quantization of FP32 ONNX models using nncf.

Generates INT8 variants used by --vpp-onnx. Two presets are supported:
  - PERFORMANCE (default): aggressive quantization for speed (_int8_perf suffix)
  - MIXED: more conservative quantization (_int8 suffix)

--calib-video指定時のみ、ArtCNN Chromaの2モデルを実写YCbCrで較正し、
MIXEDとfast_bias_correction=Falseを使用する。未指定時は従来動作を維持する。

Usage:
    python quantize_int8.py --onnx-dir /path/to/onnx --output /path/to/output
    python quantize_int8.py --onnx-dir /path/to/onnx --output /path/to/output --dry-run
"""

import argparse
import contextlib
import os
import shutil
import subprocess
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


# 実写YCbCrによる較正を適用する出力モデルを明示的に限定する。
CHROMA_MODELS = frozenset({
    "ArtCNN_R8F64_Chroma_int8_perf.onnx",
    "ArtCNN_R8F64_Chroma_DN_int8_perf.onnx",
})


def rgb_to_ycbcr(rgb):
    """CHW形式のRGBをArtCNN Chroma用のBT.601 full-range YCbCrへ変換する。"""
    r, g, b = rgb
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.stack([y, (b - y) / 1.772 + 0.5, (r - y) / 1.402 + 0.5])


def load_chroma_calibration(video):
    """先頭から毎秒最大32フレームを抽出し、64x64の実写タイルを300個返す。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--calib-videoにはPATH上のffmpegが必要です")
    size = 512
    # パイプで受け取り、一時ファイルの衝突を避ける。出力量は32フレームに制限する。
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", os.fspath(video),
         "-map", "0:v:0", "-vf", f"fps=1,scale={size}:{size}",
         "-frames:v", "32", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(f"ffmpegのデコード失敗: {proc.stderr.decode(errors='replace')[:500]}")
    per_frame = size * size * 3
    if not proc.stdout or len(proc.stdout) % per_frame:
        raise RuntimeError("動画から完全なRGBフレームを取得できませんでした")
    frames = np.frombuffer(proc.stdout, np.uint8).reshape(-1, size, size, 3)
    # 他モデルのnoise較正が使うグローバル乱数列には影響させない。
    rng = np.random.default_rng(1234)
    tiles = np.empty((300, 3, 64, 64), dtype=np.float32)
    for i in range(len(tiles)):
        frame = frames[rng.integers(len(frames))]
        y, x = rng.integers(0, size - 64 + 1, size=2)
        rgb = frame[y:y + 64, x:x + 64].astype(np.float32).transpose(2, 0, 1) / 255.0
        tiles[i] = rgb_to_ycbcr(rgb)
    return tiles


@contextlib.contextmanager
def suppress_nncf_progress():
    import nncf.common.logging.track_progress as track_progress

    original_init = track_progress.track.__init__

    def init_without_progress(self, *args, **kwargs):
        kwargs["disable"] = True
        original_init(self, *args, **kwargs)

    track_progress.track.__init__ = init_without_progress
    try:
        yield
    finally:
        track_progress.track.__init__ = original_init


def quantize_one(src_path, dst_path, input_channels, preset_name, dry_run,
                 calib_video=None, chroma_tiles=None):
    use_chroma = calib_video is not None and os.path.basename(dst_path) in CHROMA_MODELS
    if use_chroma:
        preset_name = "mixed"

    if not os.path.isfile(src_path):
        print(f"  SKIP {os.path.basename(src_path)}: source not found")
        return False

    if os.path.isfile(dst_path):
        print(f"  SKIP {os.path.basename(dst_path)}: already exists")
        return True

    if dry_run:
        print(f"  [DRY] {os.path.basename(src_path)} -> {os.path.basename(dst_path)} ({preset_name}, {'ycbcr' if use_chroma else 'noise'})")
        return True

    import nncf
    import onnx

    preset = (nncf.QuantizationPreset.PERFORMANCE if preset_name == "perf"
              else nncf.QuantizationPreset.MIXED)
    model = onnx.load(src_path)
    input_name = model.graph.input[0].name

    quantize_options = {}
    if use_chroma:
        if chroma_tiles is None:
            chroma_tiles = load_chroma_calibration(calib_video)
        calibration_data = [{input_name: chroma_tiles[i:i + 1]}
                            for i in range(len(chroma_tiles))]
        quantize_options["fast_bias_correction"] = False
    else:
        calibration_data = [
            {input_name: np.random.rand(1, input_channels, 64, 64).astype(np.float32)}
            for _ in range(300)
        ]

    with suppress_nncf_progress():
        quantized = nncf.quantize(
            model,
            nncf.Dataset(calibration_data),
            preset=preset,
            **quantize_options,
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
    parser.add_argument("--calib-video", help="対象2モデルを実写YCbCrで較正する動画。未指定時は従来のnoise較正")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"=== INT8 Quantization (nncf) ===")
    print(f"  source: {args.onnx_dir}")
    print(f"  output: {args.output}")

    chroma_tiles = None
    if args.calib_video is None:
        print("  chroma: --calib-video未指定のため従来のnoise較正を使用します")

    ok = 0
    fail = 0
    for src_sub, src_file, dst_sub, dst_file, in_ch, preset in TARGETS:
        src_path = os.path.join(args.onnx_dir, src_sub, src_file)
        dst_path = os.path.join(args.output, dst_sub, dst_file)
        try:
            # 実際に生成する対象がある場合だけデコードし、2モデルで共有する。
            if (args.calib_video is not None and dst_file in CHROMA_MODELS
                    and not args.dry_run and os.path.isfile(src_path)
                    and not os.path.isfile(dst_path) and chroma_tiles is None):
                chroma_tiles = load_chroma_calibration(args.calib_video)
            if quantize_one(src_path, dst_path, in_ch, preset, args.dry_run,
                            args.calib_video, chroma_tiles):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  FAIL {dst_file}: {e}")
            fail += 1

    print(f"done: {ok}/{ok + fail} quantized")


if __name__ == "__main__":
    main()
