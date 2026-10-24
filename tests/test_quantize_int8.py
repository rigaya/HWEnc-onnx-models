"""Chroma較正の限定適用と既存noise経路の回帰テスト。"""

import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "quantize_int8", Path(__file__).resolve().parents[1] / "quantize_int8.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


class CalibrationTest(unittest.TestCase):
    def test_bt601_full_range(self):
        rgb = np.array([[0, 1, 1, 0, 0], [0, 1, 0, 1, 0],
                        [0, 1, 0, 0, 1]], dtype=np.float32)[:, None, :]
        expected = np.array([[0, 1, .299, .587, .114],
                             [.5, .5, .331264, .168736, 1],
                             [.5, .5, 1, .081312, .418688]])[:, None, :]
        np.testing.assert_allclose(q.rgb_to_ycbcr(rgb), expected, atol=1e-6)

    def test_all_targets_quantization_options_and_data(self):
        tiles = np.full((300, 3, 64, 64), .5, np.float32)
        nncf = types.SimpleNamespace(
            QuantizationPreset=types.SimpleNamespace(PERFORMANCE="perf", MIXED="mixed"),
            Dataset=lambda data: data, quantize=Mock())
        model = types.SimpleNamespace(graph=types.SimpleNamespace(
            input=[types.SimpleNamespace(name="input")]))
        onnx = types.SimpleNamespace(load=Mock(return_value=model), save=Mock())
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, nncf=nncf, onnx=onnx), \
                patch.object(q, "suppress_nncf_progress", contextlib.nullcontext), \
                patch.object(q, "load_chroma_calibration") as load, \
                patch.object(q.os.path, "getsize", return_value=1), \
                contextlib.redirect_stdout(io.StringIO()):
            src = Path(directory) / "source.onnx"
            src.touch()
            for video in (None, "clip.mkv"):
                noise_count = 0
                for _, _, _, dst, channels, preset in q.TARGETS:
                    with self.subTest(video=video, model=dst):
                        # 同一seedの従来式と比較し、noiseの値そのものも維持されることを確認する。
                        np.random.seed(77)
                        expected = np.random.rand(1, channels, 64, 64).astype(np.float32)
                        np.random.seed(77)
                        q.quantize_one(str(src), str(Path(directory) / dst), channels,
                                       preset, False, video, tiles)
                        call = nncf.quantize.call_args
                        data = call.args[1]
                        self.assertEqual(len(data), 300)
                        self.assertEqual(data[0]["input"].shape, (1, channels, 64, 64))
                        self.assertEqual(data[0]["input"].dtype, np.float32)
                        if video and dst in q.CHROMA_MODELS:
                            self.assertEqual(call.kwargs, dict(preset="mixed", fast_bias_correction=False))
                            np.testing.assert_array_equal(data[0]["input"], tiles[:1])
                        else:
                            noise_count += 1
                            self.assertEqual(call.kwargs, dict(preset=preset))
                            np.testing.assert_array_equal(data[0]["input"], expected)
                self.assertEqual(noise_count, 10 if video else 12)
            load.assert_not_called()

    def test_cli_dry_run_does_not_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sub, src, *_ in q.TARGETS:
                (root / sub).mkdir(exist_ok=True)
                (root / sub / src).touch()
            for video in ([], ["--calib-video", "nonexistent.mkv"]):
                output = io.StringIO()
                with patch.object(sys, "argv", ["quantize_int8.py", "--onnx-dir", directory,
                                  "--output", str(root / "out"), "--dry-run"] + video), \
                        patch.object(q, "load_chroma_calibration") as load, \
                        contextlib.redirect_stdout(output):
                    q.main()
                load.assert_not_called()
                self.assertEqual(output.getvalue().count("[DRY]"), 12)
                self.assertEqual(output.getvalue().count(", ycbcr)"), 2 if video else 0)
                self.assertEqual(output.getvalue().count(", noise)"), 10 if video else 12)
                self.assertFalse((root / "out").exists())

    def test_main_shares_tiles_and_skips_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sub, src, *_ in q.TARGETS:
                (root / sub).mkdir(exist_ok=True)
                (root / sub / src).touch()
            tiles = np.zeros((300, 3, 64, 64), np.float32)
            argv = ["quantize_int8.py", "--onnx-dir", directory,
                    "--output", str(root / "out"), "--calib-video", "clip.mkv"]
            with patch.object(sys, "argv", argv), \
                    patch.object(q, "load_chroma_calibration", return_value=tiles) as load, \
                    patch.object(q, "quantize_one", return_value=True) as quantize, \
                    contextlib.redirect_stdout(io.StringIO()):
                q.main()
                load.assert_called_once_with("clip.mkv")
                self.assertEqual(quantize.call_count, 12)
                for call in quantize.call_args_list:
                    if Path(call.args[1]).name in q.CHROMA_MODELS:
                        self.assertIs(call.args[6], tiles)
                for name in q.CHROMA_MODELS:
                    path = root / "out" / "artcnn" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                load.reset_mock()
                q.main()
                load.assert_not_called()

    def test_decode_errors(self):
        with patch.object(q.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError):
                q.load_chroma_calibration("missing")
        for code, raw in ((1, b""), (0, b""), (0, b"partial")):
            with self.subTest(code=code, raw=raw), \
                    patch.object(q.shutil, "which", return_value="ffmpeg"), \
                    patch.object(q.subprocess, "run", return_value=types.SimpleNamespace(
                        returncode=code, stdout=raw, stderr=b"decode error")):
                with self.assertRaises(RuntimeError):
                    q.load_chroma_calibration("invalid")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpegが必要です")
    def test_ffmpeg_tiles_and_random_state(self):
        with tempfile.TemporaryDirectory() as directory:
            video = str(Path(directory) / "gray.mkv")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
                            "-i", "color=c=gray:s=96x80:r=2:d=2", "-c:v", "ffv1", video],
                           check=True, capture_output=True)
            state = np.random.get_state()
            tiles = q.load_chroma_calibration(video)
            after = np.random.get_state()
            np.testing.assert_array_equal(state[1], after[1])
            self.assertEqual(state[2:], after[2:])
            self.assertEqual(tiles.shape, (300, 3, 64, 64))
            self.assertEqual(tiles.dtype, np.float32)
            self.assertTrue(np.isfinite(tiles).all())
            self.assertGreater(float(tiles[:, 0].mean()), .45)
            self.assertLess(float(tiles[:, 0].mean()), .55)
            np.testing.assert_allclose(tiles[:, 1:], .5, atol=.003)


if __name__ == "__main__":
    unittest.main()
