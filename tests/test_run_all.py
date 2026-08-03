#!/usr/bin/env python3
"""run_all.pyのデインターレース用マニフェスト生成テスト。"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_all", SCRIPT_DIR / "run_all.py")
assert SPEC is not None and SPEC.loader is not None
run_all = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_all)


def read_manifest(output: Path, filename: str = "onnx_deint_models.json") -> dict:
    return json.loads((output / filename).read_text(encoding="utf-8"))


class OnnxDeintManifestTest(unittest.TestCase):
    def test_all_models_are_registered_with_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            for relative_path in (
                "stdeint/stdeint.onnx",
                "stdeint/stdeint_fast.onnx",
                "ddd/DDD.onnx",
            ):
                path = output / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            run_all.generate_onnx_deint_models_json(output, dry_run=False)

            self.assertEqual(
                read_manifest(output),
                {
                    "version": 1,
                    "models": {
                        "stdeint": {"path": "stdeint/stdeint.onnx", "architecture": "stdeint"},
                        "stdeint_fast": {"path": "stdeint/stdeint_fast.onnx", "architecture": "stdeint"},
                        "DDD": {"path": "ddd/DDD.onnx", "architecture": "ddd"},
                    },
                },
            )

    def test_missing_ddd_is_not_registered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            for model_name in ("stdeint", "stdeint_fast"):
                path = output / "stdeint" / f"{model_name}.onnx"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            run_all.generate_onnx_deint_models_json(output, dry_run=False)

            self.assertEqual(set(read_manifest(output)["models"]), {"stdeint", "stdeint_fast"})

    def test_no_models_writes_empty_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            run_all.generate_onnx_deint_models_json(output, dry_run=False)
            self.assertEqual(read_manifest(output), {"version": 1, "models": {}})

    def test_ddd_is_excluded_from_generic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            ddd_path = output / "ddd" / "DDD.onnx"
            ddd_path.parent.mkdir(parents=True, exist_ok=True)
            ddd_path.write_bytes(b"test")
            generic_path = output / "artcnn" / "example.onnx"
            generic_path.parent.mkdir(parents=True, exist_ok=True)
            generic_path.write_bytes(b"test")

            run_all.generate_models_json(output, dry_run=False)
            run_all.generate_onnx_deint_models_json(output, dry_run=False)

            self.assertNotIn("ddd", read_manifest(output, "models.json")["models"])
            self.assertIn("example", read_manifest(output, "models.json")["models"])
            self.assertIn("DDD", read_manifest(output)["models"])

    def test_dry_run_only_prints_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                run_all.generate_onnx_deint_models_json(output, dry_run=True)

            self.assertIn("onnx_deint_models.json", stdout.getvalue())
            self.assertIn("ddd/DDD.onnx", stdout.getvalue())
            self.assertFalse((output / "onnx_deint_models.json").exists())

    def test_main_dry_run_calls_new_generator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "run_all.py"),
                    "--output",
                    temp_dir,
                    "--dry-run",
                    "--skip-download",
                    "--skip-convert",
                    "--skip-int8",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[PHASE 4] generate onnx_deint_models.json", result.stdout)
            self.assertIn("ddd/DDD.onnx", result.stdout)

    def test_legacy_manifest_output_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            for model_name in ("stdeint", "stdeint_fast"):
                path = output / "stdeint" / f"{model_name}.onnx"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            run_all.generate_stdeint_ov_models_json(output, dry_run=False)

            self.assertEqual(
                read_manifest(output, "stdeint_ov_models.json"),
                {
                    "version": 1,
                    "models": {
                        "stdeint": {"path": "stdeint/stdeint.onnx"},
                        "stdeint_fast": {"path": "stdeint/stdeint_fast.onnx"},
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
