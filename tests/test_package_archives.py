#!/usr/bin/env python3
"""モデルアーカイブの収録範囲を検証するユニットテスト。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = SCRIPT_DIR / "package_archives.py"


def touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def archive_entries(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


class PackageArchivesTest(unittest.TestCase):
    def run_packager(self, input_dir: Path, dist_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(PACKAGE_SCRIPT),
                "--input",
                str(input_dir),
                "--dist",
                str(dist_dir),
                "--version",
                "test",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def create_common_fixture(self, output: Path, include_ddd: bool) -> None:
        # 必須の汎用モデルと分割対象を用意する。
        for relative_path in (
            "models.json",
            "rife_ov_models.json",
            "artcnn/example.onnx",
            "fbcnn/example.onnx",
            "nafnet/example.onnx",
            "stdeint/stdeint.onnx",
            "stdeint/stdeint_fast.onnx",
            "stdeint_ov_models.json",
            "onnx_deint_models.json",
        ):
            touch(output, relative_path)
        if include_ddd:
            touch(output, "ddd/DDD.onnx")

    def test_main_excludes_both_deint_families_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            dist = root / "dist"
            self.create_common_fixture(output, include_ddd=True)

            result = self.run_packager(output, dist)

            self.assertEqual(result.returncode, 0, result.stderr)
            main = archive_entries(dist / "onnx_models_test_main.zip")
            split = archive_entries(dist / "onnx_models_test_fbcnn_nafnet.zip")
            self.assertIn("models.json", main)
            self.assertIn("rife_ov_models.json", main)
            self.assertIn("artcnn/example.onnx", main)
            self.assertNotIn("stdeint/stdeint.onnx", main)
            self.assertNotIn("stdeint/stdeint_fast.onnx", main)
            self.assertNotIn("ddd/DDD.onnx", main)
            self.assertNotIn("stdeint_ov_models.json", main)
            self.assertNotIn("onnx_deint_models.json", main)
            self.assertEqual(split, {"fbcnn/example.onnx", "nafnet/example.onnx"})

    def test_missing_ddd_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            dist = root / "dist"
            self.create_common_fixture(output, include_ddd=False)

            result = self.run_packager(output, dist)

            self.assertEqual(result.returncode, 0, result.stderr)
            main = archive_entries(dist / "onnx_models_test_main.zip")
            self.assertIn("artcnn/example.onnx", main)
            self.assertNotIn("onnx_deint_models.json", main)
            self.assertNotIn("ddd/DDD.onnx", main)


if __name__ == "__main__":
    unittest.main()
