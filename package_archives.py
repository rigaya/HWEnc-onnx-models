#!/usr/bin/env python3
"""Package converted ONNX models into release zip archives.

Includes only generated models / manifests / per-family LICENSE.txt.
Excludes intermediate downloads (_work) and ST-DeInt (upstream license Unknown;
kept out of release archives).

Archives:
    onnx_models_<version>_main.zip
        All families except fbcnn/nafnet/stdeint.
    onnx_models_<version>_fbcnn_nafnet.zip
        fbcnn + nafnet (split for the GitHub 2GB release asset limit).

Usage:
    python package_archives.py --input /path/to/output --dist /path/to/dist --version 20260718
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

GITHUB_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

MAIN_EXCLUDE = {
    "_work",
    "fbcnn",
    "nafnet",
    "stdeint",
    "stdeint_ov_models.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="run_all.py output directory")
    parser.add_argument("--dist", required=True, type=Path, help="directory to write zip / sha256 files")
    parser.add_argument("--version", required=True, help="version/tag string used in archive names")
    return parser.parse_args()


def iter_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def write_zip(zip_path: Path, root: Path, files: list[Path]) -> None:
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(root).as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_size(path: Path) -> None:
    size = path.stat().st_size
    print(f"  {path.name}: {size / 1048576:.1f} MiB ({size} bytes)")
    if size >= GITHUB_LIMIT_BYTES:
        raise SystemExit(f"ERROR: {path.name} exceeds GitHub release asset limit (2 GiB)")


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    dist_dir = args.dist.resolve()
    version = args.version

    if not (input_dir / "models.json").is_file():
        raise SystemExit(f"ERROR: models.json not found under {input_dir}")
    if not (input_dir / "fbcnn").is_dir() or not (input_dir / "nafnet").is_dir():
        raise SystemExit(f"ERROR: fbcnn/ and nafnet/ are required under {input_dir}")

    dist_dir.mkdir(parents=True, exist_ok=True)
    main_zip = dist_dir / f"onnx_models_{version}_main.zip"
    fbcnn_zip = dist_dir / f"onnx_models_{version}_fbcnn_nafnet.zip"

    for old in dist_dir.glob("onnx_models_*"):
        if old.is_file():
            old.unlink()
    for old in dist_dir.glob("*.sha256"):
        old.unlink()
    (dist_dir / "SHA256SUMS.txt").unlink(missing_ok=True)

    main_files: list[Path] = []
    for child in sorted(input_dir.iterdir(), key=lambda path: path.name):
        if child.name in MAIN_EXCLUDE:
            continue
        if child.is_file():
            main_files.append(child)
        elif child.is_dir():
            main_files.extend(iter_files(child))

    fbcnn_files = iter_files(input_dir / "fbcnn") + iter_files(input_dir / "nafnet")

    if not main_files:
        raise SystemExit("ERROR: no entries found for main archive")
    if not fbcnn_files:
        raise SystemExit("ERROR: no entries found for fbcnn/nafnet archive")

    print("=== Packaging main archive ===")
    print(f"  input: {input_dir}")
    print(f"  zip:   {main_zip}")
    print(f"  files: {len(main_files)}")
    write_zip(main_zip, input_dir, main_files)

    print("=== Packaging fbcnn/nafnet archive ===")
    print(f"  zip:   {fbcnn_zip}")
    print(f"  files: {len(fbcnn_files)}")
    write_zip(fbcnn_zip, input_dir, fbcnn_files)

    archives = (main_zip, fbcnn_zip)

    print("=== Archive sizes ===")
    for path in archives:
        check_size(path)

    print("=== SHA-256 ===")
    sums_lines: list[str] = []
    for path in archives:
        digest = sha256_file(path)
        line = f"{digest}  {path.name}"
        sums_lines.append(line)
        (dist_dir / f"{path.name}.sha256").write_text(line + "\n", encoding="utf-8")
        print(line)

    (dist_dir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    print("=== Done ===")
    for path in archives:
        print(f"  {path}")
    print(f"  {dist_dir / 'SHA256SUMS.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
