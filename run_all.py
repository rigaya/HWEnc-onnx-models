#!/usr/bin/env python3
"""One-shot ONNX model builder for QSVEnc/NVEnc --vpp-onnx.

Runs the whole pipeline in one go:

    Phase 1: build the local venv (setup_env.sh / setup_env.bat)
    Phase 2: download repos + pretrained weights
    Phase 3: run every export_*.py to convert weights/GLSL/JSON to ONNX
    Phase 4: scan the output tree and emit models.json

Usage:
    python run_all.py --output /path/to/output_dir
    python run_all.py --output OUT --skip-download
    python run_all.py --output OUT --skip-convert
    python run_all.py --output OUT --dry-run
    python run_all.py --output OUT --jobs 4

Layout produced under --output:
    output/_work/repos/...        cloned source repos (intermediate)
    output/_work/realesrgan/...   Real-ESRGAN .pth weights (intermediate)
    output/_work/realcugan_weights/...  Real-CUGAN .pth weights (intermediate)
    output/<family>/              converted ONNX files
    output/models.json            manifest (key = stem.lower(), path relative)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if sys.platform == "win32":
    VENV_PYTHON = SCRIPT_DIR / ".venv_onnx" / "Scripts" / "python.exe"
    SETUP_ENV_SCRIPT = SCRIPT_DIR / "setup_env.bat"
else:
    VENV_PYTHON = SCRIPT_DIR / ".venv_onnx" / "bin" / "python"
    SETUP_ENV_SCRIPT = SCRIPT_DIR / "setup_env.sh"

# ---------------------------------------------------------------------------
# Download tables (ported from setup_models.py, extended with the repos the
# new export_*.py scripts need: Anime4K, websr, Real-CUGAN/ailab).
# ---------------------------------------------------------------------------

REPOS = [
    ("ArtCNN", "https://github.com/Artoriuz/ArtCNN/archive/refs/heads/main.zip", "_work/repos/ArtCNN"),
    ("KAIR", "https://github.com/cszn/KAIR/archive/refs/heads/master.zip", "_work/repos/KAIR"),
    ("Real-ESRGAN", "https://github.com/xinntao/Real-ESRGAN/archive/refs/heads/master.zip", "_work/repos/Real-ESRGAN"),
    ("waifu2x", "https://github.com/nagadomi/waifu2x/archive/refs/heads/master.zip", "_work/repos/waifu2x"),
    ("waifu2x-ncnn-vulkan", "https://github.com/nihui/waifu2x-ncnn-vulkan/archive/refs/heads/master.zip", "_work/repos/waifu2x-ncnn-vulkan"),
    ("Anime4KCPP", "https://github.com/TianZerL/Anime4KCPP/archive/refs/heads/master.zip", "_work/repos/Anime4KCPP"),
    ("ACNetGLSL", "https://github.com/TianZerL/ACNetGLSL/archive/refs/heads/master.zip", "_work/repos/ACNetGLSL"),
    ("Anime4K", "https://github.com/bloc97/Anime4K/archive/refs/heads/master.zip", "_work/repos/Anime4K"),
    ("websr", "https://github.com/sb2702/websr/archive/refs/heads/main.zip", "_work/repos/websr"),
    ("Real-CUGAN", "https://github.com/bilibili/ailab/archive/refs/heads/main.zip", "_work/repos/Real-CUGAN"),
]

REALCUGAN_WEIGHTS_URL = "https://github.com/bilibili/ailab/releases/download/Real-CUGAN/updated_weights.zip"
ANIME4K_RELEASE_URL = "https://github.com/bloc97/Anime4K/releases/download/v4.0.1/Anime4K_v4.0.zip"

# R4F32 ONNX models were removed from ArtCNN main after commit e13ac6c7 (2026-05-12).
ARTCNN_EXTRA_ONNX = [
    ("https://github.com/Artoriuz/ArtCNN/raw/e13ac6c7/ONNX/Experiments/ArtCNN_R4F32.onnx", "artcnn/ArtCNN_R4F32.onnx"),
    ("https://github.com/Artoriuz/ArtCNN/raw/e13ac6c7/ONNX/Experiments/ArtCNN_R4F32_DN.onnx", "artcnn/ArtCNN_R4F32_DN.onnx"),
]

REALESRGAN_RELEASES = [
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", "_work/realesrgan/realesr-animevideov3.pth"),
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", "_work/realesrgan/realesr-general-x4v3.pth"),
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth", "_work/realesrgan/realesr-general-wdn-x4v3.pth"),
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "_work/realesrgan/RealESRGAN_x4plus.pth"),
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", "_work/realesrgan/RealESRGAN_x4plus_anime_6B.pth"),
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", "_work/realesrgan/RealESRGAN_x2plus.pth"),
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth", "_work/realesrgan/RealESRNet_x4plus.pth"),
]

KAIR_MODELS = [
    # export_kair_rrdb.py
    "ESRGAN.pth", "RealSR_DPED.pth", "RealSR_JPEG.pth", "FSSR_DPED.pth", "FSSR_JPEG.pth",
    # export_bsrgan.py
    "BSRGAN.pth", "BSRGANx2.pth", "BSRNet.pth",
    # export_dpsr.py
    "dpsr_x2.pth", "dpsr_x3.pth", "dpsr_x4.pth", "dpsr_x4_gan.pth",
    # export_drunet.py
    "drunet_color.pth", "drunet_deblocking_color.pth", "drunet_gray.pth", "drunet_deblocking_grayscale.pth",
    # export_kair_denoise.py (DnCNN)
    "dncnn_gray_blind.pth", "dncnn_color_blind.pth", "dncnn3.pth", "dncnn_15.pth", "dncnn_25.pth", "dncnn_50.pth",
    # export_kair_denoise.py (FDnCNN)
    "fdncnn_gray.pth", "fdncnn_gray_clip.pth", "fdncnn_color.pth", "fdncnn_color_clip.pth",
    # export_kair_denoise.py (FFDNet)
    "ffdnet_gray.pth", "ffdnet_gray_clip.pth", "ffdnet_color.pth", "ffdnet_color_clip.pth",
    # export_srmd.py
    "srmd_x2.pth", "srmd_x3.pth", "srmd_x4.pth", "srmdnf_x2.pth", "srmdnf_x3.pth", "srmdnf_x4.pth",
]


# ---------------------------------------------------------------------------
# Conversion table. Placeholders are filled per-run:
#   {repos}  = output/_work/repos
#   {work}   = output/_work
#   {out}    = output          (model output root)
# Each export_*.py takes its inputs and an --output directory; failures are
# logged and skipped so one broken family does not abort the whole run.
# ---------------------------------------------------------------------------

CONVERT_COMMANDS = [
    # --- GLSL based (self-contained, read shaders straight from the repos) ---
    ("export_acnet.py", ["--glsl-dir", "{repos}/ACNetGLSL/glsl/acnet", "--output", "{out}/acnet"]),
    ("export_arnet.py", ["--glsl-dir", "{repos}/ACNetGLSL/glsl/arnet", "--output", "{out}/arnet"]),
    ("export_fsrcnnx.py", ["--param-header", "{repos}/Anime4KCPP/core/include/AC/Core/Model/Param/FSRCNNX.p", "--output", "{out}/fsrcnnx"]),
    ("export_anime4k_upscale_cnn.py", ["--glsl-dir", "{repos}/Anime4K/glsl/Upscale", "--output", "{out}/anime4k_upscale"]),
    ("export_anime4k_restore.py", ["--glsl-dir", "{repos}/Anime4K/glsl/Restore", "--output", "{out}/anime4k_restore"]),
    ("export_anime3d.py", ["--glsl-dir", "{repos}/Anime4K/glsl/Upscale", "--output", "{out}/anime3d"]),
    ("export_anime4k_gan.py", ["--glsl-dir", "{repos}/Anime4K/glsl/Upscale", "--output", "{out}/anime4k_gan"]),

    # --- .pth based (network definitions imported from the external repos) ---
    ("export_realesrgan.py", ["--models-dir", "{work}/realesrgan", "--output", "{out}/realesrgan"]),
    ("export_bsrgan.py", ["--repo-root", "{repos}/KAIR", "--weights-dir", "{repos}/KAIR/model_zoo", "--output", "{out}/bsrgan"]),
    ("export_dpsr.py", ["--repo-root", "{repos}/KAIR", "--weights-dir", "{repos}/KAIR/model_zoo", "--output", "{out}/dpsr"]),
    ("export_drunet.py", ["--repo-root", "{repos}/KAIR", "--weights-dir", "{repos}/KAIR/model_zoo", "--output", "{out}/drunet"]),
    # kair_denoise creates dncnn/fdncnn/ffdnet subdirs under --output itself.
    ("export_kair_denoise.py", ["--repo-root", "{repos}/KAIR", "--weights-dir", "{repos}/KAIR/model_zoo", "--output", "{out}"]),
    ("export_kair_rrdb.py", ["--repo-root", "{repos}/KAIR", "--weights-dir", "{repos}/KAIR/model_zoo", "--output", "{out}/esrgan"]),
    ("export_srmd.py", ["--repo-root", "{repos}/KAIR", "--weights-dir", "{repos}/KAIR/model_zoo", "--output", "{out}/srmd"]),
    ("export_realcugan.py", ["--repo-root", "{repos}/Real-CUGAN/Real-CUGAN", "--weights-dir", "{work}/realcugan_weights", "--output", "{out}/realcugan"]),

    # --- JSON based ---
    ("export_waifu2x.py", ["--models-dir", "{repos}/waifu2x/models", "--output", "{out}/waifu2x"]),
    ("export_waifu2x_cunet.py", ["--param-dir", "{repos}/waifu2x-ncnn-vulkan/models/models-cunet", "--json-dir", "{repos}/waifu2x/models/cunet/art", "--output", "{out}/waifu2x"]),
    ("export_websr.py", ["--json-dir", "{repos}/websr/weights/anime4k", "--output", "{out}/websr"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", required=True, type=Path, help="Output root directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip venv setup + downloads")
    parser.add_argument("--skip-convert", action="store_true", help="Skip the export/convert phase")
    parser.add_argument("--skip-int8", action="store_true", help="Skip INT8 quantization phase")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without doing anything")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel conversion workers")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Download helpers (ported verbatim from setup_models.py)
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, dry_run: bool) -> bool:
    if dest.exists():
        print(f"    [SKIP] {dest.name}")
        return True
    print(f"    [DL] {dest.name}")
    print(f"      {url}")
    if dry_run:
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        tmp_dest = dest.with_name(dest.name + ".tmp")
        tmp_dest.unlink(missing_ok=True)
        for _ in range(3):
            try:
                with urllib.request.urlopen(url) as response, tmp_dest.open("wb") as out_file:
                    shutil.copyfileobj(response, out_file)
                tmp_dest.replace(dest)
                return True
            except (OSError, urllib.error.URLError) as err:
                last_error = err
                tmp_dest.unlink(missing_ok=True)
        print(f"    [FAIL] {url}: {last_error}")
    except Exception as err:
        print(f"    [FAIL] {url}: {err}")
    return False


def move_inner_contents_up(dest: Path) -> None:
    inner_dirs = [item for item in dest.iterdir() if item.is_dir()]
    if not inner_dirs:
        return
    inner = inner_dirs[0]
    if inner.name == dest.name:
        return
    for item in inner.iterdir():
        shutil.move(str(item), str(dest / item.name))
    try:
        inner.rmdir()
    except OSError:
        pass


def download_repo(name: str, url: str, dest: Path, work_dir: Path, dry_run: bool) -> None:
    if dest.is_dir():
        print(f"  [SKIP] {name} (already exists: {dest})")
        return
    zip_path = work_dir / f"{name}.zip"
    print(f"  [DL] {name}")
    print(f"    {url}")
    if dry_run:
        print(f"  [UNZIP] -> {dest}")
        return
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        if not download_file(url, zip_path, False):
            return
        size = zip_path.stat().st_size / 1048576
        print(f"    {size:.1f} MB")
        print(f"  [UNZIP] -> {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dest)
        zip_path.unlink(missing_ok=True)
        move_inner_contents_up(dest)
        print("    OK")
    except Exception as err:
        print(f"    [FAIL] {name}: {err}")


# ---------------------------------------------------------------------------
# Phase 1: venv
# ---------------------------------------------------------------------------

def setup_venv(dry_run: bool) -> None:
    print("[PHASE 1] venv setup")
    print(f"  {SETUP_ENV_SCRIPT}")
    if dry_run:
        return
    try:
        if sys.platform == "win32":
            subprocess.run([str(SETUP_ENV_SCRIPT)], check=True, shell=True)
        else:
            subprocess.run(["bash", str(SETUP_ENV_SCRIPT)], check=True)
    except subprocess.CalledProcessError as err:
        print(f"  [FAIL] {SETUP_ENV_SCRIPT.name}: return code {err.returncode}")
    except Exception as err:
        print(f"  [FAIL] {SETUP_ENV_SCRIPT.name}: {err}")


# ---------------------------------------------------------------------------
# Phase 2: download
# ---------------------------------------------------------------------------

def download_phase(output: Path, dry_run: bool) -> None:
    print("[PHASE 2] download repos + weights")
    work_dir = output / "_work"

    for name, url, rel_dest in REPOS:
        download_repo(name, url, output / rel_dest, work_dir, dry_run)

    print("  Downloading pretrained weights from GitHub Releases...")
    for url, rel_dest in REALESRGAN_RELEASES:
        download_file(url, output / rel_dest, dry_run)

    for name in KAIR_MODELS:
        url = f"https://github.com/cszn/KAIR/releases/download/v1.0/{name}"
        download_file(url, output / "_work/repos/KAIR/model_zoo" / name, dry_run)

    # Anime4K Denoise GLSL are only in the v4.0.1 release zip, not the git repo
    supplement_anime4k_glsl(output, dry_run)

    # SRMD needs .mat files from kernels/ in the same dir as the .pth weights
    copy_kair_mat_files(output, dry_run)

    print("  Downloading Real-CUGAN weights...")
    download_realcugan_weights(output, dry_run)

    # ArtCNN ships pre-built ONNX (incl. int8); just copy them in, no export.
    copy_artcnn(output, dry_run)

    # R4F32 models removed from ArtCNN main; fetch from pinned commit.
    for url, rel_dest in ARTCNN_EXTRA_ONNX:
        download_file(url, output / rel_dest, dry_run)

    if not dry_run:
        try:
            work_dir.rmdir()
        except OSError:
            pass


def supplement_anime4k_glsl(output: Path, dry_run: bool) -> None:
    """Download Anime4K v4.0.1 release zip and extract Denoise GLSL shaders
    that are not in the git repo but are needed by export_anime4k_upscale_cnn.py."""
    upscale_dir = output / "_work/repos/Anime4K/glsl/Upscale"
    needed = [f"Anime4K_Upscale_Denoise_CNN_x2_{s}.glsl" for s in ("S", "M", "L", "VL", "UL")]
    if all((upscale_dir / n).exists() for n in needed):
        print(f"  [SKIP] Anime4K Denoise GLSL (already present)")
        return
    zip_path = output / "_work" / "Anime4K_v4.0.zip"
    print(f"  [DL] Anime4K v4.0.1 release (for Denoise GLSL)")
    if dry_run:
        return
    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not download_file(ANIME4K_RELEASE_URL, zip_path, False):
            return
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                fname = Path(member).name
                if fname in needed:
                    with archive.open(member) as src, (upscale_dir / fname).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        zip_path.unlink(missing_ok=True)
        copied = sum(1 for n in needed if (upscale_dir / n).exists())
        print(f"    OK ({copied}/{len(needed)} Denoise GLSL extracted)")
    except Exception as err:
        print(f"    [FAIL] Anime4K Denoise GLSL: {err}")


def copy_kair_mat_files(output: Path, dry_run: bool) -> None:
    """Copy .mat files from KAIR/kernels/ to KAIR/model_zoo/ for SRMD export."""
    kernels_dir = output / "_work/repos/KAIR/kernels"
    model_zoo = output / "_work/repos/KAIR/model_zoo"
    mat_files = ["srmd_pca_matlab.mat", "kernels_bicubicx234.mat"]
    for name in mat_files:
        src = kernels_dir / name
        dst = model_zoo / name
        if dst.exists():
            continue
        if dry_run:
            print(f"    [COPY] {src} -> {dst}")
            continue
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"    [COPY] {name} -> model_zoo/")
        else:
            print(f"    [WARN] {src} not found")


def download_realcugan_weights(output: Path, dry_run: bool) -> None:
    dest_dir = output / "_work/realcugan_weights"
    zip_path = output / "_work" / "realcugan_weights.zip"
    if dest_dir.is_dir() and any(dest_dir.glob("*.pth")):
        print(f"  [SKIP] Real-CUGAN weights (already exists: {dest_dir})")
        return
    print(f"  [DL] Real-CUGAN weights")
    print(f"    {REALCUGAN_WEIGHTS_URL}")
    if dry_run:
        return
    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not download_file(REALCUGAN_WEIGHTS_URL, zip_path, False):
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith(".pth"):
                    fname = Path(member).name
                    with archive.open(member) as src, (dest_dir / fname).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        zip_path.unlink(missing_ok=True)
        count = len(list(dest_dir.glob("*.pth")))
        print(f"    OK ({count} .pth files)")
    except Exception as err:
        print(f"    [FAIL] Real-CUGAN weights: {err}")


def copy_artcnn(output: Path, dry_run: bool) -> None:
    src = output / "_work/repos/ArtCNN/ONNX"
    dst = output / "artcnn"
    print(f"  [ARTCNN] copy {src} -> {dst}")
    if dry_run:
        return
    if not src.is_dir():
        print(f"    [SKIP] ArtCNN ONNX dir not found: {src}")
        return
    try:
        copied = 0
        for onnx_file in src.rglob("*.onnx"):
            target = dst / onnx_file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(onnx_file, target)
                copied += 1
        print(f"    OK ({copied} files)")
    except Exception as err:
        print(f"    [FAIL] ArtCNN copy: {err}")


POSTCONVERT_ALIASES = [
    ("realesrgan", "RealESRGAN_x4plus_anime_6B.onnx", "realesrgan_anime_6b.onnx"),
    ("realcugan", "up2x_latest_no_denoise.onnx", "upcunet2x_no_denoise.onnx"),
]


def create_postconvert_aliases(output: Path, dry_run: bool) -> None:
    for subdir, src_name, dst_name in POSTCONVERT_ALIASES:
        parent = output / subdir
        src = parent / src_name
        dst = parent / dst_name
        if dst.exists():
            print(f"  [SKIP] {dst_name} (already exists)")
            continue
        if dry_run:
            print(f"  [ALIAS] {subdir}/{src_name} -> {dst_name}")
            continue
        if not src.exists():
            print(f"  [WARN] {subdir}/{src_name} not found, cannot create {dst_name}")
            continue
        shutil.copy2(src, dst)
        print(f"  [ALIAS] {subdir}/{src_name} -> {dst_name}")


# ---------------------------------------------------------------------------
# Phase 3: convert
# ---------------------------------------------------------------------------

def build_commands(output: Path) -> list[list[str]]:
    fields = {
        "repos": str(output / "_work/repos"),
        "work": str(output / "_work"),
        "out": str(output),
    }
    commands: list[list[str]] = []
    for script, args in CONVERT_COMMANDS:
        cmd = [str(VENV_PYTHON), str(SCRIPT_DIR / script)]
        cmd += [a.format(**fields) for a in args]
        commands.append(cmd)
    return commands


def run_command(command: list[str], dry_run: bool) -> int:
    print("[CONVERT] " + " ".join(command))
    if dry_run:
        return 0
    try:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"[FAIL] return code {completed.returncode}: {' '.join(command)}")
        return completed.returncode
    except Exception as err:
        print(f"[FAIL] {' '.join(command)}: {err}")
        return 1


def convert_phase(output: Path, jobs: int, dry_run: bool) -> None:
    print("[PHASE 3] convert weights/GLSL/JSON -> ONNX")
    if not dry_run and not VENV_PYTHON.exists():
        print(f"  [WARN] venv python not found: {VENV_PYTHON}")
        print("  [WARN] run without --skip-download first, or run setup_env.sh")
    commands = build_commands(output)
    worker_count = max(1, jobs)
    if worker_count == 1 or dry_run:
        for command in commands:
            run_command(command, dry_run)
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_command, command, False) for command in commands]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as err:
                print(f"[FAIL] conversion worker: {err}")


# ---------------------------------------------------------------------------
# Phase 3.5: INT8 quantization
# ---------------------------------------------------------------------------

def quantize_phase(output: Path, dry_run: bool) -> None:
    print("[PHASE 3.5] INT8 quantization (nncf)")
    cmd = [str(VENV_PYTHON), str(SCRIPT_DIR / "quantize_int8.py"),
           "--onnx-dir", str(output), "--output", str(output)]
    if dry_run:
        cmd.append("--dry-run")
    run_command(cmd, False)


# ---------------------------------------------------------------------------
# License files
# ---------------------------------------------------------------------------

LICENSES_DIR = SCRIPT_DIR / "licenses"

LICENSE_MAP = {
    "acnet": "acnet.txt",
    "anime3d": "anime3d.txt",
    "anime4k_gan": "anime4k_gan.txt",
    "anime4k_restore": "anime4k_restore.txt",
    "anime4k_upscale": "anime4k_upscale.txt",
    "arnet": "arnet.txt",
    "artcnn": "artcnn.txt",
    "bsrgan": "bsrgan.txt",
    "dncnn": "dncnn.txt",
    "dpsr": "dpsr.txt",
    "drunet": "drunet.txt",
    "esrgan": "esrgan.txt",
    "fdncnn": "fdncnn.txt",
    "ffdnet": "ffdnet.txt",
    "fsrcnnx": "fsrcnnx.txt",
    "realcugan": "realcugan.txt",
    "realesrgan": "realesrgan.txt",
    "srmd": "srmd.txt",
    "waifu2x": "waifu2x.txt",
    "websr": "websr.txt",
}


def install_licenses(output: Path, dry_run: bool) -> None:
    print("[LICENSE] install license files")
    copied = 0
    for family, filename in LICENSE_MAP.items():
        src = LICENSES_DIR / filename
        dst = output / family / "LICENSE.txt"
        if not src.exists():
            print(f"  [WARN] {src} not found")
            continue
        if not (output / family).is_dir():
            if not dry_run:
                continue
        if dry_run:
            print(f"  [LICENSE] {family}/LICENSE.txt")
            copied += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"  {copied} license files installed")


# ---------------------------------------------------------------------------
# Phase 4: models.json
# ---------------------------------------------------------------------------

SCAN_EXCLUDE = {"_work"}


def generate_models_json(output: Path, dry_run: bool) -> None:
    print("[PHASE 4] generate models.json")
    output_path = output / "models.json"
    if dry_run:
        print(f"  [SCAN] {output}/**/*.onnx -> {output_path}")
        return

    models: dict[str, dict] = {}
    duplicates = 0
    for onnx_file in sorted(output.rglob("*.onnx")):
        rel = onnx_file.relative_to(output)
        if rel.parts[0] in SCAN_EXCLUDE:
            continue
        key = onnx_file.stem.lower()
        if key in models:
            duplicates += 1
            print(f"  [DUP] key '{key}' already registered, skipping {rel}")
            continue
        models[key] = {"path": str(rel.as_posix())}

    manifest = {"version": 1, "models": models}
    try:
        output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"  [OK] {output_path}")
        print(f"  models: {len(models)} (duplicates skipped: {duplicates})")
    except Exception as err:
        print(f"  [FAIL] models.json: {err}")


# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    output = args.output

    print(f"[RUN_ALL] output: {output}")
    print(f"[RUN_ALL] jobs:   {max(1, args.jobs)}")
    print(f"[RUN_ALL] dry-run: {args.dry_run}")

    if not args.skip_download:
        setup_venv(args.dry_run)
        download_phase(output, args.dry_run)
    else:
        print("[SKIP] download phase (--skip-download)")

    if not args.skip_convert:
        convert_phase(output, args.jobs, args.dry_run)
    else:
        print("[SKIP] convert phase (--skip-convert)")

    # Post-convert aliases (source ONNX must exist by now)
    create_postconvert_aliases(output, args.dry_run)

    if not args.skip_int8:
        quantize_phase(output, args.dry_run)
    else:
        print("[SKIP] INT8 quantization (--skip-int8)")

    install_licenses(output, args.dry_run)
    generate_models_json(output, args.dry_run)
    print("[RUN_ALL] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
