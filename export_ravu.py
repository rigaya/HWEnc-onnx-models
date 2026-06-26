"""Orchestrate RAVU ONNX conversion (LGPL-3.0, bjin/mpv-prescalers).

Calls the four family-specific converters to produce 21 ONNX models:
  convert_ravu_lite.py  — RAVU-Lite 2x (r2/r3/r4)           = 3
  convert_ravu_base.py  — base RAVU 2x (r2/r3/r4)           = 3
  convert_ravu_3x.py    — RAVU-3x 3x (r2/r3/r4)             = 3
  convert_ravu_zoom.py  — RAVU-Zoom fixed-ratio (r2/r3,      = 12
                          2x/3x/4x, plain + anti-ringing)
"""
import argparse, subprocess, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

CONVERSIONS = [
    ("convert_ravu_lite.py", ["2"]),
    ("convert_ravu_lite.py", ["3"]),
    ("convert_ravu_lite.py", ["4"]),
    ("convert_ravu_base.py", ["2"]),
    ("convert_ravu_base.py", ["3"]),
    ("convert_ravu_base.py", ["4"]),
    ("convert_ravu_3x.py", ["2"]),
    ("convert_ravu_3x.py", ["3"]),
    ("convert_ravu_3x.py", ["4"]),
    ("convert_ravu_zoom.py", ["2", "2"]),
    ("convert_ravu_zoom.py", ["2", "2", "ar"]),
    ("convert_ravu_zoom.py", ["2", "3"]),
    ("convert_ravu_zoom.py", ["2", "3", "ar"]),
    ("convert_ravu_zoom.py", ["2", "4"]),
    ("convert_ravu_zoom.py", ["2", "4", "ar"]),
    ("convert_ravu_zoom.py", ["3", "2"]),
    ("convert_ravu_zoom.py", ["3", "2", "ar"]),
    ("convert_ravu_zoom.py", ["3", "3"]),
    ("convert_ravu_zoom.py", ["3", "3", "ar"]),
    ("convert_ravu_zoom.py", ["3", "4"]),
    ("convert_ravu_zoom.py", ["3", "4", "ar"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", required=True,
                        help="Path to mpv-prescalers clone (source branch)")
    parser.add_argument("--output", required=True,
                        help="Output directory for RAVU ONNX files")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    failed = 0
    for script, extra_args in CONVERSIONS:
        cmd = [sys.executable, str(SCRIPT_DIR / script),
               "--upstream", args.repo_root,
               "--output-dir", str(out)] + extra_args
        label = f"{script} {' '.join(extra_args)}"
        print(f"  [RAVU] {label}")
        try:
            r = subprocess.run(cmd, check=False)
            if r.returncode != 0:
                print(f"  [FAIL] {label}")
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            failed += 1

    count = len(list(out.glob("*.onnx")))
    print(f"  [RAVU] {count} ONNX files generated ({failed} failures)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
