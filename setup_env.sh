#!/bin/bash
# Create/update the local venv for HWEnc-onnx-models.
#
# Usage:
#   bash setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv_onnx"

echo "=== Setting up Python venv ==="

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo "  Python: $(python --version)"
echo "  torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "  onnx:   $(python -c 'import onnx; print(onnx.__version__)')"
echo ""
echo "Use:"
echo "  python $SCRIPT_DIR/run_all.py --output /path/to/output --dry-run"
