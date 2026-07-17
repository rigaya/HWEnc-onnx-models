#!/bin/bash
# Thin wrapper around package_archives.py for local / CI use.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/package_archives.py" "$@"
