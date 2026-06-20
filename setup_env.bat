@echo off
REM Create/update the local venv for HWEnc-onnx-models.
REM
REM Usage:
REM   setup_env.bat

setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv_onnx"

echo === Setting up Python venv ===

if not exist "%VENV_DIR%\Scripts\python.exe" (
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

pip install --quiet --upgrade pip
pip install --quiet -r "%SCRIPT_DIR%requirements.txt"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo.
python --version
python -c "import torch; print('  torch: ' + torch.__version__)"
python -c "import onnx; print('  onnx:  ' + onnx.__version__)"
echo.
echo Use:
echo   .venv_onnx\Scripts\python run_all.py --output PATH --dry-run

endlocal
