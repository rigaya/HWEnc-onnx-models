# HWEnc-onnx-models

[**日本語版はこちら / Japanese**](README.ja.md)

Build tools for ONNX models used by the `--vpp-onnx` filter in QSVEnc / NVEnc / VCEEnc.

Generates 185 models (173 FP32 + 12 INT8) across 20 model families with a single command: download → convert → INT8 quantize → models.json.

## Quick Start

```bash
# 1. Set up venv (first time only)
bash setup_env.sh

# 2. Full build (download + convert + INT8 quantization + models.json)
.venv_onnx/bin/python run_all.py --output /path/to/output

# 3. Dry run (print plan without executing)
.venv_onnx/bin/python run_all.py --output /path/to/output --dry-run
```

## run_all.py Options

| Option | Description |
|--------|-------------|
| `--output PATH` | Output root directory (required) |
| `--skip-download` | Skip venv setup and downloads |
| `--skip-convert` | Skip FP32 ONNX conversion |
| `--skip-int8` | Skip INT8 quantization |
| `--jobs N` | Parallel conversion workers (default: 1) |
| `--dry-run` | Print plan without executing |

## Output Directory Structure

```
output/
├── repos/              # Cloned source repositories
├── realesrgan/         # Real-ESRGAN .pth weights
├── realcugan_weights/  # Real-CUGAN .pth weights
├── onnx/               # Converted ONNX models
│   ├── acnet/
│   ├── anime3d/
│   ├── anime4k_gan/
│   ├── anime4k_restore/
│   ├── anime4k_upscale/
│   ├── arnet/
│   ├── artcnn/
│   ├── bsrgan/
│   ├── dncnn/
│   ├── dpsr/
│   ├── drunet/
│   ├── esrgan/
│   ├── fdncnn/
│   ├── ffdnet/
│   ├── fsrcnnx/
│   ├── realcugan/
│   ├── realesrgan/
│   ├── srmd/
│   ├── waifu2x/
│   └── websr/
└── models.json         # Model manifest
```

## Model Families

| Family | Source Type | FP32 | INT8 | Total |
|--------|-----------|------|------|-------|
| acnet | GLSL | 12 | - | 12 |
| anime3d | GLSL | 2 | - | 2 |
| anime4k_gan | GLSL | 6 | - | 6 |
| anime4k_restore | GLSL | 6 | - | 6 |
| anime4k_upscale | GLSL | 10 | - | 10 |
| arnet | GLSL | 16 | - | 16 |
| artcnn | ONNX (upstream) | 18 | 6 | 24 |
| bsrgan | PyTorch .pth | 3 | - | 3 |
| dncnn | PyTorch .pth | 6 | - | 6 |
| dpsr | PyTorch .pth | 4 | 1 | 5 |
| drunet | PyTorch .pth | 4 | 1 | 5 |
| esrgan | PyTorch .pth | 5 | - | 5 |
| fdncnn | PyTorch .pth | 4 | - | 4 |
| ffdnet | PyTorch .pth | 4 | - | 4 |
| fsrcnnx | C++ header | 4 | - | 4 |
| realcugan | PyTorch .pth | 12 | 2 | 14 |
| realesrgan | PyTorch .pth | 8 | 2 | 10 |
| srmd | PyTorch .pth | 6 | - | 6 |
| waifu2x | JSON weights | 34 | - | 34 |
| websr | JSON weights | 9 | - | 9 |
| **Total** | | **173** | **12** | **185** |

## Conversion Source Types

### GLSL Shader Parsing
Extracts weights from Anime4K, ACNet, ARNet, and FSRCNNX shaders and builds ONNX graphs.

### PyTorch .pth Weights
Converts pretrained models from KAIR, Real-ESRGAN, Real-CUGAN, BSRGAN, etc. via `torch.onnx.export`.

### JSON Weights
Builds ONNX graphs from waifu2x and websr JSON weight files.

### Upstream ONNX
ArtCNN models are pre-built ONNX files published by the author.

### INT8 Quantization
Generates INT8 ONNX from FP32 ONNX using nncf (Neural Network Compression Framework) Post-Training Quantization.

## Requirements

- Python 3.10+
- PyTorch (CPU)
- ONNX
- NumPy
- SciPy
- nncf
- onnxruntime

`setup_env.sh` creates the venv and installs all dependencies.

## File Structure

| File | Description |
|------|-------------|
| `run_all.py` | Integrated runner (download → convert → INT8 → models.json) |
| `setup_env.sh` | Python venv setup |
| `quantize_int8.py` | INT8 quantization via nncf |
| `export_*.py` | Per-family FP32 ONNX conversion scripts (18 files) |
| `extract_anime4k_upscale_gan_glsl.py` | Anime4K GAN GLSL parsing helper |
| `requirements.txt` | Python dependencies |

## License

Each model's license follows its upstream repository:

- ArtCNN: MIT (Joao Chrisostomo)
- KAIR (BSRGAN, DPSR, DRUNet, DnCNN, FDnCNN, FFDNet, SRMD, ESRGAN): MIT (Kai Zhang)
- Real-ESRGAN: BSD-3-Clause (Xintao Wang)
- Real-CUGAN: MIT (bilibili)
- waifu2x: MIT (nagadomi)
- Anime4K: MIT (bloc97)
- ACNet/ARNet: MIT (TianZer)
- Anime4KCPP/FSRCNNX: MIT (TianZer)
- websr: MIT (sb2702)

The conversion scripts in this repository are licensed under the MIT License.
