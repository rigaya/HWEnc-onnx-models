# HWEnc-onnx-models

[**日本語版はこちら / Japanese**](README.ja.md)

Build tools for ONNX models used by `--vpp-onnx` and `--vpp-rife-ov` in
QSVEnc / NVEnc / VCEEnc, plus `--vpp-stdeint` in QSVEnc.

Generates 253 `--vpp-onnx` models (241 FP32 + 12 INT8) across 27 families,
10 RIFE models, and one ST-DeInt model with a single command: download →
convert or extract → INT8 quantize → manifests.

## Quick Start

### Linux

```bash
# 1. Set up venv (first time only)
bash setup_env.sh

# 2. Full build (download + convert + INT8 quantization + manifests)
.venv_onnx/bin/python run_all.py --output /path/to/output

# 3. Dry run (print plan without executing)
.venv_onnx/bin/python run_all.py --output /path/to/output --dry-run
```

### Windows

```bat
REM 1. Set up venv (first time only)
setup_env.bat

REM 2. Full build
.venv_onnx\Scripts\python run_all.py --output C:\path\to\output

REM 3. Dry run
.venv_onnx\Scripts\python run_all.py --output C:\path\to\output --dry-run
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
├── _work/              # Intermediate data (can be deleted after build)
│   ├── repos/          #   Cloned source repositories
│   ├── realesrgan/     #   Real-ESRGAN .pth weights
│   └── realcugan_weights/ # Real-CUGAN .pth weights
├── acnet/              # Converted ONNX models (one dir per family)
├── anime3d/
├── anime4k_gan/
├── anime4k_restore/
├── anime4k_upscale/
├── arnet/
├── artcnn/
├── bsrgan/
├── dncnn/
├── dpsr/
├── drunet/
├── edsr/
├── esrgan/
├── fdncnn/
├── ffdnet/
├── fsrcnnx/
├── hdrtvnetpp/
├── nnedi3/
├── ravu/
├── realcugan/
├── realesrgan/
├── rife/               # --vpp-rife-ov models; excluded from models.json
├── stdeint/            # --vpp-stdeint model; excluded from models.json
├── srmd/
├── waifu2x/
├── websr/
├── models.json              # --vpp-onnx model manifest
├── rife_ov_models.json      # --vpp-rife-ov model manifest
└── stdeint_ov_models.json   # --vpp-stdeint model manifest
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
| edsr | PyTorch .pth | 3 | - | 3 |
| esrgan | PyTorch .pth | 5 | - | 5 |
| fdncnn | PyTorch .pth | 4 | - | 4 |
| ffdnet | PyTorch .pth | 4 | - | 4 |
| fbcnn | PyTorch .pth | 4 | - | 4 |
| fsrcnnx | C++ header | 4 | - | 4 |
| hdrtvnetpp | PyTorch .pth | 2 | - | 2 |
| nnedi3 | Python weights | 10 | - | 10 |
| nafnet | PyTorch .pth | 4 | - | 4 |
| ravu | Python weights | 21 | - | 21 |
| realcugan | PyTorch .pth | 12 | 2 | 14 |
| realesrgan | PyTorch .pth | 8 | 2 | 10 |
| rife | upstream ONNX archive | 10 | - | 10 |
| stdeint | PyTorch .pt | 1 | - | 1 |
| srmd | PyTorch .pth | 6 | - | 6 |
| superimage | PyTorch .pth | 24 | - | 24 |
| waifu2x | JSON weights | 34 | - | 34 |
| websr | JSON weights | 9 | - | 9 |
| **--vpp-onnx Total** | | **241** | **12** | **253** |

## Conversion Source Types

### GLSL Shader Parsing
Extracts weights from Anime4K, ACNet, ARNet, and FSRCNNX shaders and builds ONNX graphs.

### Python Weights (RAVU, NNEDI3)
Converts trained weights from bjin/mpv-prescalers via `torch.onnx.export`.

### PyTorch .pth Weights
Converts pretrained models from KAIR, EDSR, Real-ESRGAN, Real-CUGAN, BSRGAN, etc. via `torch.onnx.export`.
HDRTVNet++ SDR-to-HDR models are converted from the official pretrained weights.
ST-DeInt is converted from the official pretrained weights into a dynamic H/W
model using an equivalent GridSample implementation and half-height, 6-channel
restoration output. Weaving is performed by QSVEnc after inference.

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
- onnxscript
- NumPy
- SciPy
- nncf
- onnxruntime

`setup_env.sh` (Linux) or `setup_env.bat` (Windows) creates the venv and installs all dependencies.

## File Structure

| File | Description |
|------|-------------|
| `run_all.py` | Integrated runner (download → convert → INT8 → manifests) |
| `setup_env.sh` | Python venv setup (Linux) |
| `setup_env.bat` | Python venv setup (Windows) |
| `quantize_int8.py` | INT8 quantization via nncf |
| `export_*.py` | Per-family FP32 ONNX conversion scripts |
| `export_stdeint.py` | Converts the official ST-DeInt weights to a half-height, 6-channel restoration-output ONNX model |
| `extract_rife.py` | Extracts RIFE ONNX from the official archive and verifies SHA-256 and I/O shapes |
| `convert_edsr.py` | EDSR conversion script |
| `convert_ravu_*.py` | RAVU conversion scripts (4 files, called by export_ravu.py) |
| `extract_anime4k_upscale_gan_glsl.py` | Anime4K GAN GLSL parsing helper |
| `requirements.txt` | Python dependencies |

## License

Each model's license follows its upstream repository.
A `LICENSE.txt` is placed in each `<family>/` directory with full provenance details.

| Family | License | Author / Copyright |
|--------|---------|-------------------|
| ACNet, ARNet | MIT | TianZerL (ACNetGLSL) |
| Anime4K (anime3d, anime4k_gan, anime4k_restore, anime4k_upscale) | MIT | bloc97 |
| ArtCNN | MIT | Joao Chrisostomo |
| BSRGAN | MIT (KAIR) / Apache-2.0 (BSRGAN repo) | Kai Zhang |
| DnCNN, DPSR, DRUNet, ESRGAN, FDnCNN, FFDNet, SRMD | MIT | Kai Zhang (KAIR) |
| EDSR | MIT | Sanghyun Son |
| FSRCNNX | **GPL-3.0** (igv's trained weights) | igv, nessotrin, TianZerL |
| HDRTVNet++ | MIT | Hoven Li |
| FBCNN | Apache-2.0 | Jiaxi Jiang, Kai Zhang, Radu Timofte |
| NAFNet | MIT | megvii-model |
| NNEDI3 | **LGPL-3.0** (trained weights) | bjin |
| RAVU | **LGPL-3.0** (trained weights) | bjin |
| Real-CUGAN | MIT | bilibili |
| Real-ESRGAN | BSD-3-Clause | Xintao Wang |
| RIFE v4.x | MIT | Megvii, hzwer |
| ST-DeInt | **Unknown** | Yuqing Liu et al. |
| super-image (PAN, CARN, A2N, AWSRN, MSRN) | Apache-2.0 | Eugene Siow et al. |
| waifu2x | MIT | nagadomi, nihui (ncnn topology) |
| websr | MIT | sb2702, bloc97 |

**Note on FSRCNNX:** The weights originate from
[igv/FSRCNN-TensorFlow](https://github.com/igv/FSRCNN-TensorFlow) (GPL-3.0),
not from the MIT-licensed Anime4KCPP code that bundles them.
See `licenses/fsrcnnx.txt` for details.

The conversion scripts in this repository are licensed under the MIT License.
