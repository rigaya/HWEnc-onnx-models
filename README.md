# HWEnc-onnx-models

[**日本語版はこちら / Japanese**](README.ja.md)

Build tools for ONNX models used by `--vpp-onnx` and `--vpp-rife-ov` in
QSVEnc / NVEnc / VCEEnc. QSVEnc's ONNX deinterlacer uses
`--vpp-onnx-deint` with `onnx_deint_models.json`.

Generates 257 `--vpp-onnx` models (245 FP32 + 12 INT8) across 30 families,
10 RIFE models and the MIT-licensed DDD model with a
single command: download → convert or extract → INT8 quantize → manifests.
DDD is downloaded from a fixed upstream commit and converted from its pinned
checkpoint with `export_ddd.py`.

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
├── edvr/               # QSVEnc/NVEnc --vpp-onnx frames=5
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
├── ddd/                # generated DDD.onnx and LICENSE.txt
├── srmd/
├── waifu2x/
├── websr/
├── models.json              # --vpp-onnx model manifest
├── rife_ov_models.json      # --vpp-rife-ov model manifest
└── onnx_deint_models.json   # QSVEnc --vpp-onnx-deint manifest
```

`architecture` in `onnx_deint_models.json` is an internal tensor contract
(`ddd` at present), not a public command-line switch. `model=` is the
manifest registration name; direct ONNX paths are not accepted. DDD is
distributed in the dedicated `onnx_models_<version>_deint_ddd.zip`
archive with its MIT license and a DDD-only `onnx_deint_models.json`.
ST-DeInt download, conversion, and manifest generation are all skipped.

### Deinterlacer manifests by backend

| Backend | Option | Manifest | Registered names |
|---------|--------|----------|------------------|
| QSVEnc | `--vpp-onnx-deint model=<name>` | `onnx_deint_models.json` | `DDD` |

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
| ddd | PyTorch .pt | 1 | - | 1 |
| drunet | PyTorch .pth | 4 | 1 | 5 |
| edvr | PyTorch .pth | 2 | - | 2 |
| edsr | PyTorch .pth | 3 | - | 3 |
| esrgan | PyTorch .pth | 5 | - | 5 |
| fdncnn | PyTorch .pth | 4 | - | 4 |
| ffdnet | PyTorch .pth | 4 | - | 4 |
| fbcnn | PyTorch .pth | 4 | - | 4 |
| fsrcnnx | C++ header | 4 | - | 4 |
| hdrtvnetpp | PyTorch .pth | 2 | - | 2 |
| liveaction | ONNX（作者配布） | 1 | - | 1 |
| vhs2hd | ONNX（作者配布） | 1 | - | 1 |
| nnedi3 | Python weights | 10 | - | 10 |
| nafnet | PyTorch .pth | 4 | - | 4 |
| ravu | Python weights | 21 | - | 21 |
| realcugan | PyTorch .pth | 12 | 2 | 14 |
| realesrgan | PyTorch .pth | 8 | 2 | 10 |
| rife | upstream ONNX archive | 10 | - | 10 |
| srmd | PyTorch .pth | 6 | - | 6 |
| superimage | PyTorch .pth | 24 | - | 24 |
| waifu2x | JSON weights | 34 | - | 34 |
| websr | JSON weights | 9 | - | 9 |
| **--vpp-onnx Total** | | **245** | **12** | **257** |

DDD is a dedicated deinterlacer model and is not included in the generic
`--vpp-onnx` total above. Its dynamic-H/W ONNX contract is
`(1,9,H,W) -> (1,3,H,W)`; it returns the missing field and QSVEnc performs
field weaving after inference.

## Conversion Source Types

### GLSL Shader Parsing
Extracts weights from Anime4K, ACNet, ARNet, and FSRCNNX shaders and builds ONNX graphs.

### Python Weights (RAVU, NNEDI3)
Converts trained weights from bjin/mpv-prescalers via `torch.onnx.export`.

### PyTorch .pth Weights
Converts pretrained models from KAIR, EDSR, Real-ESRGAN, Real-CUGAN, BSRGAN, etc. via `torch.onnx.export`.
HDRTVNet++ SDR-to-HDR models are converted from the official pretrained weights.

DDD is converted from the fixed MIT upstream checkpoint into a dynamic-H/W
`(1,9,H,W) -> (1,3,H,W)` missing-field model. QSVEnc performs field weaving
after inference; the model is distributed in the dedicated DDD archive.

EDVR-M is converted from the MMagic REDS checkpoint to the identical BasicSR
architecture and exported as fixed-resolution, 15-channel models containing five
RGB frames. QSVEnc/NVEnc apply `frames=5` from the registry automatically. The
deformable-convolution decomposition depends on the spatial shape, so the release
contains 128x128 and 160x88 inputs. Additional multiples-of-four can be generated
with `export_edvr.py --size WIDTHxHEIGHT`.

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
- deform-conv2d-onnx-exporter

`setup_env.sh` (Linux) or `setup_env.bat` (Windows) creates the venv and installs all dependencies.

## File Structure

| File | Description |
|------|-------------|
| `run_all.py` | Integrated runner (download → convert → INT8 → manifests) |
| `setup_env.sh` | Python venv setup (Linux) |
| `setup_env.bat` | Python venv setup (Windows) |
| `quantize_int8.py` | INT8 quantization via nncf |
| `export_*.py` | Per-family FP32 ONNX conversion scripts |
| `export_ddd.py` | Converts the fixed MIT DDD checkpoint to dynamic-H/W `(1,9,H,W) -> (1,3,H,W)` missing-field ONNX |
| `export_edvr.py` | Converts EDVR-M REDS x4 to fixed-resolution, five-frame 15-channel ONNX models |
| `extract_rife.py` | Extracts RIFE ONNX from the official archive and verifies SHA-256 and I/O shapes |
| `convert_edsr.py` | EDSR conversion script |
| `convert_ravu_*.py` | RAVU conversion scripts (4 files, called by export_ravu.py) |
| `extract_anime4k_upscale_gan_glsl.py` | Anime4K GAN GLSL parsing helper |
| `package_archives.py` | Creates main, fbcnn/nafnet, and dedicated DDD deinterlacer archives |
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
| EDVR | Apache-2.0 | BasicSR Authors / OpenMMLab |
| FSRCNNX | **GPL-3.0** (igv's trained weights) | igv, nessotrin, TianZerL |
| HDRTVNet++ | MIT | Hoven Li |
| LiveActionV1 SPAN | Apache-2.0（NOTICE同梱） | jcj83429 |
| VHS2HD RealPLKSR | CC-BY-SA-4.0 | starinspace |
| FBCNN | Apache-2.0 | Jiaxi Jiang, Kai Zhang, Radu Timofte |
| NAFNet | MIT | megvii-model |
| NNEDI3 | **LGPL-3.0** (trained weights) | bjin |
| RAVU | **LGPL-3.0** (trained weights) | bjin |
| Real-CUGAN | MIT | bilibili |
| Real-ESRGAN | BSD-3-Clause | Xintao Wang |
| RIFE v4.x | MIT | Megvii, hzwer |
| DDD | MIT | Vincent van de Schaft (fixed upstream commit) |
| super-image (PAN, CARN, A2N, AWSRN, MSRN) | Apache-2.0 | Eugene Siow et al. |
| waifu2x | MIT | nagadomi, nihui (ncnn topology) |
| websr | MIT | sb2702, bloc97 |

**Note on FSRCNNX:** The weights originate from
[igv/FSRCNN-TensorFlow](https://github.com/igv/FSRCNN-TensorFlow) (GPL-3.0),
not from the MIT-licensed Anime4KCPP code that bundles them.
See `licenses/fsrcnnx.txt` for details.

The conversion scripts in this repository are licensed under the MIT License.

## Generated result counts

With the default pipeline, the output contains 257 generic `--vpp-onnx`
models (245 FP32 + 12 INT8), 10 RIFE models, and DDD, plus manifests for the
generic, RIFE, and QSVEnc deinterlacer registries. ST-DeInt is neither
downloaded nor generated.

## 実写修復モデル

LiveActionV1 SPANは作者配布のONNXをSHA-256とonnx.checkerで検証し、改変せずに収録する。
`--vpp-onnx model=2xliveactionv1_span_490000`で選択できるRGB入力の2倍拡大モデル。
Apache-2.0のライセンス本文とNOTICEをliveaction/LICENSE.txtに同梱する。

VHS2HD RealPLKSRも作者配布のONNXを同様に検証し、改変せずに収録する。
`--vpp-onnx model=2xvhs2hd-realplksr`で選択できるRGB入力の2倍拡大モデル。
CC-BY-SA-4.0の由来と条件はvhs2hd/LICENSE.txtに同梱する。
