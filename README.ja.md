# HWEnc-onnx-models

QSVEnc / NVEnc / VCEEnc の `--vpp-onnx` フィルタで使用する ONNX モデルのビルドツール群。

22 ファミリー・209 モデル（FP32 197 + INT8 12）を、1コマンドでダウンロード → 変換 → INT8 量子化 → models.json 生成まで実行する。

## クイックスタート

### Linux

```bash
# 1. venv セットアップ（初回のみ）
bash setup_env.sh

# 2. フルビルド（ダウンロード + 変換 + INT8 量子化 + models.json 生成）
.venv_onnx/bin/python run_all.py --output /path/to/output

# 3. ドライラン（実際にはダウンロード・変換しない）
.venv_onnx/bin/python run_all.py --output /path/to/output --dry-run
```

### Windows

```bat
REM 1. venv セットアップ（初回のみ）
setup_env.bat

REM 2. フルビルド
.venv_onnx\Scripts\python run_all.py --output C:\path\to\output

REM 3. ドライラン
.venv_onnx\Scripts\python run_all.py --output C:\path\to\output --dry-run
```

## run_all.py オプション

| オプション | 説明 |
|-----------|------|
| `--output PATH` | 出力ルートディレクトリ（必須） |
| `--skip-download` | venv セットアップ + ダウンロードをスキップ |
| `--skip-convert` | FP32 ONNX 変換をスキップ |
| `--skip-int8` | INT8 量子化をスキップ |
| `--jobs N` | 変換の並列実行数（デフォルト: 1） |
| `--dry-run` | 実行せず計画のみ表示 |

## 出力ディレクトリ構造

```
output/
├── _work/              # 中間データ（ビルド後は削除可能）
│   ├── repos/          #   クローンしたソースリポジトリ
│   ├── realesrgan/     #   Real-ESRGAN .pth ウェイト
│   └── realcugan_weights/ # Real-CUGAN .pth ウェイト
├── acnet/              # 変換済み ONNX モデル（ファミリーごと）
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
├── ravu/
├── realcugan/
├── realesrgan/
├── srmd/
├── waifu2x/
├── websr/
└── models.json         # モデルマニフェスト
```

## モデル一覧

| ファミリー | ソース種別 | FP32 | INT8 | 合計 |
|-----------|-----------|------|------|------|
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
| fsrcnnx | C++ header | 4 | - | 4 |
| ravu | Python weights | 21 | - | 21 |
| realcugan | PyTorch .pth | 12 | 2 | 14 |
| realesrgan | PyTorch .pth | 8 | 2 | 10 |
| srmd | PyTorch .pth | 6 | - | 6 |
| waifu2x | JSON weights | 34 | - | 34 |
| websr | JSON weights | 9 | - | 9 |
| **合計** | | **197** | **12** | **209** |

## 変換ソース種別

### GLSL シェーダ解析
Anime4K, ACNet, ARNet, FSRCNNX のシェーダからウェイトを抽出し ONNX グラフを構築。

### Python ウェイト (RAVU)
bjin/mpv-prescalers の学習済み重み（Python形式）から `torch.onnx.export` で変換。

### PyTorch .pth ウェイト
KAIR, EDSR, Real-ESRGAN, Real-CUGAN, BSRGAN 等の学習済みモデルを `torch.onnx.export` で変換。

### JSON ウェイト
waifu2x, websr の JSON 形式ウェイトから ONNX グラフを構築。

### Upstream ONNX
ArtCNN は作者が公開している ONNX ファイルをそのまま使用。

### INT8 量子化
nncf (Neural Network Compression Framework) の Post-Training Quantization で FP32 ONNX から INT8 ONNX を生成。

## 依存関係

- Python 3.10+
- PyTorch (CPU)
- ONNX
- onnxscript
- NumPy
- SciPy
- nncf
- onnxruntime

`setup_env.sh` (Linux) / `setup_env.bat` (Windows) が venv の作成と依存関係のインストールを行う。

## ファイル構成

| ファイル | 役割 |
|---------|------|
| `run_all.py` | 統合ランナー（ダウンロード → 変換 → INT8 → models.json） |
| `setup_env.sh` | Python venv セットアップ (Linux) |
| `setup_env.bat` | Python venv セットアップ (Windows) |
| `quantize_int8.py` | nncf による INT8 量子化 |
| `export_*.py` | 各ファミリーの FP32 ONNX 変換スクリプト (19本) |
| `convert_edsr.py` | EDSR 変換スクリプト |
| `convert_ravu_*.py` | RAVU 変換スクリプト (4本、export_ravu.py から呼出) |
| `extract_anime4k_upscale_gan_glsl.py` | Anime4K GAN GLSL 解析ヘルパー |
| `requirements.txt` | Python 依存パッケージ |

## ライセンス

各モデルのライセンスは元リポジトリに準じる。
各 `<ファミリー>/` ディレクトリに `LICENSE.txt` を配置し、由来を詳述している。

| ファミリー | ライセンス | 著作者 |
|-----------|-----------|--------|
| ACNet, ARNet | MIT | TianZerL (ACNetGLSL) |
| Anime4K (anime3d, anime4k_gan, anime4k_restore, anime4k_upscale) | MIT | bloc97 |
| ArtCNN | MIT | Joao Chrisostomo |
| BSRGAN | MIT (KAIR) / Apache-2.0 (BSRGANリポジトリ) | Kai Zhang |
| DnCNN, DPSR, DRUNet, ESRGAN, FDnCNN, FFDNet, SRMD | MIT | Kai Zhang (KAIR) |
| EDSR | MIT | Sanghyun Son |
| FSRCNNX | **GPL-3.0** (igvの学習済み重み) | igv, nessotrin, TianZerL |
| RAVU | **LGPL-3.0** (学習済み重み) | bjin |
| Real-CUGAN | MIT | bilibili |
| Real-ESRGAN | BSD-3-Clause | Xintao Wang |
| waifu2x | MIT | nagadomi, nihui (ncnnトポロジー) |
| websr | MIT | sb2702, bloc97 |

**FSRCNNXについて:** 重みの出自は
[igv/FSRCNN-TensorFlow](https://github.com/igv/FSRCNN-TensorFlow) (GPL-3.0) であり、
バンドル元の Anime4KCPP (MIT) のライセンスとは異なる。
詳細は `licenses/fsrcnnx.txt` を参照。

本リポジトリの変換スクリプト自体は MIT License。
