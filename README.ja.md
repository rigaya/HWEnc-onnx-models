# HWEnc-onnx-models

QSVEnc / NVEnc / VCEEnc の `--vpp-onnx` と `--vpp-rife-ov` で使用する
ONNXモデルのビルドツール群。QSVEncのONNXデインターレーサーは
`--vpp-onnx-deint` と `onnx_deint_models.json`、NVEnc/VCEEncは現行の
`--vpp-stdeint` と `stdeint_ov_models.json` を使用する。

`--vpp-onnx` 向け28ファミリー・255モデル（FP32 243 + INT8 12）、RIFE 10モデル、
ST-DeInt 2モデルとMITライセンスのDDDを、1コマンドでダウンロード → 変換または展開 →
INT8量子化 → マニフェスト生成まで実行する。DDDは固定upstreamコミットとcheckpointから
`export_ddd.py`で再現可能に変換する。

## クイックスタート

### Linux

```bash
# 1. venv セットアップ（初回のみ）
bash setup_env.sh

# 2. フルビルド（ダウンロード + 変換 + INT8 量子化 + マニフェスト生成）
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
├── edvr/               # QSVEnc/NVEnc --vpp-onnx frames=5 用
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
├── rife/               # --vpp-rife-ov 用（models.jsonには未登録）
├── stdeint/            # ST-DeInt用（models.jsonには未登録）
├── ddd/                # 生成された DDD.onnx と LICENSE.txt
├── srmd/
├── waifu2x/
├── websr/
├── models.json              # --vpp-onnx 用モデルマニフェスト
├── rife_ov_models.json      # --vpp-rife-ov 用モデルマニフェスト
├── stdeint_ov_models.json   # NVEnc/VCEEnc --vpp-stdeint 用マニフェスト
└── onnx_deint_models.json   # QSVEnc --vpp-onnx-deint 用マニフェスト
```

`onnx_deint_models.json` の `architecture` は内部テンソル契約
（`stdeint` または `ddd`）であり、公開CLIの切替スイッチではない。
`model=` にはマニフェストの登録名を指定し、ONNXファイルの直接パスは受け付けない。
DDDは`onnx_models_<version>_deint_ddd.zip`専用アーカイブ（DDD、LICENSE、DDD専用マニフェスト）
として配布する。ST-DeIntはupstreamライセンスがUnknownのため、
リリースアーカイブから除外する。

### バックエンド別デインターレース用マニフェスト

| バックエンド | オプション | マニフェスト | 登録名 |
|-------------|-----------|--------------|--------|
| QSVEnc | `--vpp-onnx-deint model=<name>` | `onnx_deint_models.json` | `stdeint`, `stdeint_fast`, `DDD` |
| NVEnc / VCEEnc | `--vpp-stdeint model=<name>` | `stdeint_ov_models.json` | `stdeint`, `stdeint_fast` |

2つのマニフェストはバックエンドごとに使用するレジストリが異なるため分離している。
`run_all.py` は両方を生成するが、`architecture` メタデータを持つのはQSVEnc用だけである。

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
| nnedi3 | Python weights | 10 | - | 10 |
| nafnet | PyTorch .pth | 4 | - | 4 |
| ravu | Python weights | 21 | - | 21 |
| realcugan | PyTorch .pth | 12 | 2 | 14 |
| realesrgan | PyTorch .pth | 8 | 2 | 10 |
| rife | upstream ONNX archive | 10 | - | 10 |
| stdeint | PyTorch .pt | 2 | - | 2 |
| srmd | PyTorch .pth | 6 | - | 6 |
| superimage | PyTorch .pth | 24 | - | 24 |
| waifu2x | JSON weights | 34 | - | 34 |
| websr | JSON weights | 9 | - | 9 |
| **--vpp-onnx 合計** | | **243** | **12** | **255** |

DDDは専用デインターレースモデルのため、上記の汎用`--vpp-onnx`合計255には含めない。
動的H/WのONNX契約は`(1,9,H,W) -> (1,3,H,W)`で、欠落フィールドを出力し、推論後の
フィールドweaveはQSVEnc側で行う。

## 変換ソース種別

### GLSL シェーダ解析
Anime4K, ACNet, ARNet, FSRCNNX のシェーダからウェイトを抽出し ONNX グラフを構築。

### Python ウェイト (RAVU, NNEDI3)
bjin/mpv-prescalers の学習済み重みから `torch.onnx.export` で変換。

### PyTorch .pth ウェイト
KAIR, EDSR, Real-ESRGAN, Real-CUGAN, BSRGAN 等の学習済みモデルを `torch.onnx.export` で変換。
HDRTVNet++ の SDR→HDR モデルは公式の学習済み重みから変換。
ST-DeInt は公式の学習済み重みから、GridSampleによる等価な動的H/W・半高6ch復元出力モデルへ変換。
weaveは推論後にQSVEnc側で行う。
`stdeint_fast.onnx` は `--no-deform` でdeform alignmentとGridSampleを除去し、わずかな画質差と引き換えにfp16推論を高速化する。

DDDは固定MIT upstream checkpointから動的H/Wの`(1,9,H,W) -> (1,3,H,W)`欠落フィールド
モデルへ変換する。推論後のフィールドweaveはQSVEnc側で行い、DDD専用アーカイブとして配布する。

EDVR-MはMMagicのREDS学習済み重みをBasicSRの同一構造へ変換し、5フレームを
チャンネル方向へ並べた15ch入力の固定解像度ONNXとして出力する。QSVEnc/NVEncでは
登録モデル名だけで`frames=5`が適用される。deformable convolutionの分解が空間形状へ
依存するため、生成済みモデルは128x128および160x88入力専用。任意の4の倍数解像度は
`export_edvr.py --size WIDTHxHEIGHT`で追加生成できる。

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
- deform-conv2d-onnx-exporter

`setup_env.sh` (Linux) / `setup_env.bat` (Windows) が venv の作成と依存関係のインストールを行う。

## ファイル構成

| ファイル | 役割 |
|---------|------|
| `run_all.py` | 統合ランナー（ダウンロード → 変換 → INT8 → マニフェスト） |
| `setup_env.sh` | Python venv セットアップ (Linux) |
| `setup_env.bat` | Python venv セットアップ (Windows) |
| `quantize_int8.py` | nncf による INT8 量子化 |
| `export_*.py` | 各ファミリーの FP32 ONNX 変換スクリプト |
| `export_stdeint.py` | ST-DeInt公式重みを通常版と `--no-deform` 高速版の半高6ch復元出力ONNXへ変換 |
| `export_ddd.py` | 固定MIT DDD checkpointを動的H/Wの`(1,9,H,W) -> (1,3,H,W)`欠落フィールドONNXへ変換 |
| `export_edvr.py` | EDVR-M REDS x4を5フレーム・15ch入力の固定解像度ONNXへ変換 |
| `extract_rife.py` | 公式アーカイブからRIFE ONNXを展開し、SHA-256と入出力形状を検証 |
| `convert_edsr.py` | EDSR 変換スクリプト |
| `convert_ravu_*.py` | RAVU 変換スクリプト (4本、export_ravu.py から呼出) |
| `extract_anime4k_upscale_gan_glsl.py` | Anime4K GAN GLSL 解析ヘルパー |
| `package_archives.py` | main、fbcnn/nafnet、DDD専用デインターレースアーカイブを作成（ST-DeIntは分離） |
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
| EDVR | Apache-2.0 | BasicSR Authors / OpenMMLab |
| FSRCNNX | **GPL-3.0** (igvの学習済み重み) | igv, nessotrin, TianZerL |
| HDRTVNet++ | MIT | Hoven Li |
| FBCNN | Apache-2.0 | Jiaxi Jiang, Kai Zhang, Radu Timofte |
| NAFNet | MIT | megvii-model |
| NNEDI3 | **LGPL-3.0** (学習済み重み) | bjin |
| RAVU | **LGPL-3.0** (学習済み重み) | bjin |
| Real-CUGAN | MIT | bilibili |
| Real-ESRGAN | BSD-3-Clause | Xintao Wang |
| RIFE v4.x | MIT | Megvii, hzwer |
| ST-DeInt | **Unknown** | Yuqing Liuほか |
| DDD | MIT | Vincent van de Schaft（固定upstreamコミット） |
| super-image (PAN, CARN, A2N, AWSRN, MSRN) | Apache-2.0 | Eugene Siow ほか |
| waifu2x | MIT | nagadomi, nihui (ncnnトポロジー) |
| websr | MIT | sb2702, bloc97 |

**FSRCNNXについて:** 重みの出自は
[igv/FSRCNN-TensorFlow](https://github.com/igv/FSRCNN-TensorFlow) (GPL-3.0) であり、
バンドル元の Anime4KCPP (MIT) のライセンスとは異なる。
詳細は `licenses/fsrcnnx.txt` を参照。

本リポジトリの変換スクリプト自体は MIT License。

## 生成結果数

デフォルトのパイプラインでは、汎用 `--vpp-onnx` モデル255個（FP32 243 + INT8 12）、
RIFE 10個、ST-DeInt 2個、DDDと各マニフェストを生成する。リリース時はDDD専用
アーカイブを追加し、ST-DeIntは配布アーカイブから除外する。
