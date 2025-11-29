# vLLM Dynamic Inference Server on NVIDIA GB10 (ARM64)

NVIDIA Grace Blackwell (GB10 / ARM64) 環境向けに最適化された vLLM 推論サーバー環境です。
FastAPI を使用した動的なモデルロード/アンロードに対応し、複数のモデルを切り替えて使用できます。

## 特徴

- **NVIDIA Optimized**: `nvcr.io/nvidia/vllm` イメージを使用
- **Dynamic Loading**: API 経由でモデルのロード/アンロードが可能
- **ARM64 Support**: GB10 環境での動作を確認済み
- **Offline Load**: ローカルモデルマウントにより認証エラーを回避

## 必要条件

- Docker & Docker Compose
- NVIDIA GPU (Driver & Container Toolkit installed)
- Hugging Face Token (Gated Modelを使用する場合)

## セットアップ

### 1. 環境変数の設定

`.env` ファイルを作成し、Hugging Face Token を設定してください。

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. モデルのダウンロード

モデルのダウンロードは、後述の「ベンチマーク」セクションの `benchmark_full.py` を使うと自動で行われます。

手動でダウンロードする場合:

```bash
# .envからトークンを読み込んで実行
TOKEN=$(grep "^HF_TOKEN=" .env | cut -d= -f2) && \
docker compose run --rm -e HF_TOKEN=$TOKEN vllm-server \
  python3 /workspace/scripts/download_model.py --model <model_id>

# 例: Gemma 3n E2B をダウンロード
TOKEN=$(grep "^HF_TOKEN=" .env | cut -d= -f2) && \
docker compose run --rm -e HF_TOKEN=$TOKEN vllm-server \
  python3 /workspace/scripts/download_model.py --model google/gemma-3n-E2B-it
```

### 3. サーバーの起動

サーバーを起動します。起動直後はモデルはロードされていません（待機状態）。

```bash
docker compose up -d --build
```

### 4. モデルのロード

推論を行う前に、使用したいモデルをAPI経由でロードします。

```bash
# Gemma 3n E2B をロード
curl -X POST http://localhost:8001/v1/models/load \
-H "Content-Type: application/json" \
-d '{"model_path": "/workspace/models/gemma-3n-E2B-it"}'
```

ロードが完了すると `{"status":"success", ...}` が返ります。

### 5. 推論の実行

モデルロード後、OpenAI互換のAPIで推論を実行できます。

```bash
curl -X POST http://localhost:8001/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/workspace/models/gemma-3n-E2B-it",
"prompt": "Tell me a joke about programming.",
"max_tokens": 100,
"temperature": 0.7
}'
```

### 6. モデルの切り替え

別のモデルを使用したい場合、再度 `/v1/models/load` を叩くと、現在のモデルがアンロードされ、新しいモデルがロードされます。

## ベンチマーク

### 実行方法

#### 方法1: 完全自動ベンチマーク（推奨）

モデルのダウンロード→シングルストリーム→並列ベンチマークを自動実行します。

```bash
# 必要なライブラリのインストール
pip install aiohttp numpy pandas tqdm

# 完全自動ベンチマーク（ダウンロード含む）
python3 scripts/benchmark_full.py <model_id>

# 例: TinySwallow-1.5B-Instructをベンチマーク
python3 scripts/benchmark_full.py SakanaAI/TinySwallow-1.5B-Instruct

# 既にダウンロード済みの場合（ダウンロードをスキップ）
python3 scripts/benchmark_full.py google/gemma-3n-E2B-it --skip-download

# HF_TOKENを明示的に指定する場合
python3 scripts/benchmark_full.py <model_id> --hf-token <your_token>
```

結果は `benchmarks/<model_name>/` に保存されます:
- `single_stream.csv`: シングルストリームベンチマーク結果（64, 128, 256, 512, 1024トークン）
- `parallel.csv`: 並列ベンチマーク結果（4, 8, 16, 32, 64並列）

#### 方法2: 個別実行

既にモデルがロードされている状態で、個別にベンチマークを実行できます。

```bash
# シングルストリームのみ
python3 scripts/benchmark_suite.py

# 並列ベンチマークのみ
python3 scripts/benchmark_parallel.py
```

### ベンチマーク結果

#### シングルストリーム（並列度1）- 全トークン長

日本語テキスト生成における各トークン長でのスループット (TPS) です。

| Model | 64 tokens | 128 tokens | 256 tokens | 512 tokens | 1024 tokens |
|:------|----------:|-----------:|-----------:|-----------:|------------:|
| **Gemma 3n E2B-it** | 27.75 | 28.45 | 28.38 | 28.22 | **27.95** |
| **Gemma 3n E2B-it FP8** | 37.77 | 37.90 | 37.61 | 37.97 | **37.47** |
| **Gemma 3n E4B-it** | 16.69 | 16.80 | 16.74 | 16.62 | **16.54** |
| **Gemma-2-Llama-Swallow-2b-it-v0.1** | 30.87 | 30.90 | 30.74 | 30.63 | **30.25** |
| **gpt-oss-120b** | 28.59 | 29.70 | 29.89 | 30.00 | **29.88** |
| **gpt-oss-20b** | 28.59 | 29.70 | 29.89 | 30.00 | **29.88** |
| **Qwen3-0.6B** | 79.50 | 88.39 | 88.10 | 87.41 | **85.87** |
| **Qwen3-1.7B** | 40.02 | 40.49 | 40.65 | 40.79 | **40.22** |
| **Qwen3-4B-Instruct** | 19.64 | 20.13 | 20.03 | 20.17 | **20.10** |
| **Qwen3-4B-Instruct-FP8** | 36.14 | 36.83 | 36.75 | 36.98 | **36.43** |
| **Qwen3-8B** | 12.87 | 13.06 | 13.13 | 12.97 | **12.94** |
| **TinySwallow-1.5B-Instruct** | 12.89 | 50.25 | 50.03 | 49.98 | **27.36** |

*注: TPS (Tokens Per Second) = 生成トークン数 / 生成時間*

#### 並列処理ベンチマーク（1024トークン生成）

複数リクエストを同時処理した際の合計スループット (TPS) です。

| Model | 4並列 | 8並列 | 16並列 | 32並列 | 64並列 |
|:------|------:|------:|-------:|-------:|-------:|
| **Gemma 3n E2B-it** | 120.74 | 229.34 | 439.76 | 717.59 | **972.95** |
| **Gemma 3n E2B-it FP8** | 146.40 | 282.43 | 537.46 | 875.71 | **1120.40** |
| **Gemma 3n E4B-it** | 73.76 | 34.63* | 268.90 | 456.90 | **682.97** |
| **Gemma-2-Llama-Swallow-2b-it-v0.1** | 99.89 | 246.15 | 444.78 | 666.41 | **905.92** |
| **gpt-oss-120b** | 82.91 | 122.76 | 185.36 | 249.82 | **343.16** |
| **gpt-oss-20b** | 122.22 | 210.30 | 335.31 | 497.67 | **820.20** |
| **Qwen3-0.6B** | 330.05 | 591.24 | 920.48 | 1270.77 | **1639.58** |
| **Qwen3-1.7B** | 183.09 | 337.34 | 584.33 | 877.02 | **1252.75** |
| **Qwen3-4B-Instruct** | 92.65 | 174.29 | 315.32 | 494.95 | **760.35** |
| **Qwen3-4B-Instruct-FP8** | 139.04 | 253.17 | 438.30 | 686.90 | **982.30** |
| **Qwen3-8B** | 53.40 | 103.03 | 190.51 | 309.38 | **522.19** |
| **TinySwallow-1.5B-Instruct** | 104.35 | 90.95 | 142.67 | 225.40 | **557.94** |

*注: Gemma 3n E4B-itの8並列で1リクエスト失敗（タイムアウト）*

**主な知見:**
- **Qwen3-0.6B**: 64並列で1639 TPSと圧倒的なスループット。小型モデルの並列処理性能の高さを示す。
- **gpt-oss-120b**: 120Bという巨大モデルながら64並列で343 TPSを達成。パラメータ数あたりの効率が非常に高い。
- **並列効率**: 小型モデル（0.6B, 1.7B）は並列度に対してほぼ線形にスケール。大型モデルは並列度32以降で効率が低下。
- **FP8量子化**: Qwen3-4B-FP8はFP16版より約1.3倍高速で、並列時は970 TPSを達成。

※ Qwen3 Baseモデルは指示追従ではなく補完を行うため、生成内容はプロンプトの続きとなる傾向があります。

## 構成の詳細

- **Server**: FastAPI + vLLM AsyncLLMEngine
- **Port**: 8001 (Host) -> 8000 (Container)
- **API Endpoints**:
  - `POST /v1/models/load`: モデルロード
  - `POST /v1/completions`: 推論
  - `GET /health`: ヘルスチェック
