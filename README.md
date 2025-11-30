# 統合LLM推論サーバー on NVIDIA GB10 (ARM64)

NVIDIA Grace Blackwell (GB10 / ARM64) 環境向けに最適化された統合推論サーバー環境です。
単一のコンテナ内でvLLMとllama.cppの両方のエンジンをサポートし、APIパラメータで透過的に切り替えることができます。

## 特徴

### 統合アーキテクチャ
- **単一ポート (8080)**: vLLMとllama.cppを同じポートで利用可能
- **透過的な切り替え**: `engine`パラメータでエンジンを指定
- **ステータスAPI**: 各エンジンのロード状態をリアルタイムで確認
- **動的モデルロード**: APIでモデルを切り替え可能

### vLLM エンジン
- **NVIDIA Optimized**: `nvcr.io/nvidia/vllm` イメージを使用
- **High Throughput**: 並列処理に最適化
- **FP8/FP16 Support**: 高精度推論

### llama.cpp エンジン
- **Native C++**: Pythonラッパー不使用の高速実装
- **GGUF Support**: Q2〜Q8の様々な量子化レベルをサポート
- **Low Memory**: メモリ使用量を大幅に削減

### 共通機能
- **ARM64 Support**: GB10 環境での動作を確認済み
- **OpenAI Compatible API**: 統一されたAPIインターフェース
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
docker compose run --rm -e HF_TOKEN=$TOKEN unified-proxy \
  python3 /workspace/scripts/download_model.py --model <model_id>

# 例: Gemma 3n E2B をダウンロード
TOKEN=$(grep "^HF_TOKEN=" .env | cut -d= -f2) && \
docker compose run --rm -e HF_TOKEN=$TOKEN unified-proxy \
  python3 /workspace/scripts/download_model.py --model google/gemma-3n-E2B-it
```

### 3. サーバーの起動

統合プロキシサーバーを起動します。起動直後はどちらのエンジンもモデルはロードされていません（待機状態）。

```bash
docker compose up -d --build
```

サーバーが起動するまで数分かかる場合があります（特に初回ビルド時）。
ログを確認するには：

```bash
docker compose logs -f unified-proxy
```

`Uvicorn running on http://0.0.0.0:8080` と表示されれば起動完了です。

### 4. ステータスの確認

各エンジンの状態を確認できます。

```bash
curl http://localhost:8080/v1/status
```

レスポンス例:
```json
{
  "vllm": {
    "status": "idle",
    "model": null
  },
  "llamacpp": {
    "status": "idle",
    "model": null,
    "process_alive": false
  }
}
```

### 5. モデルのロード

推論を行う前に、使用したいモデルをAPI経由でロードします。

```bash
# vLLMでGemma 3n E2B をロード
curl -X POST http://localhost:8080/v1/models/load \
-H "Content-Type: application/json" \
-d '{"model_path": "/workspace/models/gemma-3n-E2B-it", "engine": "vllm"}'

# llama.cppでGGUFモデルをロード
curl -X POST http://localhost:8080/v1/models/load \
-H "Content-Type: application/json" \
-d '{"model_path": "/workspace/models/model.gguf", "engine": "llamacpp"}'
```

ロードが完了すると `{"status":"success", ...}` が返ります。

### 6. 推論の実行

モデルロード後、OpenAI互換のAPIで推論を実行できます。**`engine`パラメータは必須です。**

```bash
# vLLMで推論
curl -X POST http://localhost:8080/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "/workspace/models/gemma-3n-E2B-it",
  "prompt": "Tell me a joke about programming.",
  "max_tokens": 100,
  "temperature": 0.7,
  "engine": "vllm"
}'

# llama.cppで推論
curl -X POST http://localhost:8080/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "model.gguf",
  "prompt": "Tell me a joke about programming.",
  "max_tokens": 100,
  "temperature": 0.7,
  "engine": "llamacpp"
}'
```

### 7. モデルの切り替え

別のモデルを使用したい場合、再度 `/v1/models/load` を叩くと、現在のモデルがアンロードされ、新しいモデルがロードされます。

## ベンチマーク

### 実行方法

#### 完全自動ベンチマーク（推奨）

モデルのダウンロード→シングルストリーム→並列ベンチマークを自動実行します。

```bash
# 必要なライブラリのインストール
pip install aiohttp numpy tqdm

# vLLMでベンチマーク
python3 scripts/benchmark_full.py google/gemma-3n-E2B-it --engine vllm

# llama.cppでベンチマーク（ダウンロードは不要）
python3 scripts/benchmark_full.py model.gguf --engine llamacpp --skip-download
```

結果は `benchmarks/<model_name>/single_stream.csv` と `benchmarks/<model_name>/parallel.csv` に保存されます。

### ベンチマーク結果

#### シングルストリーム（並列度1）- 全トークン長

日本語テキスト生成における各トークン長でのスループット (TPS) です。

| Engine | Model | 64 tokens | 128 tokens | 256 tokens | 512 tokens | 1024 tokens |
|:-------|:------|----------:|-----------:|-----------:|-----------:|------------:|
| vLLM | **Gemma 3n E2B-it** | 27.75 | 28.45 | 28.38 | 28.22 | **27.95** |
| vLLM | **Gemma 3n E2B-it FP8** | 37.77 | 37.90 | 37.61 | 37.97 | **37.47** |
| vLLM | **Gemma 3n E4B-it** | 16.69 | 16.80 | 16.74 | 16.62 | **16.54** |
| vLLM | **Gemma-2-Llama-Swallow-2b-it-v0.1** | 30.87 | 30.90 | 30.74 | 30.63 | **30.25** |
| vLLM | **gpt-oss-120b** | 28.59 | 29.70 | 29.89 | 30.00 | **29.88** |
| vLLM | **gpt-oss-20b** | 28.59 | 29.70 | 29.89 | 30.00 | **29.88** |
| llama.cpp | **openai_gpt-oss-20b-Q4_K_M** | 5.00 | 83.48 | 83.30 | 83.17 | **80.17** |
| vLLM | **Qwen3-0.6B** | 79.50 | 88.39 | 88.10 | 87.41 | **85.87** |
| vLLM | **Qwen3-1.7B** | 40.02 | 40.49 | 40.65 | 40.79 | **40.22** |
| vLLM | **Qwen3-4B-Instruct** | 19.64 | 20.13 | 20.03 | 20.17 | **20.10** |
| vLLM | **Qwen3-4B-Instruct-FP8** | 36.14 | 36.83 | 36.75 | 36.98 | **36.43** |
| vLLM | **Qwen3-8B** | 12.87 | 13.06 | 13.13 | 12.97 | **12.94** |
| vLLM | **TinySwallow-1.5B-Instruct** | 12.89 | 50.25 | 50.03 | 49.98 | **27.36** |

*注: TPS (Tokens Per Second) = 生成トークン数 / 生成時間*

#### 並列処理ベンチマーク（1024トークン生成）

複数リクエストを同時処理した際の合計スループット (TPS) です。

| Engine | Model | 4並列 | 8並列 | 16並列 | 32並列 | 64並列 |
|:-------|:------|------:|------:|-------:|-------:|-------:|
| vLLM | **Gemma 3n E2B-it** | 120.74 | 229.34 | 439.76 | 717.59 | **972.95** |
| vLLM | **Gemma 3n E2B-it FP8** | 146.40 | 282.43 | 537.46 | 875.71 | **1120.40** |
| vLLM | **Gemma 3n E4B-it** | 73.76 | 34.63* | 268.90 | 456.90 | **682.97** |
| vLLM | **Gemma-2-Llama-Swallow-2b-it-v0.1** | 99.89 | 246.15 | 444.78 | 666.41 | **905.92** |
| vLLM | **gpt-oss-120b** | 82.91 | 122.76 | 185.36 | 249.82 | **343.16** |
| vLLM | **gpt-oss-20b** | 122.22 | 210.30 | 335.31 | 497.67 | **820.20** |
| llama.cpp | **openai_gpt-oss-20b-Q4_K_M** | 95.41 | 188.48 | 287.74 | 506.07 | **731.39** |
| vLLM | **Qwen3-0.6B** | 330.05 | 591.24 | 920.48 | 1270.77 | **1639.58** |
| vLLM | **Qwen3-1.7B** | 183.09 | 337.34 | 584.33 | 877.02 | **1252.75** |
| vLLM | **Qwen3-4B-Instruct** | 92.65 | 174.29 | 315.32 | 494.95 | **760.35** |
| vLLM | **Qwen3-4B-Instruct-FP8** | 139.04 | 253.17 | 438.30 | 686.90 | **982.30** |
| vLLM | **Qwen3-8B** | 53.40 | 103.03 | 190.51 | 309.38 | **522.19** |
| vLLM | **TinySwallow-1.5B-Instruct** | 104.35 | 90.95 | 142.67 | 225.40 | **557.94** |

*注: Gemma 3n E4B-itの8並列で1リクエスト失敗（タイムアウト）*

**主な知見:**
- **Qwen3-0.6B**: 64並列で1639 TPSと圧倒的なスループット。小型モデルの並列処理性能の高さを示す。
- **gpt-oss-120b**: 120Bという巨大モデルながら64並列で343 TPSを達成。パラメータ数あたりの効率が非常に高い。
- **並列効率**: 小型モデル（0.6B, 1.7B）は並列度に対してほぼ線形にスケール。大型モデルは並列度32以降で効率が低下。
- **FP8量子化**: Qwen3-4B-FP8はFP16版より約1.3倍高速で、並列時は970 TPSを達成。
- **llama.cpp + GGUF**: `openai_gpt-oss-20b-Q4_K_M`では、vLLMのFP16版と比較してシングルストリームで約2.7倍高速。並列処理ではvLLMがやや優位。

※ Qwen3 Baseモデルは指示追従ではなく補完を行うため、生成内容はプロンプトの続きとなる傾向があります。

## API仕様

### ステータス確認

```bash
curl http://localhost:8080/v1/status
```

レスポンス例:
```json
{
  "vllm": {
    "status": "loaded",
    "model": "/workspace/models/gemma-3n-E2B-it"
  },
  "llamacpp": {
    "status": "idle",
    "model": null,
    "process_alive": false
  }
}
```

ステータス値:
- `idle`: エンジンは起動しているがモデル未ロード
- `loading`: モデルロード中
- `loaded`: モデルロード完了、推論可能
- `error`: エラー発生

### モデルロード

```bash
curl -X POST http://localhost:8080/v1/models/load \
-H "Content-Type: application/json" \
-d '{
  "model_path": "/workspace/models/gemma-3n-E2B-it",
  "engine": "vllm"
}'
```

パラメータ:
- `model_path` (必須): モデルのパス
- `engine` (必須): `vllm` または `llamacpp`
- `dtype`: vLLMのみ、デフォルト `bfloat16`
- `gpu_memory_utilization`: vLLMのみ、デフォルト `0.9`
- `max_model_len`: vLLMのみ、デフォルト `8192`

### 推論

```bash
curl -X POST http://localhost:8080/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "model-name",
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.95,
  "engine": "vllm"
}'
```

パラメータ:
- `prompt` (必須): 入力プロンプト
- `engine` (必須): `vllm` または `llamacpp`
- `model`: モデル名（オプション）
- `max_tokens`: 最大生成トークン数、デフォルト `128`
- `temperature`: サンプリング温度、デフォルト `0.7`
- `top_p`: Top-pサンプリング、デフォルト `0.95`

## トラブルシューティング

### モデルロードが遅い

- 初回ロード時はモデルのダウンロードとGPUへの転送に時間がかかります
- vLLMの場合、大型モデル（120B）で5-10分程度かかることがあります
- llama.cppの場合、プロセス起動に数秒〜数十秒かかります

### GPUメモリ不足

- `gpu_memory_utilization`を0.9から0.7に下げてみてください
- FP8量子化モデルやGGUFモデルを使用してください

### llama.cppプロセスが起動しない

- ログを確認: `docker compose logs unified-proxy`
- モデルパスが正しいか確認
- GPUが正しく認識されているか確認: `nvidia-smi`

## 構成の詳細

- **Server**: FastAPI Proxy + vLLM AsyncLLMEngine + llama-server (subprocess)
- **Port**: 8080 (Host) -> 8080 (Container)
- **API Endpoints**:
  - `GET /v1/status`: エンジンステータス確認
  - `POST /v1/models/load`: モデルロード
  - `POST /v1/completions`: 推論
  - `GET /health`: ヘルスチェック

## ライセンス

MIT License
