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
- **Web UI**: モダンなチャットインターフェース（統合ポート8080）

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

### 4. Web UIへのアクセス

ブラウザで `http://localhost:8080` にアクセスすると、チャットUIが表示されます。

**Web UIの機能:**
- エンジン選択（vLLM / llama.cpp）
- モデル選択
- パラメータ調整（max_tokens, temperature）
- リアルタイムステータス表示
- 推論速度（TPS）表示

### 5. ステータスの確認

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

### 8. モデルのロード

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

### 9. 推論の実行

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

### 10. モデルの切り替え

別のモデルを使用したい場合、再度 `/v1/models/load` を叩くと、現在のモデルがアンロードされ、新しいモデルがロードされます。

## ベンチマーク

### vLLM v25.12 + V1エンジン ベンチマーク結果

> **注意**: 一部のモデルはロードに2-3分かかります。タイムアウト設定を十分に長くしてください。

#### v25.12 シングルストリーム（1024トークン生成）

| Engine | Model | TPS |
|:-------|:------|----:|
| llama.cpp | **LFM2.5-1.2B-JP-Q8_0-GGUF** | 146.1 |
| llama.cpp | **TinySwallow-Q8_0-GGUF** | 124.6 |
| Gemini | **gemini-2.5-flash-lite** | 122.7 |
| vLLM | **Qwen3-0.6B** | 100.8 |
| Transformers | **ELYZA-Diffusion** | 70.0 |
| vLLM | **LFM2.5-1.2B-JP** | 71.6 |
| vLLM | **LFM2.5-1.2B-Instruct** | 70.6 |
| vLLM | **TinySwallow-1.5B-Instruct** | 53.9 |
| vLLM | **Qwen3-1.7B** | 43.7 |
| vLLM | **gpt-oss-20b** | 42.6 |
| vLLM | **gemma-3n-E2B-FP8-dynamic** | 40.8 |
| vLLM | **Qwen3-4B-Instruct-2507-FP8** | 38.9 |
| vLLM | **gpt-oss-120b** | 30.9 |
| vLLM | **Gemma-2-Llama-Swallow-2b** | 30.3 |
| vLLM | **gemma-3n-E2B-it** | 29.1 |
| Gemini | **gemini-3-pro-preview** | 26.4 |
| vLLM | **shisa-v2.1-llama3.2-3b** | 26.1 |
| vLLM | **Qwen3-4B-Instruct-2507** | 21.0 |
| vLLM | **Ministral-3-8B-Instruct** | 19.6 |
| vLLM | **gemma-3n-E4B-it** | 17.5 |
| Gemini | **gemini-3-flash-preview** | 13.4 |
| vLLM | **Qwen3-8B** | 13.3 |
| vLLM | **shisa-v2.1-qwen3-8b** | 12.6 |
| vLLM | **shisa-v2.1-unphi4-14b** | 7.8 |
| Gemini | **gemini-2.5-flash** | 4.7 |

#### v25.12 並列処理ベンチマーク（合計TPS）

| Engine | Model | 4並列 | 8並列 | 16並列 | 32並列 | 64並列 |
|:-------|:------|------:|------:|-------:|-------:|-------:|
| vLLM | **LFM2.5-1.2B-JP** | 320 | 632 | 1155 | 1965 | **3018** |
| vLLM | **LFM2.5-1.2B-Instruct** | 241 | 504 | 915 | 1653 | **2568** |
| vLLM | **Qwen3-0.6B** | 330 | 591 | 920 | 1271 | **1640** |
| vLLM | **Qwen3-1.7B** | 183 | 337 | 584 | 877 | **1253** |
| vLLM | **gemma-3n-E2B-FP8-dynamic** | 146 | 282 | 537 | 876 | **1120** |
| vLLM | **Qwen3-4B-FP8** | 139 | 253 | 438 | 687 | **982** |
| vLLM | **gemma-3n-E2B-it** | 121 | 229 | 440 | 718 | **973** |
| vLLM | **Gemma-2-Llama-Swallow-2b** | 100 | 246 | 445 | 666 | **906** |
| vLLM | **gpt-oss-20b** | 122 | 210 | 335 | 498 | **820** |
| vLLM | **Qwen3-4B** | 93 | 174 | 315 | 495 | **760** |
| vLLM | **gemma-3n-E4B-it** | 75 | 138 | 265 | 456 | **678** |
| vLLM | **shisa-v2.1-llama3.2-3b** | 106 | 191 | 334 | 506 | **656** |
| vLLM | **Ministral-3-8B-Instruct** | 78 | 146 | 259 | 405 | **581** |
| vLLM | **TinySwallow-1.5B** | 104 | 91 | 143 | 225 | **558** |
| vLLM | **Qwen3-8B** | 53 | 103 | 191 | 309 | **522** |
| vLLM | **gpt-oss-120b** | 83 | 123 | 185 | 250 | **343** |
| vLLM | **shisa-v2.1-qwen3-8b** | 51 | 93 | 163 | 220 | - |
| vLLM | **shisa-v2.1-unphi4-14b** | 29 | 56 | - | - | - |

#### v25.12 1セッションあたりのTPS

| Engine | Model | 4並列 | 8並列 | 16並列 | 32並列 | 64並列 |
|:-------|:------|------:|------:|-------:|-------:|-------:|
| vLLM | **LFM2.5-1.2B-JP** | 80.0 | 79.0 | 72.2 | 61.4 | **47.2** |
| vLLM | **LFM2.5-1.2B-Instruct** | 60.1 | 63.0 | 57.2 | 51.6 | **40.1** |
| vLLM | **Qwen3-0.6B** | 82.5 | 73.9 | 57.5 | 39.7 | **25.6** |
| vLLM | **Qwen3-1.7B** | 45.8 | 42.2 | 36.5 | 27.4 | **19.6** |
| vLLM | **gemma-3n-E2B-FP8-dynamic** | 36.6 | 35.3 | 33.6 | 27.4 | **17.5** |
| vLLM | **Qwen3-4B-FP8** | 34.8 | 31.7 | 27.4 | 21.5 | **15.4** |
| vLLM | **gemma-3n-E2B-it** | 30.2 | 28.7 | 27.5 | 22.4 | **15.2** |
| vLLM | **Gemma-2-Llama-Swallow-2b** | 25.0 | 30.8 | 27.8 | 20.8 | **14.2** |
| vLLM | **gpt-oss-20b** | 30.6 | 26.3 | 21.0 | 15.6 | **12.8** |
| vLLM | **Qwen3-4B** | 23.2 | 21.8 | 19.7 | 15.5 | **11.9** |
| vLLM | **gemma-3n-E4B-it** | 18.7 | 17.3 | 16.6 | 14.3 | **10.6** |
| vLLM | **shisa-v2.1-llama3.2-3b** | 26.4 | 23.9 | 20.9 | 15.8 | **10.3** |
| vLLM | **Ministral-3-8B-Instruct** | 19.6 | 18.3 | 16.2 | 12.6 | **9.1** |
| vLLM | **TinySwallow-1.5B** | 26.1 | 11.4 | 8.9 | 7.0 | **8.7** |
| vLLM | **Qwen3-8B** | 13.4 | 12.9 | 11.9 | 9.7 | **8.2** |
| vLLM | **gpt-oss-120b** | 20.7 | 15.4 | 11.6 | 7.8 | **5.4** |
| vLLM | **shisa-v2.1-qwen3-8b** | 12.7 | 11.6 | 10.2 | 6.9 | - |
| vLLM | **shisa-v2.1-unphi4-14b** | 7.3 | 7.1 | - | - | - |

#### 指示追従性テスト結果（v25.12）

**全29モデルの総合ランキング（指示追従率順）**

| Rank | Model | Type | Engine | Params | 合格率 | TPS | Grade |
|-----:|:------|:-----|:-------|:-------|-------:|----:|:------|
| 1 | **Qwen3-4B-Instruct-2507-FP8** | local | vLLM | 4B | **100.0%** ⭐ | 38.9 | S |
| 2 | **Qwen3-4B-Instruct-2507** | local | vLLM | 4B | **100.0%** ⭐ | 21.0 | S |
| 3 | **gemini-3-pro-preview** | API | Gemini | - | 93.1% | 26.4 | A+ |
| 4 | **LFM2.5-1.2B-JP-Q8_0-GGUF** | local | llama.cpp | 1.2B | 89.7% | 146.1 | A |
| 5 | **gemma-3n-E4B-it** | local | vLLM | 4B | 86.2% | 17.5 | A |
| 6 | **TinySwallow-1.5B-Instruct** | local | vLLM | 1.5B | 86.2% | 53.9 | A |
| 7 | **gemini-2.5-flash-lite** | API | Gemini | - | 82.8% | 122.7 | A |
| 8 | **gemma-3n-E2B-it** | local | vLLM | 2B | 82.8% | 29.1 | A |
| 9 | **Gemma-2-Llama-Swallow-2b** | local | vLLM | 2B | 82.8% | 30.3 | A |
| 10 | **shisa-v2.1-llama3.2-3b** | local | vLLM | 3B | 82.8% | 26.1 | A |
| 11 | **shisa-v2.1-qwen2.5-3b** | local | vLLM | 3B | 79.3% | 25.4 | B+ |
| 12 | **Qwen2.5-3B-Instruct** | local | vLLM | 3B | 79.3% | 25.0 | B+ |
| 13 | **LFM2.5-1.2B-JP** | local | vLLM | 1.2B | 75.9% | 71.6 | B+ |
| 14 | **LFM2.5-1.2B-Instruct** | local | vLLM | 1.2B | 75.9% | 70.6 | B+ |
| 15 | **TinySwallow-Q8_0-GGUF** | local | llama.cpp | 1.5B | 75.9% | 124.6 | B+ |
| 16 | **gemini-3-flash-preview** | API | Gemini | - | 75.9% | 13.4 | B+ |
| 17 | **shisa-v2.1-qwen3-8b** | local | vLLM | 8B | 75.9% | 12.6 | B+ |
| 18 | **shisa-v2.1-lfm2-1.2b** | local | vLLM | 1.2B | 72.4% | - | B |
| 19 | **Qwen3-8B** | local | vLLM | 8B | 69.0% | 13.3 | B |
| 20 | **gemini-2.5-flash** | API | Gemini | - | 65.5% | 4.7 | B- |
| 21 | **Ministral-3-8B-Instruct** | local | vLLM | 8B | 65.5% | 19.6 | B- |
| 22 | **Qwen3-1.7B** | local | vLLM | 1.7B | 58.6% | 43.7 | C+ |
| 23 | **gpt-oss-20b-Q8_0-GGUF** | local | llama.cpp | 20B | 51.7% | - | C |
| 24 | **gpt-oss-20b** | local | vLLM | 20B | 41.4% | 42.6 | C |
| 25 | **Qwen3-0.6B** | local | vLLM | 0.6B | 41.4% | 100.8 | C |
| 26 | **ELYZA-Diffusion** | local | Transformers | 7B | 34.5% | 70.0 | C- |
| 27 | **gpt-oss-120b** | local | vLLM | 120B | 27.6% | 30.9 | D |
| 28 | **gemma-3n-E2B-FP8-dynamic** | local | vLLM | 2B | 17.2% | 40.8 | D |
| 29 | **shisa-v2.1-unphi4-14b** | local | vLLM | 14B | 17.2% | 7.8 | D |

**推奨モデル（指示追従性重視）:**
1. **Qwen3-4B-Instruct-2507-FP8** - 全29テスト合格、38.9 TPS（ローカル最強）
2. **gemini-3-pro-preview** - 93.1%、思考モードで高品質（API最高品質）
3. **LFM2.5-1.2B-JP-Q8_0-GGUF** - 89.7%、146.1 TPS（llama.cpp最強、超高速）
4. **TinySwallow-1.5B-Instruct** - 86.2%、1.5Bで53.9 TPS（超軽量最強）
5. **gemini-2.5-flash-lite** - 82.8%、122.7 TPS（API最速）
6. **shisa-v2.1-llama3.2-3b** - 82.8%、日本語特化で高品質

**用途別推奨モデル:**

| 用途 | 推奨モデル | TPS | 合格率 | 理由 |
|:-----|:----------|----:|-------:|:-----|
| 汎用最強 | Qwen3-4B-Instruct-2507-FP8 | 38.9 | 100.0% | 全指示100%対応。FP8で高速 |
| 高速バッチ | gemini-2.5-flash-lite | 122.7 | 82.8% | 122.7TPSで大量処理向け |
| 超軽量デバイス | TinySwallow-1.5B-Instruct | 53.9 | 86.2% | 1.5Bで86%・54TPS。最高効率 |
| 日本語特化 | shisa-v2.1-llama3.2-3b | 26.1 | 82.8% | 人格維持と日本語品質優秀 |
| ロールプレイ | Gemma-2-Llama-Swallow-2b | 30.3 | 82.8% | ロールプレイ100% |
| 安全性重視 | gemini-3-flash-preview | 13.4 | 75.9% | 安全性・人格100% |
| 高品質API | gemini-3-pro-preview | 26.4 | 93.1% | 思考モードで高品質 |
| 超高速処理 | Qwen3-0.6B | 100.8 | 41.4% | 品質より速度重視時 |

**テスト内容（全29テスト）:**
- **フォーマット（3）**: JSON出力、箇条書き、番号付きリスト
- **文字数制限（2）**: 20文字以内、約50文字
- **単語制約（2）**: 特定単語の包含/除外
- **言語制約（1）**: 日本語のみ（英語禁止）
- **構造（1）**: 序論・本論・結論の3部構成
- **視点（1）**: 子供向け説明
- **数量（1）**: 正確に5つ列挙
- **複合（1）**: 複数条件の同時達成
- **安全性（6）**: 有害コンテンツ拒否、違法行為拒否、個人情報保護、差別拒否、AI人格維持、脱獄試行拒否
- **人格（3）**: 丁寧な対応、誤情報訂正、不確実性の認識
- **ロールプレイ（8）**: 医師、シェフ、関西弁、侍、子供、ツンデレ、ナレーター、キャラ維持

---

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
| vLLM | **Gemma 3n E4B-it** | 16.25 | 16.96 | 16.95 | 16.87 | **16.80** |
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
| llama.cpp | **TinySwallow-1.5B-Instruct-Q8_0-GGUF** | 5.12 | 44.82 | 40.08 | 125.98 | **124.60** |
| vLLM | **shisa-v2.1-llama3.2-3b** | 23.88 | 24.53 | 24.33 | 24.71 | **24.63** |
| vLLM | **shisa-v2.1-qwen3-8b** | 11.49 | 12.74 | 12.69 | 12.60 | **12.58** |
| vLLM | **shisa-v2.1-unphi4-14b** | 5.28 | 7.74 | 7.76 | 7.83 | **7.82** |
| vLLM | **Ministral-3-8B-Instruct-2512** | 20.94 | 20.20 | 20.42 | 20.60 | **20.54** |
| Transformers | **ELYZA-Diffusion-Instruct-7B (steps=32)** | 30.04 | 49.78 | 64.57 | 69.18 | **70.26** |

*注: TPS (Tokens Per Second) = 生成トークン数 / 生成時間*

#### 並列処理ベンチマーク（1024トークン生成）

複数リクエストを同時処理した際の**合計スループット (TPS)** です。

| Engine | Model | 4並列 | 8並列 | 16並列 | 32並列 | 64並列 |
|:-------|:------|------:|------:|-------:|-------:|-------:|
| vLLM | **Gemma 3n E2B-it** | 120.74 | 229.34 | 439.76 | 717.59 | **972.95** |
| vLLM | **Gemma 3n E2B-it FP8** | 146.40 | 282.43 | 537.46 | 875.71 | **1120.40** |
| vLLM | **Gemma 3n E4B-it** | 74.61 | 137.99 | 265.37 | 456.19 | **677.85** |
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
| llama.cpp | **TinySwallow-1.5B-Instruct-Q8_0-GGUF** | 135.21 | 155.12 | 133.45 | 170.41 | **220.55** |
| llama.cpp | **LFM2.5-1.2B-JP-Q8_0-GGUF** | 450.13 | 449.55 | 444.72 | 446.34 | **446.04** |
| vLLM | **shisa-v2.1-llama3.2-3b** | 105.77 | 191.28 | 334.15 | 505.73 | **655.79** |
| vLLM | **shisa-v2.1-qwen3-8b** | 50.86 | 92.64 | 163.17 | 219.50 | - |
| vLLM | **shisa-v2.1-unphi4-14b** | 29.37 | 56.36 | - | - | - |
| vLLM | **Ministral-3-8B-Instruct-2512** | 78.23 | 146.26 | 259.49 | 404.59 | **581.10** |

#### 1セッションあたりのTPS（並列処理時）

各ユーザーが体感する平均TPS（= 合計TPS ÷ 並列数）です。

| Engine | Model | 4並列 | 8並列 | 16並列 | 32並列 | 64並列 |
|:-------|:------|------:|------:|-------:|-------:|-------:|
| vLLM | **Gemma 3n E2B-it** | 30.19 | 28.67 | 27.49 | 22.42 | **15.20** |
| vLLM | **Gemma 3n E2B-it FP8** | 36.60 | 35.30 | 33.59 | 27.37 | **17.51** |
| vLLM | **Gemma 3n E4B-it** | 18.65 | 17.25 | 16.59 | 14.26 | **10.59** |
| vLLM | **Gemma-2-Llama-Swallow-2b-it-v0.1** | 24.97 | 30.77 | 27.80 | 20.83 | **14.16** |
| vLLM | **gpt-oss-120b** | 20.73 | 15.35 | 11.59 | 7.81 | **5.36** |
| vLLM | **gpt-oss-20b** | 30.56 | 26.29 | 20.96 | 15.55 | **12.82** |
| llama.cpp | **openai_gpt-oss-20b-Q4_K_M** | 23.85 | 23.56 | 17.98 | 15.81 | **11.43** |
| vLLM | **Qwen3-0.6B** | 82.51 | 73.91 | 57.53 | 39.71 | **25.62** |
| vLLM | **Qwen3-1.7B** | 45.77 | 42.17 | 36.52 | 27.41 | **19.57** |
| vLLM | **Qwen3-4B-Instruct** | 23.16 | 21.79 | 19.71 | 15.47 | **11.88** |
| vLLM | **Qwen3-4B-Instruct-FP8** | 34.76 | 31.65 | 27.39 | 21.47 | **15.35** |
| vLLM | **Qwen3-8B** | 13.35 | 12.88 | 11.91 | 9.67 | **8.16** |
| vLLM | **TinySwallow-1.5B-Instruct** | 26.09 | 11.37 | 8.92 | 7.04 | **8.72** |
| llama.cpp | **TinySwallow-1.5B-Instruct-Q8_0-GGUF** | 33.80 | 19.39 | 8.34 | 5.33 | **3.45** |
| llama.cpp | **LFM2.5-1.2B-JP-Q8_0-GGUF** | 112.53 | 56.19 | 27.80 | 13.95 | **6.97** |
| vLLM | **shisa-v2.1-llama3.2-3b** | 26.44 | 23.91 | 20.88 | 15.80 | **10.25** |
| vLLM | **shisa-v2.1-qwen3-8b** | 12.72 | 11.58 | 10.20 | 6.86 | - |
| vLLM | **shisa-v2.1-unphi4-14b** | 7.34 | 7.05 | - | - | - |
| vLLM | **Ministral-3-8B-Instruct-2512** | 19.56 | 18.28 | 16.22 | 12.64 | **9.08** |

#### 並列処理時のユーザー体験に関する考察

**1. 並列効率（Parallel Efficiency）**

並列効率 = (N並列時の1セッションTPS) / (シングルストリームTPS) × 100%

| Model | シングル | 4並列効率 | 64並列効率 | 評価 |
|:------|--------:|---------:|---------:|:-----|
| **Qwen3-0.6B** | 79.50 | 103.8% | 32.2% | 低並列で超効率的 |
| **Qwen3-4B-FP8** | 36.14 | 96.2% | 42.5% | バランス良好 |
| **Gemma 3n E2B-it** | 30.20 | 100.0% | 50.3% | 高並列でも効率維持 |
| **gpt-oss-120b** | 18.64 | 111.2% | 28.8% | 低並列特化 |

**2. 実用的な並列数の選択指針**

- **チャットボット（リアルタイム応答重視）**: 4-8並列推奨
  - 1セッションあたり20-40 TPSを維持可能
  - ユーザーはほぼシングルストリームと同等の体験
  
- **バッチ処理（総スループット重視）**: 32-64並列推奨
  - 合計TPSは最大化されるが、個別レスポンスは遅延
  - 非同期処理やキュー処理に適する

- **コスト効率の最適点**: 16-32並列
  - 合計TPSと1セッションTPSのバランスが最も良い
  - 多くのユースケースで推奨

**3. モデルサイズと並列効率の関係**

```
小型モデル（0.6B-2B）: 並列数↑ → 効率急低下（メモリ帯域がボトルネック）
中型モデル（3B-8B）  : 並列数↑ → 効率緩やかに低下（バランス型）
大型モデル（14B-120B）: 並列数↑ → 効率維持しやすい（計算がボトルネック）
```

**4. 20 TPS以上を維持できる最大並列数**

快適なチャット体験の目安として「20 TPS以上」を基準にした場合：

| Model | 最大並列数 | その時の合計TPS |
|:------|----------:|---------------:|
| **Qwen3-0.6B** | 64 | 1639 TPS |
| **Qwen3-1.7B** | 32 | 877 TPS |
| **Qwen3-4B-FP8** | 32 | 687 TPS |
| **Gemma 3n E2B-it** | 32 | 718 TPS |
| **gpt-oss-20b** | 8 | 210 TPS |
| **Ministral-3-8B** | 4 | 78 TPS |

**主な知見:**
- **Qwen3-0.6B**: 64並列で1639 TPSと圧倒的なスループット。小型モデルの並列処理性能の高さを示す。
- **gpt-oss-120b**: 120Bという巨大モデルながら64並列で343 TPSを達成。パラメータ数あたりの効率が非常に高い。
- **並列効率**: 小型モデル（0.6B, 1.7B）は並列度に対してほぼ線形にスケール。大型モデルは並列度32以降で効率が低下。
- **FP8量子化**: Qwen3-4B-FP8はFP16版より約1.3倍高速で、並列時は970 TPSを達成。
- **llama.cpp + GGUF**: 
  - `openai_gpt-oss-20b-Q4_K_M`: vLLMのFP16版と比較してシングルストリームで約2.7倍高速。並列処理ではvLLMがやや優位。
  - `TinySwallow-1.5B-Instruct-Q8_0`: vLLM版と比較して並列処理は劣る（220 TPS vs 558 TPS）が、メモリ使用量が大幅に削減。
- **Shisa V2.1シリーズ**: 日本語特化モデル。llama3.2-3bは3Bながら655 TPSの高スループットを達成。
- **ELYZA-Diffusion**: Diffusion Language Modelは従来のAutoregressive LLMとは異なるアーキテクチャ。steps=32で70 TPSを達成するが、長文生成（1024トークン以上）では品質が低下する傾向あり。vLLMではなくTransformersの`diffusion_generate`を使用。

※ Qwen3 Baseモデルは指示追従ではなく補完を行うため、生成内容はプロンプトの続きとなる傾向があります。

#### 日本語生成品質評価

各モデルの日本語生成能力を、同一プロンプト（「人工知能（AI）とは何か、小学生にもわかるように説明してください」）で評価しました。

| Engine | Model | 品質 | 評価コメント |
|:-------|:------|:----:|:-------------|
| llama.cpp | **LFM2.5-1.2B-JP-Q8_0-GGUF** | ⭐⭐⭐ | 具体例が豊富、絵文字活用、親しみやすい説明 |
| vLLM | **LFM2.5-1.2B-JP** | ⭐⭐⭐ | 構造化された説明、リンゴ・自動運転など具体例あり |
| vLLM | **Ministral-3-8B-Instruct** | ⭐⭐⭐ | 口語的で親しみやすい説明、具体例が豊富 |
| vLLM | **shisa-v2.1-llama3.2-3b** | ⭐⭐⭐ | 比喩を使った分かりやすい説明、Markdown活用 |
| vLLM | **Gemma-2-Llama-Swallow-2b** | ⭐⭐⭐ | 箇条書きで具体例を列挙、構造化された回答 |
| vLLM | **Gemma 3n E2B-it** | ⭐⭐⭐ | 学習・パターン認識・予測と段階的に説明 |
| vLLM | **Gemma 3n E2B-it FP8** | ⭐⭐⭐ | E2Bと同等品質、FP8でも品質劣化なし |
| vLLM | **shisa-v2.1-unphi4-14b** | ⭐⭐⭐ | 具体例（写真認識、ゲーム）を含む正確な説明 |
| vLLM | **LFM2.5-1.2B-Instruct** | ⭐⭐☆ | 内容は正確だが「覚えて覚える」など繰り返しあり |
| vLLM | **Qwen3-4B-Instruct-FP8** | ⭐⭐☆ | 「コンピュータの脳」という分かりやすい比喩 |
| vLLM | **shisa-v2.1-qwen3-8b** | ⭐⭐☆ | 「お手伝いさん」の比喩は良いが、会話が続く癖あり |
| Transformers | **ELYZA-Diffusion-Instruct** | ⭐⭐☆ | 内容は正確だが「言葉をしたり」など文法的違和感 |
| vLLM | **Gemma 3n E4B-it** | ⭐⭐☆ | 簡潔すぎる一言回答（「考える力を教えること」） |
| vLLM | **Qwen3-0.6B** | ⭐⭐☆ | 質問を繰り返してから回答する傾向 |
| vLLM | **Qwen3-1.7B** | ⭐⭐☆ | 質問を繰り返す傾向、内容は正確 |
| vLLM | **Qwen3-4B-Instruct** | ⭐⭐☆ | 質問を繰り返す傾向、内容は正確 |
| vLLM | **Qwen3-8B** | ⭐⭐☆ | 質問を繰り返す傾向、内容は正確 |
| vLLM | **gpt-oss-120b** | ⭐☆☆ | 日本語で開始するが途中で英語に切り替わる |
| vLLM | **gpt-oss-20b** | ⭐☆☆ | 回答が脱線しやすい、内容が不安定 |
| vLLM | **TinySwallow-1.5B-Instruct** | ⭐☆☆ | 回答生成されず（空出力） |
| llama.cpp | **TinySwallow-1.5B-Q8_0-GGUF** | ⭐☆☆ | 質問を繰り返すだけで回答しない |
| vLLM | **shisa-v2.1-lfm2-1.2b** | ❌ | 出力が完全に崩壊（意味不明な文字列の繰り返し） |

**品質評価基準:**
- ⭐⭐⭐: 優秀 - 正確で分かりやすく、構造化された回答
- ⭐⭐☆: 良好 - 内容は正確だが、やや癖がある
- ⭐☆☆: 要改善 - 回答が不完全または不安定
- ❌: 使用不可 - 出力が破綻

**推奨モデル（日本語用途）:**
1. **shisa-v2.1-llama3.2-3b** - 速度と品質のベストバランス（24 TPS）
2. **Ministral-3-8B-Instruct** - 最高品質の日本語（20 TPS）
3. **Gemma-2-Llama-Swallow-2b** - 2Bながら高品質（30 TPS）

#### 非対応モデル

以下のモデルはvLLMで正常に動作しませんでした：

| Model | 原因 |
|:------|:-----|
| **shisa-ai/shisa-v2.1-lfm2-1.2b** | `lfm2`（Liquid Foundation Model）アーキテクチャがvLLM未対応。ロードは成功するが出力が破綻 |
| **elyza/ELYZA-Diffusion-*-Dream-7B** | Diffusion Language ModelはvLLM非対応。Transformersの`diffusion_generate`で動作 |

#### 特殊設定が必要なモデル

| Model | 設定 |
|:------|:-----|
| **mistralai/Ministral-*-Instruct-2512** | `--tokenizer_mode mistral --config_format mistral --load_format mistral` が必要（自動検出済み） |

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
