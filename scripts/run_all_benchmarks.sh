#!/bin/bash
# 全モデルのベンチマーク実行スクリプト
# nohup ./scripts/run_all_benchmarks.sh > benchmark_all.log 2>&1 &

set -e
cd /home/k/develop/services/vllm_gb10

LOG_FILE="benchmark_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "ベンチマーク開始: $(date)"
echo "=========================================="

# サーバーが起動していなければ起動
echo "サーバー起動確認..."
docker compose up -d
sleep 10

# サーバーの準備ができるまで待機
echo "サーバー準備待機..."
for i in {1..60}; do
    if curl -s http://localhost:8081/health > /dev/null 2>&1; then
        echo "サーバー準備完了"
        break
    fi
    echo "待機中... ($i/60)"
    sleep 5
done

# ベンチマーク対象モデル
MODELS=(
    "mistralai/Ministral-3-8B-Instruct-2512"
    "shisa-ai/shisa-v2.1-lfm2-1.2b"
    "shisa-ai/shisa-v2.1-llama3.2-3b"
    "shisa-ai/shisa-v2.1-qwen3-8b"
    "shisa-ai/shisa-v2.1-unphi4-14b"
    "elyza/ELYZA-Diffusion-Base-1.0-Dream-7B"
    "elyza/ELYZA-Diffusion-Instruct-1.0-Dream-7B"
)

TOTAL=${#MODELS[@]}
CURRENT=0

for MODEL in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] ベンチマーク: $MODEL"
    echo "開始時刻: $(date)"
    echo "=========================================="
    
    # vLLMでベンチマーク実行（ポート8081を使用）
    python3 scripts/benchmark_full.py "$MODEL" --engine vllm --skip-download --api-base http://localhost:8081 || {
        echo "警告: $MODEL のベンチマークが失敗しました。続行します..."
    }
    
    echo "[$CURRENT/$TOTAL] 完了: $MODEL"
    echo "完了時刻: $(date)"
done

echo ""
echo "=========================================="
echo "全ベンチマーク完了: $(date)"
echo "=========================================="
