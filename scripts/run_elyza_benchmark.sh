#!/bin/bash
# ELYZA Diffusion モデルのベンチマーク実行スクリプト
# nohup ./scripts/run_elyza_benchmark.sh > benchmark_elyza.log 2>&1 &

set -e
cd /home/k/develop/services/vllm_gb10

LOG_FILE="benchmark_elyza_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "ELYZA Diffusion ベンチマーク開始: $(date)"
echo "=========================================="

# Dockerコンテナ内で実行
MODELS=(
    "elyza/ELYZA-Diffusion-Base-1.0-Dream-7B"
    "elyza/ELYZA-Diffusion-Instruct-1.0-Dream-7B"
)

# 異なるsteps数でベンチマーク
STEPS_LIST=(256 128 64 32)

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    echo ""
    echo "=========================================="
    echo "モデル: $MODEL"
    echo "=========================================="
    
    for STEPS in "${STEPS_LIST[@]}"; do
        echo ""
        echo "--- Steps: $STEPS ---"
        echo "開始時刻: $(date)"
        
        docker compose run --rm unified-proxy python3 /workspace/scripts/benchmark_elyza_diffusion.py \
            --model "/workspace/models/$MODEL_NAME" \
            --steps "$STEPS" || {
            echo "警告: $MODEL (steps=$STEPS) のベンチマークが失敗しました"
        }
        
        echo "完了時刻: $(date)"
    done
done

echo ""
echo "=========================================="
echo "全ベンチマーク完了: $(date)"
echo "=========================================="
