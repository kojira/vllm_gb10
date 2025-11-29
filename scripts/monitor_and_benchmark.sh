#!/bin/bash

MODEL_NAME="gpt-oss-120b"
MODEL_DIR="/home/k/develop/services/vllm_gb10/models/${MODEL_NAME}"
CHECK_INTERVAL=30

echo "Monitoring download progress for ${MODEL_NAME}..."
echo "Checking every ${CHECK_INTERVAL} seconds..."
echo ""

while true; do
    # モデルディレクトリが存在し、config.jsonが存在するかチェック
    if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download appears complete. Verifying..."
        
        # 数秒待ってファイルが安定しているか確認
        sleep 5
        
        # もう一度確認
        if [ -f "${MODEL_DIR}/config.json" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download confirmed complete!"
            echo ""
            echo "Starting benchmark..."
            
            # ベンチマークスクリプトのモデルリストを更新
            cd /home/k/develop/services/vllm_gb10
            
            # benchmark_suite.pyのMODELS行を更新
            sed -i 's|MODELS = \[.*\]|MODELS = [\n    "/workspace/models/gpt-oss-120b"\n]|' scripts/benchmark_suite.py
            
            # 仮想環境をアクティベートしてベンチマーク実行
            source venv/bin/activate
            python3 scripts/benchmark_suite.py
            
            # 結果を分割
            LATEST_DIR=$(ls -td benchmarks/2025* | head -1)
            if [ -n "${LATEST_DIR}" ]; then
                sed -i "s|INPUT_CSV = \"benchmarks/[^\"]*\"|INPUT_CSV = \"${LATEST_DIR}/result.csv\"|" scripts/split_results.py
                python3 scripts/split_results.py
                echo ""
                echo "Benchmark completed. Results saved to ${LATEST_DIR}"
            fi
            
            break
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download in progress... (checking ${MODEL_DIR})"
    fi
    
    sleep ${CHECK_INTERVAL}
done

echo ""
echo "All tasks completed!"

