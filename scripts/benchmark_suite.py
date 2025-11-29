import asyncio
import aiohttp
import time
import json
import os
import csv
import argparse
from datetime import datetime

# モデル定義（ローカルパス）
MODELS = [
    "/workspace/models/gpt-oss-120b"
]

# プロンプト定義
PROMPTS = [
    {
        "target": 64,
        "text": "人工知能（AI）とは何か、小学生にもわかるように一言で説明してください。"
    },
    {
        "target": 128,
        "text": "日本の四季（春・夏・秋・冬）それぞれの魅力を、箇条書きで簡潔に紹介してください。"
    },
    {
        "target": 256,
        "text": "『深夜のコンビニ』という題名で、店内の独特の静けさや客の様子を、五感（視覚・聴覚・嗅覚など）を使った表現を入れて描写してください。"
    },
    {
        "target": 512,
        "text": "リモートワークの導入によるメリットとデメリットについて、働く側と企業側の双方の視点から詳しく論じてください。それぞれ3つ以上の具体的なポイントを挙げ、理由とともに説明してください。"
    },
    {
        "target": 1024,
        "text": "22世紀、人類が火星に移住した後の日常を描いたSF小説を書いてください。主人公は火星生まれの最初の世代です。朝起きてから学校へ行くまでの様子を、火星の重力、窓から見える赤茶色の風景、特殊な食事、未来のテクノロジーなどの詳細な設定を盛り込みながら、できるだけ長く、詳細に書き続けてください。"
    }
]

API_BASE = "http://localhost:8001"

async def load_model(session, model_path):
    print(f"Loading model: {model_path} ...")
    start = time.time()
    timeout = aiohttp.ClientTimeout(total=1800)  # 30分
    async with session.post(f"{API_BASE}/v1/models/load", json={"model_path": model_path}, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            print(f"Failed to load model: {text}")
            return False
        result = await resp.json()
        print(f"Model loaded in {time.time() - start:.2f}s")
        return True

async def run_inference(session, model_path, prompt_data):
    target_tokens = prompt_data["target"]
    prompt_text = prompt_data["text"]
    
    # 少し余裕を持たせる
    max_tokens = int(target_tokens * 1.5)
    
    payload = {
        "model": model_path,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False
    }
    
    print(f"  Running inference for target {target_tokens} tokens...")
    start_time = time.time()
    async with session.post(f"{API_BASE}/v1/completions", json=payload) as resp:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        if resp.status != 200:
            text = await resp.text()
            print(f"    Failed: {text}")
            return None

        result = await resp.json()
        usage = result["usage"]
        completion_tokens = usage["completion_tokens"]
        tps = completion_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0
        output_text = result["choices"][0]["text"]
        
        print(f"    Success: {completion_tokens} tokens generated, {latency_ms:.2f}ms, {tps:.2f} TPS")
        
        return {
            "model": os.path.basename(model_path),
            "target_tokens": target_tokens,
            "actual_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "tps": tps,
            "prompt": prompt_text,
            "output_text": output_text
        }

async def main():
    # 結果保存ディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"benchmarks/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/result.csv"
    
    print(f"Benchmark started. Results will be saved to {csv_path}")
    
    # CSVヘッダー
    fieldnames = ["model", "target_tokens", "actual_tokens", "latency_ms", "tps", "prompt", "output_text"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    async with aiohttp.ClientSession() as session:
        for model_path in MODELS:
            # モデルロード
            if not await load_model(session, model_path):
                continue
            
            # ウォームアップ（短いリクエストで一度推論を通しておく）
            print("  Warming up...")
            await run_inference(session, model_path, {"target": 10, "text": "Hi"})
            
            # プロンプトごとに実行
            for prompt in PROMPTS:
                result = await run_inference(session, model_path, prompt)
                if result:
                    # 結果をCSVに書き込み（都度書き込みでデータ消失防ぐ）
                    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(result)
            
            print("-" * 50)

    print("Benchmark completed.")

if __name__ == "__main__":
    asyncio.run(main())

