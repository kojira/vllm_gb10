import asyncio
import aiohttp
import time
import csv
import os
from datetime import datetime
from typing import List

# 設定
API_BASE = "http://localhost:8001"
CONCURRENCY_LEVELS = [4, 8, 16, 32, 64]
TARGET_TOKENS = 1024  # 並列ベンチマークは1024トークンで統一

# モデル定義（ローカルパス）
MODELS = [
    "/workspace/models/gemma-3n-E2B-it",
    "/workspace/models/gemma-3n-E4B-it",
    "/workspace/models/Qwen3-0.6B",
    "/workspace/models/Qwen3-1.7B",
    "/workspace/models/Qwen3-4B-Instruct-2507",
    "/workspace/models/Qwen3-4B-Instruct-2507-FP8",
    "/workspace/models/Qwen3-8B",
    "/workspace/models/gpt-oss-20b",
    "/workspace/models/gpt-oss-120b",
]

# 日本語プロンプト（1024トークン生成用）
PROMPT = "22世紀、人類が火星に移住した後の日常を描いたSF小説を書いてください。主人公は火星生まれの最初の世代です。朝起きてから学校へ行くまでの様子を、火星の重力、窓から見える赤茶色の風景、特殊な食事、未来のテクノロジーなどの詳細な設定を盛り込みながら、できるだけ長く、詳細に書き続けてください。"

async def load_model(session: aiohttp.ClientSession, model_path: str) -> bool:
    """モデルをロード"""
    print(f"Loading model: {model_path} ...")
    start = time.time()
    timeout = aiohttp.ClientTimeout(total=1800)  # 30分
    try:
        async with session.post(
            f"{API_BASE}/v1/models/load",
            json={"model_path": model_path},
            timeout=timeout
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"Failed to load model: {text}")
                return False
            result = await resp.json()
            elapsed = time.time() - start
            print(f"Model loaded in {elapsed:.2f}s")
            return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

async def run_single_inference(
    session: aiohttp.ClientSession,
    model_path: str,
    prompt: str,
    max_tokens: int
) -> dict:
    """単一の推論を実行"""
    payload = {
        "model": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    
    start_time = time.time()
    timeout = aiohttp.ClientTimeout(total=300)  # 5分
    try:
        async with session.post(
            f"{API_BASE}/v1/completions",
            json=payload,
            timeout=timeout
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {
                    "success": False,
                    "error": text,
                    "latency": 0,
                    "tokens": 0
                }
            result = await resp.json()
            end_time = time.time()
            latency = end_time - start_time
            tokens = result["usage"]["completion_tokens"]
            return {
                "success": True,
                "latency": latency,
                "tokens": tokens,
                "text": result["choices"][0]["text"]
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": 0,
            "tokens": 0
        }

async def run_parallel_benchmark(
    session: aiohttp.ClientSession,
    model_path: str,
    concurrency: int
) -> dict:
    """並列ベンチマークを実行"""
    print(f"  Running {concurrency} concurrent requests...")
    
    # 並列リクエストを生成
    tasks = [
        run_single_inference(session, model_path, PROMPT, int(TARGET_TOKENS * 1.5))
        for _ in range(concurrency)
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # 統計を計算
    successful = [r for r in results if r["success"]]
    failed = len(results) - len(successful)
    
    if not successful:
        return {
            "concurrency": concurrency,
            "total_time": end_time - start_time,
            "successful": 0,
            "failed": failed,
            "total_tokens": 0,
            "throughput_tps": 0,
            "avg_latency": 0,
            "min_latency": 0,
            "max_latency": 0,
        }
    
    total_tokens = sum(r["tokens"] for r in successful)
    total_time = end_time - start_time
    throughput = total_tokens / total_time if total_time > 0 else 0
    
    latencies = [r["latency"] for r in successful]
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"    Success: {len(successful)}/{concurrency} requests, "
          f"{total_tokens} tokens in {total_time:.2f}s, "
          f"{throughput:.2f} TPS")
    
    return {
        "concurrency": concurrency,
        "total_time": total_time,
        "successful": len(successful),
        "failed": failed,
        "total_tokens": total_tokens,
        "throughput_tps": throughput,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
    }

async def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("benchmarks", f"parallel_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "parallel_results.csv")
    
    print(f"Parallel benchmark started. Results will be saved to {results_file}")
    
    # CSVヘッダーを書き込み
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "concurrency", "total_time_s", "successful", "failed",
            "total_tokens", "throughput_tps", "avg_latency_s", "min_latency_s", "max_latency_s"
        ])
    
    async with aiohttp.ClientSession() as session:
        for model_path in MODELS:
            model_name = model_path.split("/")[-1]
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            
            # モデルをロード
            if not await load_model(session, model_path):
                print(f"Skipping {model_name} due to load failure.")
                continue
            
            # ウォームアップ
            print("  Warming up...")
            await run_single_inference(session, model_path, "Hello", 10)
            
            # 各並列度でベンチマーク
            for concurrency in CONCURRENCY_LEVELS:
                result = await run_parallel_benchmark(session, model_path, concurrency)
                
                # 結果をCSVに追記
                with open(results_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        model_name,
                        result["concurrency"],
                        f"{result['total_time']:.2f}",
                        result["successful"],
                        result["failed"],
                        result["total_tokens"],
                        f"{result['throughput_tps']:.2f}",
                        f"{result['avg_latency']:.2f}",
                        f"{result['min_latency']:.2f}",
                        f"{result['max_latency']:.2f}",
                    ])
    
    print(f"\n{'='*60}")
    print(f"Parallel benchmark completed. Results saved to {results_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())

