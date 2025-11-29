#!/usr/bin/env python3 -u
"""
完全自動ベンチマークスクリプト
モデルのダウンロード→シングルストリームベンチマーク→並列ベンチマークを自動実行
"""

import sys
import os

# 標準出力のバッファリングを無効化
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import argparse
import asyncio
import aiohttp
import subprocess
import sys
import time
import os
import csv
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# 設定
API_BASE = "http://localhost:8001"
CONCURRENCY_LEVELS = [4, 8, 16, 32, 64]
TARGET_TOKENS = 1024

# プロンプト定義（シングルストリーム用）
SINGLE_PROMPTS = [
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

# 並列ベンチマーク用プロンプト
PARALLEL_PROMPT = SINGLE_PROMPTS[-1]["text"]  # 1024トークン用

def download_model(model_id: str, hf_token: str = None) -> bool:
    """モデルをダウンロード"""
    print(f"\n{'='*60}", flush=True)
    print(f"Step 1: Downloading model {model_id}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # HF_TOKENを環境変数または.envから取得
    if not hf_token:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.strip().split("=", 1)[1]
                        break
    
    if not hf_token:
        print("Warning: HF_TOKEN not found. Download may fail for gated models.")
    
    # Dockerコンテナ内でダウンロードスクリプトを実行
    cmd = [
        "docker", "compose", "run", "--rm",
        "-e", f"HF_TOKEN={hf_token}" if hf_token else "HF_TOKEN=",
        "vllm-server",
        "python3", "/workspace/scripts/download_model.py",
        "--model", model_id
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            check=True,
            capture_output=False
        )
        print(f"✓ Model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Model download failed: {e}")
        return False

async def load_model(session: aiohttp.ClientSession, model_path: str) -> bool:
    """モデルをロード"""
    print(f"\nLoading model: {model_path} ...")
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

async def run_single_stream_benchmark(
    session: aiohttp.ClientSession,
    model_path: str,
    model_name: str,
    output_dir: Path
) -> bool:
    """シングルストリームベンチマーク"""
    print(f"\n{'='*60}")
    print(f"Step 2: Single Stream Benchmark")
    print(f"{'='*60}")
    
    single_csv = output_dir / "single_stream.csv"
    
    # CSVヘッダー
    with open(single_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "target_tokens", "actual_tokens", "latency_ms", "tps", "prompt", "output_text"
        ])
    
    # ウォームアップ
    print("  Warming up...")
    await run_single_inference(session, model_path, "Hello", 10)
    
    # 各プロンプトでベンチマーク
    with tqdm(total=len(SINGLE_PROMPTS), desc="Single-stream", unit="prompt") as pbar:
        for prompt_data in SINGLE_PROMPTS:
            target_tokens = prompt_data["target"]
            prompt_text = prompt_data["text"]
            max_tokens = int(target_tokens * 1.5)
            
            pbar.set_postfix_str(f"{target_tokens} tokens")
            
            result = await run_single_inference(session, model_path, prompt_text, max_tokens)
            
            if result["success"]:
                latency_ms = result["latency"] * 1000
                tps = result["tokens"] / result["latency"] if result["latency"] > 0 else 0
                pbar.write(f"  {target_tokens} tokens: {result['tokens']} generated, {tps:.2f} TPS")
            
            with open(single_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name,
                    target_tokens,
                    result["tokens"],
                    latency_ms,
                    tps,
                    prompt_text,
                    result["text"]
                ])
        else:
            print(f"    Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n✓ Single stream results saved to {single_csv}")
    return True

async def run_parallel_benchmark(
    session: aiohttp.ClientSession,
    model_path: str,
    model_name: str,
    output_dir: Path
) -> bool:
    """並列ベンチマーク"""
    print(f"\n{'='*60}")
    print(f"Step 3: Parallel Benchmark")
    print(f"{'='*60}")
    
    parallel_csv = output_dir / "parallel.csv"
    
    # CSVヘッダー
    with open(parallel_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "concurrency", "total_time_s", "successful", "failed",
            "total_tokens", "throughput_tps", "avg_latency_s", "min_latency_s", "max_latency_s"
        ])
    
    # 各並列度でベンチマーク
    for concurrency in CONCURRENCY_LEVELS:
        print(f"  Running {concurrency} concurrent requests...")
        
        # 並列リクエストを生成
        tasks = [
            run_single_inference(session, model_path, PARALLEL_PROMPT, int(TARGET_TOKENS * 1.5))
            for _ in range(concurrency)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # 統計を計算
        successful = [r for r in results if r["success"]]
        failed = len(results) - len(successful)
        
        if successful:
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
            
            with open(parallel_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name,
                    concurrency,
                    f"{total_time:.2f}",
                    len(successful),
                    failed,
                    total_tokens,
                    f"{throughput:.2f}",
                    f"{avg_latency:.2f}",
                    f"{min_latency:.2f}",
                    f"{max_latency:.2f}",
                ])
        else:
            print(f"    Failed: All {concurrency} requests failed")
    
    print(f"\n✓ Parallel results saved to {parallel_csv}")
    return True

async def run_benchmarks(model_id: str, model_path: str, model_name: str, output_dir: Path):
    """ベンチマークを実行"""
    async with aiohttp.ClientSession() as session:
        # モデルをロード
        if not await load_model(session, model_path):
            print("Failed to load model. Aborting benchmarks.")
            return False
        
        # シングルストリームベンチマーク
        if not await run_single_stream_benchmark(session, model_path, model_name, output_dir):
            print("Single stream benchmark failed")
            return False
        
        # 並列ベンチマーク
        if not await run_parallel_benchmark(session, model_path, model_name, output_dir):
            print("Parallel benchmark failed")
            return False
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description="完全自動ベンチマーク: モデルダウンロード→シングル→並列ベンチマーク"
    )
    parser.add_argument(
        "model_id",
        help="Hugging Face model ID (例: google/gemma-3n-E2B-it, Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token (省略時は.envから読み込み)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="ダウンロードをスキップ（既にダウンロード済みの場合）"
    )
    
    args = parser.parse_args()
    
    # モデル名とパスを決定
    model_name = args.model_id.split("/")[-1]
    model_path = f"/workspace/models/{model_name}"
    
    # 出力ディレクトリを作成（モデル名のみ、タイムスタンプなし）
    output_dir = Path(__file__).parent.parent / "benchmarks" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Full Benchmark: {args.model_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Step 1: ダウンロード
    if not args.skip_download:
        if not download_model(args.model_id, args.hf_token):
            print("\n✗ Benchmark aborted due to download failure")
            sys.exit(1)
    else:
        print(f"\nSkipping download (--skip-download specified)")
    
    # Step 2 & 3: ベンチマーク実行
    success = asyncio.run(run_benchmarks(args.model_id, model_path, model_name, output_dir))
    
    if success:
        print(f"\n{'='*60}")
        print(f"✓ Full benchmark completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
    else:
        print(f"\n✗ Benchmark failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

