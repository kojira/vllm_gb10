import argparse
import asyncio
import time
import aiohttp
import numpy as np
import json

async def send_request(session, url, prompt, max_new_tokens, model):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "stream": False
    }
    start_req_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            end_req_time = time.time()
            latency = end_req_time - start_req_time
            
            if response.status == 200:
                result = await response.json()
                usage = result["usage"]
                return {
                    "success": True,
                    "latency": latency,
                    "tokens": usage["completion_tokens"]
                }
            else:
                text = await response.text()
                return {"success": False, "error": f"{response.status} - {text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def benchmark(args):
    url = f"http://{args.host}:{args.port}/v1/completions"
    print(f"Starting benchmark on {url}...")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Total Requests: {args.requests}")

    prompt = "Tell me a short joke about AI."

    async with aiohttp.ClientSession() as session:
        # ウォームアップ
        print("Warming up...")
        await send_request(session, url, "Hi", 10, args.model)
        
        start_time = time.time()
        tasks = []
        for _ in range(args.requests):
            tasks.append(asyncio.create_task(send_request(session, url, prompt, args.max_new_tokens, args.model)))
            if len(tasks) % args.concurrency == 0:
                await asyncio.sleep(0.01)
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        valid_results = [r for r in results if r["success"]]
        failed_count = len(results) - len(valid_results)
        
        if not valid_results:
            print("All requests failed.")
            return

        total_tokens = sum(r["tokens"] for r in valid_results)
        avg_latency = np.mean([r["latency"] for r in valid_results])
        p95_latency = np.percentile([r["latency"] for r in valid_results], 95)
        
        print("\n--- Benchmark Results ---")
        print(f"Successful Requests: {len(valid_results)}/{args.requests}")
        print(f"Failed Requests: {failed_count}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Total Tokens: {total_tokens}")
        print(f"Throughput (TPS): {total_tokens / total_time:.2f} tokens/s")
        print(f"Request Throughput (RPS): {len(valid_results) / total_time:.2f} req/s")
        print(f"Avg Latency: {avg_latency:.4f}s")
        print(f"P95 Latency: {p95_latency:.4f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark vLLM Server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrency level")
    parser.add_argument("--requests", type=int, default=100, help="Total requests")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument("--model", type=str, default="/workspace/models/gemma-3n-E2B-it", help="Model path")
    
    args = parser.parse_args()
    asyncio.run(benchmark(args))

