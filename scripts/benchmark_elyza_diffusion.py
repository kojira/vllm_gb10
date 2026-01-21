#!/usr/bin/env python3
"""
ELYZA Diffusion Language Model ベンチマークスクリプト
Transformers の diffusion_generate を使用
"""

import sys
import os
import argparse
import time
import csv
import gc
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

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


def load_model(model_path: str):
    """モデルとトークナイザーをロード"""
    print(f"Loading model: {model_path}")
    start = time.time()
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")
    return model, tokenizer


def run_single_inference(model, tokenizer, prompt: str, max_tokens: int, steps: int = 256) -> dict:
    """単一の推論を実行"""
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    prompt_len = input_ids.size(1)
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                steps=steps,
                temperature=0.5,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.5
            )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # diffusion_generateの戻り値を処理
        # 戻り値がTensorの場合とオブジェクトの場合がある
        if hasattr(output, 'sequences'):
            generated_ids = output.sequences[0][prompt_len:]
        elif isinstance(output, torch.Tensor):
            generated_ids = output[0][prompt_len:]
        else:
            # その他の形式の場合
            generated_ids = output[0][prompt_len:] if len(output.shape) > 1 else output[prompt_len:]
        
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
        
        return {
            "success": True,
            "latency": latency,
            "tokens": num_tokens,
            "text": generated_text
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
            "latency": 0,
            "tokens": 0
        }


def run_single_stream_benchmark(model, tokenizer, model_name: str, output_dir: Path, steps: int) -> bool:
    """シングルストリームベンチマーク"""
    print(f"\n{'='*60}")
    print(f"Single Stream Benchmark (steps={steps})")
    print(f"{'='*60}")
    
    single_csv = output_dir / f"single_stream_steps{steps}.csv"
    
    # CSVヘッダー
    with open(single_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "target_tokens", "actual_tokens", "latency_ms", "tps", "steps", "prompt", "output_text"
        ])
    
    # ウォームアップ
    print("  Warming up...")
    run_single_inference(model, tokenizer, "Hello", 10, steps=steps)
    
    # 各プロンプトでベンチマーク
    with tqdm(total=len(SINGLE_PROMPTS), desc="Single-stream", unit="prompt") as pbar:
        for prompt_data in SINGLE_PROMPTS:
            target_tokens = prompt_data["target"]
            prompt_text = prompt_data["text"]
            max_tokens = int(target_tokens * 1.5)
            
            pbar.set_postfix_str(f"{target_tokens} tokens")
            
            result = run_single_inference(model, tokenizer, prompt_text, max_tokens, steps=steps)
            
            if result["success"]:
                latency_ms = result["latency"] * 1000
                tps = result["tokens"] / result["latency"] if result["latency"] > 0 else 0
                pbar.write(f"  {target_tokens} tokens: {result['tokens']} generated, {tps:.2f} TPS, {latency_ms:.0f}ms")
                
                with open(single_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        model_name,
                        target_tokens,
                        result["tokens"],
                        latency_ms,
                        tps,
                        steps,
                        prompt_text,
                        result.get("text", "")
                    ])
            else:
                pbar.write(f"  {target_tokens} tokens: Failed - {result.get('error', 'Unknown error')}")
            
            pbar.update(1)
    
    print(f"\n✓ Single stream results saved to {single_csv}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="ELYZA Diffusion Language Model ベンチマーク"
    )
    parser.add_argument(
        "--model",
        default="elyza/ELYZA-Diffusion-Instruct-1.0-Dream-7B",
        help="モデルパスまたはHugging Face ID"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Diffusion steps (default: 256, 低いほど高速)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ"
    )
    
    args = parser.parse_args()
    
    # モデル名を決定
    if args.model.startswith("/"):
        model_path = args.model
        model_name = Path(args.model).name
    else:
        model_name = args.model.split("/")[-1]
        # ローカルパスを優先
        local_path = f"/workspace/models/{model_name}"
        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = args.model
    
    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "benchmarks" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ELYZA Diffusion Benchmark")
    print(f"Model: {model_path}")
    print(f"Steps: {args.steps}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # モデルロード
    model, tokenizer = load_model(model_path)
    
    # ベンチマーク実行
    run_single_stream_benchmark(model, tokenizer, model_name, output_dir, args.steps)
    
    # クリーンアップ
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"✓ Benchmark completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
