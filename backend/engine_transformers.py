"""
Transformers engine for Diffusion Language Models (ELYZA Diffusion, etc.)
"""
import gc
import time
import torch
import asyncio
from typing import Optional, AsyncGenerator

from .state import state
from .utils import sanitize_utf8
from .database import add_message

# Global lock for model loading
_load_lock = asyncio.Lock()


async def load_transformers_model(model_path: str):
    """Load a Transformers model (specifically for Diffusion LLMs)"""
    async with _load_lock:
        try:
            state.transformers_status = "loading"
            state.transformers_progress = 0.0
            state.transformers_progress_message = "モデルをロード中..."
            
            # Unload existing model first
            if state.transformers_model is not None:
                await unload_transformers_model()
            
            state.transformers_progress = 0.2
            state.transformers_progress_message = "モデルファイルを読み込み中..."
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model, tokenizer = await loop.run_in_executor(
                None,
                _load_transformers_model_sync,
                model_path
            )
            
            state.transformers_model = model
            state.transformers_tokenizer = tokenizer
            state.transformers_current_model = model_path
            state.transformers_status = "loaded"
            state.transformers_progress = 1.0
            state.transformers_progress_message = "ロード完了"
            
            print(f"✅ Transformers model loaded: {model_path}")
            
        except Exception as e:
            state.transformers_status = "error"
            state.transformers_progress = 0.0
            state.transformers_progress_message = f"エラー: {str(e)}"
            print(f"❌ Transformers model load error: {e}")
            raise


def _load_transformers_model_sync(model_path: str):
    """Synchronous model loading (runs in thread pool)"""
    from transformers import AutoModel, AutoTokenizer
    
    print(f"Transformers: Loading model from {model_path}")
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    return model, tokenizer


async def unload_transformers_model():
    """Unload the Transformers model"""
    try:
        if state.transformers_model is not None:
            del state.transformers_model
            state.transformers_model = None
        
        if state.transformers_tokenizer is not None:
            del state.transformers_tokenizer
            state.transformers_tokenizer = None
        
        state.transformers_current_model = None
        state.transformers_status = "idle"
        state.transformers_progress = 0.0
        state.transformers_progress_message = ""
        
        gc.collect()
        torch.cuda.empty_cache()
        
        print("✅ Transformers model unloaded")
        
    except Exception as e:
        print(f"❌ Transformers unload error: {e}")
        state.transformers_status = "error"
        state.transformers_progress_message = f"アンロードエラー: {str(e)}"


def _is_diffusion_model(model_path: str) -> bool:
    """Check if the model is a Diffusion LLM"""
    diffusion_keywords = ["diffusion", "dream", "ELYZA-Diffusion"]
    return any(kw.lower() in model_path.lower() for kw in diffusion_keywords)


async def transformers_generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    steps: int = 64  # 32は日本語品質が低下、64以上推奨
) -> dict:
    """Generate text using Transformers model"""
    
    if state.transformers_model is None or state.transformers_tokenizer is None:
        raise RuntimeError("No Transformers model loaded")
    
    model = state.transformers_model
    tokenizer = state.transformers_tokenizer
    
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        # inputs is a BatchEncoding, extract tensors explicitly
        input_ids = inputs["input_ids"].clone().to("cuda")
        attention_mask = inputs["attention_mask"].clone().to("cuda") if "attention_mask" in inputs else None
    except Exception:
        # Fallback for models without chat template
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].clone().to("cuda")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.clone().to("cuda")
    
    # Ensure attention_mask exists
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids).to("cuda")
    
    # Run generation in thread pool
    loop = asyncio.get_event_loop()
    
    start_time = time.time()
    
    if _is_diffusion_model(state.transformers_current_model or ""):
        # Use diffusion_generate for Diffusion LLMs
        output = await loop.run_in_executor(
            None,
            _diffusion_generate_sync,
            model, input_ids, attention_mask, max_tokens, temperature, top_p, steps
        )
    else:
        # Use standard generate for regular models
        output = await loop.run_in_executor(
            None,
            _standard_generate_sync,
            model, input_ids, attention_mask, max_tokens, temperature, top_p
        )
    
    latency = time.time() - start_time
    
    # Decode output
    generated_tokens = output[0][input_ids.size(1):]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    generated_text = sanitize_utf8(generated_text)
    
    return {
        "text": generated_text,
        "tokens": len(generated_tokens),
        "latency": latency,
        "tps": len(generated_tokens) / latency if latency > 0 else 0
    }


def _diffusion_generate_sync(model, input_ids, attention_mask, max_tokens, temperature, top_p, steps):
    """Synchronous diffusion generation"""
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg="entropy",
            alg_temp=0.5
        )
    return output


def _standard_generate_sync(model, input_ids, attention_mask, max_tokens, temperature, top_p):
    """Synchronous standard generation"""
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    return output


async def transformers_stream_generator(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    session_id: Optional[str],
    user_prompt: str,
    steps: int = 64
) -> AsyncGenerator[str, None]:
    """Stream generator for Transformers (non-streaming, returns all at once)
    
    Note: Diffusion models don't support true streaming, so we generate all at once
    and yield it as a single chunk.
    """
    import json
    
    try:
        result = await transformers_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            steps=steps
        )
        
        text = result["text"]
        tps = result["tps"]
        
        # Output preview
        output_preview = text[:200] + "..." if len(text) > 200 else text
        print(f"📤 OUTPUT [transformers] tokens={result['tokens']}, {tps:.2f} TPS: {output_preview}")
        
        # Save to session
        if session_id:
            add_message(session_id, "user", user_prompt)
            add_message(session_id, "assistant", text)
        
        # Yield the complete response as SSE (matching vLLM format for frontend compatibility)
        data = {
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
        
        # Send done signal with stats
        done_data = {
            "choices": [{
                "text": "",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "completion_tokens": result["tokens"],
                "tps": tps
            }
        }
        yield f"data: {json.dumps(done_data)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
