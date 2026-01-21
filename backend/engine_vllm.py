"""
vLLM engine management
"""
import os
import gc
import torch
import asyncio
import json
import time
from typing import Optional, List, Dict
from fastapi import HTTPException

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from .state import state
from .utils import sanitize_utf8, build_vllm_prompt_with_history
from .database import add_message


def _is_mistral_model(model_path: str) -> bool:
    """Check if the model is a Mistral model requiring special options"""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_type = config.get("model_type", "")
            architectures = config.get("architectures", [])
            # Check for Mistral-specific model types
            if "mistral" in model_type.lower():
                return True
            for arch in architectures:
                if "mistral" in arch.lower():
                    return True
        except Exception:
            pass
    # Also check model path name
    if "mistral" in model_path.lower() or "ministral" in model_path.lower():
        return True
    return False


def _load_vllm_model_sync(model_path: str, dtype: str, gpu_memory_utilization: float, max_model_len: int):
    """Synchronous vLLM model loading (runs in thread pool)"""
    # Check if this is a Mistral model
    is_mistral = _is_mistral_model(model_path)
    
    if is_mistral:
        print(f"vLLM: Detected Mistral model, using mistral-specific options")
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral"
        )
    else:
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True
        )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def load_vllm_model(model_path: str, dtype: str, gpu_memory_utilization: float, max_model_len: int):
    """Load a model using vLLM engine"""
    state.vllm_status = "loading"
    state.vllm_progress = 0.0
    state.vllm_progress_message = "初期化中..."
    
    try:
        # Check if same model already loaded
        if state.vllm_engine is not None and state.vllm_current_model == model_path:
            print(f"vLLM: Model {model_path} already loaded, skipping reload")
            state.vllm_status = "loaded"
            state.vllm_progress = 1.0
            state.vllm_progress_message = "ロード済み"
            return {"status": "success", "message": f"Model already loaded: {model_path}"}
        
        # Unload existing engine
        if state.vllm_engine is not None:
            state.vllm_progress = 0.1
            state.vllm_progress_message = "既存モデルをアンロード中..."
            print(f"vLLM: Unloading existing model: {state.vllm_current_model}")
            del state.vllm_engine
            gc.collect()
            torch.cuda.empty_cache()
            state.vllm_engine = None
            state.vllm_current_model = None
            await asyncio.sleep(0)  # Yield to event loop
        
        # Validate path
        state.vllm_progress = 0.2
        state.vllm_progress_message = "モデルパスを検証中..."
        if not os.path.exists(model_path):
            state.vllm_status = "error"
            state.vllm_progress = 0.0
            state.vllm_progress_message = "エラー: モデルが見つかりません"
            raise HTTPException(status_code=400, detail=f"Model path not found: {model_path}")
        
        state.vllm_progress = 0.3
        state.vllm_progress_message = "vLLMエンジンを初期化中..."
        print(f"vLLM: Loading model {model_path}...")
        
        state.vllm_progress = 0.5
        state.vllm_progress_message = "モデルをロード中..."
        
        # Run blocking model load in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        state.vllm_engine = await loop.run_in_executor(
            None,  # Use default thread pool
            _load_vllm_model_sync,
            model_path, dtype, gpu_memory_utilization, max_model_len
        )
        
        state.vllm_progress = 0.9
        state.vllm_progress_message = "最終初期化中..."
        state.vllm_current_model = model_path
        
        state.vllm_status = "loaded"
        state.vllm_progress = 1.0
        state.vllm_progress_message = "ロード完了"
        print(f"vLLM: Model loaded successfully")
        return {"status": "success", "message": f"Model loaded: {model_path}"}
        
    except Exception as e:
        print(f"vLLM: Error loading model: {e}")
        state.vllm_status = "error"
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


async def unload_vllm_model():
    """Unload vLLM model and free GPU memory"""
    if state.vllm_engine is None:
        return {"status": "success", "message": "No vLLM model loaded"}
    
    print(f"vLLM: Unloading model: {state.vllm_current_model}")
    del state.vllm_engine
    gc.collect()
    torch.cuda.empty_cache()
    state.vllm_engine = None
    state.vllm_current_model = None
    state.vllm_status = "idle"
    state.vllm_progress = 0.0
    state.vllm_progress_message = ""
    print("vLLM: Model unloaded and GPU cache cleared")
    return {"status": "success", "message": "vLLM model unloaded"}


async def vllm_stream_generator(request_id: str, results_generator, session_id: Optional[str] = None, user_prompt: Optional[str] = None):
    """Generator for vLLM streaming responses"""
    previous_text = ""
    full_response = ""
    async for request_output in results_generator:
        current_text = request_output.outputs[0].text
        finish_reason = request_output.outputs[0].finish_reason
        
        # Send only the delta (new text since last update)
        delta_text = current_text[len(previous_text):]
        
        # If this is the final chunk, sanitize to remove incomplete UTF-8
        if finish_reason:
            sanitized_current = sanitize_utf8(current_text)
            delta_text = sanitized_current[len(previous_text):]
            full_response = sanitized_current
        
        if delta_text or finish_reason:
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": state.vllm_current_model,
                "choices": [
                    {
                        "text": delta_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        previous_text = current_text
        if not finish_reason:
            full_response = current_text
    
    # Sanitize final output
    full_response = sanitize_utf8(full_response)
    
    # Log output
    output_preview = full_response[:200] + "..." if len(full_response) > 200 else full_response
    print(f"📤 OUTPUT [vllm/stream] len={len(full_response)}: {output_preview}")
    
    # Save to session after streaming completes
    if session_id and user_prompt:
        add_message(session_id, "user", user_prompt)
        add_message(session_id, "assistant", full_response)
    
    yield "data: [DONE]\n\n"


async def generate_vllm_completion(
    prompt: str,
    model: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    session_id: Optional[str],
    all_messages: List[Dict]
):
    """Generate completion using vLLM engine"""
    if state.vllm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM: No model loaded")
    
    if state.vllm_status != "loaded":
        raise HTTPException(status_code=503, detail=f"vLLM: Engine status is {state.vllm_status}")
    
    # Build prompt with history if we have messages
    if all_messages:
        formatted_prompt = build_vllm_prompt_with_history(all_messages, state.vllm_current_model)
    else:
        from .utils import apply_chat_template
        formatted_prompt = apply_chat_template(prompt, state.vllm_current_model)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    request_id = random_uuid()
    results_generator = state.vllm_engine.generate(formatted_prompt, sampling_params, request_id)
    
    return request_id, results_generator

