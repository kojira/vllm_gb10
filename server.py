import os
import gc
import torch
import uuid
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

app = FastAPI(title="vLLM Dynamic Inference Server")

# Global engine instance
llm_engine: Optional[AsyncLLMEngine] = None
current_model_path: Optional[str] = None

class LoadModelRequest(BaseModel):
    model_path: str
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False

@app.post("/v1/models/load")
async def load_model(request: LoadModelRequest):
    global llm_engine
    global current_model_path
    
    print(f"Received request to load model: {request.model_path}")

    # Check if the same model is already loaded
    if llm_engine is not None and current_model_path == request.model_path:
        print(f"Model {request.model_path} is already loaded. Skipping reload.")
        return {"status": "success", "message": f"Model already loaded: {request.model_path}"}

    # Unload existing engine if any
    if llm_engine is not None:
        print(f"Unloading existing model: {current_model_path}...")
        del llm_engine
        gc.collect()
        torch.cuda.empty_cache()
        llm_engine = None
        current_model_path = None
        print("Model unloaded and GPU cache cleared.")

    # Validate path
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=400, detail=f"Model path not found: {request.model_path}")

    try:
        print(f"Initializing vLLM engine with {request.model_path}...")
        engine_args = AsyncEngineArgs(
            model=request.model_path,
            dtype=request.dtype,
            gpu_memory_utilization=request.gpu_memory_utilization,
            max_model_len=request.max_model_len,
            trust_remote_code=True,
            enforce_eager=True
        )
        llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        current_model_path = request.model_path
        print("Model loaded successfully.")
        return {"status": "success", "message": f"Model loaded: {request.model_path}"}
    except Exception as e:
        print(f"Error loading model: {e}")
        # Attempt cleanup on failure
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    global llm_engine
    
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="No model loaded. Please load a model via /v1/models/load first.")

    request_id = random_uuid()
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    results_generator = llm_engine.generate(request.prompt, sampling_params, request_id)

    # Simple non-streaming implementation for benchmark
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
         raise HTTPException(status_code=500, detail="Generation failed")

    text_output = final_output.outputs[0].text
    
    # OpenAI-compatible-ish response format for compatibility with benchmark script
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model or "unknown",
        "choices": [
            {
                "text": text_output,
                "index": 0,
                "logprobs": None,
                "finish_reason": final_output.outputs[0].finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(final_output.outputs[0].token_ids),
            "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids)
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm_engine is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

