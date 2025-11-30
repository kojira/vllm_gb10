import os
import gc
import torch
import asyncio
import subprocess
import signal
import time
import json
import aiohttp
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

app = FastAPI(title="Unified LLM Inference Proxy (vLLM + llama.cpp)")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のオリジンに制限すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine state management
class EngineState:
    # vLLM state
    vllm_engine: Optional[AsyncLLMEngine] = None
    vllm_current_model: Optional[str] = None
    vllm_status: str = "idle"  # idle/loading/loaded/error
    vllm_loading_task: Optional[asyncio.Task] = None
    
    # llama.cpp state
    llamacpp_process: Optional[subprocess.Popen] = None
    llamacpp_current_model: Optional[str] = None
    llamacpp_status: str = "idle"  # idle/loading/loaded/error
    llamacpp_port: int = 8002
    llamacpp_loading_task: Optional[asyncio.Task] = None

state = EngineState()

class LoadModelRequest(BaseModel):
    model_path: str
    engine: str  # "vllm" or "llamacpp"
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
    engine: str  # "vllm" or "llamacpp" - REQUIRED

async def load_vllm_model(model_path: str, dtype: str, gpu_memory_utilization: float, max_model_len: int):
    """Load a model using vLLM engine"""
    global state
    
    state.vllm_status = "loading"
    
    try:
        # Check if same model already loaded
        if state.vllm_engine is not None and state.vllm_current_model == model_path:
            print(f"vLLM: Model {model_path} already loaded, skipping reload")
            state.vllm_status = "loaded"
            return {"status": "success", "message": f"Model already loaded: {model_path}"}
        
        # Unload existing engine
        if state.vllm_engine is not None:
            print(f"vLLM: Unloading existing model: {state.vllm_current_model}")
            del state.vllm_engine
            gc.collect()
            torch.cuda.empty_cache()
            state.vllm_engine = None
            state.vllm_current_model = None
        
        # Validate path
        if not os.path.exists(model_path):
            state.vllm_status = "error"
            raise HTTPException(status_code=400, detail=f"Model path not found: {model_path}")
        
        print(f"vLLM: Loading model {model_path}...")
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True
        )
        state.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        state.vllm_current_model = model_path
        state.vllm_status = "loaded"
        print(f"vLLM: Model loaded successfully")
        return {"status": "success", "message": f"Model loaded: {model_path}"}
        
    except Exception as e:
        print(f"vLLM: Error loading model: {e}")
        state.vllm_status = "error"
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

async def load_llamacpp_model(model_path: str):
    """Load a model using llama.cpp server"""
    global state
    
    state.llamacpp_status = "loading"
    
    try:
        # Check if same model already loaded
        if state.llamacpp_process is not None and state.llamacpp_current_model == model_path:
            # Check if process is still running
            if state.llamacpp_process.poll() is None:
                print(f"llama.cpp: Model {model_path} already loaded, skipping reload")
                state.llamacpp_status = "loaded"
                return {"status": "success", "message": f"Model already loaded: {model_path}"}
        
        # Stop existing process
        if state.llamacpp_process is not None:
            print(f"llama.cpp: Stopping existing process for model: {state.llamacpp_current_model}")
            try:
                state.llamacpp_process.send_signal(signal.SIGTERM)
                state.llamacpp_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                state.llamacpp_process.kill()
                state.llamacpp_process.wait()
            state.llamacpp_process = None
            state.llamacpp_current_model = None
            # Wait a bit for port to be released
            await asyncio.sleep(2)
        
        # Validate path
        if not os.path.exists(model_path):
            state.llamacpp_status = "error"
            raise HTTPException(status_code=400, detail=f"Model path not found: {model_path}")
        
        print(f"llama.cpp: Starting server with model {model_path}...")
        
        # Start llama-server process
        cmd = [
            "/workspace/llama.cpp/build/bin/llama-server",
            "--host", "127.0.0.1",
            "--port", str(state.llamacpp_port),
            "--model", model_path,
            "--n-gpu-layers", "-1",
            "--ctx-size", "8192",
            "--parallel", "64"
        ]
        
        state.llamacpp_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        state.llamacpp_current_model = model_path
        
        # Wait for server to be ready
        max_wait = 120  # 2 minutes
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < max_wait:
                try:
                    async with session.get(f"http://127.0.0.1:{state.llamacpp_port}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            state.llamacpp_status = "loaded"
                            print(f"llama.cpp: Server ready")
                            return {"status": "success", "message": f"Model loaded: {model_path}"}
                except:
                    pass
                
                # Check if process died
                if state.llamacpp_process.poll() is not None:
                    state.llamacpp_status = "error"
                    raise HTTPException(status_code=500, detail="llama-server process died during startup")
                
                await asyncio.sleep(1)
        
        # Timeout
        state.llamacpp_status = "error"
        if state.llamacpp_process:
            state.llamacpp_process.kill()
        raise HTTPException(status_code=500, detail="llama-server startup timeout")
        
    except Exception as e:
        print(f"llama.cpp: Error loading model: {e}")
        state.llamacpp_status = "error"
        if state.llamacpp_process:
            try:
                state.llamacpp_process.kill()
            except:
                pass
            state.llamacpp_process = None
        raise

@app.post("/v1/models/load")
async def load_model(request: LoadModelRequest):
    """Load a model on the specified engine"""
    
    if request.engine not in ["vllm", "llamacpp"]:
        raise HTTPException(status_code=400, detail="engine must be 'vllm' or 'llamacpp'")
    
    if request.engine == "vllm":
        # Cancel any ongoing loading task
        if state.vllm_loading_task and not state.vllm_loading_task.done():
            state.vllm_loading_task.cancel()
        
        # Start loading in background
        state.vllm_loading_task = asyncio.create_task(
            load_vllm_model(request.model_path, request.dtype, request.gpu_memory_utilization, request.max_model_len)
        )
        
        # Wait for completion
        return await state.vllm_loading_task
        
    else:  # llamacpp
        # Cancel any ongoing loading task
        if state.llamacpp_loading_task and not state.llamacpp_loading_task.done():
            state.llamacpp_loading_task.cancel()
        
        # Start loading in background
        state.llamacpp_loading_task = asyncio.create_task(
            load_llamacpp_model(request.model_path)
        )
        
        # Wait for completion
        return await state.llamacpp_loading_task

async def vllm_stream_generator(request_id: str, results_generator):
    """Generator for vLLM streaming responses"""
    previous_text = ""
    async for request_output in results_generator:
        current_text = request_output.outputs[0].text
        # Send only the delta (new text since last update)
        delta_text = current_text[len(previous_text):]
        
        if delta_text or request_output.outputs[0].finish_reason:
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
                        "finish_reason": request_output.outputs[0].finish_reason
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        previous_text = current_text
    
    yield "data: [DONE]\n\n"

async def llamacpp_stream_generator(url: str, payload: dict):
    """Generator for llama.cpp streaming responses"""
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            async for line in resp.content:
                if line:
                    yield line

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Generate completions using the specified engine"""
    
    if request.engine not in ["vllm", "llamacpp"]:
        raise HTTPException(status_code=400, detail="engine must be 'vllm' or 'llamacpp'")
    
    if request.engine == "vllm":
        # Use vLLM
        if state.vllm_engine is None:
            raise HTTPException(status_code=503, detail="vLLM: No model loaded. Please load a model first.")
        
        if state.vllm_status != "loaded":
            raise HTTPException(status_code=503, detail=f"vLLM: Engine status is {state.vllm_status}")
        
        request_id = random_uuid()
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        results_generator = state.vllm_engine.generate(request.prompt, sampling_params, request_id)
        
        if request.stream:
            # Streaming response
            return StreamingResponse(
                vllm_stream_generator(request_id, results_generator),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output is None:
                raise HTTPException(status_code=500, detail="Generation failed")
            
            text_output = final_output.outputs[0].text
            
            return {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model or state.vllm_current_model,
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
    
    else:  # llamacpp
        # Forward to llama.cpp server
        if state.llamacpp_process is None or state.llamacpp_process.poll() is not None:
            raise HTTPException(status_code=503, detail="llama.cpp: No model loaded or server died")
        
        if state.llamacpp_status != "loaded":
            raise HTTPException(status_code=503, detail=f"llama.cpp: Engine status is {state.llamacpp_status}")
        
        # Forward request to llama-server
        payload = {
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream
        }
        
        url = f"http://127.0.0.1:{state.llamacpp_port}/completion"
        
        if request.stream:
            # Streaming response
            return StreamingResponse(
                llamacpp_stream_generator(url, payload),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise HTTPException(status_code=resp.status, detail=f"llama.cpp error: {text}")
                        
                        result = await resp.json()
                        
                        # Convert to OpenAI format
                        return {
                            "id": f"llamacpp-{int(time.time())}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": request.model or state.llamacpp_current_model,
                            "choices": [
                                {
                                    "text": result.get("content", ""),
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": "stop" if result.get("stop", False) else "length"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": result.get("tokens_evaluated", 0),
                                "completion_tokens": result.get("tokens_predicted", 0),
                                "total_tokens": result.get("tokens_evaluated", 0) + result.get("tokens_predicted", 0)
                            }
                        }
            except aiohttp.ClientError as e:
                raise HTTPException(status_code=503, detail=f"llama.cpp communication error: {str(e)}")

@app.get("/v1/status")
async def get_status():
    """Get the status of both engines"""
    return {
        "vllm": {
            "status": state.vllm_status,
            "model": state.vllm_current_model
        },
        "llamacpp": {
            "status": state.llamacpp_status,
            "model": state.llamacpp_current_model,
            "process_alive": state.llamacpp_process is not None and state.llamacpp_process.poll() is None
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "vllm_loaded": state.vllm_engine is not None,
        "llamacpp_loaded": state.llamacpp_process is not None and state.llamacpp_process.poll() is None
    }

@app.get("/")
async def root():
    """フロントエンドのindex.htmlを返す"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Unified LLM Inference Proxy API"}

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down proxy server...")
    
    # Stop llama.cpp process
    if state.llamacpp_process is not None:
        try:
            state.llamacpp_process.send_signal(signal.SIGTERM)
            state.llamacpp_process.wait(timeout=5)
        except:
            state.llamacpp_process.kill()
    
    # Cleanup vLLM
    if state.vllm_engine is not None:
        del state.vllm_engine
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Cleanup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

