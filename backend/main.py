"""
FastAPI application for Unified LLM Inference Proxy
"""
import os
import gc
import torch
import asyncio
import signal
import time
import aiohttp
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .models import (
    LoadModelRequest, CompletionRequest, DownloadModelRequest,
    CreateSessionRequest, UnloadModelRequest
)
from .state import state
from .database import (
    init_database, create_session, get_session, list_sessions,
    delete_session, add_message, get_session_messages, update_session_model
)
from .utils import sanitize_utf8, build_vllm_prompt_with_history
from .engine_vllm import (
    load_vllm_model, unload_vllm_model, vllm_stream_generator
)
from .engine_llamacpp import (
    load_llamacpp_model, unload_llamacpp_model, llamacpp_stream_generator
)
from .engine_transformers import (
    load_transformers_model, unload_transformers_model, 
    transformers_generate, transformers_stream_generator
)
from .download import download_model_task, list_downloaded_models

# vLLM imports
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Create FastAPI app
app = FastAPI(title="Unified LLM Inference Proxy (vLLM + llama.cpp)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("=" * 60)
print("🚀 Unified LLM Inference Proxy Server Starting...")
print("   Modular Backend Version 1.0")
print("=" * 60)

# Initialize database on startup
init_database()


# ============== Model Management API ==============

@app.post("/v1/models/load")
async def api_load_model(request: LoadModelRequest):
    """Load a model on the specified engine (non-blocking)"""
    if request.engine not in ["vllm", "llamacpp", "transformers"]:
        raise HTTPException(status_code=400, detail="engine must be 'vllm', 'llamacpp', or 'transformers'")
    
    if request.engine == "vllm":
        if state.vllm_status == "loading":
            return {"status": "already_loading", "message": "vLLM is already loading a model"}
        
        if state.vllm_loading_task and not state.vllm_loading_task.done():
            state.vllm_loading_task.cancel()
        
        state.vllm_loading_task = asyncio.create_task(
            load_vllm_model(request.model_path, request.dtype, request.gpu_memory_utilization, request.max_model_len)
        )
        return {"status": "loading", "message": f"Started loading model: {request.model_path}", "engine": "vllm"}
    
    elif request.engine == "llamacpp":
        if state.llamacpp_status == "loading":
            return {"status": "already_loading", "message": "llama.cpp is already loading a model"}
        
        if state.llamacpp_loading_task and not state.llamacpp_loading_task.done():
            state.llamacpp_loading_task.cancel()
        
        state.llamacpp_loading_task = asyncio.create_task(
            load_llamacpp_model(request.model_path)
        )
        return {"status": "loading", "message": f"Started loading model: {request.model_path}", "engine": "llamacpp"}
    
    else:  # transformers
        if state.transformers_status == "loading":
            return {"status": "already_loading", "message": "Transformers is already loading a model"}
        
        if state.transformers_loading_task and not state.transformers_loading_task.done():
            state.transformers_loading_task.cancel()
        
        state.transformers_loading_task = asyncio.create_task(
            load_transformers_model(request.model_path)
        )
        return {"status": "loading", "message": f"Started loading model: {request.model_path}", "engine": "transformers"}


@app.post("/v1/models/unload")
async def api_unload_model(request: UnloadModelRequest):
    """Unload a model from the specified engine (non-blocking)"""
    if request.engine not in ["vllm", "llamacpp", "transformers"]:
        raise HTTPException(status_code=400, detail="engine must be 'vllm', 'llamacpp', or 'transformers'")
    
    if request.engine == "vllm":
        asyncio.create_task(unload_vllm_model())
        return {"status": "unloading", "message": "Started unloading vLLM model", "engine": "vllm"}
    elif request.engine == "llamacpp":
        asyncio.create_task(unload_llamacpp_model())
        return {"status": "unloading", "message": "Started unloading llama.cpp model", "engine": "llamacpp"}
    else:  # transformers
        asyncio.create_task(unload_transformers_model())
        return {"status": "unloading", "message": "Started unloading Transformers model", "engine": "transformers"}


@app.post("/v1/models/download")
async def api_download_model(request: DownloadModelRequest):
    """Download a model from Hugging Face"""
    if state.download_status == "downloading":
        raise HTTPException(status_code=409, detail="Another download is in progress")
    
    result = await download_model_task(request.model_id, request.filename)
    return result


@app.get("/v1/models/list")
async def api_list_models():
    """List downloaded models"""
    models = list_downloaded_models()
    return {"models": models}


# ============== Completion API ==============

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Generate completions using the specified engine"""
    # Log input
    prompt_preview = request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
    print(f"📥 INPUT [{request.engine}] session={request.session_id or 'none'}: {prompt_preview}")
    
    if request.engine not in ["vllm", "llamacpp", "transformers"]:
        raise HTTPException(status_code=400, detail="engine must be 'vllm', 'llamacpp', or 'transformers'")
    
    # Handle session-based conversation history
    session_messages = []
    if request.session_id:
        session = get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
        session_messages = get_session_messages(request.session_id)
        if request.engine == "vllm":
            current_model = state.vllm_current_model
        elif request.engine == "llamacpp":
            current_model = state.llamacpp_current_model
        else:
            current_model = state.transformers_current_model
        if current_model:
            update_session_model(request.session_id, request.engine, current_model)
    
    all_messages = session_messages + [{"role": "user", "content": request.prompt}]
    
    if request.engine == "vllm":
        return await _handle_vllm_completion(request, all_messages)
    elif request.engine == "llamacpp":
        return await _handle_llamacpp_completion(request, all_messages)
    else:  # transformers
        return await _handle_transformers_completion(request, all_messages)


async def _handle_vllm_completion(request: CompletionRequest, all_messages: list):
    """Handle vLLM completion request"""
    if state.vllm_engine is None:
        raise HTTPException(status_code=503, detail="vLLM: No model loaded. Please load a model first.")
    
    if state.vllm_status != "loaded":
        raise HTTPException(status_code=503, detail=f"vLLM: Engine status is {state.vllm_status}")
    
    formatted_prompt = build_vllm_prompt_with_history(all_messages, state.vllm_current_model)
    
    request_id = random_uuid()
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )
    
    results_generator = state.vllm_engine.generate(formatted_prompt, sampling_params, request_id)
    
    if request.stream:
        return StreamingResponse(
            vllm_stream_generator(request_id, results_generator, request.session_id, request.prompt),
            media_type="text/event-stream"
        )
    else:
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        text_output = sanitize_utf8(final_output.outputs[0].text)
        
        output_preview = text_output[:200] + "..." if len(text_output) > 200 else text_output
        print(f"📤 OUTPUT [vllm] tokens={len(final_output.outputs[0].token_ids)}: {output_preview}")
        
        if request.session_id:
            add_message(request.session_id, "user", request.prompt)
            add_message(request.session_id, "assistant", text_output)
        
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


async def _handle_llamacpp_completion(request: CompletionRequest, all_messages: list):
    """Handle llama.cpp completion request"""
    if state.llamacpp_process is None or state.llamacpp_process.poll() is not None:
        raise HTTPException(status_code=503, detail="llama.cpp: No model loaded or server died")
    
    if state.llamacpp_status != "loaded":
        raise HTTPException(status_code=503, detail=f"llama.cpp: Engine status is {state.llamacpp_status}")
    
    messages_for_llama = [{"role": msg["role"], "content": msg["content"]} for msg in all_messages]
    
    payload = {
        "model": request.model or state.llamacpp_current_model,
        "messages": messages_for_llama,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": request.stream
    }
    
    url = f"http://127.0.0.1:{state.llamacpp_port}/v1/chat/completions"
    
    if request.stream:
        return StreamingResponse(
            llamacpp_stream_generator(url, payload, request.model or state.llamacpp_current_model, request.session_id, request.prompt),
            media_type="text/event-stream"
        )
    else:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise HTTPException(status_code=resp.status, detail=f"llama.cpp error: {text}")
                    
                    result = await resp.json()
                    
                    choices = result.get("choices", [])
                    content = ""
                    finish_reason = "stop"
                    if choices:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        finish_reason = choices[0].get("finish_reason", "stop")
                    
                    content = sanitize_utf8(content)
                    usage = result.get("usage", {})
                    
                    output_preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"📤 OUTPUT [llamacpp] tokens={usage.get('completion_tokens', '?')}: {output_preview}")
                    
                    if request.session_id:
                        add_message(request.session_id, "user", request.prompt)
                        add_message(request.session_id, "assistant", content)
                    
                    return {
                        "id": result.get("id", f"llamacpp-{int(time.time())}"),
                        "object": "text_completion",
                        "created": result.get("created", int(time.time())),
                        "model": request.model or state.llamacpp_current_model,
                        "choices": [
                            {
                                "text": content,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": finish_reason
                            }
                        ],
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    }
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=503, detail=f"llama.cpp communication error: {str(e)}")


async def _handle_transformers_completion(request: CompletionRequest, all_messages: list):
    """Handle Transformers completion request (for Diffusion LLMs)"""
    if state.transformers_model is None:
        raise HTTPException(status_code=503, detail="Transformers: No model loaded. Please load a model first.")
    
    if state.transformers_status != "loaded":
        raise HTTPException(status_code=503, detail=f"Transformers: Engine status is {state.transformers_status}")
    
    # Build prompt from messages (simple concatenation for now)
    prompt = all_messages[-1]["content"] if all_messages else request.prompt
    
    if request.stream:
        return StreamingResponse(
            transformers_stream_generator(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                session_id=request.session_id,
                user_prompt=request.prompt,
                steps=request.steps
            ),
            media_type="text/event-stream"
        )
    else:
        try:
            result = await transformers_generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                steps=request.steps
            )
            
            text_output = result["text"]
            output_preview = text_output[:200] + "..." if len(text_output) > 200 else text_output
            print(f"📤 OUTPUT [transformers] tokens={result['tokens']}, {result['tps']:.2f} TPS: {output_preview}")
            
            if request.session_id:
                add_message(request.session_id, "user", request.prompt)
                add_message(request.session_id, "assistant", text_output)
            
            return {
                "id": f"transformers-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model or state.transformers_current_model,
                "choices": [
                    {
                        "text": text_output,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # Not tracked for transformers
                    "completion_tokens": result["tokens"],
                    "total_tokens": result["tokens"]
                },
                "tps": result["tps"]
            }
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ Transformers generation error: {error_detail}")
            raise HTTPException(status_code=500, detail=f"Transformers generation error: {str(e)}\n{error_detail}")


# ============== Session Management API ==============

@app.post("/v1/sessions")
async def api_create_session(request: CreateSessionRequest):
    """Create a new chat session"""
    session_id = create_session(request.title, request.engine, request.model)
    return {"id": session_id, "message": "Session created"}


@app.get("/v1/sessions")
async def api_list_sessions(limit: int = 50):
    """List all sessions"""
    sessions = list_sessions(limit)
    return {"sessions": sessions}


@app.get("/v1/sessions/{session_id}")
async def api_get_session(session_id: str):
    """Get a session with its messages"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session["messages"] = get_session_messages(session_id)
    return session


@app.delete("/v1/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """Delete a session"""
    if delete_session(session_id):
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# ============== Status API ==============

@app.get("/v1/status")
async def get_status():
    """Get the status of all engines with progress information"""
    return {
        "vllm": {
            "status": state.vllm_status,
            "model": state.vllm_current_model,
            "progress": state.vllm_progress,
            "progress_message": state.vllm_progress_message
        },
        "llamacpp": {
            "status": state.llamacpp_status,
            "model": state.llamacpp_current_model,
            "process_alive": state.llamacpp_process is not None and state.llamacpp_process.poll() is None,
            "progress": state.llamacpp_progress,
            "progress_message": state.llamacpp_progress_message
        },
        "transformers": {
            "status": state.transformers_status,
            "model": state.transformers_current_model,
            "progress": state.transformers_progress,
            "progress_message": state.transformers_progress_message
        },
        "download": {
            "status": state.download_status,
            "model_id": state.download_model_id,
            "progress": state.download_progress,
            "message": state.download_message
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "vllm_loaded": state.vllm_engine is not None,
        "llamacpp_loaded": state.llamacpp_process is not None and state.llamacpp_process.poll() is None,
        "transformers_loaded": state.transformers_model is not None
    }


# ============== Frontend ==============

@app.get("/")
async def root():
    """Serve frontend index.html"""
    # Try multiple possible paths
    possible_paths = [
        "/workspace/frontend/dist/index.html",  # React build
        "/workspace/frontend/index.html",  # Development
        os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"),  # Relative
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return FileResponse(path)
    
    return {"message": "Unified LLM Inference Proxy API", "docs": "/docs"}


# ============== Shutdown ==============

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down proxy server...")
    
    if state.llamacpp_process is not None:
        try:
            state.llamacpp_process.send_signal(signal.SIGTERM)
            state.llamacpp_process.wait(timeout=5)
        except:
            state.llamacpp_process.kill()
    
    if state.vllm_engine is not None:
        del state.vllm_engine
        gc.collect()
        torch.cuda.empty_cache()
    
    if state.transformers_model is not None:
        del state.transformers_model
        del state.transformers_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Cleanup complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

