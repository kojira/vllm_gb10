"""
llama.cpp engine management
"""
import os
import re
import asyncio
import subprocess
import signal
import time
import json
from typing import Optional, List, Dict
import aiohttp
from fastapi import HTTPException

from .state import state
from .utils import sanitize_utf8
from .database import add_message


async def monitor_llamacpp_output(process):
    """Monitor llama-server stdout for progress information"""
    total_tensors = None
    loaded_tensors = 0
    
    try:
        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None, process.stdout.readline
            )
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Parse total tensors count
            if "loaded meta data" in line and "tensors" in line:
                match = re.search(r'and (\d+) tensors', line)
                if match:
                    total_tensors = int(match.group(1))
                    state.llamacpp_progress = 0.1
                    state.llamacpp_progress_message = f"メタデータ読み込み完了 ({total_tensors}個のテンソル)"
                    print(f"llama.cpp: Found {total_tensors} tensors to load")
            
            # Parse tensor loading progress
            elif "llama_model_load: - tensor" in line:
                match = re.search(r'tensor\s+(\d+):', line)
                if match and total_tensors:
                    loaded_tensors = int(match.group(1)) + 1
                    progress = 0.1 + (loaded_tensors / total_tensors) * 0.8
                    state.llamacpp_progress = progress
                    state.llamacpp_progress_message = f"テンソルロード中 ({loaded_tensors}/{total_tensors})"
            
            # Server ready message
            elif "HTTP server listening" in line or "server is listening" in line:
                state.llamacpp_progress = 0.95
                state.llamacpp_progress_message = "サーバー起動中..."
                print(f"llama.cpp: Server is starting...")
                
    except Exception as e:
        print(f"llama.cpp: Error monitoring output: {e}")


async def load_llamacpp_model(model_path: str):
    """Load a model using llama.cpp server"""
    state.llamacpp_status = "loading"
    state.llamacpp_progress = 0.0
    state.llamacpp_progress_message = "初期化中..."
    
    try:
        # Check if same model already loaded
        if state.llamacpp_process is not None and state.llamacpp_current_model == model_path:
            if state.llamacpp_process.poll() is None:
                print(f"llama.cpp: Model {model_path} already loaded, skipping reload")
                state.llamacpp_status = "loaded"
                state.llamacpp_progress = 1.0
                state.llamacpp_progress_message = "ロード済み"
                return {"status": "success", "message": f"Model already loaded: {model_path}"}
        
        # Stop existing process
        if state.llamacpp_process is not None:
            state.llamacpp_progress = 0.05
            state.llamacpp_progress_message = "既存プロセスを停止中..."
            print(f"llama.cpp: Stopping existing process for model: {state.llamacpp_current_model}")
            state.llamacpp_process.send_signal(signal.SIGTERM)
            # Non-blocking wait with timeout
            for _ in range(100):  # 10 seconds max
                if state.llamacpp_process.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            else:
                state.llamacpp_process.kill()
                await asyncio.sleep(0.1)
            state.llamacpp_process = None
            state.llamacpp_current_model = None
            await asyncio.sleep(2)
        
        # Validate path
        if not os.path.exists(model_path):
            state.llamacpp_status = "error"
            state.llamacpp_progress = 0.0
            state.llamacpp_progress_message = "エラー: モデルが見つかりません"
            raise HTTPException(status_code=400, detail=f"Model path not found: {model_path}")
        
        state.llamacpp_progress = 0.1
        state.llamacpp_progress_message = "llama-serverを起動中..."
        print(f"llama.cpp: Starting server with model {model_path}...")
        
        # Start llama-server process
        cmd = [
            "/workspace/llama.cpp/build/bin/llama-server",
            "--host", "127.0.0.1",
            "--port", str(state.llamacpp_port),
            "--model", model_path,
            "--n-gpu-layers", "-1",
            "-c", "32768",
            "--parallel", "4"
        ]
        
        state.llamacpp_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        state.llamacpp_current_model = model_path
        
        # Start monitoring output in background
        asyncio.create_task(monitor_llamacpp_output(state.llamacpp_process))
        
        # Wait for server to be ready
        max_wait = 600
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < max_wait:
                try:
                    async with session.get(f"http://127.0.0.1:{state.llamacpp_port}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            state.llamacpp_status = "loaded"
                            state.llamacpp_progress = 1.0
                            state.llamacpp_progress_message = "ロード完了"
                            print(f"llama.cpp: Server ready")
                            return {"status": "success", "message": f"Model loaded: {model_path}"}
                except:
                    pass
                
                if state.llamacpp_process.poll() is not None:
                    state.llamacpp_status = "error"
                    try:
                        stdout, stderr = state.llamacpp_process.communicate(timeout=1)
                        error_msg = f"llama-server process died during startup. Output: {stdout[-500:] if stdout else 'N/A'}"
                    except:
                        error_msg = "llama-server process died during startup (no output captured)"
                    print(f"llama.cpp: {error_msg}")
                    state.llamacpp_process = None
                    state.llamacpp_current_model = None
                    raise HTTPException(status_code=500, detail=error_msg)
                
                await asyncio.sleep(1)
        
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


async def unload_llamacpp_model():
    """Unload llama.cpp model by stopping the server process"""
    if state.llamacpp_process is None:
        return {"status": "success", "message": "No llama.cpp model loaded"}
    
    print(f"llama.cpp: Stopping server for model: {state.llamacpp_current_model}")
    try:
        state.llamacpp_process.send_signal(signal.SIGTERM)
        for _ in range(100):
            if state.llamacpp_process.poll() is not None:
                break
            await asyncio.sleep(0.1)
        else:
            state.llamacpp_process.kill()
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"llama.cpp: Error stopping process: {e}")
    
    state.llamacpp_process = None
    state.llamacpp_current_model = None
    state.llamacpp_status = "idle"
    state.llamacpp_progress = 0.0
    state.llamacpp_progress_message = ""
    print("llama.cpp: Server stopped")
    await asyncio.sleep(1)
    return {"status": "success", "message": "llama.cpp model unloaded"}


async def llamacpp_stream_generator(url: str, payload: dict, model_name: str, session_id: Optional[str] = None, user_prompt: Optional[str] = None):
    """Generator for llama.cpp streaming responses using /v1/chat/completions format"""
    full_response = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=f"llama.cpp error: {error_text}")
                async for line in resp.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str == '[DONE]':
                                if full_response:
                                    full_response = sanitize_utf8(full_response)
                                    output_preview = full_response[:200] + "..." if len(full_response) > 200 else full_response
                                    print(f"📤 OUTPUT [llamacpp/stream] len={len(full_response)}: {output_preview}")
                                    if session_id and user_prompt:
                                        add_message(session_id, "user", user_prompt)
                                        add_message(session_id, "assistant", full_response)
                                    full_response = ""
                                yield b"data: [DONE]\n\n"
                                break
                            try:
                                llama_data = json.loads(data_str)
                                choices = llama_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content") or ""
                                    finish_reason = choices[0].get("finish_reason")
                                    
                                    full_response += content
                                    
                                    openai_chunk = {
                                        "id": llama_data.get("id", f"llamacpp-{int(time.time())}"),
                                        "object": "text_completion",
                                        "created": llama_data.get("created", int(time.time())),
                                        "model": model_name,
                                        "choices": [{
                                            "text": content,
                                            "index": 0,
                                            "logprobs": None,
                                            "finish_reason": finish_reason
                                        }]
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
                                    
                                    if finish_reason:
                                        full_response = sanitize_utf8(full_response)
                                        output_preview = full_response[:200] + "..." if len(full_response) > 200 else full_response
                                        print(f"📤 OUTPUT [llamacpp/stream] len={len(full_response)}: {output_preview}")
                                        if session_id and user_prompt:
                                            add_message(session_id, "user", user_prompt)
                                            add_message(session_id, "assistant", full_response)
                                        full_response = ""
                                        yield b"data: [DONE]\n\n"
                            except json.JSONDecodeError:
                                pass
    except aiohttp.ClientConnectorError as e:
        error_msg = f"llama.cpp server not available (is the model loaded?): {str(e)}"
        print(f"ERROR: {error_msg}")
        yield f"data: {json.dumps({'error': error_msg})}\n\n".encode()
    except Exception as e:
        error_msg = f"llama.cpp streaming error: {str(e)}"
        print(f"ERROR: {error_msg}")
        yield f"data: {json.dumps({'error': error_msg})}\n\n".encode()


async def generate_llamacpp_completion(
    prompt: str,
    model: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    session_id: Optional[str],
    all_messages: List[Dict]
):
    """Generate completion using llama.cpp server"""
    if state.llamacpp_process is None or state.llamacpp_process.poll() is not None:
        raise HTTPException(status_code=503, detail="llama.cpp: No model loaded or server died")
    
    if state.llamacpp_status != "loaded":
        raise HTTPException(status_code=503, detail=f"llama.cpp: Engine status is {state.llamacpp_status}")
    
    # Build messages array with conversation history
    messages_for_llama = [{"role": msg["role"], "content": msg["content"]} for msg in all_messages]
    
    payload = {
        "model": model or state.llamacpp_current_model,
        "messages": messages_for_llama,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream
    }
    
    url = f"http://127.0.0.1:{state.llamacpp_port}/v1/chat/completions"
    
    return url, payload

