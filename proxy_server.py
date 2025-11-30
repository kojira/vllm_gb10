import os
import gc
import torch
import asyncio
import subprocess
import signal
import time
import json
import re
import uuid
import sqlite3
from datetime import datetime
from contextlib import contextmanager
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

print("=" * 60)
print("🚀 Unified LLM Inference Proxy Server Starting...")
print("   Hot-reload test: Version 5.0 - Session History Support!")
print("=" * 60)

# SQLite Database Setup
DB_PATH = "/workspace/data/chat_history.db"

def init_database():
    """Initialize SQLite database with sessions and messages tables"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            engine TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    
    # Index for faster session lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
    
    conn.commit()
    conn.close()
    print(f"📦 Database initialized: {DB_PATH}")

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Initialize database on startup
init_database()

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
    vllm_progress: float = 0.0  # 0.0 to 1.0
    vllm_progress_message: str = ""
    
    # llama.cpp state
    llamacpp_process: Optional[subprocess.Popen] = None
    llamacpp_current_model: Optional[str] = None
    llamacpp_status: str = "idle"  # idle/loading/loaded/error
    llamacpp_port: int = 8002
    llamacpp_loading_task: Optional[asyncio.Task] = None
    llamacpp_progress: float = 0.0  # 0.0 to 1.0
    llamacpp_progress_message: str = ""
    
    # Download state
    download_status: str = "idle"  # idle/downloading/completed/error
    download_progress: float = 0.0
    download_message: str = ""
    download_model_id: Optional[str] = None

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
    session_id: Optional[str] = None  # Optional: session ID for conversation history

class DownloadModelRequest(BaseModel):
    model_id: str  # Hugging Face model ID (e.g., "google/gemma-3n-E2B-it")
    filename: Optional[str] = None  # Optional: specific file to download (for GGUF models)

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    engine: Optional[str] = None
    model: Optional[str] = None

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: str

class SessionResponse(BaseModel):
    id: str
    title: Optional[str]
    engine: Optional[str]
    model: Optional[str]
    created_at: str
    updated_at: str
    messages: Optional[List[MessageResponse]] = None

# Session Management Functions
def create_session(title: Optional[str] = None, engine: Optional[str] = None, model: Optional[str] = None) -> str:
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, title, engine, model) VALUES (?, ?, ?, ?)",
            (session_id, title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}", engine, model)
        )
        conn.commit()
    return session_id

def get_session(session_id: str) -> Optional[Dict]:
    """Get session by ID with messages"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        if not session:
            return None
        
        cursor.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        messages = cursor.fetchall()
        
        return {
            "id": session["id"],
            "title": session["title"],
            "engine": session["engine"],
            "model": session["model"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "messages": [dict(m) for m in messages]
        }

def list_sessions(limit: int = 50) -> List[Dict]:
    """List all sessions ordered by updated_at"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, engine, model, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        )
        sessions = cursor.fetchall()
        return [dict(s) for s in sessions]

def delete_session(session_id: str) -> bool:
    """Delete a session and its messages"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0

def add_message(session_id: str, role: str, content: str) -> int:
    """Add a message to a session"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        # Update session's updated_at
        cursor.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )
        conn.commit()
        return cursor.lastrowid

def get_session_messages(session_id: str) -> List[Dict]:
    """Get all messages for a session in order"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

def update_session_model(session_id: str, engine: str, model: str):
    """Update session's engine and model"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET engine = ?, model = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (engine, model, session_id)
        )
        conn.commit()

def sanitize_utf8(text: str) -> str:
    """Remove incomplete UTF-8 characters from the end of text"""
    if not text:
        return text
    
    # Encode to bytes and check for incomplete sequences at the end
    try:
        encoded = text.encode('utf-8')
    except UnicodeEncodeError:
        # If encoding fails, try to recover
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Check last few bytes for incomplete multi-byte sequences
    valid_length = len(encoded)
    
    for i in range(min(4, len(encoded))):
        idx = len(encoded) - 1 - i
        byte = encoded[idx]
        
        # 4-byte sequence start (11110xxx)
        if (byte & 0xF8) == 0xF0:
            if i < 3:  # Not enough bytes for complete sequence
                valid_length = idx
            break
        # 3-byte sequence start (1110xxxx)
        elif (byte & 0xF0) == 0xE0:
            if i < 2:
                valid_length = idx
            break
        # 2-byte sequence start (110xxxxx)
        elif (byte & 0xE0) == 0xC0:
            if i < 1:
                valid_length = idx
            break
        # Continuation byte (10xxxxxx) - keep looking
        elif (byte & 0xC0) == 0x80:
            continue
        # ASCII or valid start - we're good
        else:
            break
    
    if valid_length < len(encoded):
        return encoded[:valid_length].decode('utf-8', errors='ignore')
    
    return text


def apply_chat_template(prompt: str, model_path: str) -> str:
    """Apply appropriate chat template based on model type"""
    model_name_lower = model_path.lower()
    
    # Gemma format
    if "gemma" in model_name_lower:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Qwen format (if instruct)
    elif "qwen" in model_name_lower and "instruct" in model_name_lower:
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Swallow/TinySwallow format (alpaca-style)
    elif "swallow" in model_name_lower and "instruct" in model_name_lower:
        return f"### 指示:\n{prompt}\n\n### 応答:\n"
    
    # Default: no template (for base models)
    else:
        return prompt


def build_vllm_prompt_with_history(messages: List[Dict], model_path: str) -> str:
    """Build a prompt with conversation history for vLLM"""
    model_name_lower = model_path.lower()
    
    # Gemma format
    if "gemma" in model_name_lower:
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>")
        prompt_parts.append("<start_of_turn>model\n")
        return "\n".join(prompt_parts)
    
    # Qwen format (ChatML)
    elif "qwen" in model_name_lower and "instruct" in model_name_lower:
        prompt_parts = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>"]
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg['content']}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    # Swallow/TinySwallow format (alpaca-style) - doesn't support multi-turn well
    elif "swallow" in model_name_lower and "instruct" in model_name_lower:
        # For Swallow, we'll concatenate the conversation
        conversation = ""
        for msg in messages:
            if msg["role"] == "user":
                conversation += f"ユーザー: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation += f"アシスタント: {msg['content']}\n"
        return f"### 指示:\n以下の会話を続けてください。\n\n{conversation}\n### 応答:\n"
    
    # Default: simple concatenation
    else:
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)


def _load_vllm_model_sync(model_path: str, dtype: str, gpu_memory_utilization: float, max_model_len: int):
    """Synchronous vLLM model loading (runs in thread pool)"""
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
    global state
    
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

async def monitor_llamacpp_output(process):
    """Monitor llama-server stdout for progress information"""
    global state
    
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
            # Example: "llama_model_load: loaded meta data with 20 key-value pairs and 291 tensors"
            if "loaded meta data" in line and "tensors" in line:
                match = re.search(r'and (\d+) tensors', line)
                if match:
                    total_tensors = int(match.group(1))
                    state.llamacpp_progress = 0.1
                    state.llamacpp_progress_message = f"メタデータ読み込み完了 ({total_tensors}個のテンソル)"
                    print(f"llama.cpp: Found {total_tensors} tensors to load")
            
            # Parse tensor loading progress
            # Example: "llama_model_load: - tensor    0:                token_embd.weight q8_0"
            elif "llama_model_load: - tensor" in line:
                match = re.search(r'tensor\s+(\d+):', line)
                if match and total_tensors:
                    loaded_tensors = int(match.group(1)) + 1
                    progress = 0.1 + (loaded_tensors / total_tensors) * 0.8  # 10% to 90%
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
    global state
    
    state.llamacpp_status = "loading"
    state.llamacpp_progress = 0.0
    state.llamacpp_progress_message = "初期化中..."
    
    try:
        # Check if same model already loaded
        if state.llamacpp_process is not None and state.llamacpp_current_model == model_path:
            # Check if process is still running
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
                # Force kill if still running
                state.llamacpp_process.kill()
                await asyncio.sleep(0.1)
            state.llamacpp_process = None
            state.llamacpp_current_model = None
            # Wait a bit for port to be released
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
            "-c", "32768",  # コンテキストサイズ (32K)
            "--parallel", "4"  # 並列数を減らす（n_ctx/parallel = 8192/スロット）
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
        max_wait = 600  # 10 minutes (large models like 120B need more time)
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
                
                # Check if process died
                if state.llamacpp_process.poll() is not None:
                    state.llamacpp_status = "error"
                    # Capture output for debugging
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

async def unload_vllm_model():
    """Unload vLLM model and free GPU memory"""
    global state
    
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

async def unload_llamacpp_model():
    """Unload llama.cpp model by stopping the server process"""
    global state
    
    if state.llamacpp_process is None:
        return {"status": "success", "message": "No llama.cpp model loaded"}
    
    print(f"llama.cpp: Stopping server for model: {state.llamacpp_current_model}")
    try:
        state.llamacpp_process.send_signal(signal.SIGTERM)
        # Non-blocking wait
        for _ in range(100):  # 10 seconds max
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
    await asyncio.sleep(1)  # Wait for port to be released
    return {"status": "success", "message": "llama.cpp model unloaded"}

class UnloadModelRequest(BaseModel):
    engine: str

@app.post("/v1/models/unload")
async def unload_model(request: UnloadModelRequest):
    """Unload a model from the specified engine (non-blocking)"""
    
    if request.engine not in ["vllm", "llamacpp"]:
        raise HTTPException(status_code=400, detail="engine must be 'vllm' or 'llamacpp'")
    
    if request.engine == "vllm":
        # Run unload in background
        asyncio.create_task(unload_vllm_model())
        return {"status": "unloading", "message": "Started unloading vLLM model", "engine": "vllm"}
    else:  # llamacpp
        # Run unload in background
        asyncio.create_task(unload_llamacpp_model())
        return {"status": "unloading", "message": "Started unloading llama.cpp model", "engine": "llamacpp"}

async def download_model_task(model_id: str, filename: Optional[str] = None):
    """Download a model from Hugging Face
    
    Args:
        model_id: Hugging Face model ID (e.g., "google/gemma-3n-E2B-it")
        filename: Optional specific file to download (for GGUF models, e.g., "model-Q4_K_M.gguf")
    """
    global state
    
    state.download_status = "downloading"
    state.download_progress = 0.0
    state.download_model_id = model_id
    state.download_message = f"ダウンロード開始: {model_id}"
    
    try:
        # モデル名からローカルパスを生成
        model_name = model_id.split("/")[-1]
        local_path = f"/workspace/models/{model_name}"
        
        # 特定ファイルのダウンロードの場合、そのファイルが存在するかチェック
        if filename:
            target_path = os.path.join(local_path, filename)
            # ディレクトリ指定の場合は、その中に.ggufファイルがあるかチェック
            if not filename.endswith('.gguf'):
                if os.path.isdir(target_path):
                    gguf_files = [f for f in os.listdir(target_path) if f.endswith('.gguf')]
                    if gguf_files:
                        state.download_status = "completed"
                        state.download_progress = 1.0
                        state.download_message = f"既にダウンロード済み: {filename} ({len(gguf_files)}ファイル)"
                        return {"status": "success", "message": f"Already downloaded: {filename}", "path": target_path}
            else:
                # 単一ファイル指定の場合
                if os.path.exists(target_path):
                    state.download_status = "completed"
                    state.download_progress = 1.0
                    state.download_message = f"既にダウンロード済み: {filename}"
                    return {"status": "success", "message": f"Already downloaded: {filename}", "path": target_path}
        else:
            # 全体ダウンロードの場合
            if os.path.exists(local_path) and os.listdir(local_path):
                # config.jsonがあれば完全なモデルとみなす
                if os.path.exists(os.path.join(local_path, "config.json")):
                    state.download_status = "completed"
                    state.download_progress = 1.0
                    state.download_message = f"既にダウンロード済み: {model_name}"
                    return {"status": "success", "message": f"Already downloaded: {model_name}", "path": local_path}
        
        state.download_message = f"ダウンロード中: {model_id}" + (f" ({filename})" if filename else "")
        state.download_progress = 0.1
        
        # hf download を使ってダウンロード（huggingface-cli downloadは非推奨）
        cmd = ["hf", "download", model_id]
        
        # 特定ファイル/ディレクトリの指定がある場合
        if filename:
            if not filename.endswith('.gguf'):
                # ディレクトリ指定の場合、HF APIからファイル一覧を取得
                state.download_message = f"ファイル一覧を取得中: {filename}"
                try:
                    async with aiohttp.ClientSession() as session:
                        api_url = f"https://huggingface.co/api/models/{model_id}/tree/main/{filename}"
                        async with session.get(api_url) as resp:
                            if resp.status == 200:
                                files_info = await resp.json()
                                gguf_files = [f["path"] for f in files_info if f["path"].endswith(".gguf")]
                                if gguf_files:
                                    cmd.extend(gguf_files)
                                else:
                                    raise HTTPException(status_code=404, detail=f"No .gguf files found in {filename}")
                            else:
                                raise HTTPException(status_code=resp.status, detail=f"Failed to list files in {filename}")
                except aiohttp.ClientError as e:
                    raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
            else:
                # 単一ファイル指定
                cmd.append(filename)
        
        cmd.extend(["--local-dir", local_path])
        
        print(f"Download command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # 出力を監視してプログレスを更新
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            line_text = line.decode().strip()
            print(f"Download: {line_text}")
            
            # プログレスを推定（ファイル数やサイズから）
            if "Downloading" in line_text or "Fetching" in line_text:
                state.download_progress = min(state.download_progress + 0.05, 0.9)
                state.download_message = line_text[:80]
            elif "%" in line_text:
                # パーセンテージを抽出
                match = re.search(r'(\d+)%', line_text)
                if match:
                    percent = int(match.group(1))
                    state.download_progress = percent / 100.0
                    state.download_message = line_text[:80]
        
        await process.wait()
        
        if process.returncode == 0:
            state.download_status = "completed"
            state.download_progress = 1.0
            result_path = os.path.join(local_path, filename) if filename else local_path
            state.download_message = f"ダウンロード完了: {filename or model_name}"
            return {"status": "success", "message": f"Downloaded: {filename or model_name}", "path": result_path}
        else:
            state.download_status = "error"
            state.download_progress = 0.0
            state.download_message = f"ダウンロード失敗: {model_id}"
            raise HTTPException(status_code=500, detail=f"Download failed for {model_id}")
            
    except Exception as e:
        state.download_status = "error"
        state.download_progress = 0.0
        state.download_message = f"エラー: {str(e)}"
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/models/download")
async def download_model(request: DownloadModelRequest):
    """Download a model from Hugging Face
    
    For GGUF models, specify the filename parameter to download a specific quantization.
    Example: model_id="bartowski/openai_gpt-oss-120b-GGUF", filename="openai_gpt-oss-120b-Q4_K_M.gguf"
    """
    
    # 既にダウンロード中なら拒否
    if state.download_status == "downloading":
        raise HTTPException(status_code=409, detail="Another download is in progress")
    
    # バックグラウンドでダウンロード開始
    result = await download_model_task(request.model_id, request.filename)
    return result

@app.get("/v1/models/list")
async def list_models():
    """List downloaded models"""
    models_dir = "/workspace/models"
    models = []
    
    if os.path.exists(models_dir):
        for name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, name)
            if os.path.isdir(model_path):
                # GGUFファイルがあるかチェック（直下とサブディレクトリ）
                gguf_files = []
                for item in os.listdir(model_path):
                    item_path = os.path.join(model_path, item)
                    if item.endswith('.gguf'):
                        gguf_files.append(item)
                    elif os.path.isdir(item_path) and not item.startswith('.'):
                        # サブディレクトリ内のGGUFファイルもチェック
                        for subitem in os.listdir(item_path):
                            if subitem.endswith('.gguf'):
                                gguf_files.append(f"{item}/{subitem}")
                
                model_type = "gguf" if gguf_files else "transformers"
                models.append({
                    "name": name,
                    "path": model_path,
                    "type": model_type,
                    "gguf_files": gguf_files if gguf_files else None
                })
    
    return {"models": models}

@app.post("/v1/models/load")
async def load_model(request: LoadModelRequest):
    """Load a model on the specified engine (non-blocking)"""
    
    if request.engine not in ["vllm", "llamacpp"]:
        raise HTTPException(status_code=400, detail="engine must be 'vllm' or 'llamacpp'")
    
    if request.engine == "vllm":
        # Check if already loading
        if state.vllm_status == "loading":
            return {"status": "already_loading", "message": "vLLM is already loading a model"}
        
        # Cancel any ongoing loading task
        if state.vllm_loading_task and not state.vllm_loading_task.done():
            state.vllm_loading_task.cancel()
        
        # Start loading in background (non-blocking)
        state.vllm_loading_task = asyncio.create_task(
            load_vllm_model(request.model_path, request.dtype, request.gpu_memory_utilization, request.max_model_len)
        )
        
        # Return immediately - client should poll /v1/status
        return {"status": "loading", "message": f"Started loading model: {request.model_path}", "engine": "vllm"}
        
    else:  # llamacpp
        # Check if already loading
        if state.llamacpp_status == "loading":
            return {"status": "already_loading", "message": "llama.cpp is already loading a model"}
        
        # Cancel any ongoing loading task
        if state.llamacpp_loading_task and not state.llamacpp_loading_task.done():
            state.llamacpp_loading_task.cancel()
        
        # Start loading in background (non-blocking)
        state.llamacpp_loading_task = asyncio.create_task(
            load_llamacpp_model(request.model_path)
        )
        
        # Return immediately - client should poll /v1/status
        return {"status": "loading", "message": f"Started loading model: {request.model_path}", "engine": "llamacpp"}

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
                                # [DONE]で終了する場合のみログ（finish_reasonで終了した場合は既にログ済み）
                                if full_response:
                                    full_response = sanitize_utf8(full_response)  # サニタイズ
                                    output_preview = full_response[:200] + "..." if len(full_response) > 200 else full_response
                                    print(f"📤 OUTPUT [llamacpp/stream] len={len(full_response)}: {output_preview}")
                                    # Save to session after streaming completes
                                    if session_id and user_prompt:
                                        add_message(session_id, "user", user_prompt)
                                        add_message(session_id, "assistant", full_response)
                                    full_response = ""  # 重複防止
                                yield b"data: [DONE]\n\n"
                                break
                            try:
                                llama_data = json.loads(data_str)
                                # /v1/chat/completions format: extract delta content
                                choices = llama_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content") or ""  # Handle None
                                    finish_reason = choices[0].get("finish_reason")
                                    
                                    # Accumulate full response
                                    full_response += content
                                    
                                    # Convert to our standard format
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
                                        # finish_reasonで終了する場合 - サニタイズして保存
                                        full_response = sanitize_utf8(full_response)
                                        output_preview = full_response[:200] + "..." if len(full_response) > 200 else full_response
                                        print(f"📤 OUTPUT [llamacpp/stream] len={len(full_response)}: {output_preview}")
                                        # Save to session after streaming completes
                                        if session_id and user_prompt:
                                            add_message(session_id, "user", user_prompt)
                                            add_message(session_id, "assistant", full_response)
                                        full_response = ""  # 重複防止
                                        yield b"data: [DONE]\n\n"
                            except json.JSONDecodeError:
                                pass
    except aiohttp.ClientConnectorError as e:
        error_msg = f"llama.cpp server not available (is the model loaded?): {str(e)}"
        print(f"ERROR: {error_msg}")
        # Send error as SSE
        yield f"data: {json.dumps({'error': error_msg})}\n\n".encode()
    except Exception as e:
        error_msg = f"llama.cpp streaming error: {str(e)}"
        print(f"ERROR: {error_msg}")
        yield f"data: {json.dumps({'error': error_msg})}\n\n".encode()

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Generate completions using the specified engine"""
    
    # Log input
    prompt_preview = request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
    print(f"📥 INPUT [{request.engine}] session={request.session_id or 'none'}: {prompt_preview}")
    
    if request.engine not in ["vllm", "llamacpp"]:
        raise HTTPException(status_code=400, detail="engine must be 'vllm' or 'llamacpp'")
    
    # Handle session-based conversation history
    session_messages = []
    if request.session_id:
        session = get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
        session_messages = get_session_messages(request.session_id)
        # Update session with current engine/model
        current_model = state.vllm_current_model if request.engine == "vllm" else state.llamacpp_current_model
        if current_model:
            update_session_model(request.session_id, request.engine, current_model)
    
    # Add current user message to history
    all_messages = session_messages + [{"role": "user", "content": request.prompt}]
    
    if request.engine == "vllm":
        # Use vLLM
        if state.vllm_engine is None:
            raise HTTPException(status_code=503, detail="vLLM: No model loaded. Please load a model first.")
        
        if state.vllm_status != "loaded":
            raise HTTPException(status_code=503, detail=f"vLLM: Engine status is {state.vllm_status}")
        
        # Build prompt with conversation history for vLLM
        formatted_prompt = build_vllm_prompt_with_history(all_messages, state.vllm_current_model)
        
        request_id = random_uuid()
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        results_generator = state.vllm_engine.generate(formatted_prompt, sampling_params, request_id)
        
        if request.stream:
            # Streaming response
            return StreamingResponse(
                vllm_stream_generator(request_id, results_generator, request.session_id, request.prompt),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output is None:
                raise HTTPException(status_code=500, detail="Generation failed")
            
            # Sanitize output to remove incomplete UTF-8 characters
            text_output = sanitize_utf8(final_output.outputs[0].text)
            
            # Log output
            output_preview = text_output[:200] + "..." if len(text_output) > 200 else text_output
            print(f"📤 OUTPUT [vllm] tokens={len(final_output.outputs[0].token_ids)}: {output_preview}")
            
            # Save messages to session if session_id is provided
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
    
    else:  # llamacpp
        # Forward to llama.cpp server
        if state.llamacpp_process is None or state.llamacpp_process.poll() is not None:
            raise HTTPException(status_code=503, detail="llama.cpp: No model loaded or server died")
        
        if state.llamacpp_status != "loaded":
            raise HTTPException(status_code=503, detail=f"llama.cpp: Engine status is {state.llamacpp_status}")
        
        # Use /v1/chat/completions endpoint to leverage GGUF's embedded chat template
        # Build messages array with conversation history
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
            # Streaming response
            return StreamingResponse(
                llamacpp_stream_generator(url, payload, request.model or state.llamacpp_current_model, request.session_id, request.prompt),
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
                        
                        # Parse /v1/chat/completions response format
                        choices = result.get("choices", [])
                        content = ""
                        finish_reason = "stop"
                        if choices:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")
                            finish_reason = choices[0].get("finish_reason", "stop")
                        
                        # Sanitize output to remove incomplete UTF-8 characters
                        content = sanitize_utf8(content)
                        
                        usage = result.get("usage", {})
                        
                        # Log output
                        output_preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"📤 OUTPUT [llamacpp] tokens={usage.get('completion_tokens', '?')}: {output_preview}")
                        
                        # Save messages to session if session_id is provided
                        if request.session_id:
                            add_message(request.session_id, "user", request.prompt)
                            add_message(request.session_id, "assistant", content)
                        
                        # Convert to our standard format
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
    return session

@app.delete("/v1/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """Delete a session"""
    if delete_session(session_id):
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/v1/status")
async def get_status():
    """Get the status of both engines with progress information"""
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


