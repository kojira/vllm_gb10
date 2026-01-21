"""
Pydantic models for API requests and responses
"""
from typing import Optional, List
from pydantic import BaseModel


class LoadModelRequest(BaseModel):
    model_path: str
    engine: str
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192


class CompletionRequest(BaseModel):
    prompt: str
    engine: str
    model: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    session_id: Optional[str] = None
    steps: int = 64  # Diffusionモデル用のsteps（32は品質低下、64-128推奨）


class DownloadModelRequest(BaseModel):
    model_id: str
    filename: Optional[str] = None


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
    title: str
    engine: Optional[str]
    model: Optional[str]
    created_at: str
    updated_at: str
    messages: Optional[List[MessageResponse]] = None


class UnloadModelRequest(BaseModel):
    engine: str

