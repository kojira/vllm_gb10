"""
Shared state management for engine instances
"""
from typing import Optional
import asyncio


class EngineState:
    """Global state for managing vLLM and llama.cpp engines"""
    
    def __init__(self):
        # vLLM state
        self.vllm_engine = None
        self.vllm_current_model: Optional[str] = None
        self.vllm_status: str = "idle"  # idle, loading, loaded, error
        self.vllm_loading_task: Optional[asyncio.Task] = None
        self.vllm_progress: float = 0.0
        self.vllm_progress_message: str = ""
        
        # llama.cpp state
        self.llamacpp_process = None
        self.llamacpp_current_model: Optional[str] = None
        self.llamacpp_status: str = "idle"  # idle, loading, loaded, error
        self.llamacpp_loading_task: Optional[asyncio.Task] = None
        self.llamacpp_port: int = 8002
        self.llamacpp_progress: float = 0.0
        self.llamacpp_progress_message: str = ""
        
        # Transformers state (for Diffusion LLM like ELYZA)
        self.transformers_model = None
        self.transformers_tokenizer = None
        self.transformers_current_model: Optional[str] = None
        self.transformers_status: str = "idle"  # idle, loading, loaded, error
        self.transformers_loading_task: Optional[asyncio.Task] = None
        self.transformers_progress: float = 0.0
        self.transformers_progress_message: str = ""
        
        # Download state
        self.download_status: str = "idle"  # idle, downloading, completed, error
        self.download_progress: float = 0.0
        self.download_message: str = ""
        self.download_model_id: Optional[str] = None


# Global state instance
state = EngineState()

