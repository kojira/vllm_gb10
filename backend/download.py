"""
Model download functionality
"""
import os
import re
import asyncio
import aiohttp
from typing import Optional
from fastapi import HTTPException

from .state import state


async def download_model_task(model_id: str, filename: Optional[str] = None):
    """Download a model from Hugging Face
    
    Args:
        model_id: Hugging Face model ID (e.g., "google/gemma-3n-E2B-it")
        filename: Optional specific file to download (for GGUF models, e.g., "model-Q4_K_M.gguf")
    """
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
            if not filename.endswith('.gguf'):
                if os.path.isdir(target_path):
                    gguf_files = [f for f in os.listdir(target_path) if f.endswith('.gguf')]
                    if gguf_files:
                        state.download_status = "completed"
                        state.download_progress = 1.0
                        state.download_message = f"既にダウンロード済み: {filename} ({len(gguf_files)}ファイル)"
                        return {"status": "success", "message": f"Already downloaded: {filename}", "path": target_path}
            else:
                if os.path.exists(target_path):
                    state.download_status = "completed"
                    state.download_progress = 1.0
                    state.download_message = f"既にダウンロード済み: {filename}"
                    return {"status": "success", "message": f"Already downloaded: {filename}", "path": target_path}
        else:
            if os.path.exists(local_path) and os.listdir(local_path):
                if os.path.exists(os.path.join(local_path, "config.json")):
                    state.download_status = "completed"
                    state.download_progress = 1.0
                    state.download_message = f"既にダウンロード済み: {model_name}"
                    return {"status": "success", "message": f"Already downloaded: {model_name}", "path": local_path}
        
        state.download_message = f"ダウンロード中: {model_id}" + (f" ({filename})" if filename else "")
        state.download_progress = 0.1
        
        # hf download を使ってダウンロード
        cmd = ["hf", "download", model_id]
        
        if filename:
            if not filename.endswith('.gguf'):
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
                cmd.append(filename)
        
        cmd.extend(["--local-dir", local_path])
        
        print(f"Download command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            line_text = line.decode().strip()
            print(f"Download: {line_text}")
            
            if "Downloading" in line_text or "Fetching" in line_text:
                state.download_progress = min(state.download_progress + 0.05, 0.9)
                state.download_message = line_text[:80]
            elif "%" in line_text:
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


def list_downloaded_models():
    """List downloaded models"""
    models_dir = "/workspace/models"
    models = []
    
    if os.path.exists(models_dir):
        for name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, name)
            if os.path.isdir(model_path):
                gguf_files = []
                for item in os.listdir(model_path):
                    item_path = os.path.join(model_path, item)
                    if item.endswith('.gguf'):
                        gguf_files.append(item)
                    elif os.path.isdir(item_path) and not item.startswith('.'):
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
    
    return models

