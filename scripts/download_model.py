import os
import sys
import argparse
from huggingface_hub import snapshot_download

def download_model(model_id, local_dir):
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN environment variable is not set. Public models may work, but gated models will fail.")

    print(f"Starting download of {model_id}...")
    print(f"Destination: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=token,
        )
        print(f"Download completed successfully: {local_dir}")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face model")
    parser.add_argument("--model", type=str, default="google/gemma-3n-E2B-it", help="Hugging Face Model ID")
    # モデルIDの最後の部分をディレクトリ名にする
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    local_path = f"/workspace/models/{model_name}"
    
    download_model(args.model, local_path)
