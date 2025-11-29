# NVIDIAの最適化済みvLLMイメージをベースに使用
FROM nvcr.io/nvidia/vllm:25.09-py3

# Gemma 3nに必要なtimmライブラリ、およびAPIサーバー用ライブラリを追加
RUN pip install --no-cache-dir timm einops huggingface_hub[cli] fastapi uvicorn

# 作業ディレクトリ
WORKDIR /workspace

# スクリプトとサーバーコードをコピー
COPY scripts/ /workspace/scripts/
COPY server.py /workspace/server.py

# デフォルトのコマンドはdocker-composeで上書きするが、CMDとしてはuvicornを指定しておく
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
