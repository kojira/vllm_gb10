# NVIDIAの最適化済みvLLMイメージをベースに使用
FROM nvcr.io/nvidia/vllm:25.09-py3

# llama.cppのビルドに必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# llama.cppのクローンとビルド
WORKDIR /workspace
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="89;90" \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" && \
    cmake --build build --config Release -j$(nproc) --target llama-server

# 必要なPythonライブラリを追加
RUN pip install --no-cache-dir timm einops huggingface_hub[cli] fastapi uvicorn aiohttp

# 作業ディレクトリ
WORKDIR /workspace

# スクリプトとサーバーコードをコピー
COPY scripts/ /workspace/scripts/
COPY proxy_server.py /workspace/proxy_server.py
# frontendはホストからマウントする

# デフォルトのコマンドはdocker-composeで上書きするが、CMDとしてはproxy_serverを指定
CMD ["uvicorn", "proxy_server:app", "--host", "0.0.0.0", "--port", "8080"]
