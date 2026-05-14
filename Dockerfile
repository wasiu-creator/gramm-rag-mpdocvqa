# GraMM-RAG — MP-DocVQA end-to-end pipeline
# Base: CUDA 12.4 + Python 3.11 (Lambda-compatible)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HOME=/app/.cache/huggingface

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        git wget curl build-essential libgl1 libglib2.0-0 \
        poppler-utils tesseract-ocr \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch (CUDA 12.4) ───────────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# ── PyTorch Geometric ─────────────────────────────────────────────────────────
RUN pip install torch-geometric && \
    pip install torch-scatter torch-sparse \
        -f https://data.pyg.org/whl/torch-2.3.0+cu124.html

# ── Copy and install project requirements ─────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# ── spaCy model ───────────────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_lg

# ── Copy project (flat layout: src/ at root alongside notebook) ──────────────
COPY src/                /app/src/
COPY gen_notebook.py     /app/gen_notebook.py
COPY mpdocvqa_e2e.ipynb  /app/mpdocvqa_e2e.ipynb

# ── Create directory structure (data + artefacts mounted as volumes) ──────────
RUN mkdir -p \
    /app/data/mpdocvqa/ocr \
    /app/data/mpdocvqa/images \
    /app/data/mpdocvqa/imdbs \
    /app/parsed/mpdocvqa \
    /app/embeddings/mpdocvqa \
    /app/graphs/mpdocvqa \
    /app/results/models/hgt_mpdocvqa \
    /app/results/models/router_mpdocvqa \
    /app/results/figures \
    /app/.cache/huggingface

WORKDIR /app

# ── Jupyter Lab on port 8888 ─────────────────────────────────────────────────
EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
