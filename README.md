# GraMM-RAG — MP-DocVQA End-to-End Pipeline

Graph-augmented Multimodal RAG with Reward-Based Self-Healing Retrieval,
benchmarked on **MP-DocVQA** (5,187 questions across 927 documents, val split).

**Model:** Llama-3.3-70B-Instruct-Turbo (Together.ai) | **Primary metric:** ANLS

---

## Repository Layout

```
gramm-rag-mpdocvqa/
├── src/                       # Pipeline source (parsing, graph, retrieval, gen, eval)
├── mpdocvqa_e2e.ipynb         # The single notebook — run top-to-bottom
├── gen_notebook.py            # Regenerates the notebook from source
├── Dockerfile                 # CUDA 12.4 + Python 3.11 base image
├── docker-compose.yml         # GPU runtime + data/cache mounts
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API keys
└── README.md
```

> **MP-DocVQA data is NOT in this repo** (imdb_val.npy alone is 241 MB).
> You must supply OCR JSONs, page images, and the imdb numpy file at runtime
> via the volume mounts in `docker-compose.yml`.

---

## Requirements

- GPU with ≥ 16 GB VRAM (A100 recommended)
- Python 3.11
- CUDA 12.4
- `TOGETHER_API_KEY` (required for generation)
- `OPENAI_API_KEY` (optional — enables KG triplet extraction)
- MP-DocVQA dataset (OCR JSONs + page images + `imdb_val.npy`)

---

## Quick Start (Docker — recommended for GPU runs on Lambda / new machine)

```bash
# 1. Clone
git clone https://github.com/wasiu-creator/gramm-rag-mpdocvqa.git
cd gramm-rag-mpdocvqa

# 2. Provide MP-DocVQA data via local volume mounts
mkdir -p data_volume/{ocr,images,imdbs}
cp -r /path/to/mpdocvqa/ocr/*           data_volume/ocr/
cp -r /path/to/mpdocvqa/images/*        data_volume/images/
cp    /path/to/mpdocvqa/imdb_val.npy    data_volume/imdbs/

# 3. Set API keys
cp .env.example .env
# Edit .env and paste your real TOGETHER_API_KEY and OPENAI_API_KEY

# 4. Build and run
docker compose --env-file .env build
docker compose --env-file .env up -d

# 5. Open JupyterLab in browser
#    http://<server-ip>:8888
#    Open mpdocvqa_e2e.ipynb → Kernel → Restart Kernel and Run All Cells
```

---

## Quick Start (Bare Python)

```bash
# 1. Clone and enter
git clone https://github.com/wasiu-creator/gramm-rag-mpdocvqa.git
cd gramm-rag-mpdocvqa

# 2. Create venv
python3.11 -m venv venv && source venv/bin/activate

# 3. Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 4. Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.3.0+cu124.html

# 5. Install remaining dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 6. Set API keys
cp .env.example .env
# Edit .env with real keys

# 7. Provide MP-DocVQA data
mkdir -p data/mpdocvqa/{ocr,images,imdbs}
cp -r /path/to/mpdocvqa/ocr/*           data/mpdocvqa/ocr/
cp -r /path/to/mpdocvqa/images/*        data/mpdocvqa/images/
cp    /path/to/mpdocvqa/imdb_val.npy    data/mpdocvqa/imdbs/

# 8. Launch Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
#    Open mpdocvqa_e2e.ipynb → Kernel → Restart Kernel and Run All Cells
```

---

## Pipeline Overview

| Phase | Description |
|---|---|
| 0   | Setup & configuration (loads `.env` for API keys, detects GPU) |
| 1   | Load 5,187 QA records from `imdb_val.npy` |
| 1.5 | Exploratory data analysis (7 charts) |
| 2   | Verify OCR + image file coverage |
| 3   | Parse documents (Textract OCR) + temporal annotation |
| 4   | Compute E5-Mistral embeddings → PyG graphs |
| 5   | Train HGT (50 epochs, evidence-guided InfoNCE) |
| 6   | Fine-tune DeBERTa-v3 query router (3 epochs) |
| 7   | Tune reward function (α, β, λ, τ grid search) |
| 8   | Flat-vector RAG baseline (FAISS + Llama-3.3-70B) |
| 9   | GraMM-RAG evaluation × 3 seeds (42, 123, 456) |
| 10  | Results comparison table |

**Estimated runtime:** ~6–12 hours on A100 80 GB
**Estimated API cost:** ~$5–10 (Llama-3.3-70B × 3 seeds × 5,187 questions)

---

## Outputs

Written to `results/` (mounted on the host as `cache/results/` when using Docker):

```
results/
├── figures/                            # EDA charts
├── models/hgt_mpdocvqa/best_model.pt
├── models/router_mpdocvqa/
├── models/reward_mpdocvqa.json
├── baseline_vector_mpdocvqa.json
├── gramm_mpdocvqa_s42.json
├── gramm_mpdocvqa_s123.json
├── gramm_mpdocvqa_s456.json
└── summary_mpdocvqa.json
```

---

## Pilot Mode (CPU-friendly smoke test)

For a fast smoke-test (~30 min on CPU, no GPU needed), set in Phase 0 of the notebook:

```python
N_PILOT = 100   # set None for full 5,187-question run
```

Pilot mode skips heavy E5 embedding (random-init fallback) and trains
HGT/router for fewer epochs.
