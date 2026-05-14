"""
src/graph/embeddings.py
───────────────────────
Node embedding computation for GraMM-RAG.

Step 4: Embed all text nodes with E5-Mistral-7B (4-bit quantized, ~4GB VRAM)
        and all figure/image nodes with SigLIP-SO400M (~1.5GB VRAM).

⚠ Sequential processing is REQUIRED on RTX 5070 (8GB VRAM):
   1. Load E5-Mistral → embed all text → unload → save
   2. Load SigLIP     → embed all images → unload → save
   3. Load projection layers (tiny) → project to 256-dim → save

Both models running simultaneously would exceed 8GB VRAM.
"""

import os
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECTION_DIM = 256   # all nodes projected to this common dimension


# ─── Linear projection layers ──────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Single-layer linear projection to PROJECTION_DIM."""
    def __init__(self, in_dim: int, out_dim: int = PROJECTION_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ─── Text embedding (E5-Mistral-7B 4-bit or E5-large-v2 fallback) ────────────

def load_text_model(use_4bit: bool = True, device: str = "cuda"):
    """
    Load E5-Mistral-7B-Instruct with 4-bit quantization (~4GB VRAM).
    Falls back to E5-large-v2 (~1.3GB) if 4-bit loading fails.
    """
    try:
        from sentence_transformers import SentenceTransformer
        if use_4bit:
            logger.info("Loading E5-Mistral-7B-Instruct (4-bit quantized)")
            model = SentenceTransformer(
                "intfloat/e5-mistral-7b-instruct",
                model_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                },
                device=device,
            )
            embed_dim = 4096
        else:
            raise RuntimeError("Skipping 4-bit, using fallback.")
    except Exception as e:
        logger.warning(f"E5-Mistral 4-bit failed ({e}). Using E5-large-v2 fallback.")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/e5-large-v2", device=device)
        embed_dim = 1024

    return model, embed_dim


def embed_text_nodes(
    elements: list,
    model,
    batch_size: int = 16,   # ← EXTEND: increase to 32+ when VRAM allows
    device: str = "cuda",
) -> torch.Tensor:
    """
    Embed all text-type elements.

    Args:
        elements:   List of parsed element dicts (from mineru_wrapper).
        model:      Loaded SentenceTransformer model.
        batch_size: Inference batch size.
                    # LIMIT: set small (16) to fit VRAM during dev.
                    # ← EXTEND: increase for faster throughput in full runs.

    Returns:
        Tensor of shape [N_text, embed_dim].
    """
    text_elems = [e for e in elements if e["type"] in ("text", "section", "equation")]
    texts = []
    for e in text_elems:
        # E5-Mistral expects "Instruct: <task>\nQuery: <text>" format
        texts.append(f"Represent this document passage: {e['text']}")

    if not texts:
        return torch.zeros(0, PROJECTION_DIM)

    logger.info(f"Embedding {len(texts)} text nodes (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return embeddings.cpu()


# ─── Image embedding (SigLIP-SO400M) ──────────────────────────────────────────

def load_image_model(device: str = "cuda"):
    """
    Load SigLIP-SO400M (~1.5GB VRAM). Safe to run on RTX 5070 alone.
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("Run: pip install open-clip-torch")

    logger.info("Loading SigLIP-SO400M (ViT-SO400M-14-SigLIP)")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP", pretrained="webli"
    )
    model = model.to(device).eval()
    embed_dim = 1152   # SigLIP-SO400M output dimension
    return model, preprocess, embed_dim


def embed_image_nodes(
    elements: list,
    model,
    preprocess,
    image_root: str,
    batch_size: int = 8,    # ← EXTEND: increase when VRAM allows
    device: str = "cuda",
) -> torch.Tensor:
    """
    Embed all figure/table elements using their extracted image crops.

    Args:
        elements:   List of parsed element dicts.
        model:      Loaded SigLIP model.
        preprocess: SigLIP image transform.
        image_root: Directory containing image crops (MinerU artifact dir).
        batch_size: Inference batch size.
                    # LIMIT: 8 to fit VRAM.
                    # ← EXTEND: increase for faster throughput.

    Returns:
        Tensor of shape [N_figure, embed_dim].
    """
    from PIL import Image

    fig_elems = [e for e in elements if e["type"] in ("figure", "table")]
    if not fig_elems:
        return torch.zeros(0, 1152)

    images = []
    for e in fig_elems:
        img_path = Path(image_root) / e.get("image_path", "")
        if img_path.exists():
            images.append(preprocess(Image.open(img_path).convert("RGB")))
        else:
            # Create blank placeholder if image crop not found
            logger.warning(f"Image not found for element {e['id']}, using blank.")
            images.append(torch.zeros(3, 224, 224))

    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack(images[i : i + batch_size]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = model.encode_image(batch)
        embeddings.append(feats.cpu())

    return torch.cat(embeddings, dim=0)


# ─── Project all embeddings to 256-dim ────────────────────────────────────────

def project_embeddings(
    text_emb: torch.Tensor,
    img_emb: torch.Tensor,
    text_proj: ProjectionHead,
    img_proj: ProjectionHead,
) -> tuple:
    """Apply linear projection to common 256-dim space."""
    text_proj.eval()
    img_proj.eval()
    with torch.no_grad():
        text_out = text_proj(text_emb) if text_emb.shape[0] > 0 else text_emb
        img_out = img_proj(img_emb) if img_emb.shape[0] > 0 else img_emb
    return text_out, img_out


# ─── Main driver ───────────────────────────────────────────────────────────────

def compute_and_save_embeddings(
    parsed: dict,
    embeddings_dir: str = None,   # canonical param name
    image_root: str = "",
    device: str = None,
    force: bool = False,
    # ── Aliases used by 03_build_graphs.ipynb ────────────────────────────────
    emb_dir: str = None,          # alias for embeddings_dir
    text_batch_size: int = 16,    # LIMIT: 16 for 8GB VRAM; ← EXTEND: 32+
    image_batch_size: int = 8,    # LIMIT: 8 for 8GB VRAM;  ← EXTEND: 16+
) -> dict:
    """
    Full embedding pipeline for one document. Saves .pt tensors to disk.

    Outputs:
      embeddings/{doc_id}_text.pt  — shape [N_text, 256]
      embeddings/{doc_id}_img.pt   — shape [N_fig,  256]

    Sequential processing: text model → unload → image model → unload → project.

    Args:
        parsed:         Parsed document dict (from mineru_wrapper/mpdocvqa_parser).
        embeddings_dir: Output directory for .pt files.
        emb_dir:        Alias for embeddings_dir (accepted from notebook).
        image_root:     Root dir for image crops (MinerU artifacts).
        device:         'cuda' or 'cpu' (auto-detected if None).
        force:          Recompute even if cached.
        text_batch_size: # LIMIT: batch size for text encoding (fit 8GB VRAM).
                         # ← EXTEND: increase for faster throughput.
        image_batch_size: # LIMIT: batch size for image encoding.
                          # ← EXTEND: increase when VRAM allows.
    """
    # Resolve embeddings_dir from either canonical param or alias
    out_dir = embeddings_dir or emb_dir
    if out_dir is None:
        raise ValueError("Provide either embeddings_dir or emb_dir.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    doc_id = parsed["doc_id"]
    out_text = Path(out_dir) / f"{doc_id}_text.pt"
    out_img  = Path(out_dir) / f"{doc_id}_img.pt"

    if out_text.exists() and out_img.exists() and not force:
        logger.info(f"Embeddings cached for {doc_id}, skipping.")
        return {
            "text": torch.load(out_text, weights_only=True),
            "img":  torch.load(out_img,  weights_only=True),
        }

    elements = parsed["elements"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Text embeddings ──────────────────────────────────────────────
    text_model, text_dim = load_text_model(device=device)
    raw_text_emb = embed_text_nodes(
        elements, text_model, batch_size=text_batch_size, device=device
    )
    del text_model   # free VRAM before loading image model
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"Text embeddings: {raw_text_emb.shape}")

    # ── Step 2: Image embeddings ─────────────────────────────────────────────
    has_images = any(e.get("type") in ("figure", "image") for e in elements)
    if has_images:
        img_model, preprocess, img_dim = load_image_model(device=device)
        raw_img_emb = embed_image_nodes(
            elements, img_model, preprocess, image_root,
            batch_size=image_batch_size, device=device
        )
        del img_model
        if device == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"Image embeddings: {raw_img_emb.shape}")
    else:
        # No figure/image elements — skip SigLIP loading entirely (saves ~5s per doc)
        raw_img_emb = torch.zeros(0, PROJECTION_DIM)
        img_dim = PROJECTION_DIM
        logger.info("No image elements — skipping SigLIP (image embeddings: empty)")

    # ── Step 3: Project to 256-dim ───────────────────────────────────────────
    text_proj = ProjectionHead(text_dim if text_dim > 0 and raw_text_emb.shape[0] > 0
                               else (raw_text_emb.shape[-1] if raw_text_emb.shape[0] > 0
                                     else PROJECTION_DIM))
    img_proj  = ProjectionHead(img_dim  if img_dim  > 0 and raw_img_emb.shape[0]  > 0
                               else (raw_img_emb.shape[-1]  if raw_img_emb.shape[0]  > 0
                                     else PROJECTION_DIM))

    # Projection layers start randomly initialised; jointly trained with HGT in Step 8.
    text_emb, img_emb = project_embeddings(raw_text_emb, raw_img_emb, text_proj, img_proj)

    torch.save(text_emb, out_text)
    torch.save(img_emb,  out_img)
    logger.info(f"Saved embeddings for {doc_id} → {out_dir}")

    return {"text": text_emb, "img": img_emb}
