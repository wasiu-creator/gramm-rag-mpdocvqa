"""
src/retrieval/router.py
───────────────────────
Query router for GraMM-RAG (Step 9).

Fine-tuned DeBERTa-v3-base (86M params) for 3-class classification:
  0 = GRAPH   → complex, multi-hop, cross-page questions
  1 = VECTOR  → simple single-fact extraction
  2 = HYBRID  → ambiguous / multi-modal questions

VRAM: ~2GB. Fits on RTX 5070.
Training: 3 epochs, lr=2e-5, batch_size=16.
Inference: CPU-friendly (<1ms per query).

Annotation rules (for creating training data):
  GRAPH  ← cross-page questions (use MMLongBench evidence metadata)
  VECTOR ← single-fact extraction (direct look-up)
  HYBRID ← ambiguous multi-modal questions
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

LABEL_NAMES = ["GRAPH", "VECTOR", "HYBRID"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for i, n in enumerate(LABEL_NAMES)}


class QueryRouter:
    """
    Wraps a fine-tuned DeBERTa-v3-base sequence classifier.
    Falls back to a simple keyword heuristic if no trained model is available.
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self.tokenizer = None

        if model_dir and Path(model_dir).exists():
            self._load_model(model_dir)
        else:
            logger.warning(
                "No trained router model found. Using keyword heuristic fallback. "
                "Train the router first with 06_train_router.ipynb."
            )

    def _load_model(self, model_dir: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        logger.info(f"Loading router from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

    def route(self, query: str) -> str:
        """
        Classify a query into GRAPH / VECTOR / HYBRID.

        Args:
            query: Natural language question string.

        Returns:
            One of "GRAPH", "VECTOR", "HYBRID".
        """
        if self.model is not None:
            return self._model_route(query)
        return self._heuristic_route(query)

    def _model_route(self, query: str) -> str:
        enc = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        label_id = logits.argmax(dim=-1).item()
        return ID2LABEL[label_id]

    def _heuristic_route(self, query: str) -> str:
        """
        Keyword-based fallback router (used before DeBERTa is trained).
        Covers the most common cross-page / multi-hop patterns.
        """
        q = query.lower()

        graph_keywords = [
            "compare", "difference between", "how does", "across",
            "throughout", "over the years", "trend", "relationship between",
            "summarize", "summarise", "overall", "in total", "across all",
            "multiple", "various sections", "across pages",
        ]
        vector_keywords = [
            "what is the", "what was the", "how many", "when did",
            "who is", "which year", "what date", "name the",
            "list the", "what percentage",
        ]

        if any(k in q for k in graph_keywords):
            return "GRAPH"
        if any(k in q for k in vector_keywords):
            return "VECTOR"
        return "HYBRID"


def build_router_training_data(
    benchmark_datasets: list[dict],
    max_samples: int = 5000,  # ← EXTEND: increase for better router accuracy
) -> list[dict]:
    """
    Auto-annotate query-route pairs from benchmark metadata.

    Rules:
      - MMLongBench questions with evidence_pages spanning >1 page → GRAPH
      - Questions with single evidence page AND single-hop phrasing → VECTOR
      - All others → HYBRID

    # LIMIT: max_samples=5000. Full annotation ideally 5000+ samples.
    # ← EXTEND: increase max_samples and add more benchmark sources.
    """
    samples = []
    for item in benchmark_datasets:
        if len(samples) >= max_samples:
            break
        query = item.get("question", "")
        evidence = item.get("evidence_pages", item.get("page_ids", []))

        if isinstance(evidence, list) and len(evidence) > 1:
            label = "GRAPH"
        elif _is_simple_factoid(query):
            label = "VECTOR"
        else:
            label = "HYBRID"

        samples.append({"query": query, "label": label})

    logger.info(
        f"Built {len(samples)} router training samples. "
        "Recommend manual review of ~10% for quality."
    )
    return samples


def _is_simple_factoid(query: str) -> bool:
    """Heuristic to detect single-fact extraction questions."""
    q = query.lower().strip()
    starters = ("what is", "what was", "how many", "when did", "who is",
                 "which", "name the", "what percentage", "what date")
    return any(q.startswith(s) for s in starters) and len(q.split()) < 12
