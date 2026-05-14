"""
src/retrieval/vector_retrieval.py
──────────────────────────────────
FAISS-based vector retrieval for GraMM-RAG (used in VECTOR and HYBRID routes).

Builds a flat FAISS index over all document node embeddings.
At query time: encode query with E5-Mistral → search FAISS → return top-K.

Complements graph retrieval: HYBRID route merges FAISS results with HGT results.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    FAISS L2 (cosine after normalisation) index over all document nodes.
    One index per document, stored in embeddings/{doc_id}_faiss.index.
    """

    def __init__(self):
        self.index = None
        self.node_registry = []   # list of {doc_id, node_type, local_idx, text}

    def build(
        self,
        node_emb_dict: dict,       # {node_type: Tensor[N, D]}
        parsed: dict,              # for storing text metadata
        doc_id: str,
    ) -> None:
        """Build FAISS index from node embeddings for one document."""
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu  (or faiss-gpu)")

        elements = parsed.get("elements", [])
        type_to_elems = {}
        for e in elements:
            type_to_elems.setdefault(e["type"], []).append(e)

        all_vecs = []
        self.node_registry = []

        for node_type, embs in node_emb_dict.items():
            if embs.shape[0] == 0:
                continue
            normed = F.normalize(embs, dim=-1).numpy().astype("float32")
            typed_elems = type_to_elems.get(node_type, [])

            for local_idx in range(embs.shape[0]):
                text = typed_elems[local_idx]["text"] if local_idx < len(typed_elems) else ""
                self.node_registry.append({
                    "doc_id": doc_id,
                    "node_type": node_type,
                    "local_idx": local_idx,
                    "text": text,
                })
                all_vecs.append(normed[local_idx])

        if not all_vecs:
            logger.warning(f"No vectors to index for {doc_id}")
            return

        vecs = np.stack(all_vecs)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner product = cosine on L2-normed vecs
        self.index.add(vecs)
        logger.info(f"FAISS index built: {len(all_vecs)} nodes, dim={dim}")

    def search(
        self,
        query_emb: torch.Tensor,   # [D] raw query embedding (will be normalised)
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search the FAISS index for top-K similar nodes.

        Returns list of dicts: {doc_id, node_type, local_idx, text, score}.
        """
        if self.index is None:
            logger.warning("FAISS index not built. Call build() first.")
            return []

        q = F.normalize(query_emb.unsqueeze(0), dim=-1).numpy().astype("float32")
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.node_registry):
                continue
            entry = dict(self.node_registry[idx])
            entry["score"] = float(score)
            results.append(entry)

        return sorted(results, key=lambda r: r["score"], reverse=True)

    def save(self, path: str) -> None:
        """Save FAISS index to disk."""
        import faiss
        if self.index:
            faiss.write_index(self.index, path)
            logger.info(f"FAISS index saved: {path}")

    def load(self, path: str) -> None:
        """Load FAISS index from disk."""
        import faiss
        self.index = faiss.read_index(path)
        logger.info(f"FAISS index loaded: {path}")


def merge_graph_vector_results(
    graph_nodes: list[dict],
    vector_nodes: list[dict],
    graph_weight: float = 0.6,
    vector_weight: float = 0.4,
    top_k: int = 10,
) -> list[dict]:
    """
    Merge graph (HGT) and vector (FAISS) retrieval results for HYBRID routing.

    Uses a simple weighted score fusion. De-duplicates by (node_type, local_idx).

    Args:
        graph_nodes:    Results from HGT retrieve_top_k().
        vector_nodes:   Results from VectorIndex.search().
        graph_weight:   Score weight for graph results.
        vector_weight:  Score weight for vector results.
        top_k:          Number of final merged results to return.

    Returns:
        Merged and re-ranked list of node dicts.
    """
    # Normalise scores within each set to [0,1]
    def _normalise(nodes):
        if not nodes:
            return nodes
        scores = [n["score"] for n in nodes]
        mn, mx = min(scores), max(scores)
        rng = mx - mn if mx != mn else 1.0
        for n in nodes:
            n["norm_score"] = (n["score"] - mn) / rng
        return nodes

    graph_nodes = _normalise(graph_nodes)
    vector_nodes = _normalise(vector_nodes)

    # Build a combined dict keyed by (node_type, local_idx)
    combined = {}
    for n in graph_nodes:
        key = (n["node_type"], n["local_idx"])
        combined[key] = {**n, "fused_score": graph_weight * n.get("norm_score", 0)}

    for n in vector_nodes:
        key = (n["node_type"], n["local_idx"])
        if key in combined:
            combined[key]["fused_score"] += vector_weight * n.get("norm_score", 0)
        else:
            combined[key] = {**n, "fused_score": vector_weight * n.get("norm_score", 0)}

    merged = sorted(combined.values(), key=lambda x: x["fused_score"], reverse=True)
    return merged[:top_k]
