"""
src/graph/builder.py
─────────────────────
PyG HeteroData graph construction for GraMM-RAG.

Step 6: Build one PyG HeteroData object per document combining:
  - 5 node types:  text, table, figure, equation, section
  - 8 edge types:  spatial, reading_order, hierarchy, temporal,
                   coref, crosspage, caption, kg

All node features projected to 256-dim (see embeddings.py).
Graph runs entirely on CPU. ~0.3 sec/page. ~30 min total for all benchmarks.

Saves:  graphs/{doc_id}.pt
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import HeteroData

from src.graph.edges import (
    compute_spatial_edges,
    compute_reading_order_edges,
    compute_hierarchy_edges,
    compute_temporal_edges,
    compute_coref_edges,
    compute_crosspage_edges,
    compute_caption_edges,
    compute_kg_edges,
)

logger = logging.getLogger(__name__)

PROJECTION_DIM = 256   # must match embeddings.py


def build_graph(
    doc_id: str,
    parsed: dict,
    embeddings: dict,    # {"text": Tensor[N_text,256], "img": Tensor[N_fig,256]}
    triplets: list[dict],
) -> HeteroData:
    """
    Build a PyG HeteroData heterogeneous graph for one document.

    Args:
        doc_id:     Document identifier string.
        parsed:     Output of mineru_wrapper.parse_pdf() + temporal annotation.
        embeddings: Output of compute_and_save_embeddings().
        triplets:   Output of extract_all_triplets() (may be empty list).

    Returns:
        PyG HeteroData object, also saved to graphs/{doc_id}.pt.
    """
    elements = parsed["elements"]

    # ── Partition elements by type ────────────────────────────────────────────
    type_groups: dict[str, list[tuple]] = {
        "text": [], "table": [], "figure": [], "equation": [], "section": []
    }
    for global_idx, e in enumerate(elements):
        t = e["type"]
        if t in type_groups:
            type_groups[t].append((global_idx, e))

    # ── Index mappings: global element index → per-type index ─────────────────
    global_to_type_idx: dict[int, tuple[str, int]] = {}
    for node_type, group in type_groups.items():
        for local_idx, (global_idx, _) in enumerate(group):
            global_to_type_idx[global_idx] = (node_type, local_idx)

    # ── Build node feature matrices ───────────────────────────────────────────
    data = HeteroData()

    text_emb = embeddings.get("text", torch.zeros(0, PROJECTION_DIM))
    # Accept both "img" (canonical) and "image" (notebook alias)
    img_emb = embeddings.get("img", embeddings.get("image", torch.zeros(0, PROJECTION_DIM)))

    # Text-type nodes share the E5-Mistral embedding tensor
    # We interleave: text, section, equation → all get text embeddings
    n_text_like = len(type_groups["text"]) + len(type_groups["section"]) + len(type_groups["equation"])
    text_emb_padded = _pad_or_trim(text_emb, n_text_like, PROJECTION_DIM)

    n_fig_like = len(type_groups["figure"]) + len(type_groups["table"])
    img_emb_padded = _pad_or_trim(img_emb, n_fig_like, PROJECTION_DIM)

    # Assign features per node type (split the pooled tensors)
    cursor_text, cursor_img = 0, 0
    for node_type in ("text", "section", "equation"):
        n = len(type_groups[node_type])
        if n > 0:
            data[node_type].x = text_emb_padded[cursor_text : cursor_text + n]
        else:
            data[node_type].x = torch.zeros(0, PROJECTION_DIM)
        cursor_text += n

    for node_type in ("figure", "table"):
        n = len(type_groups[node_type])
        if n > 0:
            data[node_type].x = img_emb_padded[cursor_img : cursor_img + n]
        else:
            data[node_type].x = torch.zeros(0, PROJECTION_DIM)
        cursor_img += n

    # Store element metadata for later retrieval
    data.doc_id = doc_id
    data.element_ids = [e["id"] for _, e in sorted(
        [(gi, e) for gi, e in enumerate(elements)], key=lambda x: x[0]
    )]

    # ── Build edge indices (8 types) ──────────────────────────────────────────
    # All edge construction functions operate on global element indices.
    # We remap to per-type indices for each edge relation.

    # 1. Spatial: text/section → table/figure (proximity on same page)
    spatial_ei = compute_spatial_edges(elements)
    data["text", "spatial", "table"].edge_index = _remap_global_edges(
        spatial_ei, global_to_type_idx, src_type="text", dst_type="table"
    )

    # 2. Reading order: text → text (consecutive in reading order)
    ro_ei = compute_reading_order_edges(elements)
    data["text", "reading_order", "text"].edge_index = _remap_global_edges(
        ro_ei, global_to_type_idx, src_type="text", dst_type="text"
    )

    # 3. Hierarchy: section → text (section header → children)
    hier_ei = compute_hierarchy_edges(elements)
    data["section", "hierarchy", "text"].edge_index = _remap_global_edges(
        hier_ei, global_to_type_idx, src_type="section", dst_type="text"
    )

    # 4. Temporal: text → text (same resolved temporal marker)
    temp_ei = compute_temporal_edges(elements)
    data["text", "temporal", "text"].edge_index = _remap_global_edges(
        temp_ei, global_to_type_idx, src_type="text", dst_type="text"
    )

    # 5. Coreference: text → text (shared named entities, spaCy)
    coref_ei = compute_coref_edges(elements)
    data["text", "coref", "text"].edge_index = _remap_global_edges(
        coref_ei, global_to_type_idx, src_type="text", dst_type="text"
    )

    # 6. Cross-page: text → text (same entity on different pages)
    cross_ei = compute_crosspage_edges(elements)
    data["text", "crosspage", "text"].edge_index = _remap_global_edges(
        cross_ei, global_to_type_idx, src_type="text", dst_type="text"
    )

    # 7. Caption: text → figure (caption text → its figure/table)
    cap_ei = compute_caption_edges(elements)
    data["text", "caption", "figure"].edge_index = _remap_global_edges(
        cap_ei, global_to_type_idx, src_type="text", dst_type="figure"
    )

    # 8. KG: text → text (shared KG entities from GPT-4o-mini extraction)
    kg_ei = compute_kg_edges(elements, triplets)
    data["text", "kg", "text"].edge_index = _remap_global_edges(
        kg_ei, global_to_type_idx, src_type="text", dst_type="text"
    )

    logger.info(
        f"Built graph for {doc_id}: "
        f"{sum(len(g) for g in type_groups.values())} nodes, "
        f"8 edge types"
    )
    return data


def _pad_or_trim(tensor: torch.Tensor, target_n: int, dim: int) -> torch.Tensor:
    """Pad with zeros or trim to have exactly target_n rows."""
    if tensor.shape[0] == target_n:
        return tensor
    if tensor.shape[0] > target_n:
        return tensor[:target_n]
    pad = torch.zeros(target_n - tensor.shape[0], dim)
    return torch.cat([tensor, pad], dim=0)


def _remap_global_edges(
    edge_index: torch.Tensor,
    global_to_type_idx: dict,
    src_type: str,
    dst_type: str,
) -> torch.Tensor:
    """
    Filter an edge_index (global element indices) to only edges between
    the given src_type and dst_type, and remap to per-type local indices.
    """
    if edge_index.shape[1] == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    src_list, dst_list = [], []
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        s_info = global_to_type_idx.get(s)
        d_info = global_to_type_idx.get(d)
        if s_info is None or d_info is None:
            continue
        if s_info[0] == src_type and d_info[0] == dst_type:
            src_list.append(s_info[1])
            dst_list.append(d_info[1])

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src_list, dst_list], dtype=torch.long)


def save_graph(data: HeteroData, graphs_dir: str, doc_id: str) -> None:
    """Save a HeteroData graph to disk."""
    Path(graphs_dir).mkdir(parents=True, exist_ok=True)
    path = Path(graphs_dir) / f"{doc_id}.pt"
    torch.save(data, path)
    logger.info(f"Saved graph: {path}")


def load_graph(graphs_dir: str, doc_id: str) -> Optional[HeteroData]:
    """Load a cached graph from disk."""
    path = Path(graphs_dir) / f"{doc_id}.pt"
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)
