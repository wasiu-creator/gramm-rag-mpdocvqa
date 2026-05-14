"""
src/graph/edges.py
──────────────────
Edge construction for GraMM-RAG document graphs.

Step 5: KG triplet extraction via GPT-4o-mini API.
Step 6: Deterministic edge types (spatial, reading_order, hierarchy, temporal,
        coref, crosspage, caption, kg).

API cost (Step 5): ~3,000 pages × ~$0.015/page = ~$45 total.
Budget $60 with buffer. Rate limit: 0.5s sleep between calls.

# LIMIT: In dev/testing we only call extract_triplets() on the first
# MAX_TRIPLET_ELEMENTS elements per document to save cost.
# ← EXTEND: remove the cap in build_kg_edges() for full paper runs.
MAX_TRIPLET_ELEMENTS = 20   # ← EXTEND: set to None for full paper run
"""

import json
import logging
import time
import re
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# LIMIT: cap triplet extraction per document to save API cost during testing.
# ← EXTEND: increase or remove this limit for the full paper run.
MAX_TRIPLET_ELEMENTS = 20


# ─── Step 5: KG triplet extraction ────────────────────────────────────────────

def extract_triplets(
    text_content: str,
    client,                       # openai.OpenAI() instance
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    Extract (subject, predicate, object) triplets from a text passage.

    Follows RAG-Anything's prompting strategy with JSON-mode output.

    Args:
        text_content: One text element's content string.
        client:       openai.OpenAI() instance.
        model:        Model to call (default: gpt-4o-mini, cheapest).

    Returns:
        List of dicts with keys: subject, predicate, object.
    """
    if not text_content.strip():
        return []

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge graph extraction assistant. "
                        "Extract all factual (subject, predicate, object) triplets "
                        "from the provided text. Return a JSON object with a single "
                        "key 'triplets' containing a list of objects, each with keys: "
                        "'subject', 'predicate', 'object'. Be concise. "
                        "Only extract clearly stated facts, not implied ones."
                    ),
                },
                {"role": "user", "content": text_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("triplets", [])

    except Exception as e:
        logger.warning(f"Triplet extraction failed: {e}")
        return []


def extract_all_triplets(
    parsed: dict,
    client,
    sleep_between_calls: float = 0.5,   # rate limit
) -> list[dict]:
    """
    Extract KG triplets for all text elements in a parsed document.

    # LIMIT: Only processes the first MAX_TRIPLET_ELEMENTS text nodes.
    # ← EXTEND: remove the [:MAX_TRIPLET_ELEMENTS] slice for full paper run.
    """
    text_elems = [e for e in parsed["elements"] if e["type"] in ("text", "section")]

    # LIMIT: cap to MAX_TRIPLET_ELEMENTS per document during dev/testing
    # ← EXTEND: remove this slice for full paper run
    if MAX_TRIPLET_ELEMENTS is not None:
        text_elems = text_elems[:MAX_TRIPLET_ELEMENTS]
        logger.info(
            f"[LIMIT] Capping triplet extraction to {MAX_TRIPLET_ELEMENTS} elements. "
            "Extend MAX_TRIPLET_ELEMENTS for full run."
        )

    all_triplets = []
    for i, elem in enumerate(text_elems):
        triplets = extract_triplets(elem["text"], client)
        for t in triplets:
            t["source_element_id"] = elem["id"]
        all_triplets.extend(triplets)
        time.sleep(sleep_between_calls)   # OpenAI rate limiting

    logger.info(f"Extracted {len(all_triplets)} triplets from {len(text_elems)} elements")
    return all_triplets


# ─── Step 6: Deterministic edge construction ──────────────────────────────────

def compute_spatial_edges(
    elements: list[dict],
    threshold_pt: float = 20.0,   # vertical gap threshold in points
) -> torch.Tensor:
    """
    Spatial edges between text/section nodes and nearby table/figure nodes
    on the same page, based on bounding-box proximity.

    Returns edge_index of shape [2, E].
    """
    id_to_idx = {e["id"]: i for i, e in enumerate(elements)}
    text_types = {"text", "section", "equation"}
    table_types = {"table", "figure"}

    src, dst = [], []
    for i, a in enumerate(elements):
        if a["type"] not in text_types:
            continue
        for j, b in enumerate(elements):
            if b["type"] not in table_types or a["page_no"] != b["page_no"]:
                continue
            if _bbox_nearby(a.get("bbox", []), b.get("bbox", []), threshold_pt):
                src.append(i)
                dst.append(j)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _bbox_nearby(bbox_a: list, bbox_b: list, threshold: float) -> bool:
    """Return True if two bboxes are within threshold points vertically."""
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return False
    a_bottom = bbox_a[3]
    b_top = bbox_b[1]
    a_top = bbox_a[1]
    b_bottom = bbox_b[3]
    gap = min(abs(a_bottom - b_top), abs(b_bottom - a_top))
    return gap < threshold


def compute_reading_order_edges(elements: list[dict]) -> torch.Tensor:
    """
    Reading-order edges between consecutive text nodes on the same page.
    Connects element[i] → element[i+1] for text-type nodes in reading order.

    Returns edge_index of shape [2, E].
    """
    text_elems = [
        (e["reading_order"], i)
        for i, e in enumerate(elements)
        if e["type"] in ("text", "section")
    ]
    text_elems.sort()   # sort by reading_order

    src, dst = [], []
    for k in range(len(text_elems) - 1):
        _, i = text_elems[k]
        _, j = text_elems[k + 1]
        src.append(i)
        dst.append(j)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def compute_hierarchy_edges(elements: list[dict]) -> torch.Tensor:
    """
    Hierarchy edges: section headers → child text nodes.
    A text node is a child of the most recent preceding section header.

    Returns edge_index of shape [2, E] (section → text).
    """
    src, dst = [], []
    current_section_idx = None

    sorted_elems = sorted(enumerate(elements), key=lambda x: x[1]["reading_order"])
    for idx, e in sorted_elems:
        if e["type"] == "section":
            current_section_idx = idx
        elif e["type"] in ("text", "table", "figure", "equation"):
            if current_section_idx is not None:
                src.append(current_section_idx)
                dst.append(idx)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def compute_temporal_edges(elements: list[dict]) -> torch.Tensor:
    """
    Temporal edges between text nodes that share the same resolved temporal marker.
    E.g., two paragraphs both referencing "FY2023" get connected.

    Returns edge_index of shape [2, E] (text → text).
    """
    # Group elements by their absolute temporal expressions
    temporal_to_elems: dict[str, list[int]] = {}
    for i, e in enumerate(elements):
        if e["type"] not in ("text", "section"):
            continue
        for marker in e.get("temporal_markers", []):
            key = marker.get("resolved_text", marker["text"]).upper()
            temporal_to_elems.setdefault(key, []).append(i)

    src, dst = [], []
    for elems_with_same_time in temporal_to_elems.values():
        if len(elems_with_same_time) < 2:
            continue
        for a in elems_with_same_time:
            for b in elems_with_same_time:
                if a != b:
                    src.append(a)
                    dst.append(b)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def compute_coref_edges(elements: list[dict]) -> torch.Tensor:
    """
    Coreference edges between text nodes sharing named entities (spaCy NER).
    Connects any two text nodes that mention the same entity.

    Returns edge_index of shape [2, E].
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_lg")
    except Exception:
        logger.warning("spaCy model not found. Skipping coreference edges.")
        return torch.zeros(2, 0, dtype=torch.long)

    entity_to_elems: dict[str, list[int]] = {}
    for i, e in enumerate(elements):
        if e["type"] not in ("text", "section"):
            continue
        doc = nlp(e["text"][:500])   # truncate for speed
        for ent in doc.ents:
            key = f"{ent.label_}:{ent.text.lower()}"
            entity_to_elems.setdefault(key, []).append(i)

    src, dst = [], []
    for elems in entity_to_elems.values():
        if len(elems) < 2:
            continue
        for a in elems:
            for b in elems:
                if a != b:
                    src.append(a)
                    dst.append(b)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def compute_crosspage_edges(elements: list[dict]) -> torch.Tensor:
    """
    Cross-page edges: text nodes on different pages that share named entities.
    Subset of coreference edges, filtered to only cross-page pairs.

    Returns edge_index of shape [2, E] (text → table/text on different page).
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_lg")
    except Exception:
        logger.warning("spaCy model not found. Skipping cross-page edges.")
        return torch.zeros(2, 0, dtype=torch.long)

    entity_to_elems: dict[str, list[tuple]] = {}   # key → [(idx, page_no), ...]
    for i, e in enumerate(elements):
        if e["type"] not in ("text", "section", "table"):
            continue
        doc = nlp(e["text"][:500])
        for ent in doc.ents:
            key = f"{ent.label_}:{ent.text.lower()}"
            entity_to_elems.setdefault(key, []).append((i, e["page_no"]))

    src, dst = [], []
    for pairs in entity_to_elems.values():
        for (a, pa) in pairs:
            for (b, pb) in pairs:
                if a != b and pa != pb:   # different pages only
                    src.append(a)
                    dst.append(b)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def compute_caption_edges(elements: list[dict]) -> torch.Tensor:
    """
    Caption edges: text nodes that are captions → their figure/table node.
    Detected heuristically: text immediately before/after a figure on same page.

    Returns edge_index of shape [2, E] (text → figure/table).
    """
    src, dst = [], []
    fig_types = {"figure", "table"}

    sorted_elems = sorted(enumerate(elements), key=lambda x: x[1]["reading_order"])
    indices = [idx for idx, _ in sorted_elems]
    elems_sorted = [e for _, e in sorted_elems]

    for k, (idx, e) in enumerate(zip(indices, elems_sorted)):
        if e["type"] not in fig_types:
            continue
        # Check immediate neighbours in reading order on the same page
        for offset in (-1, 1):
            j = k + offset
            if 0 <= j < len(elems_sorted):
                neighbour_idx, neighbour = indices[j], elems_sorted[j]
                if (neighbour["type"] == "text"
                        and neighbour["page_no"] == e["page_no"]):
                    text = neighbour["text"].lower().strip()
                    if text.startswith(("figure", "fig.", "table", "tab.")):
                        src.append(neighbour_idx)
                        dst.append(idx)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def compute_kg_edges(
    elements: list[dict],
    triplets: list[dict],
) -> torch.Tensor:
    """
    KG edges: connect source text elements based on shared entity mentions
    in extracted triplets.

    Args:
        elements: Parsed document elements.
        triplets: Output of extract_all_triplets().

    Returns:
        edge_index of shape [2, E] (text → text).
    """
    # Map element_id → index
    id_to_idx = {e["id"]: i for i, e in enumerate(elements)}

    # Group by subject+object pairs
    entity_to_elem_ids: dict[str, list[str]] = {}
    for t in triplets:
        for role in ("subject", "object"):
            key = t.get(role, "").lower().strip()
            if key:
                entity_to_elem_ids.setdefault(key, []).append(
                    t.get("source_element_id", "")
                )

    src, dst = [], []
    for elem_ids in entity_to_elem_ids.values():
        valid = [id_to_idx[eid] for eid in elem_ids if eid in id_to_idx]
        for a in valid:
            for b in valid:
                if a != b:
                    src.append(a)
                    dst.append(b)

    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)
