"""
src/parsing/mpdocvqa_parser.py
────────────────────────────────
Parse MP-DocVQA OCR JSON files into the GraMM-RAG element format.

Actual MP-DocVQA OCR format (AWS Textract, one file per page):
  File naming: {doc_id}_p{page_idx}.json  (0-indexed, in ocr/ocr/ subdir)
  Content:
    {
      "PAGE": [...],
      "LINE": [
        {
          "BlockType": "LINE",
          "Confidence": 91.5,
          "Text": "USA Petroleum",
          "Geometry": {
            "BoundingBox": {"Width": 0.163, "Height": 0.032, "Left": 0.724, "Top": 0.043}
          }
        }, ...
      ],
      "WORD": [...]
    }

BoundingBox coordinates are normalised (0–1). We scale to 1000×1000 for
compatibility with spatial edge computation in edges.py.

Output element format:
  {
    "id":            "<doc_id>_p<page_no>_<local_idx>",
    "type":          "text",
    "text":          "...",
    "bbox":          [x0, y0, x1, y1],   # 0–1000 scale
    "page_no":       int,
    "reading_order": int,
    "confidence":    float,
  }
"""

import json
import logging
import pathlib
import re
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

PAGE_SCALE = 1000   # normalised bbox → pixel-like units for edge computation


def _textract_page_to_elements(
    ocr_data: dict,
    page_no: int,
    doc_id: str,
    start_idx: int = 0,
) -> list:
    """
    Convert AWS Textract LINE blocks from one page to GraMM-RAG elements.
    Uses LINE blocks (already word-grouped by Textract) as the text unit.
    """
    elements = []
    lines = ocr_data.get("LINE", [])

    for line in lines:
        text = line.get("Text", "").strip()
        if not text:
            continue
        bb = line.get("Geometry", {}).get("BoundingBox", {})
        left   = bb.get("Left",   0.0)
        top    = bb.get("Top",    0.0)
        width  = bb.get("Width",  0.0)
        height = bb.get("Height", 0.0)

        x0 = left  * PAGE_SCALE
        y0 = top   * PAGE_SCALE
        x1 = (left  + width)  * PAGE_SCALE
        y1 = (top   + height) * PAGE_SCALE

        elements.append({
            "id":            f"{doc_id}_p{page_no}_{start_idx + len(elements)}",
            "type":          "text",
            "text":          text,
            "bbox":          [x0, y0, x1, y1],
            "page_no":       page_no,
            "reading_order": start_idx + len(elements),
            "confidence":    float(line.get("Confidence", 0.0)),
        })

    return elements


def parse_mpdocvqa_doc_from_pages(
    page_files: list,
    doc_id: str,
) -> dict:
    """
    Build one GraMM-RAG document from a sorted list of per-page Textract JSON files.

    Args:
        page_files: Paths to {doc_id}_p{N}.json files, in page order.
        doc_id:     Document identifier.

    Returns:
        GraMM-RAG parsed document dict.
    """
    # Sort by page index extracted from filename
    def page_idx(p):
        m = re.search(r"_p(\d+)$", pathlib.Path(p).stem)
        return int(m.group(1)) if m else 0

    page_files_sorted = sorted(page_files, key=page_idx)

    elements = []
    for page_file in page_files_sorted:
        p_idx = page_idx(page_file)
        try:
            with open(page_file) as f:
                ocr_data = json.load(f)
        except Exception as e:
            logger.warning(f"  Could not read {page_file}: {e}")
            continue

        page_elems = _textract_page_to_elements(
            ocr_data, p_idx, doc_id, start_idx=len(elements)
        )
        elements.extend(page_elems)

    return {
        "doc_id":    doc_id,
        "num_pages": len(page_files_sorted),
        "elements":  elements,
        "source":    "mpdocvqa_textract",
    }


def parse_mpdocvqa_doc(ocr_path: pathlib.Path, doc_id: str) -> dict:
    """
    Parse a single MP-DocVQA OCR file — supports both:
    - Legacy per-doc format: {"pages": [{"page_no": N, "words": [...]}]}
    - Textract per-page format: {"LINE": [...], "WORD": [...]}

    For the Textract format this function handles a single page file.
    For multi-page documents use parse_mpdocvqa_doc_from_pages().
    """
    with open(ocr_path) as f:
        ocr = json.load(f)

    # Textract format (per-page file)
    if "LINE" in ocr or "WORD" in ocr:
        elements = _textract_page_to_elements(ocr, page_no=0, doc_id=doc_id)
        return {
            "doc_id":    doc_id,
            "num_pages": 1,
            "elements":  elements,
            "source":    "mpdocvqa_textract",
        }

    # Legacy per-doc format
    pages_data = ocr.get("pages", [])
    if not pages_data:
        pages_data = [{"page_no": 1, "words": ocr.get("words", []),
                       "width": ocr.get("width", 1000), "height": ocr.get("height", 1000)}]

    elements = []
    for page in pages_data:
        page_no = page.get("page_no", 1)
        words = page.get("words", [])
        norm_words = []
        for w in words:
            bbox = w.get("bbox") or [
                w.get("left", 0), w.get("top", 0),
                w.get("right", 10), w.get("bottom", 10),
            ]
            norm_words.append({"text": str(w.get("text", "")), "bbox": bbox})
        elements.extend(_words_to_elements(norm_words, page_no, doc_id, len(elements)))

    return {
        "doc_id":    doc_id,
        "num_pages": len(pages_data),
        "elements":  elements,
        "source":    "mpdocvqa_ocr",
    }


def _words_to_elements(words: list, page_no: int, doc_id: str, start_idx: int = 0) -> list:
    """Group consecutive words into text-block elements by y-midpoint proximity."""
    if not words:
        return []

    elements = []
    current_words = [words[0]]
    current_bbox = list(words[0]["bbox"])

    def flush():
        text = " ".join(w["text"] for w in current_words).strip()
        if text:
            elements.append({
                "id": f"{doc_id}_p{page_no}_{start_idx + len(elements)}",
                "type": "text",
                "text": text,
                "bbox": current_bbox[:],
                "page_no": page_no,
                "reading_order": start_idx + len(elements),
            })

    for w in words[1:]:
        bbox = w["bbox"]
        y_mid = (bbox[1] + bbox[3]) / 2
        prev_y_mid = (current_bbox[1] + current_bbox[3]) / 2
        if abs(y_mid - prev_y_mid) <= 12:
            current_words.append(w)
            current_bbox[0] = min(current_bbox[0], bbox[0])
            current_bbox[1] = min(current_bbox[1], bbox[1])
            current_bbox[2] = max(current_bbox[2], bbox[2])
            current_bbox[3] = max(current_bbox[3], bbox[3])
        else:
            flush()
            current_words = [w]
            current_bbox = list(bbox)

    flush()
    return elements


def parse_mpdocvqa_docs(
    ocr_dir: pathlib.Path,
    images_dir: pathlib.Path,
    parsed_dir: pathlib.Path,
    max_docs: Optional[int] = 20,   # LIMIT: 20 for dev  # <- EXTEND: None for full run
) -> list:
    """
    Parse all (up to max_docs) MP-DocVQA documents.

    Handles the actual dataset layout:
      ocr_dir/            (e.g. data/mpdocvqa/ocr/)
        ocr/              (nested subdir matching parent name)
          {doc_id}_p0.json
          {doc_id}_p1.json
          ...

    Args:
        ocr_dir:    Top-level OCR directory (data/mpdocvqa/ocr/).
        images_dir: Images root (data/mpdocvqa/images/) — reserved for Step 4.
        parsed_dir: Output directory for parsed JSON files.
        max_docs:   Maximum documents to parse.
                    # LIMIT: 20 for dev.
                    # <- EXTEND: None for full paper run (~6000 docs).

    Returns:
        List of parsed document dicts.
    """
    ocr_dir    = pathlib.Path(ocr_dir)
    parsed_dir = pathlib.Path(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    if not ocr_dir.exists():
        logger.warning(f"OCR directory not found: {ocr_dir}. "
                       "Run 01_download_datasets.ipynb first.")
        return []

    # Resolve actual directory containing *.json files
    # The dataset ships as: ocr_dir/ocr/*.json  (nested subdir same name)
    actual_dir = ocr_dir
    json_files  = list(ocr_dir.glob("*.json"))
    if not json_files:
        nested = ocr_dir / ocr_dir.name   # e.g. data/mpdocvqa/ocr/ocr/
        if nested.exists():
            actual_dir = nested
            json_files = list(nested.glob("*.json"))
        if not json_files:
            # Try any one-level subdirectory
            for subdir in ocr_dir.iterdir():
                if subdir.is_dir():
                    cands = list(subdir.glob("*.json"))
                    if cands:
                        actual_dir = subdir
                        json_files = cands
                        break

    if not json_files:
        logger.warning(f"No OCR JSON files found under {ocr_dir}")
        return []

    # Group per-page files by doc_id
    doc_pages: dict[str, list] = defaultdict(list)
    per_page_pattern = re.compile(r"^(.+)_p(\d+)$")

    for f in json_files:
        m = per_page_pattern.match(f.stem)
        if m:
            doc_pages[m.group(1)].append(f)
        else:
            # Single-file-per-doc (legacy format): treat whole file as doc
            doc_pages[f.stem].append(f)

    doc_ids = sorted(doc_pages.keys())
    if max_docs is not None:
        doc_ids = doc_ids[:max_docs]   # LIMIT: cap at max_docs

    logger.info(f"parse_mpdocvqa_docs: {len(doc_ids)} docs to parse "
                f"(OCR dir: {actual_dir})")

    docs = []
    for doc_id in doc_ids:
        out_path = parsed_dir / f"{doc_id}.json"

        if out_path.exists():
            try:
                docs.append(json.loads(out_path.read_text()))
                continue
            except Exception:
                pass

        try:
            page_files = doc_pages[doc_id]
            if len(page_files) > 1 or per_page_pattern.match(page_files[0].stem):
                parsed = parse_mpdocvqa_doc_from_pages(page_files, doc_id)
            else:
                parsed = parse_mpdocvqa_doc(page_files[0], doc_id)

            out_path.write_text(json.dumps(parsed, indent=2))
            docs.append(parsed)
            logger.info(f"  Parsed {doc_id}: {len(parsed['elements'])} elements "
                        f"across {parsed['num_pages']} pages")
        except Exception as e:
            logger.error(f"  Failed to parse {doc_id}: {e}")

    logger.info(f"parse_mpdocvqa_docs: {len(docs)} docs → {parsed_dir}")
    return docs
