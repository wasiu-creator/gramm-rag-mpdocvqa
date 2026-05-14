"""
src/parsing/mineru_wrapper.py
─────────────────────────────
MinerU PDF parsing wrapper for GraMM-RAG.

MinerU (magic-pdf) converts PDFs into structured JSON with:
  - bounding boxes per element
  - element types: text / table / figure / equation / section header
  - reading order

VRAM: ~6GB. Fits on RTX 5070 (8GB). Run parse_all_pdfs() overnight for
large benchmarks (~4 hrs for MP-DocVQA full set).

Install:
    pip install magic-pdf[full]
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: str, output_dir: str) -> dict:
    """
    Parse a single PDF with MinerU and return structured JSON.

    Args:
        pdf_path:   Absolute path to the PDF file.
        output_dir: Directory to write MinerU output artifacts (images, etc.)

    Returns:
        Parsed result dict with keys: elements, metadata.
        Each element has: text, type, bbox, page_no, reading_order.
    """
    try:
        # MinerU imports — only imported when actually called so the rest of the
        # codebase can be imported even if magic-pdf is not installed.
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.pipe.UNIPipe import UNIPipe
    except ImportError as e:
        raise ImportError(
            "MinerU (magic-pdf) is not installed. "
            "Run: pip install magic-pdf[full]"
        ) from e

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pdf_bytes = open(pdf_path, "rb").read()
    writer = FileBasedDataWriter(output_dir)

    pipe = UNIPipe(
        pdf_bytes,
        jso_useful_key={},       # use default extraction settings
        image_writer=writer,
    )
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    raw = pipe.get_parsed_result()   # list of page dicts from MinerU

    # Normalise MinerU output into our internal format
    elements = []
    reading_order_counter = 0
    for page_idx, page in enumerate(raw):
        for elem in page.get("layout", []):
            el_type = _normalise_type(elem.get("type", "text"))
            elements.append({
                "id": f"p{page_idx}_e{reading_order_counter}",
                "type": el_type,
                "text": elem.get("text", "") or elem.get("markdown", ""),
                "bbox": elem.get("bbox", []),          # [x0, y0, x1, y1] in pts
                "page_no": page_idx,
                "reading_order": reading_order_counter,
            })
            reading_order_counter += 1

    result = {
        "doc_id": Path(pdf_path).stem,
        "pdf_path": str(pdf_path),
        "num_pages": len(raw),
        "elements": elements,
    }
    return result


def _normalise_type(raw_type: str) -> str:
    """Map MinerU element types to GraMM-RAG canonical types."""
    mapping = {
        "text": "text",
        "para": "text",
        "title": "section",
        "interline_equation": "equation",
        "table": "table",
        "figure": "figure",
        "image": "figure",
    }
    return mapping.get(raw_type.lower(), "text")


def parse_all_pdfs(
    pdf_dir: str,
    parsed_dir: str,
    max_docs: Optional[int] = None,   # ← EXTEND: set to None for full paper run
) -> None:
    """
    Batch-parse all PDFs in pdf_dir, saving JSON to parsed_dir.

    Args:
        pdf_dir:   Directory containing PDF files.
        parsed_dir: Output directory for parsed JSON files.
        max_docs:  Maximum number of PDFs to process. Set to None for all.
                   # LIMIT: Currently capped to save time during dev/testing.
                   # ← EXTEND: remove or increase for full paper run.
    """
    pdf_paths = sorted(Path(pdf_dir).glob("**/*.pdf"))

    # LIMIT: cap number of documents for end-to-end testing
    # ← EXTEND: remove the slice below for the full paper run
    if max_docs is not None:
        pdf_paths = pdf_paths[:max_docs]
        logger.info(f"[LIMIT] Capped to {max_docs} PDFs for dev testing. "
                    "Extend max_docs for full run.")

    logger.info(f"Parsing {len(pdf_paths)} PDFs from {pdf_dir}")
    Path(parsed_dir).mkdir(parents=True, exist_ok=True)

    for i, pdf_path in enumerate(pdf_paths):
        doc_id = pdf_path.stem
        out_path = Path(parsed_dir) / f"{doc_id}.json"

        if out_path.exists():
            logger.info(f"[{i+1}/{len(pdf_paths)}] Skipping (cached): {doc_id}")
            continue

        logger.info(f"[{i+1}/{len(pdf_paths)}] Parsing: {doc_id}")
        try:
            art_dir = Path(parsed_dir) / "artifacts" / doc_id
            result = parse_pdf(str(pdf_path), str(art_dir))
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to parse {doc_id}: {e}")


def load_parsed(parsed_dir: str, doc_id: str) -> Optional[dict]:
    """Load a cached parsed JSON result for a given doc_id."""
    path = Path(parsed_dir) / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
