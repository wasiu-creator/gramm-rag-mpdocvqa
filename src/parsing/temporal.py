"""
src/parsing/temporal.py
───────────────────────
Temporal marker extraction for GraMM-RAG.

Step 3: Apply regex patterns first; fall back to GPT-4o-mini API for
        ambiguous / implied temporal references.

API cost estimate: ~$0.50 per benchmark (very few ambiguous cases).
Budget: ~$5 total across all benchmarks.
"""

import re
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Regex patterns (deterministic, free) ─────────────────────────────────────
PATTERNS = [
    # Fiscal year: FY2023, FY 2023
    (r"\bFY\s?\d{4}\b", "fiscal_year"),
    # Quarter: Q1 2023, Q3-2022
    (r"\bQ[1-4][\s\-]?\d{4}\b", "quarter"),
    # ISO date: 2023-04-15
    (r"\b\d{4}-\d{2}-\d{2}\b", "iso_date"),
    # US date: 04/15/2023
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "us_date"),
    # Year alone (4-digit, 1900–2099): loosely matched
    (r"\b(19|20)\d{2}\b", "year"),
    # Natural language date: "January 15, 2023" or "15 January 2023"
    (
        r"\b(?:January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        "natural",
    ),
    (
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\s+\d{4}\b",
        "natural",
    ),
    # Relative: "last year", "previous quarter", "this month" — flag for LLM
    (r"\b(?:last|previous|this|next|current)\s+(?:year|quarter|month|week)\b", "relative"),
]

# Compile for speed
_COMPILED = [(re.compile(p, re.IGNORECASE), t) for p, t in PATTERNS]


def extract_temporal(text: str) -> list[dict]:
    """
    Extract temporal markers from a text string using regex.

    Returns list of dicts with keys: text, type, start, end, needs_llm.
    'needs_llm' is True when the match is a relative reference that requires
    document context to resolve to an absolute date.
    """
    markers = []
    seen_spans = set()

    for compiled_re, ttype in _COMPILED:
        for match in compiled_re.finditer(text):
            span = (match.start(), match.end())
            # Avoid duplicate / overlapping matches
            if any(s[0] <= span[0] < s[1] for s in seen_spans):
                continue
            seen_spans.add(span)
            markers.append({
                "text": match.group(),
                "type": ttype,
                "start": match.start(),
                "end": match.end(),
                "needs_llm": ttype == "relative",   # flag ambiguous cases
            })

    return sorted(markers, key=lambda m: m["start"])


def resolve_relative_temporal(
    text: str,
    markers: list[dict],
    client,                      # openai.OpenAI() instance
    model: str = "gpt-4o-mini",  # cheap model for disambiguation
) -> list[dict]:
    """
    Use GPT-4o-mini to resolve relative temporal references ("last year", etc.)
    to absolute dates, using surrounding document context.

    Only called for markers where needs_llm=True.
    API cost: ~$0.001 per call (very cheap).

    Args:
        text:    Full paragraph / element text providing context.
        markers: Output of extract_temporal() — only relative ones are sent.
        client:  openai.OpenAI() instance.
        model:   OpenAI model to use. Default: gpt-4o-mini.

    Returns:
        Updated markers list with 'resolved_text' and 'resolved_type' added.
    """
    relative_markers = [m for m in markers if m["needs_llm"]]
    if not relative_markers:
        return markers   # nothing to resolve

    prompt = (
        "Given the following document excerpt, resolve each relative temporal "
        "expression to an absolute date or date range if possible. "
        "Return a JSON array with one object per expression, each with keys: "
        "'original', 'resolved', 'type' (one of: year, quarter, fiscal_year, "
        "iso_date, natural, unresolvable).\n\n"
        f"Text: {text}\n\n"
        f"Expressions: {[m['text'] for m in relative_markers]}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a temporal resolution assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        resolved = json.loads(resp.choices[0].message.content)
        resolved_list = resolved.get("items", resolved)  # handle both formats

        # Merge resolved data back into the relevant markers
        orig_to_resolved = {r["original"]: r for r in resolved_list}
        for m in markers:
            if m["needs_llm"] and m["text"] in orig_to_resolved:
                r = orig_to_resolved[m["text"]]
                m["resolved_text"] = r.get("resolved", m["text"])
                m["resolved_type"] = r.get("type", m["type"])

        time.sleep(0.5)   # rate-limit: stay within OpenAI tier limits

    except Exception as e:
        logger.warning(f"GPT-4o-mini temporal resolution failed: {e}")

    return markers


def annotate_parsed_elements(
    parsed: dict,
    openai_client=None,
    use_llm_fallback: bool = True,
) -> dict:
    """
    Annotate every text element in a parsed document with temporal markers.

    Args:
        parsed:          Output of mineru_wrapper.parse_pdf().
        openai_client:   openai.OpenAI() instance for LLM fallback.
        use_llm_fallback: Whether to call GPT-4o-mini for relative markers.

    Returns:
        Parsed dict with 'temporal_markers' added to each element.
    """
    for elem in parsed.get("elements", []):
        text = elem.get("text", "")
        markers = extract_temporal(text)

        if use_llm_fallback and openai_client and any(m["needs_llm"] for m in markers):
            markers = resolve_relative_temporal(text, markers, openai_client)

        elem["temporal_markers"] = markers

    return parsed
