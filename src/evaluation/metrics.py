"""
src/evaluation/metrics.py
──────────────────────────
Evaluation metrics for GraMM-RAG (Step 14).

Implements:
  - ANLS  (Average Normalised Levenshtein Similarity) — DocVQA standard
  - F1    (token-level F1, SQuAD-style)
  - Accuracy (exact match after normalisation)
  - Abstention Precision/Recall (unanswerable-question detection)
  - APPA  (Answer-Page Prediction Accuracy) — MP-DocVQA retrieval localisation:
           did retrieval surface the page that actually holds the answer?

  NOTE: `compute_appa()` (abstention) and `compute_answer_page_accuracy()`
  (the proposal's APPA) are DIFFERENT metrics that historically shared the
  "APPA" acronym. They are kept distinct here on purpose.

All metrics operate on lists of (prediction, gold_answer) pairs.
"""

import re
import string
from collections import Counter
from typing import Union


# ─── Text normalisation ────────────────────────────────────────────────────────

def normalise_answer(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─── Levenshtein distance (for ANLS) ──────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def anls_score(prediction: str, gold: str, threshold: float = 0.5) -> float:
    """
    ANLS for a single prediction/gold pair.
    Score = max(0, 1 - NLS) if NLS < threshold, else 0.
    NLS = edit_distance / max(len(pred), len(gold)).
    """
    pred = normalise_answer(prediction)
    g = normalise_answer(gold)
    if not pred and not g:
        return 1.0
    if not pred or not g:
        return 0.0
    dist = levenshtein(pred, g)
    nls = dist / max(len(pred), len(g))
    return max(0.0, 1.0 - nls) if nls < threshold else 0.0


def compute_anls(
    predictions: list[str],
    gold_answers: list[Union[str, list[str]]],
    threshold: float = 0.5,
) -> float:
    """
    ANLS averaged over all examples.
    gold_answers may be a list of strings (multiple valid answers).
    """
    assert len(predictions) == len(gold_answers), "Length mismatch"
    scores = []
    for pred, gold in zip(predictions, gold_answers):
        if isinstance(gold, list):
            score = max(anls_score(pred, g, threshold) for g in gold)
        else:
            score = anls_score(pred, gold, threshold)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ─── Token-level F1 ───────────────────────────────────────────────────────────

def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style token-level F1 for a single pair."""
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(
    predictions: list[str],
    gold_answers: list[Union[str, list[str]]],
) -> float:
    """Macro-averaged token F1 over all examples."""
    assert len(predictions) == len(gold_answers)
    scores = []
    for pred, gold in zip(predictions, gold_answers):
        if isinstance(gold, list):
            score = max(token_f1(pred, g) for g in gold)
        else:
            score = token_f1(pred, gold)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ─── Exact match accuracy ──────────────────────────────────────────────────────

def compute_accuracy(
    predictions: list[str],
    gold_answers: list[Union[str, list[str]]],
) -> float:
    """Exact match accuracy after normalisation."""
    assert len(predictions) == len(gold_answers)
    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        pred_n = normalise_answer(pred)
        if isinstance(gold, list):
            correct += int(any(pred_n == normalise_answer(g) for g in gold))
        else:
            correct += int(pred_n == normalise_answer(gold))
    return correct / len(predictions) if predictions else 0.0


# ─── APPA: Abstention Precision & Recall ──────────────────────────────────────

def compute_appa(
    predictions: list[str],
    is_answerable: list[bool],
    refused_flags: list[bool],
) -> dict:
    """
    APPA (Abstention Precision / Recall) for unanswerable question detection.

    Args:
        predictions:   Model answer strings.
        is_answerable: True if the question has a ground-truth answer.
        refused_flags: True if the model refused to answer (R < τ).

    Returns:
        Dict with keys: abstention_precision, abstention_recall, abstention_f1,
                        overall_accuracy (answerable only).
    """
    assert len(predictions) == len(is_answerable) == len(refused_flags)

    # Abstention = model said "I don't know" (refused=True)
    # Unanswerable = ground truth has no answer (is_answerable=False)
    tp = sum(1 for a, r in zip(is_answerable, refused_flags) if not a and r)
    fp = sum(1 for a, r in zip(is_answerable, refused_flags) if a and r)
    fn = sum(1 for a, r in zip(is_answerable, refused_flags) if not a and not r)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "abstention_precision": precision,
        "abstention_recall":    recall,
        "abstention_f1":        f1,
        "abstention_tp":        tp,
        "abstention_fp":        fp,
        "abstention_fn":        fn,
    }


# ─── APPA: Answer-Page Prediction Accuracy (retrieval localisation) ───────────
# Distinct from compute_appa() above (abstention). This is the proposal's
# MP-DocVQA metric: of the top-k retrieved nodes, do any sit on the page
# that actually contains the gold answer?

def compute_answer_page_accuracy(
    retrieved_pages: list,
    gold_pages: list,
    k_values: tuple = (1, 5, 10),
) -> dict:
    """
    APPA — Answer-Page Prediction Accuracy.

    Args:
        retrieved_pages: list (len = n questions). Element i is an ordered
                         list of page numbers for that question's retrieved
                         nodes, ranked by retrieval score (index 0 = top node).
        gold_pages:      list (len = n questions). Element i is the gold
                         answer page (int), or a list/set of acceptable pages.
        k_values:        cut-offs to report success@k at.

    Returns:
        {'appa@1':.., 'appa@5':.., 'appa@10':.., 'appa':..}
        where 'appa' aliases the largest-k value (full retrieved set).
    """
    assert len(retrieved_pages) == len(gold_pages), "Length mismatch"
    n = len(gold_pages)
    if n == 0:
        out = {f"appa@{k}": 0.0 for k in k_values}
        out["appa"] = 0.0
        return out

    out = {}
    for k in k_values:
        hits = 0
        for pages, gold in zip(retrieved_pages, gold_pages):
            gold_set = (set(gold) if isinstance(gold, (list, set, tuple))
                        else {gold})
            if any(p in gold_set for p in list(pages)[:k]):
                hits += 1
        out[f"appa@{k}"] = hits / n
    out["appa"] = out[f"appa@{max(k_values)}"]
    return out


# ─── Consolidated metric report ───────────────────────────────────────────────

def compute_all_metrics(
    predictions: list[str],
    gold_answers: list[Union[str, list[str]]],
    is_answerable: list[bool],
    refused_flags: list[bool],
    retrieved_pages: list = None,
    gold_pages: list = None,
) -> dict:
    """
    Run all metrics and return a consolidated results dict.

    If `retrieved_pages` and `gold_pages` are supplied, APPA (Answer-Page
    Prediction Accuracy) is included. Omitting them keeps the call
    backward-compatible with callers that only have answer-level outputs.
    """
    result = {
        "anls":  compute_anls(predictions, gold_answers),
        "f1":    compute_f1(predictions, gold_answers),
        "accuracy": compute_accuracy(predictions, gold_answers),
        **compute_appa(predictions, is_answerable, refused_flags),
        "n_total":       len(predictions),
        "n_answerable":  sum(is_answerable),
        "n_unanswerable": sum(not a for a in is_answerable),
        "n_refused":      sum(refused_flags),
    }
    if retrieved_pages is not None and gold_pages is not None:
        result.update(
            compute_answer_page_accuracy(retrieved_pages, gold_pages))
    return result
