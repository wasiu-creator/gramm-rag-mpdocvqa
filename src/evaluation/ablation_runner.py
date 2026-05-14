"""
src/evaluation/ablation_runner.py
───────────────────────────────────
7-variant ablation runner for GraMM-RAG (Step 13).

Each variant disables exactly one pipeline component:
  1. no_layout   — remove spatial + reading_order + hierarchy edges
  2. no_temporal — remove temporal edges
  3. no_gnn      — replace HGT with keyword graph traversal
  4. no_routing  — always use GRAPH route (no DeBERTa router)
  5. no_healing  — skip reward check / self-healing expansion
  6. vector_only — disable graph entirely, keep self-healing on FAISS results
  7. gnn_compare — swap HGTConv for GATConv / SAGEConv (done in Step 11)

Ablations run on DocBench only (faster), 3 seeds.
"""

import logging
import copy
from typing import Optional

logger = logging.getLogger(__name__)


# Ablation config flags — each dict specifies what to disable/override
ABLATION_CONFIGS = {
    "no_layout": {
        "disable_edge_types": ["spatial", "reading_order", "hierarchy"],
        "description": "Remove all layout-based edges (spatial, reading order, hierarchy)",
    },
    "no_temporal": {
        "disable_edge_types": ["temporal"],
        "description": "Remove temporal edges",
    },
    "no_gnn": {
        "use_keyword_traversal": True,
        "disable_edge_types": [],
        "description": "Replace HGT with keyword-based BFS graph traversal",
    },
    "no_routing": {
        "always_graph": True,
        "description": "Force GRAPH route for all queries (no DeBERTa router)",
    },
    "no_healing": {
        "skip_reward_check": True,
        "description": "Skip self-healing reward check and expansion",
    },
    "vector_only": {
        "disable_graph": True,
        "keep_healing": True,
        "description": "Disable graph retrieval, use FAISS only (with self-healing)",
    },
    # gnn_compare is handled separately in 08_gnn_comparison.ipynb (Step 11)
}


class AblationRunner:
    """
    Runs all 7 ablation variants by patching a GraMM-RAG pipeline config.
    Reuses the same parsed graphs and embeddings — only the retrieval/routing
    logic changes per variant.
    """

    def __init__(self, base_pipeline_fn, results_dir: str = "results/ablations"):
        """
        Args:
            base_pipeline_fn: Callable that accepts (config_overrides, seed) and
                              returns evaluation results dict.
            results_dir:      Directory to save ablation result JSONs.
        """
        self.pipeline_fn = base_pipeline_fn
        self.results_dir = results_dir

    def run_variant(
        self,
        variant_name: str,
        seeds: list = None,
        benchmark: str = "docbench",
        max_questions: Optional[int] = 100,   # ← EXTEND: set to None for full run
    ) -> dict:
        """
        Run one ablation variant across all seeds.

        Args:
            variant_name:  Key from ABLATION_CONFIGS.
            seeds:         List of random seeds (default [42, 123, 456]).
            benchmark:     Benchmark to evaluate on (default: docbench).
            max_questions: Cap on questions per seed.
                           # LIMIT: 100 for dev testing.
                           # ← EXTEND: set to None for full paper ablations.

        Returns:
            Dict with per-seed results and aggregate stats.
        """
        if seeds is None:
            seeds = [42, 123, 456]

        if variant_name not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation variant: {variant_name}. "
                             f"Choose from: {list(ABLATION_CONFIGS)}")

        config_overrides = copy.deepcopy(ABLATION_CONFIGS[variant_name])
        config_overrides["benchmark"] = benchmark
        config_overrides["max_questions"] = max_questions

        logger.info(f"Running ablation: {variant_name} — {config_overrides['description']}")

        per_seed_results = {}
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            try:
                result = self.pipeline_fn(config_overrides, seed=seed)
                per_seed_results[seed] = result
            except Exception as e:
                logger.error(f"  Failed seed {seed}: {e}")
                per_seed_results[seed] = {"error": str(e)}

        # Aggregate across seeds
        agg = self._aggregate_seeds(per_seed_results)
        return {
            "variant":     variant_name,
            "description": config_overrides["description"],
            "benchmark":   benchmark,
            "seeds":       per_seed_results,
            "aggregate":   agg,
        }

    def run_all(
        self,
        seeds: list = None,
        benchmark: str = "docbench",
        max_questions: Optional[int] = 100,   # ← EXTEND: None for full paper run
    ) -> dict:
        """Run all 7 ablation variants and return combined results dict."""
        all_results = {}
        for variant_name in ABLATION_CONFIGS:
            all_results[variant_name] = self.run_variant(
                variant_name, seeds=seeds, benchmark=benchmark,
                max_questions=max_questions,
            )
        return all_results

    def _aggregate_seeds(self, per_seed_results: dict) -> dict:
        """Compute mean ± std across seeds for each metric."""
        import numpy as np
        metrics = ["anls", "f1", "accuracy", "abstention_f1"]
        agg = {}
        for metric in metrics:
            vals = [
                r[metric] for r in per_seed_results.values()
                if isinstance(r, dict) and metric in r
            ]
            if vals:
                agg[metric] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "n":    len(vals),
                }
        return agg


# ─── Keyword-BFS traversal (fallback for no_gnn ablation) ─────────────────────

def keyword_graph_traversal(
    query: str,
    graph_data,
    parsed: dict,
    top_k: int = 10,
) -> list[dict]:
    """
    Simple BFS graph traversal using keyword overlap (no GNN).
    Used as the retrieval mechanism in the 'no_gnn' ablation variant.

    Scores nodes by token overlap between query and node text,
    then expands via one hop of graph edges.
    """
    query_tokens = set(query.lower().split())
    elements = parsed.get("elements", [])

    scored = []
    for i, elem in enumerate(elements):
        text_tokens = set(elem.get("text", "").lower().split())
        overlap = len(query_tokens & text_tokens)
        if overlap > 0:
            scored.append({"global_idx": i, "score": overlap / len(query_tokens),
                           "node_type": elem["type"], "local_idx": i})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
