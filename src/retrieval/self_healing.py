"""
src/retrieval/self_healing.py
──────────────────────────────
Self-healing retrieval via reward function R(q, K) for GraMM-RAG (Step 10).

R(q, K) = α·C(q,K) + β·S(q,K) + λ·T(q,K)

Where:
  C(q,K) = graph coverage score    (fraction of query entities covered by K)
  S(q,K) = vector similarity score  (mean cosine similarity of top-K nodes)
  T(q,K) = temporal coherence score (agreement of temporal markers in K)

  α, β, λ are tuned via grid search (Step 10 / tune_reward.py).
  τ (tau) is the refusal threshold: if R < τ, trigger expansion or refuse.

Default params (will be overridden by models/reward/best_params.json after tuning):
  α=0.40, β=0.35, λ=0.25, τ=0.60
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default reward hyperparameters (overridden after grid search in Step 10)
DEFAULT_ALPHA = 0.40
DEFAULT_BETA = 0.35
DEFAULT_LAMBDA = 0.25
DEFAULT_TAU = 0.60


class RewardFunction:
    """
    Computes R(q, K) and drives self-healing expansion / refusal decisions.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        lambda_: float = DEFAULT_LAMBDA,
        tau: float = DEFAULT_TAU,
        params_path: Optional[str] = None,
    ):
        if params_path and Path(params_path).exists():
            self._load_params(params_path)
        else:
            self.alpha = alpha
            self.beta = beta
            self.lambda_ = lambda_
            self.tau = tau

        logger.info(
            f"RewardFunction: α={self.alpha}, β={self.beta}, "
            f"λ={self.lambda_}, τ={self.tau}"
        )

    def _load_params(self, path: str):
        with open(path) as f:
            p = json.load(f)
        self.alpha = p["alpha"]
        self.beta = p["beta"]
        self.lambda_ = p["lambda_"]
        self.tau = p["tau"]
        logger.info(f"Loaded reward params from {path}")

    def compute(
        self,
        query_emb: torch.Tensor,         # [D] L2-normalised query embedding
        retrieved_nodes: list[dict],      # output of retrieve_top_k()
        node_emb_dict: dict,             # {node_type: Tensor[N, D]}
        parsed: dict,                    # parsed document (for temporal info)
        query_entities: list[str],       # spaCy NER entities from query
    ) -> dict:
        """
        Compute R(q, K) for a retrieved candidate set K.

        Returns dict with keys: reward, coverage, similarity, temporal, should_refuse, should_expand.
        """
        if not retrieved_nodes:
            return {
                "reward": 0.0, "coverage": 0.0,
                "similarity": 0.0, "temporal": 0.0,
                "should_refuse": True, "should_expand": False,
            }

        C = self._coverage_score(retrieved_nodes, parsed, query_entities)
        S = self._similarity_score(query_emb, retrieved_nodes, node_emb_dict)
        T = self._temporal_coherence_score(retrieved_nodes, parsed)

        R = self.alpha * C + self.beta * S + self.lambda_ * T
        R = float(max(0.0, min(1.0, R)))   # clamp to [0, 1]

        result = {
            "reward": R,
            "coverage": C,
            "similarity": S,
            "temporal": T,
            "should_refuse": R < self.tau * 0.6,     # very low → refuse outright
            "should_expand": self.tau * 0.6 <= R < self.tau,  # medium → expand
        }
        return result

    def _coverage_score(
        self,
        retrieved_nodes: list[dict],
        parsed: dict,
        query_entities: list[str],
    ) -> float:
        """
        C(q, K): fraction of query named entities that appear in retrieved text.
        """
        if not query_entities:
            return 0.5   # neutral if no entities extracted

        elements = parsed.get("elements", [])
        retrieved_texts = []
        for node in retrieved_nodes:
            local_idx = node["local_idx"]
            ntype = node["node_type"]
            typed = [e for e in elements if e["type"] == ntype]
            if local_idx < len(typed):
                retrieved_texts.append(typed[local_idx].get("text", "").lower())

        combined = " ".join(retrieved_texts)
        found = sum(1 for ent in query_entities if ent.lower() in combined)
        return found / len(query_entities)

    def _similarity_score(
        self,
        query_emb: torch.Tensor,
        retrieved_nodes: list[dict],
        node_emb_dict: dict,
    ) -> float:
        """
        S(q, K): mean cosine similarity between query and retrieved node embeddings.
        """
        sims = [n["score"] for n in retrieved_nodes if "score" in n]
        if not sims:
            return 0.0
        return float(sum(sims) / len(sims))

    def _temporal_coherence_score(
        self,
        retrieved_nodes: list[dict],
        parsed: dict,
    ) -> float:
        """
        T(q, K): temporal coherence — are retrieved nodes from consistent time periods?
        Score = 1.0 if all temporal markers agree, 0.5 if no markers, 0.0 if conflict.
        """
        elements = parsed.get("elements", [])
        temporal_sets = []

        for node in retrieved_nodes:
            local_idx = node["local_idx"]
            ntype = node["node_type"]
            typed = [e for e in elements if e["type"] == ntype]
            if local_idx < len(typed):
                markers = typed[local_idx].get("temporal_markers", [])
                time_texts = frozenset(
                    m.get("resolved_text", m["text"]).upper()
                    for m in markers
                    if m["type"] in ("fiscal_year", "quarter", "year", "iso_date")
                )
                if time_texts:
                    temporal_sets.append(time_texts)

        if not temporal_sets:
            return 0.5   # neutral: no temporal info

        # Check overlap: if any two retrieved nodes share no temporal markers → conflict
        all_times = temporal_sets[0]
        for ts in temporal_sets[1:]:
            all_times = all_times & ts

        if all_times:
            return 1.0   # all agree on at least one time period
        return 0.0        # complete conflict


class SelfHealingRetriever:
    """
    Wraps HGT retrieval + RewardFunction into an iterative self-healing loop.

    Loop:
      1. Retrieve top-K nodes
      2. Compute R(q, K)
      3. If R < τ and rounds < max_rounds: expand K and retry
      4. If R < τ*0.6: refuse to answer
      5. Otherwise: return K for generation
    """

    def __init__(
        self,
        hgt_model,
        reward_fn: RewardFunction,
        top_k: int = 10,
        expansion_k: int = 5,
        max_rounds: int = 2,
    ):
        self.hgt = hgt_model
        self.reward = reward_fn
        self.top_k = top_k
        self.expansion_k = expansion_k
        self.max_rounds = max_rounds

    def retrieve(
        self,
        query_emb: torch.Tensor,
        graph_data,                 # PyG HeteroData
        parsed: dict,
        query_entities: list[str],
    ) -> dict:
        """
        Run self-healing retrieval for one query.

        Returns:
            {
              nodes:         list of retrieved node dicts,
              reward:        final R(q, K),
              refused:       True if R < hard threshold,
              rounds:        number of expansion rounds used,
              reward_detail: full reward breakdown dict,
            }
        """
        from src.retrieval.hgt_model import retrieve_top_k

        # Encode graph nodes through HGT
        with torch.no_grad():
            node_emb_dict = self.hgt.encode_nodes(
                graph_data.x_dict if hasattr(graph_data, "x_dict") else {},
                graph_data.edge_index_dict if hasattr(graph_data, "edge_index_dict") else {},
            )

        k = self.top_k
        for round_i in range(self.max_rounds + 1):
            nodes = retrieve_top_k(query_emb, node_emb_dict, top_k=k)
            reward_detail = self.reward.compute(
                query_emb, nodes, node_emb_dict, parsed, query_entities
            )

            logger.debug(
                f"Round {round_i}: R={reward_detail['reward']:.3f}, "
                f"k={k}, expand={reward_detail['should_expand']}, "
                f"refuse={reward_detail['should_refuse']}"
            )

            if reward_detail["should_refuse"]:
                return {
                    "nodes": nodes,
                    "reward": reward_detail["reward"],
                    "refused": True,
                    "rounds": round_i,
                    "reward_detail": reward_detail,
                }

            if not reward_detail["should_expand"] or round_i >= self.max_rounds:
                break

            # Expand: retrieve more nodes next round
            k += self.expansion_k

        return {
            "nodes": nodes,
            "reward": reward_detail["reward"],
            "refused": False,
            "rounds": round_i,
            "reward_detail": reward_detail,
        }
