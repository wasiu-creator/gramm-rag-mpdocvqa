"""
src/evaluation/bootstrap.py
────────────────────────────
Paired bootstrap significance testing for GraMM-RAG (Step 14).

Used to test whether GraMM-RAG significantly outperforms each baseline
across all benchmarks. Computes:
  - Mean difference in metric score
  - p-value (one-sided: does A beat B?)
  - 95% bootstrap confidence interval

Reference: Efron & Tibshirani (1994). "An Introduction to the Bootstrap."
"""

import numpy as np
from typing import Callable, Optional


def paired_bootstrap(
    scores_a: np.ndarray,          # per-question scores for method A (GraMM-RAG)
    scores_b: np.ndarray,          # per-question scores for method B (baseline)
    n_bootstrap: int = 1000,       # number of bootstrap resamples
    alpha: float = 0.05,           # significance level (two-sided: 2.5%–97.5% CI)
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap significance test.

    Args:
        scores_a:    Per-sample metric scores for method A, shape [N].
        scores_b:    Per-sample metric scores for method B, shape [N].
        n_bootstrap: Number of bootstrap samples.
        alpha:       Significance level for CI (default 0.05 → 95% CI).
        seed:        Random seed for reproducibility.

    Returns:
        Dict with keys:
          diff        — observed mean difference (A - B)
          p_value     — one-sided p-value (prob bootstrap diff ≤ 0)
          ci_low      — lower bound of (1-alpha)% CI
          ci_high     — upper bound of (1-alpha)% CI
          significant — True if p_value < alpha
    """
    assert len(scores_a) == len(scores_b), "Score arrays must be same length"
    rng = np.random.default_rng(seed)
    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)
    diffs = scores_a - scores_b
    observed_diff = diffs.mean()

    # Bootstrap resampling
    boot_means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    # One-sided p-value: P(boot_mean ≤ 0) under H0 that A = B
    p_value = float((boot_means <= 0).mean())
    ci_low, ci_high = float(np.percentile(boot_means, 100 * alpha / 2)), \
                      float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "diff":        float(observed_diff),
        "p_value":     p_value,
        "ci_low":      ci_low,
        "ci_high":     ci_high,
        "significant": p_value < alpha,
        "n_samples":   len(scores_a),
        "n_bootstrap": n_bootstrap,
    }


def run_all_significance_tests(
    gramm_results: dict,     # {benchmark: {metric: [per_sample_scores]}}
    baseline_results: dict,  # {baseline_name: {benchmark: {metric: [scores]}}}
    metrics: list = None,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Run paired bootstrap for GraMM-RAG vs every baseline on every benchmark.

    Args:
        gramm_results:    Dict of GraMM-RAG per-sample scores.
        baseline_results: Dict of baseline per-sample scores.
        metrics:          List of metric names to test (default: anls, f1, accuracy).
        n_bootstrap:      Bootstrap resamples.

    Returns:
        Nested dict: {baseline: {benchmark: {metric: bootstrap_result_dict}}}
    """
    if metrics is None:
        metrics = ["anls", "f1", "accuracy"]

    results = {}
    for baseline_name, baseline_data in baseline_results.items():
        results[baseline_name] = {}
        for benchmark, gramm_bench in gramm_results.items():
            results[baseline_name][benchmark] = {}
            baseline_bench = baseline_data.get(benchmark, {})
            for metric in metrics:
                a_scores = np.array(gramm_bench.get(metric, []))
                b_scores = np.array(baseline_bench.get(metric, []))
                if len(a_scores) == 0 or len(b_scores) == 0:
                    continue
                # Align lengths (take minimum for mismatched runs)
                n = min(len(a_scores), len(b_scores))
                results[baseline_name][benchmark][metric] = paired_bootstrap(
                    a_scores[:n], b_scores[:n], n_bootstrap=n_bootstrap
                )

    return results


def format_significance_table(test_results: dict) -> str:
    """
    Format significance test results as a readable text table.
    Suitable for copy-paste into the paper.
    """
    lines = []
    lines.append(f"{'Baseline':<20} {'Benchmark':<15} {'Metric':<10} "
                 f"{'Diff':>8} {'p-value':>10} {'95% CI':>20} {'Sig?':>6}")
    lines.append("-" * 75)

    for baseline, bench_data in test_results.items():
        for benchmark, metric_data in bench_data.items():
            for metric, r in metric_data.items():
                ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                sig_str = "✓" if r["significant"] else "✗"
                lines.append(
                    f"{baseline:<20} {benchmark:<15} {metric:<10} "
                    f"{r['diff']:>+8.4f} {r['p_value']:>10.4f} {ci_str:>20} {sig_str:>6}"
                )

    return "\n".join(lines)
