"""
Statistical analysis for spectral-vlm.

Per (layer, metric, view, agg) cell computes:
  - Cohen's d
  - Mann-Whitney U p-value
  - Bootstrap 95% CI on Cohen's d (≥10k resamples)
  - Benjamini-Hochberg FDR (q-values) across the full grid

All contrasts are one-sided (direction inferred from mean difference).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Effect-size computation
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: (mean_a - mean_b) / pooled_std."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def bootstrap_ci_d(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% CI on Cohen's d."""
    rng = np.random.default_rng(seed)
    ds = []
    na, nb = len(a), len(b)
    for _ in range(n_resamples):
        ra = a[rng.integers(0, na, size=na)]
        rb = b[rng.integers(0, nb, size=nb)]
        ds.append(cohens_d(ra, rb))
    alpha = (1 - ci) / 2
    lo = float(np.quantile(ds, alpha))
    hi = float(np.quantile(ds, 1 - alpha))
    return lo, hi


# ---------------------------------------------------------------------------
# BH-FDR correction
# ---------------------------------------------------------------------------

def bh_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    Returns q-values (adjusted p-values) in the same order as input.
    """
    n = len(p_values)
    if n == 0:
        return np.array([])
    order = np.argsort(p_values)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    q = np.minimum(1.0, p_values * n / ranks)
    # Ensure monotonicity (take cumulative min from right).
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    return q


# ---------------------------------------------------------------------------
# Per-cell test
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    layer: int
    metric: str
    view: str
    agg: str
    mean_a: float
    mean_b: float
    d: float
    d_lo: float   # 95% CI lower
    d_hi: float   # 95% CI upper
    p_value: float
    q_value: float = float("nan")   # filled in after FDR correction
    n_a: int = 0
    n_b: int = 0


def test_cell(
    values_a: np.ndarray,
    values_b: np.ndarray,
    layer: int,
    metric: str,
    view: str,
    agg: str,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> CellResult:
    d     = cohens_d(values_a, values_b)
    lo, hi = bootstrap_ci_d(values_a, values_b, n_bootstrap, seed=seed)
    _, p  = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    return CellResult(
        layer=layer, metric=metric, view=view, agg=agg,
        mean_a=float(np.mean(values_a)), mean_b=float(np.mean(values_b)),
        d=d, d_lo=lo, d_hi=hi, p_value=float(p),
        n_a=len(values_a), n_b=len(values_b),
    )


# ---------------------------------------------------------------------------
# Full grid analysis
# ---------------------------------------------------------------------------

def run_full_analysis(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    layer_col: str = "layer",
    value_col: str = "value",
    metric_col: str = "metric",
    view_col: str = "view",
    agg_col: str = "agg",
    n_bootstrap: int = 10_000,
    fdr_alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs the full layer × metric × view × agg grid test for one contrast
    (group_a vs group_b).

    Input df must have columns:
        [group_col, layer_col, metric_col, view_col, agg_col, value_col]
    with one row per (example, layer, metric, view, agg).

    Returns a DataFrame with one row per cell, columns:
        layer, metric, view, agg, mean_a, mean_b, d, d_lo, d_hi, p_value, q_value, n_a, n_b
    """
    df_a = df[df[group_col] == group_a]
    df_b = df[df[group_col] == group_b]

    cells: List[CellResult] = []

    layers  = sorted(df[layer_col].unique())
    metrics = sorted(df[metric_col].unique())
    views   = sorted(df[view_col].unique())
    aggs    = sorted(df[agg_col].unique())

    for layer in layers:
        for metric in metrics:
            for view in views:
                for agg in aggs:
                    mask = (
                        (df[layer_col]  == layer) &
                        (df[metric_col] == metric) &
                        (df[view_col]   == view) &
                        (df[agg_col]    == agg)
                    )
                    va = df_a[mask][value_col].dropna().values
                    vb = df_b[mask][value_col].dropna().values
                    if len(va) < 2 or len(vb) < 2:
                        continue
                    cells.append(test_cell(va, vb, layer, metric, view, agg,
                                           n_bootstrap, seed))

    if not cells:
        logger.warning("No cells computed — check group labels and data columns.")
        return pd.DataFrame()

    # BH-FDR across the entire grid.
    p_vals = np.array([c.p_value for c in cells])
    q_vals = bh_fdr(p_vals, alpha=fdr_alpha)
    for c, q in zip(cells, q_vals):
        c.q_value = float(q)

    rows = [
        {
            "layer": c.layer, "metric": c.metric, "view": c.view, "agg": c.agg,
            "mean_a": c.mean_a, "mean_b": c.mean_b,
            "d": c.d, "d_lo": c.d_lo, "d_hi": c.d_hi,
            "p_value": c.p_value, "q_value": c.q_value,
            "n_a": c.n_a, "n_b": c.n_b,
        }
        for c in cells
    ]
    result_df = pd.DataFrame(rows)
    result_df["significant"] = result_df["q_value"] < fdr_alpha
    return result_df
