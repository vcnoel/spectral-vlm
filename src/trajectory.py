"""
Assemble and serialise the spectral trajectory for one example.

The trajectory is a numpy array of shape
    [n_layers, n_metrics * n_views * n_aggs]
= [n_layers, 4 * 3 * 2] = [n_layers, 24]

Column ordering follows:
    for metric in [fiedler, entropy, smoothness, hfer]:
      for view in [vv, tv, full]:
        for agg in [mean, max]:
          → one column

This flat vector is what gets passed to the logistic-regression classifier.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from .spectral import LayerSpectralMetrics, METRIC_NAMES, VIEW_NAMES, AGG_NAMES

N_METRICS = len(METRIC_NAMES)
N_VIEWS   = len(VIEW_NAMES)
N_AGGS    = len(AGG_NAMES)
N_FEATURES_PER_LAYER = N_METRICS * N_VIEWS * N_AGGS   # 24


def feature_names() -> List[str]:
    cols = []
    for m in METRIC_NAMES:
        for v in VIEW_NAMES:
            for a in AGG_NAMES:
                cols.append(f"{m}_{v}_{a}")
    return cols


def trajectory_to_array(layer_metrics: List[LayerSpectralMetrics]) -> np.ndarray:
    """
    Converts list of per-layer metrics into [n_layers, 24] float32 array.
    """
    n_layers = len(layer_metrics)
    traj = np.zeros((n_layers, N_FEATURES_PER_LAYER), dtype=np.float32)
    for li, lm in enumerate(layer_metrics):
        traj[li] = lm.data.reshape(-1)  # [4, 3, 2] → [24]
    return traj


def trajectory_to_flat(layer_metrics: List[LayerSpectralMetrics]) -> np.ndarray:
    """Flatten entire trajectory to a 1-D vector of length n_layers*24."""
    return trajectory_to_array(layer_metrics).reshape(-1)


def trajectory_to_dataframe(
    layer_metrics: List[LayerSpectralMetrics],
    example_id: str = "",
    label: str = "",
) -> pd.DataFrame:
    """
    Long-form DataFrame with one row per (layer, feature).
    Columns: example_id, label, layer, metric, view, agg, value.
    """
    rows = []
    for li, lm in enumerate(layer_metrics):
        for mi, m in enumerate(METRIC_NAMES):
            for vi, v in enumerate(VIEW_NAMES):
                for ai, a in enumerate(AGG_NAMES):
                    rows.append({
                        "example_id": example_id,
                        "label": label,
                        "layer": li,
                        "metric": m,
                        "view": v,
                        "agg": a,
                        "value": float(lm.data[mi, vi, ai]),
                    })
    return pd.DataFrame(rows)
