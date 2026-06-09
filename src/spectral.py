"""
Spectral analysis of VLM attention using spectral_trust primitives.

Given a ForwardResult, computes the four GSP metrics
(Fiedler λ₂, spectral entropy, smoothness/Dirichlet energy, HFER)
across three views (A_vv, A_tv, A_full) with two head aggregations
(mean, max), for every layer.

Output shape: [n_layers, n_metrics=4, n_views=3, n_aggs=2]
Metric order : [fiedler, entropy, smoothness, hfer]
View order   : [vv, tv, full]
Agg order    : [mean, max]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer, SpectralDiagnostics
from .extract import ForwardResult

logger = logging.getLogger(__name__)

# Canonical ordering (must match trajectory.py)
METRIC_NAMES = ["fiedler", "entropy", "smoothness", "hfer"]
VIEW_NAMES   = ["vv", "tv", "full"]
AGG_NAMES    = ["mean", "max"]


def _diag_to_vec(d: SpectralDiagnostics) -> np.ndarray:
    """Extract the 4 metrics from a SpectralDiagnostics in canonical order."""
    return np.array([d.fiedler_value, d.spectral_entropy, d.smoothness_index, d.hfer],
                    dtype=np.float32)


def _build_signal(W: torch.Tensor) -> torch.Tensor:
    """
    Per-node attention mass, mean-centred → [n, 1] float32.
    W is the symmetrised adjacency [n, n].
    """
    mass = W.sum(dim=-1, keepdim=True).float()  # [n, 1]
    mass = mass - mass.mean()
    return mass


def _analyze_square(
    A_heads: torch.Tensor,          # [H, n, n]
    graph: GraphConstructor,
    analyzer: SpectralAnalyzer,
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean_metrics [4], max_metrics [4]) for a square view.
    mean: aggregate heads uniformly, build graph, compute metrics once.
    max : compute metrics per head, take element-wise max.
    """
    H, n, _ = A_heads.shape

    # ---- mean aggregation ----
    A_mean = A_heads.mean(dim=0, keepdim=True)           # [1, n, n] — already mean
    A_mean_sym = graph.symmetrize_attention(A_mean.unsqueeze(0)).squeeze(0)  # [1, n, n]
    adj_mean = A_mean_sym.squeeze(0)                     # [n, n]
    lap_mean = graph.construct_laplacian(adj_mean.unsqueeze(0)).squeeze(0)
    sig_mean = _build_signal(adj_mean)
    diag_mean = analyzer.analyze_layer(sig_mean, lap_mean.unsqueeze(0), layer_idx)
    mean_metrics = _diag_to_vec(diag_mean)

    # ---- max aggregation (per-head then element-wise max) ----
    head_metrics = np.zeros((H, 4), dtype=np.float32)
    for h in range(H):
        Ah = A_heads[h].unsqueeze(0)                     # [1, n, n]
        Ah_sym = graph.symmetrize_attention(Ah.unsqueeze(0)).squeeze(0).squeeze(0)  # [n, n]
        lap_h = graph.construct_laplacian(Ah_sym.unsqueeze(0)).squeeze(0)
        sig_h = _build_signal(Ah_sym)
        try:
            d = analyzer.analyze_layer(sig_h, lap_h.unsqueeze(0), layer_idx)
            head_metrics[h] = _diag_to_vec(d)
        except Exception as e:
            logger.debug("Head %d layer %d failed: %s", h, layer_idx, e)
    max_metrics = head_metrics.max(axis=0)

    return mean_metrics, max_metrics


def _analyze_tv(
    A_tv_heads: torch.Tensor,       # [H, n_t, n_v]  text→visual rectangular
    graph: GraphConstructor,
    analyzer: SpectralAnalyzer,
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Co-attention view: A_tv @ A_tv.T → [n_t, n_t] symmetric.
    Then same mean/max pipeline as _analyze_square.
    """
    H, n_t, n_v = A_tv_heads.shape

    # Build [H, n_t, n_t] co-attention matrices.
    # Use float32 to avoid bf16 matmul issues on CPU.
    A_tv_f = A_tv_heads.float()
    co_attn = torch.bmm(A_tv_f, A_tv_f.transpose(1, 2))  # [H, n_t, n_t]
    # Normalise rows so they still behave like attention weights.
    row_sum = co_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    co_attn = co_attn / row_sum

    return _analyze_square(co_attn, graph, analyzer, layer_idx)


class LayerSpectralMetrics:
    """All 4 metrics × 3 views × 2 aggs for one layer → shape [4, 3, 2]."""

    def __init__(self, data: np.ndarray):
        assert data.shape == (4, 3, 2), data.shape
        self.data = data  # [n_metrics, n_views, n_aggs]

    def get(self, metric: str, view: str, agg: str) -> float:
        mi = METRIC_NAMES.index(metric)
        vi = VIEW_NAMES.index(view)
        ai = AGG_NAMES.index(agg)
        return float(self.data[mi, vi, ai])

    def to_flat_dict(self) -> dict:
        out = {}
        for mi, m in enumerate(METRIC_NAMES):
            for vi, v in enumerate(VIEW_NAMES):
                for ai, a in enumerate(AGG_NAMES):
                    out[f"{m}_{v}_{a}"] = float(self.data[mi, vi, ai])
        return out


def compute_spectral_trajectory(
    result: ForwardResult,
    gsp_cfg: Optional[GSPConfig] = None,
) -> List[LayerSpectralMetrics]:
    """
    Main entry point.  Iterates over layers and returns one
    LayerSpectralMetrics per layer — i.e. a list of length n_layers.

    Uses spectral_trust.GraphConstructor + SpectralAnalyzer under the hood.
    """
    if gsp_cfg is None:
        gsp_cfg = GSPConfig(
            head_aggregation="uniform",
            symmetrization="symmetric",
            normalization="sym",
            hfer_cutoff_ratio=0.25,
            num_eigenvalues=50,
            eigen_solver="dense",
            save_intermediate=False,
            verbose=False,
        )

    graph = GraphConstructor(gsp_cfg)
    analyzer = SpectralAnalyzer(gsp_cfg)

    v_start = result.visual_start
    v_end   = result.visual_end
    seq_len = result.seq_len
    n_layers = len(result.attentions)

    trajectory: List[LayerSpectralMetrics] = []

    for layer_idx, A in enumerate(result.attentions):
        # A: [H, seq_len, seq_len]  float32  CPU
        if A.shape[-1] != seq_len:
            logger.warning("Layer %d: seq_len mismatch (%d vs %d), skipping.",
                           layer_idx, A.shape[-1], seq_len)
            trajectory.append(LayerSpectralMetrics(np.zeros((4, 3, 2), dtype=np.float32)))
            continue

        data = np.zeros((4, 3, 2), dtype=np.float32)

        try:
            # View: A_vv — visual→visual
            A_vv = A[:, v_start:v_end, v_start:v_end]          # [H, n_v, n_v]
            mean_vv, max_vv = _analyze_square(A_vv, graph, analyzer, layer_idx)
            data[:, 0, 0] = mean_vv
            data[:, 0, 1] = max_vv
        except Exception as e:
            logger.warning("Layer %d vv failed: %s", layer_idx, e)

        try:
            # View: A_tv — text→visual (text tokens are outside visual span)
            # Determine text-token indices (everything that is NOT visual AND before EOS).
            # Simple approach: all token positions outside [v_start, v_end).
            t_indices = list(range(0, v_start)) + list(range(v_end, seq_len))
            if t_indices:
                A_tv = A[:, t_indices, :][:, :, v_start:v_end]  # [H, n_t, n_v]
                mean_tv, max_tv = _analyze_tv(A_tv, graph, analyzer, layer_idx)
                data[:, 1, 0] = mean_tv
                data[:, 1, 1] = max_tv
        except Exception as e:
            logger.warning("Layer %d tv failed: %s", layer_idx, e)

        try:
            # View: A_full — entire sequence
            mean_full, max_full = _analyze_square(A, graph, analyzer, layer_idx)
            data[:, 2, 0] = mean_full
            data[:, 2, 1] = max_full
        except Exception as e:
            logger.warning("Layer %d full failed: %s", layer_idx, e)

        trajectory.append(LayerSpectralMetrics(data))

        if (layer_idx + 1) % 8 == 0:
            logger.debug("Processed %d/%d layers", layer_idx + 1, n_layers)

    return trajectory
