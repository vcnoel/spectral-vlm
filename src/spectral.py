"""
Spectral analysis of VLM attention using spectral_trust.

Graph construction  : spectral_trust.GraphConstructor  — runs on CUDA tensors.
Eigendecomposition  : spectral_trust.SpectralAnalyzer  — Lanczos via ARPACK
                      (eigen_solver="sparse", k=50 eigenvalues).
Only the final eigen step moves to CPU/numpy; everything upstream stays on GPU.

Output shape per example: [n_layers, n_metrics=4, n_views=3, n_aggs=2]
Metric order : [fiedler, entropy, smoothness, hfer]
View order   : [vv, tv, full]
Agg order    : [mean, max]
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer, SpectralDiagnostics
from .extract import ForwardResult

logger = logging.getLogger(__name__)

METRIC_NAMES = ["fiedler", "entropy", "smoothness", "hfer"]
VIEW_NAMES   = ["vv", "tv", "full"]
AGG_NAMES    = ["mean", "max"]


def _make_gsp_cfg(hfer_cutoff: float = 0.25) -> GSPConfig:
    return GSPConfig(
        head_aggregation="uniform",
        symmetrization="symmetric",
        normalization="sym",
        hfer_cutoff_ratio=hfer_cutoff,
        eigen_solver="sparse",      # Lanczos via ARPACK
        num_eigenvalues=50,
        save_intermediate=False,
        verbose=False,
    )


def _diag_to_vec(d: SpectralDiagnostics) -> np.ndarray:
    return np.array([d.fiedler_value, d.spectral_entropy,
                     d.smoothness_index, d.hfer], dtype=np.float32)


def _attention_mass_signal(adj: torch.Tensor) -> torch.Tensor:
    """Row-sum of adjacency, mean-centred → [n, 1] float32 on same device."""
    mass = adj.sum(dim=-1, keepdim=True).float()
    return mass - mass.mean()


def _analyze_adj(
    adj: torch.Tensor,              # [n, n] symmetrised, float32, on device
    graph: GraphConstructor,
    analyzer: SpectralAnalyzer,
    layer_idx: int,
) -> np.ndarray:
    """Build Laplacian on GPU, compute signal on GPU, run Lanczos on CPU."""
    lap = graph.construct_laplacian(adj.unsqueeze(0))   # [1, n, n], stays on device
    sig = _attention_mass_signal(adj)                   # [n, 1], stays on device
    # analyze_layer accepts torch tensors — handles .cpu().numpy() internally.
    d = analyzer.analyze_layer(sig, lap, layer_idx)
    return _diag_to_vec(d)


def _square_views(
    A_heads: torch.Tensor,          # [H, n, n] on device
    graph: GraphConstructor,
    analyzer: SpectralAnalyzer,
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (mean_metrics[4], max_metrics[4]) for a square view."""
    A_f = A_heads.float()
    A_sym = 0.5 * (A_f + A_f.transpose(-2, -1))        # [H, n, n]

    # mean: uniform aggregate across heads
    adj_mean = A_sym.mean(dim=0)                        # [n, n]
    mean_m = _analyze_adj(adj_mean, graph, analyzer, layer_idx)

    # max: per-head metrics, element-wise max
    head_m = np.stack([
        _analyze_adj(A_sym[h], graph, analyzer, layer_idx)
        for h in range(A_sym.shape[0])
    ])
    max_m = head_m.max(axis=0)

    return mean_m, max_m


def _tv_views(
    A_tv_heads: torch.Tensor,       # [H, n_t, n_v] on device
    graph: GraphConstructor,
    analyzer: SpectralAnalyzer,
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Co-attention A_tv @ A_tv^T → [H, n_t, n_t], then square pipeline."""
    A_f = A_tv_heads.float()
    co  = torch.bmm(A_f, A_f.transpose(1, 2))          # [H, n_t, n_t]
    co  = co / co.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return _square_views(co, graph, analyzer, layer_idx)


# ---------------------------------------------------------------------------
# Public container
# ---------------------------------------------------------------------------

class LayerSpectralMetrics:
    """4 metrics × 3 views × 2 aggs for one layer → shape [4, 3, 2]."""

    def __init__(self, data: np.ndarray):
        assert data.shape == (4, 3, 2), data.shape
        self.data = data

    def get(self, metric: str, view: str, agg: str) -> float:
        return float(self.data[METRIC_NAMES.index(metric),
                               VIEW_NAMES.index(view),
                               AGG_NAMES.index(agg)])

    def to_flat_dict(self) -> dict:
        return {
            f"{m}_{v}_{a}": float(self.data[mi, vi, ai])
            for mi, m in enumerate(METRIC_NAMES)
            for vi, v in enumerate(VIEW_NAMES)
            for ai, a in enumerate(AGG_NAMES)
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_spectral_trajectory(
    result: ForwardResult,
    gsp_cfg: Optional[GSPConfig] = None,
    device: Optional[torch.device] = None,
) -> List[LayerSpectralMetrics]:
    """
    One LayerSpectralMetrics per transformer layer.
    Graph construction is GPU-resident; Lanczos eigen runs on CPU via spectral-trust.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gsp_cfg is None:
        gsp_cfg = _make_gsp_cfg()

    graph    = GraphConstructor(gsp_cfg)
    analyzer = SpectralAnalyzer(gsp_cfg)

    v_start = result.visual_start
    v_end   = result.visual_end
    seq_len = result.seq_len
    t_idx   = list(range(0, v_start)) + list(range(v_end, seq_len))

    trajectory: List[LayerSpectralMetrics] = []

    for layer_idx, A_cpu in enumerate(result.attentions):
        # Move this layer's attention to GPU once.
        A = A_cpu.to(device, dtype=torch.float32)       # [H, seq, seq]
        data = np.zeros((4, 3, 2), dtype=np.float32)

        # A_vv — visual → visual
        try:
            data[:, 0, 0], data[:, 0, 1] = _square_views(
                A[:, v_start:v_end, v_start:v_end], graph, analyzer, layer_idx)
        except Exception as e:
            logger.debug("Layer %d vv: %s", layer_idx, e)

        # A_tv — text → visual co-attention
        try:
            if t_idx:
                t_tensor = torch.tensor(t_idx, device=device)
                A_tv = A.index_select(1, t_tensor)[:, :, v_start:v_end]
                data[:, 1, 0], data[:, 1, 1] = _tv_views(
                    A_tv, graph, analyzer, layer_idx)
        except Exception as e:
            logger.debug("Layer %d tv: %s", layer_idx, e)

        # A_full — entire sequence
        try:
            data[:, 2, 0], data[:, 2, 1] = _square_views(
                A, graph, analyzer, layer_idx)
        except Exception as e:
            logger.debug("Layer %d full: %s", layer_idx, e)

        trajectory.append(LayerSpectralMetrics(data))
        del A
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return trajectory
