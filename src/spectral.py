"""
GPU-native spectral analysis of VLM attention.

Graph construction  : spectral_trust.GraphConstructor  — CUDA tensors throughout.
Eigendecomposition  : torch.linalg.eigh                — CUDA LAPACK, full spectrum.
All metric arithmetic stays on GPU; only float scalars hit the CPU at the end.
scipy / numpy are never called in the hot path.

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

from spectral_trust import GSPConfig, GraphConstructor
from .extract import ForwardResult

logger = logging.getLogger(__name__)

METRIC_NAMES = ["fiedler", "entropy", "smoothness", "hfer"]
VIEW_NAMES   = ["vv", "tv", "full"]
AGG_NAMES    = ["mean", "max"]

_HFER_CUTOFF = 0.25   # top-25 % of spectrum = high-frequency


def _make_gsp_cfg() -> GSPConfig:
    return GSPConfig(
        head_aggregation="uniform",
        symmetrization="symmetric",
        normalization="sym",
        hfer_cutoff_ratio=_HFER_CUTOFF,
        save_intermediate=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# GPU metric kernel — no numpy, no scipy
# ---------------------------------------------------------------------------

@torch.no_grad()
def _metrics_from_adj(adj: torch.Tensor, graph: GraphConstructor) -> np.ndarray:
    """
    adj : [n, n] symmetrised adjacency, float32, on CUDA.
    Returns float32 ndarray [4] = [fiedler, entropy, smoothness, hfer].
    Everything runs on GPU; only 4 scalars are transferred to CPU at the end.
    """
    n = adj.shape[0]
    if n < 3:
        return np.zeros(4, dtype=np.float32)

    # Symmetric normalised Laplacian — stays on GPU.
    lap = graph.construct_laplacian(adj.unsqueeze(0)).squeeze(0)  # [n, n]

    # ── Attention-mass signal: degree vector, mean-centred ───────────────
    degree = adj.sum(dim=-1)                        # [n]
    signal = (degree - degree.mean()).unsqueeze(1)  # [n, 1]
    xTx    = (signal * signal).sum().clamp(min=1e-8)

    # ── Smoothness: x^T L x / x^T x  (pure matmul, no eigen needed) ─────
    smoothness = ((signal.T @ lap @ signal).squeeze() / xTx).item()

    # ── Full spectrum on GPU — CUDA LAPACK divide-and-conquer ────────────
    eigenvalues, eigenvectors = torch.linalg.eigh(lap)   # ascending, [n] / [n,n]
    eigenvalues = eigenvalues.clamp(min=0.0)

    # Fiedler: λ₂
    fiedler = eigenvalues[1].item() if n > 1 else 0.0

    # Spectral entropy: p_i = λ_i / Σλ  (skip trivial zero eigenvalue)
    ev      = eigenvalues[1:]
    ev_sum  = ev.sum().clamp(min=1e-8)
    p       = (ev / ev_sum).clamp(min=1e-12)
    entropy = -(p * p.log()).sum().item()

    # HFER: fraction of signal energy in top-25 % high-freq eigenvectors
    cutoff   = int((1.0 - _HFER_CUTOFF) * n)
    sig_hat  = eigenvectors.T @ signal               # [n, 1]
    energies = sig_hat.squeeze().pow(2)              # [n]
    total    = energies.sum().clamp(min=1e-8)
    hfer     = (energies[cutoff:].sum() / total).clamp(0.0, 1.0).item()

    return np.array([fiedler, entropy, smoothness, hfer], dtype=np.float32)


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------

def _square_views(
    A_heads: torch.Tensor,          # [H, n, n] on device
    graph: GraphConstructor,
) -> Tuple[np.ndarray, np.ndarray]:
    """mean-agg and max-agg metrics for a square view."""
    A_sym = 0.5 * (A_heads + A_heads.transpose(-2, -1))   # [H, n, n]

    # mean across heads
    mean_m = _metrics_from_adj(A_sym.mean(dim=0), graph)

    # per-head → element-wise max
    head_m = np.stack([_metrics_from_adj(A_sym[h], graph) for h in range(A_sym.shape[0])])
    return mean_m, head_m.max(axis=0)


def _tv_views(
    A_tv: torch.Tensor,             # [H, n_t, n_v] on device
    graph: GraphConstructor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Co-attention A_tv @ A_tv^T → [H, n_t, n_t], then square pipeline."""
    co  = torch.bmm(A_tv, A_tv.transpose(1, 2))            # [H, n_t, n_t]
    co  = co / co.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return _square_views(co, graph)


# ---------------------------------------------------------------------------
# Public container
# ---------------------------------------------------------------------------

class LayerSpectralMetrics:
    def __init__(self, data: np.ndarray):
        assert data.shape == (4, 3, 2)
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
    """One LayerSpectralMetrics per transformer layer. Fully GPU-resident."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gsp_cfg is None:
        gsp_cfg = _make_gsp_cfg()

    graph   = GraphConstructor(gsp_cfg)
    v_start = result.visual_start
    v_end   = result.visual_end
    seq_len = result.seq_len
    t_idx   = list(range(0, v_start)) + list(range(v_end, seq_len))

    trajectory: List[LayerSpectralMetrics] = []

    for layer_idx, A_cpu in enumerate(result.attentions):
        A    = A_cpu.to(device, dtype=torch.float32)    # [H, seq, seq] → GPU
        data = np.zeros((4, 3, 2), dtype=np.float32)

        try:
            data[:, 0, 0], data[:, 0, 1] = _square_views(
                A[:, v_start:v_end, v_start:v_end], graph)
        except Exception as e:
            logger.debug("Layer %d vv: %s", layer_idx, e)

        try:
            if t_idx:
                t_tensor = torch.tensor(t_idx, device=device)
                data[:, 1, 0], data[:, 1, 1] = _tv_views(
                    A.index_select(1, t_tensor)[:, :, v_start:v_end], graph)
        except Exception as e:
            logger.debug("Layer %d tv: %s", layer_idx, e)

        try:
            data[:, 2, 0], data[:, 2, 1] = _square_views(A, graph)
        except Exception as e:
            logger.debug("Layer %d full: %s", layer_idx, e)

        trajectory.append(LayerSpectralMetrics(data))
        del A
        torch.cuda.empty_cache()

    return trajectory
