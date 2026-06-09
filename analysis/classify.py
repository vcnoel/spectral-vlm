"""
Trajectory classifier for spectral-vlm.

L2-regularised logistic regression on the flattened per-example trajectory
[n_layers × 24 features].  Train/test split is grouped by base_image_id so
that an input and its attacked variant never straddle the split (no leakage).

Reports AUROC with bootstrap 95% CI for both contrasts:
    1. clean vs attacked
    2. resisted vs hijacked  ← go/no-go number

Ablation: single-best-layer vs full-trajectory is also reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_feature_matrix(
    records: List[Dict],
    label_col: str = "label",
    group_col: str = "base_id",
    trajectory_col: str = "trajectory_flat",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y, groups) where:
        X      : [n, n_layers*24] float32
        y      : [n] int  (0/1)
        groups : [n] str  (for GroupShuffleSplit)
    """
    X = np.stack([r[trajectory_col] for r in records]).astype(np.float32)
    labels = [r[label_col] for r in records]
    unique_labels = sorted(set(labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels], dtype=int)
    groups = np.array([r[group_col] for r in records])
    return X, y, groups


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class ClassifierResult:
    contrast: str          # e.g. "clean_vs_attacked"
    auroc: float
    auroc_lo: float        # 95% CI lower
    auroc_hi: float        # 95% CI upper
    n_train: int
    n_test: int
    label_names: List[str]
    best_single_layer_auroc: Optional[float] = None
    best_single_layer_idx: Optional[int] = None


def _bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    alpha = (1 - ci) / 2
    return float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha))


def train_and_evaluate(
    records: List[Dict],
    label_col: str = "label",
    group_col: str = "base_id",
    trajectory_col: str = "trajectory_flat",
    contrast_name: str = "contrast",
    n_splits: int = 5,
    test_size: float = 0.25,
    C: float = 1.0,
    n_bootstrap: int = 10_000,
    n_layers: int = 34,
    n_features_per_layer: int = 24,
    seed: int = 42,
) -> ClassifierResult:
    """
    Grouped cross-validation logistic regression + AUROC with bootstrap CI.
    n_splits CV folds are averaged.
    """
    X, y, groups = build_feature_matrix(records, label_col, group_col, trajectory_col)
    unique_labels = sorted(set(records[i][label_col] for i in range(len(records))))

    if len(np.unique(y)) < 2:
        raise ValueError(f"Only one class present for contrast '{contrast_name}'. "
                         "Cannot compute AUROC.")

    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

    all_y_true, all_y_score = [], []
    n_train_total, n_test_total = 0, 0

    for train_idx, test_idx in gss.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                  class_weight="balanced", random_state=seed)
        clf.fit(X_tr, y_tr)

        if len(np.unique(y_te)) < 2:
            continue

        y_score = clf.predict_proba(X_te)[:, 1]
        all_y_true.append(y_te)
        all_y_score.append(y_score)
        n_train_total += len(train_idx)
        n_test_total  += len(test_idx)

    if not all_y_true:
        raise ValueError("No valid CV folds (all test sets single-class).")

    y_true_all  = np.concatenate(all_y_true)
    y_score_all = np.concatenate(all_y_score)
    auroc = roc_auc_score(y_true_all, y_score_all)
    lo, hi = _bootstrap_auroc(y_true_all, y_score_all, n_bootstrap, seed=seed)

    # ---- Single-layer ablation ----
    best_sl_auc, best_sl_idx = 0.0, 0
    feat_per_layer = n_features_per_layer

    for li in range(n_layers):
        col_start = li * feat_per_layer
        col_end   = col_start + feat_per_layer
        X_sl = X[:, col_start:col_end]

        sl_y_true, sl_y_score = [], []
        for train_idx, test_idx in gss.split(X_sl, y, groups):
            X_tr, X_te = X_sl[train_idx], X_sl[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_te)) < 2:
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs",
                                      class_weight="balanced", random_state=seed)
            clf.fit(X_tr, y_tr)
            sl_y_score.append(clf.predict_proba(X_te)[:, 1])
            sl_y_true.append(y_te)

        if not sl_y_true:
            continue

        sl_auc = roc_auc_score(np.concatenate(sl_y_true), np.concatenate(sl_y_score))
        if sl_auc > best_sl_auc:
            best_sl_auc = sl_auc
            best_sl_idx = li

    return ClassifierResult(
        contrast=contrast_name,
        auroc=auroc,
        auroc_lo=lo,
        auroc_hi=hi,
        n_train=n_train_total // n_splits,
        n_test=n_test_total // n_splits,
        label_names=unique_labels,
        best_single_layer_auroc=best_sl_auc,
        best_single_layer_idx=best_sl_idx,
    )
