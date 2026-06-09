"""
Visualisation for spectral-vlm results.

Two main plot types:
  1. Per-layer effect-size curves (Cohen's d vs layer, for each metric×view×agg cell).
  2. Trajectory heatmaps (layer × feature, one row per example, sorted by label).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# 1. Effect-size curves
# ---------------------------------------------------------------------------

def plot_effect_size_curves(
    stats_df: pd.DataFrame,
    contrast_name: str,
    output_dir: Path,
    views: List[str] = ("vv", "tv", "full"),
    aggs: List[str] = ("mean", "max"),
    figsize_per_panel: tuple = (5, 3),
    fdr_alpha: float = 0.05,
) -> None:
    """
    One subplot per (view, agg) pair.  Each subplot shows d vs layer curves,
    one curve per metric.  Significant layers (q < fdr_alpha) are marked.
    """
    metrics = sorted(stats_df["metric"].unique())
    n_panels = len(views) * len(aggs)
    ncols = len(aggs)
    nrows = len(views)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    fig.suptitle(f"Per-layer Cohen's d — {contrast_name}", fontsize=12)

    cmap = plt.cm.tab10
    metric_colors = {m: cmap(i) for i, m in enumerate(metrics)}

    for ri, view in enumerate(views):
        for ci, agg in enumerate(aggs):
            ax = axes[ri][ci]
            sub = stats_df[(stats_df["view"] == view) & (stats_df["agg"] == agg)]

            for metric in metrics:
                msub = sub[sub["metric"] == metric].sort_values("layer")
                if msub.empty:
                    continue
                layers = msub["layer"].values
                d_vals = msub["d"].values
                d_lo   = msub["d_lo"].values
                d_hi   = msub["d_hi"].values
                sig    = msub["q_value"].values < fdr_alpha

                color = metric_colors[metric]
                ax.plot(layers, d_vals, color=color, label=metric, linewidth=1.5)
                ax.fill_between(layers, d_lo, d_hi, color=color, alpha=0.15)
                # Mark significant layers.
                sig_layers = layers[sig]
                sig_d      = d_vals[sig]
                ax.scatter(sig_layers, sig_d, color=color, s=30, zorder=5)

            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_title(f"view={view}, agg={agg}", fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cohen's d")
            ax.grid(True, alpha=0.3)
            if ri == 0 and ci == ncols - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_path = output_dir / f"effect_size_{contrast_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 2. Trajectory heatmaps
# ---------------------------------------------------------------------------

def plot_trajectory_heatmap(
    records: List[dict],
    label_col: str = "label",
    trajectory_col: str = "trajectory_flat",
    output_dir: Path = Path("outputs"),
    title: str = "Trajectory Heatmap",
    max_examples: int = 100,
    n_layers: int = 34,
    n_features_per_layer: int = 24,
) -> None:
    """
    Heatmap of [examples × (layer*feature)], sorted by label.
    Rows = examples, columns = trajectory feature index.
    """
    records_sub = records[:max_examples]
    X = np.stack([r[trajectory_col] for r in records_sub])
    labels = [r[label_col] for r in records_sub]

    # Sort by label.
    order = np.argsort(labels)
    X = X[order]
    sorted_labels = [labels[i] for i in order]

    # Row-wise z-score.
    X_z = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(16, max(4, len(records_sub) * 0.15)))
    sns.heatmap(X_z, ax=ax, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.5})

    # Add label colour bar on the y-axis.
    unique_labels = sorted(set(sorted_labels))
    label_to_col  = {l: plt.cm.Set1(i / max(1, len(unique_labels) - 1))
                     for i, l in enumerate(unique_labels)}
    label_colors  = [label_to_col[l] for l in sorted_labels]

    for idx, color in enumerate(label_colors):
        ax.add_patch(plt.Rectangle((-0.8, idx), 0.5, 1,
                                   color=color, transform=ax.transData, clip_on=False))

    # Legend.
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=label_to_col[l], label=l) for l in unique_labels]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.12, 1),
              fontsize=8)

    # Layer boundary lines.
    for li in range(1, n_layers):
        ax.axvline(li * n_features_per_layer, color="white", linewidth=0.3, alpha=0.5)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Trajectory feature (layer × metric × view × agg)")
    ax.set_ylabel("Example (sorted by label)")

    out_path = output_dir / f"trajectory_heatmap_{title.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 3. AUROC summary bar chart
# ---------------------------------------------------------------------------

def plot_auroc_summary(
    results: List[dict],  # each: {'contrast': str, 'auroc': float, 'lo': float, 'hi': float}
    output_dir: Path,
    title: str = "AUROC by Contrast",
) -> None:
    contrasts = [r["contrast"] for r in results]
    aurocs    = [r["auroc"]    for r in results]
    errs_lo   = [r["auroc"] - r["lo"] for r in results]
    errs_hi   = [r["hi"] - r["auroc"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(contrasts))
    ax.barh(x, aurocs, xerr=[errs_lo, errs_hi], color=["steelblue", "tomato"][:len(contrasts)],
            capsize=5, height=0.4)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(x)
    ax.set_yticklabels(contrasts)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "auroc_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
