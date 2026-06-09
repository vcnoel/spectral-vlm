"""
Phase 0 orchestrator: FigStep + MM-SafetyBench (SD+Typography subset).

Steps:
  1. Load model (Gemma 4 4B, bf16, eager).
  2. Load FigStep + MM-SafetyBench paired examples.
  3. For each example (attacked + clean), run forward pass and compute
     spectral trajectory.
  4. Label attacked examples as hijacked / resisted.
  5. Add benign-text-overlay confound-control examples.
  6. Serialise trajectories + labels to outputs/phase0_records.pkl.
  7. Run statistical analysis (two contrasts: clean vs attacked, resisted vs hijacked).
  8. Run trajectory classifier and report AUROC.
  9. Save plots.

Run with:
    conda activate gemma_spectral
    python run_phase0.py [--model google/gemma-4-4b-it] [--n-figstep 280] [--n-mmsb 280]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.extract import ExtractionConfig, VLMExtractor
from src.spectral import compute_spectral_trajectory
from src.trajectory import trajectory_to_flat, trajectory_to_dataframe, feature_names
from src.label import label_response
from data.load_figstep import load_figstep
from data.load_mmsafetybench import load_mmsafetybench
from data.overlay_benign import make_benign_overlay_set
from analysis.stats import run_full_analysis
from analysis.classify import train_and_evaluate
from analysis.plots import plot_effect_size_curves, plot_trajectory_heatmap, plot_auroc_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("run_phase0")

OUTPUT_DIR = Path("outputs/phase0")


# ---------------------------------------------------------------------------
# Extraction helper
# ---------------------------------------------------------------------------

def extract_record(
    extractor: VLMExtractor,
    example: Dict,
    label: Optional[str] = None,
    base_id: Optional[str] = None,
) -> Optional[Dict]:
    """Run extraction + spectral for one example. Returns None on failure."""
    try:
        result = extractor.extract_attentions(example["image"], example["prompt"])
        trajectory = compute_spectral_trajectory(result)
        traj_flat  = trajectory_to_flat(trajectory)

        if label is None:
            label = label_response(result.generated_text)

        return {
            "example_id"     : example["example_id"],
            "base_id"        : base_id or example["example_id"].replace("_clean", ""),
            "split"          : example["split"],
            "source"         : example.get("source", "unknown"),
            "category"       : example.get("category", example.get("scenario", "unknown")),
            "label"          : label,
            "generated_text" : result.generated_text,
            "visual_start"   : result.visual_start,
            "visual_end"     : result.visual_end,
            "seq_len"        : result.seq_len,
            "trajectory"     : trajectory,       # List[LayerSpectralMetrics]
            "trajectory_flat": traj_flat,        # 1-D np.ndarray
        }
    except Exception as e:
        logger.error("Failed on example '%s': %s", example.get("example_id", "?"), e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load model ----
    cfg = ExtractionConfig(
        model_id=args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        max_new_tokens=256,
        analyze_token_offset=0,
    )
    extractor = VLMExtractor(cfg)
    logger.info("Loading model %s …", args.model)
    extractor.load()

    # ---- 2. Load datasets ----
    logger.info("Loading FigStep …")
    figstep_pairs = load_figstep(n_samples=args.n_figstep)
    logger.info("Loading MM-SafetyBench …")
    mmsb_pairs    = load_mmsafetybench(n_samples=args.n_mmsb)

    all_pairs = figstep_pairs + mmsb_pairs
    logger.info("Total pairs: %d", len(all_pairs))

    # ---- 3. Extract trajectories for all examples ----
    records: List[Dict] = []
    base_images = []     # collect for benign-overlay confound set

    for pair in all_pairs:
        base_id = pair["example_id"]

        # Attacked variant.
        atk = pair["attacked"]
        rec_atk = extract_record(extractor, atk, label=None, base_id=base_id)
        if rec_atk is not None:
            records.append(rec_atk)
            base_images.append(atk["image"])

        # Clean variant (always labelled 'clean' for H1 contrast).
        cln = pair["clean"]
        rec_cln = extract_record(extractor, cln, label="clean", base_id=base_id)
        if rec_cln is not None:
            records.append(rec_cln)

    # ---- 4. Benign-overlay confound set ----
    logger.info("Generating benign-overlay confound set …")
    overlay_examples = make_benign_overlay_set(base_images[:min(50, len(base_images))])
    for ov_ex in overlay_examples:
        rec = extract_record(extractor, ov_ex, label="benign_overlay",
                             base_id=ov_ex["example_id"])
        if rec is not None:
            records.append(rec)

    logger.info("Total records: %d", len(records))
    label_counts = pd.Series([r["label"] for r in records]).value_counts()
    logger.info("Label distribution:\n%s", label_counts.to_string())

    # ---- 5. Serialise ----
    records_path = OUTPUT_DIR / "phase0_records.pkl"
    with open(records_path, "wb") as f:
        pickle.dump(records, f)
    logger.info("Records saved to %s", records_path)

    # ---- 6. Build long-form DataFrame for stats ----
    n_layers = len(records[0]["trajectory"])
    all_dfs = []
    for r in records:
        df_r = trajectory_to_dataframe(
            r["trajectory"],
            example_id=r["example_id"],
            label=r["label"],
        )
        df_r["split"]   = r["split"]
        df_r["base_id"] = r["base_id"]
        all_dfs.append(df_r)
    long_df = pd.concat(all_dfs, ignore_index=True)
    long_df.to_parquet(OUTPUT_DIR / "trajectories_long.parquet")
    logger.info("Long-form DataFrame: %d rows", len(long_df))

    # ---- 7. Statistical analysis ----
    logger.info("=== Contrast 1: clean vs attacked ===")
    attacked_labels = long_df["label"].isin(["hijacked", "resisted"])
    long_df_h1 = long_df[long_df["label"].isin(["clean"]) |
                         long_df["label"].isin(["hijacked", "resisted"])].copy()
    long_df_h1["group"] = long_df_h1["label"].apply(
        lambda x: "attacked" if x in ("hijacked", "resisted") else "clean"
    )
    stats_h1 = run_full_analysis(
        long_df_h1, group_col="group",
        group_a="attacked", group_b="clean",
        n_bootstrap=args.n_bootstrap,
    )
    stats_h1.to_csv(OUTPUT_DIR / "stats_clean_vs_attacked.csv", index=False)
    logger.info("Top cells by |d| (clean vs attacked):\n%s",
                stats_h1.nlargest(10, "d")[["layer","metric","view","agg","d","q_value"]].to_string())

    logger.info("=== Contrast 2: resisted vs hijacked (H2) ===")
    long_df_h2 = long_df[long_df["label"].isin(["resisted", "hijacked"])].copy()
    stats_h2 = run_full_analysis(
        long_df_h2, group_col="label",
        group_a="hijacked", group_b="resisted",
        n_bootstrap=args.n_bootstrap,
    )
    stats_h2.to_csv(OUTPUT_DIR / "stats_resisted_vs_hijacked.csv", index=False)
    logger.info("Top cells by |d| (resisted vs hijacked):\n%s",
                stats_h2.nlargest(10, "d")[["layer","metric","view","agg","d","q_value"]].to_string())

    # ---- 8. Trajectory classifier ----
    classifier_results = []

    logger.info("=== Classifier: clean vs attacked ===")
    recs_h1 = [r for r in records if r["label"] in ("clean", "hijacked", "resisted")]
    for r in recs_h1:
        r["label_binary"] = "attacked" if r["label"] in ("hijacked", "resisted") else "clean"
    try:
        clf_h1 = train_and_evaluate(
            recs_h1, label_col="label_binary", group_col="base_id",
            contrast_name="clean_vs_attacked",
            n_layers=n_layers, n_bootstrap=args.n_bootstrap,
        )
        logger.info("AUROC clean vs attacked: %.3f [%.3f, %.3f]",
                    clf_h1.auroc, clf_h1.auroc_lo, clf_h1.auroc_hi)
        logger.info("Best single layer: %d (AUROC=%.3f)",
                    clf_h1.best_single_layer_idx, clf_h1.best_single_layer_auroc)
        classifier_results.append({
            "contrast": "clean_vs_attacked",
            "auroc"   : clf_h1.auroc,
            "lo"      : clf_h1.auroc_lo,
            "hi"      : clf_h1.auroc_hi,
        })
    except Exception as e:
        logger.error("Classifier H1 failed: %s", e)

    logger.info("=== Classifier: resisted vs hijacked (go/no-go) ===")
    recs_h2 = [r for r in records if r["label"] in ("resisted", "hijacked")]
    if len(recs_h2) >= 4:
        try:
            clf_h2 = train_and_evaluate(
                recs_h2, label_col="label", group_col="base_id",
                contrast_name="resisted_vs_hijacked",
                n_layers=n_layers, n_bootstrap=args.n_bootstrap,
            )
            logger.info("AUROC resisted vs hijacked: %.3f [%.3f, %.3f]  ← go/no-go",
                        clf_h2.auroc, clf_h2.auroc_lo, clf_h2.auroc_hi)
            logger.info("Best single layer: %d (AUROC=%.3f)",
                        clf_h2.best_single_layer_idx, clf_h2.best_single_layer_auroc)
            classifier_results.append({
                "contrast": "resisted_vs_hijacked",
                "auroc"   : clf_h2.auroc,
                "lo"      : clf_h2.auroc_lo,
                "hi"      : clf_h2.auroc_hi,
            })
        except Exception as e:
            logger.error("Classifier H2 failed: %s", e)
    else:
        logger.warning("Too few hijacked/resisted examples for H2 classifier (%d).", len(recs_h2))

    # ---- 9. Plots ----
    plot_effect_size_curves(stats_h1, "clean_vs_attacked", OUTPUT_DIR)
    plot_effect_size_curves(stats_h2, "resisted_vs_hijacked", OUTPUT_DIR)
    plot_trajectory_heatmap(records, label_col="label",
                            output_dir=OUTPUT_DIR,
                            title="Phase0 Trajectory",
                            n_layers=n_layers)
    if classifier_results:
        plot_auroc_summary(classifier_results, OUTPUT_DIR)

    logger.info("=== Phase 0 complete. Outputs in %s ===", OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: spectral-vlm attack analysis")
    parser.add_argument("--model",       default="google/gemma-4-4b-it")
    parser.add_argument("--n-figstep",   type=int, default=280)
    parser.add_argument("--n-mmsb",      type=int, default=280)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    main(args)
