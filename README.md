# Spectral Signatures of Visual Injection Attacks on VLMs

> **Research project** — Valentin Noël

Investigates whether successful visual prompt injection attacks on open-weight vision-language models (VLMs) leave a measurable signature in the **spectral geometry of the language model's attention graph** — specifically in the visual-token and cross-modal attention blocks — that a clean image does not.

If confirmed, this yields a **training-free detection layer** for visual injection attacks, directly addressing the threat identified in recent safety literature as *"a far harder detection and mitigation challenge"* than text prompt injection.

---

## Hypotheses

| # | Hypothesis | Controls |
|---|---|---|
| **H1** | For a fixed base query/image, the attacked variant shifts at least one spectral metric in `A_vv` and/or `A_tv` relative to the clean variant. | Paired (clean ↔ attacked) design |
| **H2** | Among *attacked* inputs, the signature separates **hijacked** (model complied) from **resisted** (model refused), attack type held fixed. | Within-attack-type contrast — rules out text-presence artefacts |
| **H3** | The discriminative signal concentrates in a small set of layers, consistent with a cross-modal capture mechanism. | Layer-wise effect-size curves |
| **H4** | The qualitative signature survives across ≥2 vision-encoder families. | Cross-architecture replication |

**H2 AUROC is the go/no-go number for the Oxford pitch.**

---

## Method

Each attention matrix `A ∈ ℝⁿˣⁿ` (post-softmax) per (layer, head) is treated as a weighted graph over the token sequence `[visual_tokens | text_tokens]`. Three views are sliced:

- **`A_vv`** — visual → visual (image self-structure)
- **`A_tv`** — text → visual, symmetrised as co-attention `A_tv @ A_tvᵀ` (cross-modal capture channel)
- **`A_full`** — entire sequence

Per view and per head-aggregation (mean / max across heads), four GSP metrics are computed via [`spectral-trust`](https://pypi.org/project/spectral-trust/):

| Metric | Symbol | Description |
|---|---|---|
| Fiedler value | λ₂ | Algebraic connectivity of the attention graph |
| Spectral entropy | H | Spread of signal energy across eigenbasis |
| Smoothness (Dirichlet energy) | s | Graph-signal roughness of attention mass |
| HFER | — | High-frequency energy ratio (top-25% of spectrum) |

This produces a **spectral trajectory** of shape `[n_layers, 24]` per example, which feeds a grouped logistic-regression classifier.

---

## Repository layout

```
spectral-vlm/
  configs/          # model + metric configs (YAML)
  data/
    load_figstep.py         # Phase 0 — FigStep loader
    load_mmsafetybench.py   # Phase 0 — MM-SafetyBench SD+TYPO
    load_jailbreakv.py      # Phase 2 — JailBreakV-28K miniset
    attack_pgd.py           # Phase 1 — PGD perturbation crafter
    overlay_benign.py       # Confound-control image generator
  src/
    extract.py      # VLM loader (bf16 / eager), visual span detection, attention extraction
    spectral.py     # GSP metrics on A_vv / A_tv / A_full via spectral-trust
    trajectory.py   # Assemble [n_layers, 24] feature tensor
    label.py        # Behavioural labeller (hijacked / resisted)
  analysis/
    stats.py        # Cohen's d, Mann-Whitney U, bootstrap CI, BH-FDR
    classify.py     # Grouped logistic regression, AUROC with bootstrap CI
    plots.py        # Effect-size curves, trajectory heatmaps
  smoke_test.py     # M0 + M1 validation (single forward pass)
  run_phase0.py     # Phase 0 orchestrator
```

---

## Milestones

- **M0** — Model loads in bf16/eager; 256-token visual span correctly identified. ✅
- **M1** — All four metrics on A_vv / A_tv / A_full are finite and vary across layers. ✅
- **M2** — Phase 0 dataset (FigStep + MM-SafetyBench) run and labelled.
- **M3** — Full stats table (d, q-values per layer × metric) + per-layer effect-size plots.
- **M4** — Trajectory classifier AUROC for `resisted vs hijacked`. **Go/no-go.**
- **M5** — Cross-architecture replication (Qwen2.5-VL-3B, SmolVLM2-2.2B).
- **M6** — Phase 2 transferability correlation on JailBreakV-28K mini.

---

## Setup

```bash
conda activate gemma_spectral
pip install -U transformers accelerate pillow datasets spectral-trust
```

**Smoke test (M0 + M1):**
```bash
python smoke_test.py --model google/gemma-4-E4B-it
```

**Phase 0 full run:**
```bash
python run_phase0.py --model google/gemma-4-E4B-it --n-figstep 280 --n-mmsb 280
```

---

## Models

| Role | Model | Notes |
|---|---|---|
| Primary | `google/gemma-4-E4B-it` | bf16, SigLIP encoder, 256 fixed image tokens |
| Cross-arch A | `Qwen/Qwen2.5-VL-3B-Instruct` | Dynamic resolution |
| Cross-arch B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Lightweight, fast iteration |

---

## Datasets

- **FigStep** (Gong et al.) — Harmful instructions as typographic images; miniset (~280 samples).
- **MM-SafetyBench** (Liu et al., ECCV 2024) — SD+Typography subset (ASR ~77%).
- **JailBreakV-28K** (Luo et al.) — Phase 2 transferability; miniset (~280 samples).

---

## Statistical analysis

Per (layer, metric, view, aggregation) cell:

- Cohen's d with 95% bootstrap CI (10 k resamples)
- Mann-Whitney U p-value
- Benjamini-Hochberg FDR correction (q-values) across the full layer × metric grid
- Trajectory classifier: L2-regularised logistic regression, grouped CV split (base image held out), AUROC with bootstrap CI

---

*Hardware target: single RTX 4080 Super (16 GB VRAM). All inference in bf16.*
