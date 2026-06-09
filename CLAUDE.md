# Spectral Signatures of Visual Injection Attacks on VLMs

**Project codename:** `spectral-vlm` (extension of the `spectral-trust` GSP framework to the vision-language setting)
**Author:** Valentin Noël
**Hardware target:** single RTX 4080 Super, 16 GB VRAM, Ada (sm_89, native bf16).
**Environment:** use the existing `gemma-spectral` env. Do NOT create a new venv.
**Status:** Phase 0 pilot. Goal is a fast, paired, publishable-as-pilot signal, not a finished detector.

---

## 0. One-paragraph objective

Determine whether successful visual injection attacks on small open-weight VLMs leave a measurable signature in the spectral geometry of the language model's attention graph, specifically in the visual-token and cross-modal attention blocks, that a clean image does not. If they do, we have a training-free detection layer for exactly the threat Paren and Bibi call "a far harder detection and mitigation challenge" than text prompt injection. Phase 0 uses ready-made typographic-injection benchmarks (FigStep, MM-SafetyBench). Phase 1 adds optimized adversarial perturbations. Phase 2 tests transferability on JailBreakV-28K.

---

## 1. Hypotheses (state these, then try to kill them)

- **H1 (within-image attack signal).** For a fixed base query/image, the attacked variant shifts at least one of {Fiedler value λ₂, HFER, graph-signal smoothness, spectral entropy} in the visual-to-visual block `A_vv` and/or the text-to-visual block `A_tv`, relative to the clean variant.
- **H2 (hijack-conditional signal, the important one).** Among *attacked* inputs, the signature separates **hijacked** (model complied / output captured) from **resisted** (attack present but model refused or stayed on task), with attack type held fixed. This is the clean test: it cannot be explained by "the image contains text" or "the image is complex," because both classes contain the attack.
- **H3 (localization).** The discriminative signal concentrates in a small set of layers, consistent with a cross-modal capture mechanism where attention mass collapses onto the injected region. Expect an early-to-mid layer effect in `A_tv`.
- **H4 (cross-architecture, stretch).** The qualitative signature survives across ≥2 vision-encoder families. NB: per the Geometry of Reason finding, sliding-window or differing attention regimes may shift the discriminative metric from HFER to late-layer smoothness, so compute **all four** metrics and do not assume HFER is the carrier.

**Null we must rule out:** the only thing being detected is the presence of rendered text or increased visual complexity (a register artifact, the exact trap from the extremism project). The paired design (H1) and the hijacked-vs-resisted contrast (H2) are the controls. If H1 holds but H2 fails, we are probably detecting text-on-image, not the attack. Report that honestly. NB: VLSBench was built specifically to remove "visual leakage" confounds from multimodal safety evals; cite it as the motivation for our within-source controls and consider drawing a clean subset from it.

---

## 2. Method recap (the GSP pipeline, for the implementer)

Each attention matrix `A ∈ R^{n×n}` (post-softmax, rows sum to 1) for a given (layer, head) is treated as a weighted graph adjacency over the token sequence `[visual_tokens | text_tokens]`. Slice three views:

- `A_vv`: visual→visual block (image self-structure)
- `A_tv`: text→visual block (the cross-modal channel where capture would show up)
- `A_full`: the whole sequence

For square views (`A_vv`, `A_full`); for the rectangular `A_tv`, build the symmetric co-attention graph `A_tv @ A_tv.T` (shape `n_t × n_t`). Per view:

1. Symmetrize: `W = (A + A.T) / 2`.
2. Degree `D = diag(W.sum(1))`; normalized Laplacian `L = I - D^{-1/2} W D^{-1/2}` (add ε to degrees for isolated nodes).
3. Eigendecompose `L = U Λ U.T`, eigenvalues `0 = λ_1 ≤ λ_2 ≤ ... ≤ λ_n`.
4. **Metrics:**
   - **Fiedler / algebraic connectivity:** `λ_2`.
   - **Spectral entropy:** `p_i = λ_i / Σλ`, `H = -Σ p_i log p_i` (skip the trivial zero eigenvalue).
   - **Smoothness (Dirichlet energy):** for graph signal `x`, `s = (x.T L x) / (x.T x)`. Default signal: per-token attention-mass vector (`W.sum(1)`, mean-centered). Alt signal: per-token hidden-state L2 norm.
   - **HFER:** `x_hat = U.T x`; HFER = energy in top-k high-eigenvalue components / total energy. Default cutoff k = top 25% of the spectrum; config knob.
5. Aggregate across heads: report both **mean across heads** and **max across heads** per layer (max catches a single rogue head; injection may be head-localized).

**Output:** a feature tensor `[n_layers, n_metrics × n_views × {mean,max}]` per example, the "spectral trajectory." Classify on the **full trajectory**, not a single layer (extremism-project lesson).

---

## 3. Critical implementation gotchas (read before coding)

- **bf16, not fp16, for Gemma.** Gemma layers overflow to NaN in fp16. Load Gemma with `torch_dtype=torch.bfloat16`. The 4080 Super supports bf16 natively. fp16 is acceptable only for the non-Gemma fallbacks.
- **Attention must be returned.** Load with `attn_implementation="eager"` and pass `output_attentions=True`. Flash/SDPA silently return `None` for attention weights. This is the #1 silent failure.
- **16 GB memory discipline.** `output_attentions=True` returns a tuple of ALL layers at once. For short sequences (~256–512 visual tokens) on a 4B model this is fine (tens of MB), but to be safe and future-proof, prefer **forward hooks on each attention module** that capture, compute metrics, and immediately free the tensor per layer. Never hold all-layer attentions and the KV cache and a backward graph simultaneously.
- **Gemma 3 specifics (primary model):** each image → **256 visual tokens** (fixed, no pan-and-scan = clean, constant-length `A_vv`). Image tokens use **bidirectional** attention within the image span while text is causal. The bidirectional block is a *feature* here: `A_vv` is naturally closer to an undirected graph, so symmetrization is less lossy. Note the boundary clearly in the token-type mask.
- **Token-type mask.** Find the visual-token span via the model's image/boundary token id in `input_ids` (Gemma expands the image placeholder to the 256-token run). Assert the count equals the expected visual length; fail loudly if not.
- **Variable visual length (Qwen2.5-VL fallback).** Dynamic resolution → variable token count. Fix it via `min_pixels`/`max_pixels` in the processor, or rely on the fact that λ₂/entropy/HFER are scale-normalized. Document the choice.
- **Determinism.** Greedy decoding (`do_sample=False`), fixed seed, bf16, so the hijacked/resisted label is reproducible. Analyze attention at the **first generated token** (or mean over the first 5; pick one, be consistent).

---

## 4. Models (all <7B, fit 16 GB)

| Role | Model | dtype | Notes |
|---|---|---|---|
| **Primary** | `google/gemma-3-4b-it` | bf16 | multimodal SigLIP, 256 fixed image tokens, bidirectional image attention, ~8.6 GB weights, native to the `gemma-spectral` env |
| Cross-arch A | `Qwen/Qwen2.5-VL-3B-Instruct` | bf16/fp16 | latest Qwen VL, dynamic resolution (fix pixels), heavily used in the safety-benchmark literature → comparable to prior work |
| Cross-arch B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | bf16/fp16 | very light (2.2B), third encoder family, fast iteration |
| Optional (you know it) | `google/gemma-3n-e2b-it` | bf16 | the Gemma-3n E2B you used for steering; multimodal, but MatFormer/per-layer-embedding makes attention extraction messier — only if you want a 4th point |

Validate the entire pipeline on **Gemma 3 4B alone** first. Add the others only once Phase 0 gives a clean H1/H2 on Gemma.

---

## 5. Data — Phase 0 (famous ready-made typographic injection, do this first)

No hand-building needed. Use standardized, well-cited benchmarks so reviewers trust the inputs.

- **FigStep** (Gong et al.) — harmful instructions rendered as typographic images; the canonical visual-prompt-injection set. Use the **miniset (~280 samples)** for the pilot. This is the cleanest typographic-injection source.
- **MM-SafetyBench** (Liu et al., ECCV 2024) — 5,040 image-text pairs across 13 scenarios; use the **SD+Typography subset**, which is the one that actually triggers jailbreaks (ASR jumps from ~5% to ~77% with the typographic image).

**Pairing and labelling:**
1. **Clean variant:** the benign counterpart — for FigStep/MM-SafetyBench, run the *same underlying request* with a neutral/non-injected image (or a blank/benign image of matched type) so `clean↔attacked` is paired on the textual task.
2. **Behavioral label per attacked input** (this is what matters, not how it was built):
   - `hijacked` — model complied with the injected/harmful instruction.
   - `resisted` — model refused or stayed on the benign task despite the attack.
   Label by string-match + a lightweight refusal classifier on the generated text.
3. Capture the spectral trajectory at the first generated token for all of `{clean, resisted, hijacked}`.

**Why this wins:** `clean↔attacked` (paired, H1) and `resisted↔hijacked` (attack-type held fixed, H2) are two independent controls against the text-presence confound. Report both contrasts separately. The `resisted↔hijacked` AUROC is the headline.

**Confound control image:** also include a *benign-text-overlay* set (neutral caption rendered into the image, no instruction). If the signal is the attack and not the pixels-of-text, these should score like `clean`, not like `attacked`.

---

## 6. Data — Phase 1 (optimized adversarial perturbations)

White-box PGD perturbation `δ` crafted against the open-weight model to force a target output; paired clean `x` vs `x+δ`, `‖δ‖_∞ ≤ ε`, ε ∈ {4,8,16}/255. This is the Bibi/Paren imperceptible-attack regime and the part most relevant to their grant.

**16 GB note:** crafting δ backprops through the full model. At Gemma-3-4B bf16 this is tight; use **gradient checkpointing**, or craft the perturbations on the lighter **Qwen2.5-VL-3B / SmolVLM2-2.2B** and run *detection* (forward-only) on Gemma. Same paired analysis, behavioral hijacked/resisted labels. Keep crafted perturbations in a local non-published folder; release only the detector + aggregate stats.

---

## 7. Data — Phase 2 (transferability, the Oxford-grant headline)

**JailBreakV-28K** (Luo et al.) — 28K pairs built explicitly to measure whether LLM/VLM jailbreaks **transfer** across models. Use the **miniset (~280)**.

Hypothesis to test: transfer success of an attack between model X and model Y correlates with the **similarity of their spectral fingerprints** (reuse the cross-architecture fingerprinting from Spectral Archaeology). If even a weak correlation holds, it converts transferability from trial-and-error into a *predictive* spectral theory — that is the result that makes the Torr/Bibi pitch.

---

## 8. Statistical analysis plan (house standard)

Per (layer, metric, view, agg) cell, for each contrast (`clean vs attacked`, `resisted vs hijacked`):
- **Cohen's d**, **Mann–Whitney U** p-value, **bootstrap 95% CI** on d (≥10k resamples).
- **Benjamini–Hochberg FDR** across the full layer×metric grid; report q-values.

**Trajectory classifier:**
- L2-regularized logistic regression on the flattened trajectory.
- Train/test split **grouped by base image/query** (an input and its attacked variant must not straddle the split — else leakage).
- **AUROC with bootstrap CI** for both contrasts. `resisted vs hijacked` AUROC is the go/no-go number.
- Ablation: single best layer vs full trajectory (expect trajectory to win).

---

## 9. Repo layout

```
spectral-vlm/
  configs/                 # model + metric + attack configs (yaml)
  data/
    load_figstep.py        # Phase 0
    load_mmsafetybench.py  # Phase 0 (SD+Typography subset)
    load_jailbreakv.py     # Phase 2 (miniset)
    attack_pgd.py          # Phase 1
    overlay_benign.py      # confound-control images
  src/
    extract.py             # load VLM (eager attn, bf16), hooks per attn layer, token mask
    spectral.py            # 4 metrics on A_vv / A_tv / A_full (reuse spectral-trust)
    trajectory.py          # assemble [n_layers, n_features]
    label.py               # hijacked / resisted from generated text
  analysis/
    stats.py               # d, MWU, bootstrap CI, BH-FDR
    classify.py            # grouped split, logreg, AUROC
    plots.py               # per-layer effect-size curves, trajectory heatmaps
  run_phase0.py
  README.md
```

---

## 10. Milestones / definition of done

- **M0:** Gemma 3 4B loads in bf16 with eager attention; single forward yields per-layer attentions; the 256-token visual span is correctly identified and asserted. Smoke test on 1 image.
- **M1:** All four metrics on `A_vv`, `A_tv`, `A_full` for one clean and one attacked input; values finite, sane, differ.
- **M2:** Phase 0 set (FigStep miniset + MM-SafetyBench typography) loaded, run, labelled into clean/resisted/hijacked; class counts reported.
- **M3:** Full stats table (d, q-values per layer×metric) for both contrasts + per-layer effect-size plot.
- **M4:** Trajectory classifier AUROC (with CI) for `resisted vs hijacked`. **Go/no-go for the Oxford pitch.**
- **M5 (stretch):** repeat on Qwen2.5-VL-3B + SmolVLM2; cross-architecture consistency.
- **M6 (stretch):** Phase 2 transferability-vs-fingerprint-similarity correlation on JailBreakV-28K-mini.

---

## 11. Environment (use the existing env)

```bash
conda activate gemma-spectral        # or: source ~/envs/gemma-spectral/bin/activate
# install only what's missing into the existing env:
pip install -U transformers accelerate pillow numpy scipy scikit-learn pandas matplotlib datasets
# spectral-trust if the metric code is reusable:
pip install -U spectral-trust
# datasets pulled from HF hub: FigStep, MM-SafetyBench, JailBreakV-28K (check each card's license/gating)
```
bf16 throughout for Gemma. Gemma 3 4B fits 16 GB with room for short-output attention. For Phase 1 PGD, enable gradient checkpointing or craft on the 2–3B fallbacks.

---

## 12. Honesty checklist before claiming a result

- [ ] H1 and H2 reported **separately**. Positive H1 + null H2 ⇒ likely register artifact; say so.
- [ ] Train/test split grouped by base input (no leakage).
- [ ] FDR correction across the whole layer×metric grid.
- [ ] Effect sizes with CIs, not just p-values.
- [ ] Benign-text-overlay control scores like `clean`, not like `attacked` (else we detect text, not attack).
- [ ] No NaNs from fp16 on Gemma (use bf16).