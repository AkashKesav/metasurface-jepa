# Architecture Issues → Literature-Grounded Fixes

**Date:** 2026-09-02
**Method:** issues enumerated from the repo's own evidence (`checkpoints/milestone_b/EXPERIMENT_LOG.md`, `BUGLOG.md`, `docs/implementation/unified_jepa/VICREG_ROOTCAUSE_CLOSURE.md`), then matched against papers located via firecrawl web search; every abstract below was scraped from arXiv and read directly (not search snippets). Firecrawl search IDs recorded for feedback/credit tracking in the session scratchpad.

> **Citation caveat (Standing Rule 4):** the 2026-dated arXiv IDs below (2603.16209, 2607.23531) were verified live at scrape time, but must be **re-checked independently before being cited in any REPORT.md**.

---

## Issue 1 — EMA-JEPA dimensional collapse on real training runs

**Evidence.** Kaggle Phase 0 (`jepa`): `cos_err ≈ 0.000261` with escalating collapse votes → COLLAPSED @ step 200; cross-sample cosine ≈ 1, effective rank ~1–13 (EXPERIMENT_LOG.md Phase 0; synthetic-collapsed-anchor diagnostics in BUGLOG.md Tier 1). This is representation (dimensional/redundancy) collapse, not a prediction-quality failure.

**Literature.**
- **C-JEPA** — Mo & Tong, *Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning* (arXiv:2410.19560): names this exact defect — "the inefficacy of Exponential Moving Average (EMA) from I-JEPA in preventing entire collapse" — and shows VICReg-style variance/covariance regularization integrated on the JEPA embeddings prevents it, with improved convergence. **This directly validates the repo's `jepa_vicreg`/`jepa_vicreg2` rung design.**
- **LeJEPA** — Balestriero & LeCun (arXiv:2511.08544, v3): isotropic Gaussian is the provably optimal JEPA embedding distribution; SIGReg enforces it with a **single** λ trade-off, no EMA, no stop-gradient, no schedulers. Validates the `lejepa` rung (currently untested post device-fix).

**Status.** Not a code defect — the screening ladder was built to decide exactly this. The decisive 800-step run has still never executed (blocked on dataset staging + cloud session).

---

## Issue 2 — Why collapse recurs even under VICReg: conditional concentration (the deep issue)

**Evidence.** Kaggle Phase 1 (`jepa_vicreg`, λ_var=0.1, λ_cov=0.04) still COLLAPSED at steps 400–500. Not a blanket VICReg failure (EXPERIMENT_LOG.md), but something structural remains unexplained.

**Literature.**
- **The JEPA Paradox in Language** (arXiv:2607.23531, MBZUAI/VinUniversity): deterministic latent prediction only works under **conditional concentration** — the target given the context must lie near a single point. When multiple valid completions exist (e.g. block-masked regions that could plausibly fill in many ways), the MSE/cosine-optimal prediction is a *centroid of many structures* → centroid degeneracy → collapse pressure, regardless of regularizers. Their empirical failure sequence: MI saturation → elevated target variance → train–validation instability → **effective-rank degeneration** → cosine collapse → poor transfer.

**Implication for this repo.** If block-masked metasurface patches at high mask ratios lack conditional concentration, no amount of VICReg/Barlow/SIGReg will fully rescue deterministic JEPA — the fix is the one-shot stochastic latent path `Ẑ_y = P(Z_x, S, ε)` (Milestone H, §7.4), not more ladder rungs. The paper's caution also flags the Milestone-H averaging distortion problem in advance.

**Code fix applied (this commit).** `src/diagnostics/representation_health.py::collapse_trend()` + numeric `eff_rank_r*` logging in `scripts/train/train_milestone_b.py`: a log-only trajectory warning that fires when effective rank declines >25% across the validation window — the paper's precursor — *before* the per-validation vote thresholds fire. It **never** feeds `classify_health` (threshold semantics are Standing-Rule-3 operator decisions). Regression tests: `tests/test_collapse_trend.py` (8).

---

## Issue 3 — Barlow rung scale domination (historical, resolved)

**Evidence.** At `lambda_bt=1.0`, `barlow_ratio ≈ 0.9997` — the jepa_barlow phase was effectively Barlow-only training with a cosmetic L_J (BUGLOG.md Tier 4).

**Fix (already in).** Tier 5 dimension-normalized the Barlow off-diagonal terms (`src/losses/barlow.py`); measured L_BT ≈ 0.997 on the same synthetic data. Literature support: LeJEPA's single-λ philosophy argues against hand-tuned multi-regularizer weights. Do not re-tune λ_bt without a new demonstrated regression (BUGLOG directive).

---

## Issue 4 — Goal-ignoring / guidance-gap risk (Failure Mode 2, future milestones)

**Evidence (anticipated, not yet observed).** §13/§9/§3.5.1: predictor may learn `P(Z_x, S) ≈ P(Z_x)`. The null-gap diagnostic exists (`FixedValidation.null_gap`) and currently shows a non-collapsed gap on smoke checkpoints.

**Literature.**
- **Factored Classifier-Free Guidance** — Xia et al. (arXiv:2506.14399, ICML 2026): a single global guidance scale causes spurious attribute changes in counterfactual generation; attribute-wise factored guidance mitigates it. Applicable to Milestone D/F CFG if the guidance-gap diagnostic stays flat across diverse goals: the single scalar `w` over the 16 goal tokens could be factored instead of globally scaled.

**Status.** Mechanism exists (`src/predictor/guidance.py`); factored variant is a documented option for Milestone F's sweep — not pulled forward (Standing Rule 1).

---

## Issue 5 — Physics-loss dominance risk (Failure Mode 1) and an alternative design

**Evidence (anticipated).** §13/§5/§4.1: physics loss can turn the latent into "whatever the decoder/surrogate needs". Ablation D is the mandated check.

**Literature.**
- **Physics-guided diffusion models for inverse design of disordered metamaterials** — Xie et al. (arXiv:2603.16209, Mar 2026): instead of baking physics loss into training, use the differentiable solver as **sampling-time guidance** — the generator trains once, physics is injected at inference. This eliminates the L_S-vs-L_J tug-of-war *by construction* and decouples retraining from task changes.

**Status.** Not a defect in current code; a documented design alternative if Ablation D shows L_S dragging L_J out of tolerance. Requires operator decision before any implementation (Standing Rules 1–2).

---

## Fixes applied in this change

1. `src/diagnostics/representation_health.py` — new `collapse_trend()` early-warning readout (Issue 2).
2. `scripts/train/train_milestone_b.py` — per-validation numeric `eff_rank_r{r}` metric + trajectory warning print (Issue 2).
3. `tests/test_collapse_trend.py` — 8 regression tests.
4. This document.

## Explicitly NOT fixed here (operator decisions, per Standing Rules / BUGLOG DEFERRED)

- Ladder composition and λ values for the real screening run (incl. λ_bt) — operator sign-off required before Kaggle.
- Mask-ratio policy A/B, VICReg/SIGReg placement, winner-selection criteria, base-init hashing (BUGLOG DEFERRED section).
- refs_model per-run nondeterminism flipping smoke-scale HEALTHY/WARNING (BUGLOG Batch 4 risk) — needs a decided seed or base-init load for refs_model.
- Switching to stochastic latents (Issue 2's structural fix) — that is Milestone H scope; premature under Standing Rule 1 until Milestones B–F resolve.
- Factored guidance / sampling-time physics guidance — Issues 4–5 are design options pending observed failures.

## Path to training (what "pushed to training" actually requires)

1. Code pushed to `origin/main` (this change) — the cloud notebook pulls it.
2. **Stage the MetaDiT dataset + weights** as a Kaggle Dataset / Drive folder per `CLOUD_TRAINING.md` — they are absent locally (`data/metadit/` is `.gitkeep` only), and no data-dependent run can execute until staged.
3. Run the screening/training in `notebooks/cloud_train_runner.ipynb` on Kaggle (local machine is dev-only per AGENTS.md): the pending decisive run is the multi-objective screening (`train_milestone_b.py --config configs/milestone_b.yaml` with the configured objective set / `train_unified.py --config configs/unified.yaml` for the unified model).
4. Push `checkpoints/<milestone>/REPORT.md` back; **human operator review** (Compute Environment step 3) before any next milestone.

The `collapse_trend` warning will print `[collapse-trend WARNING @ step N]` in the Kaggle log as soon as effective rank trends down >25% across validations — an early abort signal that did not exist during Phases 0–1.
