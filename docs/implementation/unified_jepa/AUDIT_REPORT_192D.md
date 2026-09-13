# Audit Report — 192-D unified architecture (retirement + defect + maths audit)

**Date:** 2026-09-13
**Baseline:** `f557eb6` ("Add diagnostic protocols for model evaluation and analysis"), operator directive: *"use this commit only, the newer ones are broken"*.
**Scope:** (a) retire the legacy 384-D Milestone-B path; (b) audit the 192-D unified
occupancy–parameter–spectrum JEPA for defects and fix them; (c) audit the maths and the
encoder→decoder data flow for scaling/normalization correctness.
**Authority:** `architecture_v5.md` (architecture), `00_MASTER_EXECUTION.md` + phase MDs (execution),
repository code/tests (interfaces). Deviations requested by the operator are recorded in
`AGENTS.md` → Dated operator overrides.

---

## 1. Legacy 384-D path — retired and deleted

Operator decision (2026-09-13): delete the legacy code, keep historical reports/docs; legacy stays
recoverable at `f557eb6`. Superseded retention clauses are recorded as a dated override in
`AGENTS.md` and `00_MASTER_EXECUTION.md`.

Commit `57a2cf9` (retirement), `b492f6e` (docs/workflow):

- **Surgery before deletion** (the 192-D path imported legacy files at import time):
  - `Attention`/`CrossAttention`/`TransformerBlock`/`get_2d_sincos_pos_embed` → new
    `src/encoders/blocks.py` (used by `occupancy_encoder`, `fusion_encoder`, `gclct`).
  - `VICRegProjector` → `src/losses/vicreg.py`.
  - `decoders/__init__.py` now exports `OccupancyDecoder`/`ScalarDecoder` (was re-exporting the
    legacy `GeometryDecoder` as the package default).
  - `assembly.py` keeps only `UnifiedJEPA`/`build_unified_model`/`set_spectrum_path`/save-load;
    `train/engine.py` keeps only the checkpoint/EMA/RNG core; `runtime/physics_controls.py`
    drops the legacy `compute_physics_metrics`.
  - `UnifiedJEPA.geometry_decoder` → `occupancy_decoder` (it holds an `OccupancyDecoder`).
- **Deleted:** `context_encoder`, `geometry_encoder`, `geometry_decoder`, the objectives registry
  (`objectives`/`objective_modules`/`barlow`/`sigreg`/`geometry_reconstruction`),
  `reference/direct_masked_generator`, `configs/milestone_b.yaml`, 2 legacy trainers, 19 legacy
  eval/diagnostic/preflight scripts, 29 legacy test files, 3 stray root scripts.
- **Kept as history:** `checkpoints/**` reports, `docs/design_doc.md` (banner: historical),
  `docs/implementation/**`.
- **Tests:** mixed files trimmed to their still-valid coverage; data-dependent tests now skip
  loudly; `test_old_checkpoint_not_compatible` uses a synthesized 384-D state dict.
- **Verified:** full suite green at the new count; `repo_static_audit.py` → 0 findings;
  `train_unified.py --no-train --use-synthetic-smoke` forward smoke OK.

---

## 2. Defect audit — findings and fixes (each fix = one commit with a regression test)

| ID | Finding (cause) | Fix | Commit |
|---|---|---|---|
| B1 | `set_spectrum_path` attached the released MetaDiT encoder with `requires_grad=True` (SpectrumPath's freeze branch is dead because it is constructed with `released=None`), so it entered the optimizer's parameter set | Freeze + `eval()` at attach time; test builds a real dummy checkpoint and asserts frozen/excluded | `83a9b3e` |
| B2 | `EMAEncoder.total_steps` defaults to 1 and the trainer never called `set_total_steps`, pinning momentum at 0.999 from step 1 instead of ramping 0.996→0.999 | `model.set_total_steps(total_steps)`; report exposes `total_steps`/`ema_total_steps` | `b1f2c5a` |
| B3 | The resume path never called `restore_ema_state` (EMA momentum counters were write-only) | Restore on resume, then re-apply this run's schedule; schedule-length changes are reported loudly | `d467b04` |
| B4 | The private curriculum RNG (mask ratio / scalar regime / goal dropout) was not checkpointed — resumed runs restarted the stream from the seed | Persist/restore `curriculum_rng_state` (loud warning for checkpoints without it) | `d176e8d` |
| B5 | On goal-dropped (null) steps the physics term still targeted the sample's true spectrum — the very condition that was dropped (goal-ignoring pressure, §8.3) | Skip `L_phys` on null steps (and skip the wasted released-encoder call); VICReg/scalar still train that branch | `e511fe3` |
| B6 | Validation ran one pooled pass at a hardcoded mask ratio with all scalars known; `easy/hard_mask_ratio` were dead config | Per-stratum validation (easy = low mask + known; hard = full mask + all unknown), guidance gap on the hard stratum; `easy_mask_ratio` redefined 0.0 → 0.25 (a 0 % mask is not a valid loss stratum) | `957bcd5` |
| B7 | Evaluator fed the hard-thresholded, GT-retained occupancy into IoU/F1/fraction → visible-region IoU was identically 1.0 and raw occupancy quality was never measured | `decode_occupancy_prob` (raw sigmoid) for all occupancy diagnostics; retention kept only for the deployed geometry | `adba3c5` |
| B8 | `factorize_geometry` crashed on legitimate all-empty occupancy samples (unconditional h/r positivity assert) | Positivity asserted only for samples containing occupied pixels | `309d238` |
| B9 | `load_surrogate` silently returned a random-initialized surrogate for non-plain-dict checkpoints (or dicts containing `"prediction"`) — every physics loss would have been meaningless | Accept raw/wrapped state dicts (matching `external/metadit/metric.py`), raise a diagnostic error otherwise | `e2c6f94` |
| B10 | The decoder's FiLM saw only the 3 effective values — a known and a predicted scalar of the same magnitude were indistinguishable (the encoder carries flags, §3.2) | 6-dim `[value, known-flag]×3` conditioning; `decode_occupancy_logits/prob` builders; protocol script updated | `7459a78` |
| B11 | Scenario B never exercised an r-known case; `ScenarioInputs` leaked CPU masks and hardcoded a 2-row flag tensor | l→h→r rotation, device move, single authoritative flag helper | `dd07755` |
| B12 | The shuffled-spectrum control drew an unseeded derangement (non-reproducible for B > 2) | Explicit `seed` (evaluator passes seed+tag) | `dd07755` |
| B13 | Collapse check averaged the *untresholded* probability (different definition from the evaluator) and built an autograd graph outside `no_grad` | Testable helper: raw-sigmoid thresholding + per-sample fraction variability (min/std/max) | `dd07755` |
| B14 | Guidance gap used mean-absolute difference ÷ global std, not the spec's per-sample L2 ÷ σ; the sweep fixed scalars known so the hard stratum was unreachable | Shared `normalized_gap_stats` (per-sample L2/per-sample σ) used by both the diagnostic and `cfg_forward`; sweep reports both scalar strata | `f0b0663` |
| B15 | Three doc/code mismatches (ScalarDecoder init, `decode_geometry` return, `z_y_normalized`) | Docstrings/comments corrected to the code | `5d39578` |
| B16 | `need_attn=True` was accepted and silently ignored | Predictor weights returned as `out["attn_weights"]` | `5d39578` |
| B17 | Dead config keys (`variant`, `weights.metadit`, `data.use_synthetic`, `train.epochs`) and silently-ignored valid ones (`data.num_workers`, `loss.scalar_loss_type`); invalid regimes ignored | Wire the valid keys, delete the dead ones, `_validate_config()` (unknown keys warn; bad regimes / out-of-range ratios / incoherent staging raise) | `ba5b561` |
| B18 | Maths/safety: `hard_forward` in training without STE = zero-gradient surrogate input; degenerate target spectra amplified by a silent 1e-6 std floor; `n_film_blocks ≠ geo_depth` surfaced as an opaque IndexError; the per-step frozen guard missed the surrogate; the requested mask ratio was silently assumed equal to the achieved coverage; `OccupancyTokenLoss` unreachable; `cfg_forward` left the model in eval and had an unused arg; the evaluator silently fell back to a hardcoded surrogate path; a checkpoint-writing test could overwrite/delete live checkpoints | Guards added (STE, spectrum std, FiLM count, surrogate-gradient); achieved mask fraction logged per bucket and per validation stratum; dead code removed; `cfg_forward` restores the caller's mode; the evaluator's surrogate path fails loudly; the test refuses to run when live checkpoints exist | `3b6e406` |

---

## 3. Maths & scaling audit (forward data flow + objective/physics/evaluators)

Two independent static passes (encoder→decoder data flow; losses/physics/eval maths), cross-checked
against the released MetaDiT convention (`external/metadit/datapipe.py`, `model/surrogate.py`,
`metric.py`) and `architecture_v5.md`.

### 3.1 Verified correct (no change)

- **Geometry factorisation**: `factorize ↔ assemble` invert exactly (`occupancy × r/5`,
  `occupancy × h`, `l/3` everywhere) and match `src/data/dataset.py` + the released datapipe.
- **Spectrum path**: raw `[real, imag]` enters the frozen encoder — the same convention MetaDiT
  trained it with; no normalisation is expected or applied.
- **FiLM conventions**: encoder/decoder use `γ·x + β` (γ→1, β→0 at init), the predictor uses
  `x·(1+γ) + β` (γ→0, β→0) — both are exact identities at step 0.
- **Token geometry**: encoder flatten order ↔ decoder reshape order match (row-major 16×16);
  16→32→64 upsampling and the pixel-retention interleave match `apply_mask_to_pixels`.
- **VICReg maths**: hinge-mean variance (eps inside the sqrt), unbiased statistics, `(N-1)`
  covariance divisor `/D`, branch aggregation (½ variance / sum covariance) — canonical.
- **Physics normalisation**: a single per-sample normalisation by the *target's* std, applied to
  both prediction and target, consistently across loss types; no double normalisation.
- **STE**: `hard + soft − soft.detach()` — binary forward (in-distribution for the surrogate),
  real gradient through the soft path, training-only.
- **CFG combine**: `z_null + w·(z_real − z_null)`; `p` is the null-replacement probability.
- **Evaluation maths**: target-std-normalised spectrum L1; standard IoU/F1/precision/recall.

### 3.2 Scale-balance observations (no defect proven; runs will confirm)

1. **Loss-term scale at init** (analytic): weighted `L_var` ≈ 16.6 > `L_inv` ≈ 5.7 ≫ `L_scalar` ≈ 0.2
   > `L_cov` ≈ 0.05. The scalar head's mean-bias init is what keeps the physical-unit L1 small; the
   scalar term is if anything *under*-weighted relative to the latent terms. Log the per-term shares
   at step 0 and after the variance plateau before re-weighting (no change made).
2. **Raw scalars are never standardised** anywhere before learning: `r ≈ 4.25` is ≈ 5.7× `h ≈ 0.75`
   and enters the scalar trunk, the encoder FiLM heads, and the decoder FiLM raw (the `r/5`, `l/3`
   rescaling happens only at the surrogate boundary, after all learning). This is a conditioning
   risk, not an arithmetic error. **Decided 2026-09-13: keep raw** (§4) — per-scalar
   standardisation was considered and declined.
3. **Zero placeholders for unknown scalars** (`value = 0, flag = 1/0`) sit outside the physical
   ranges; the flag carries missingness, so it is learnable, but it is a discontinuity rather than
   a smooth missing-value encoding. Observation only.

### 3.3 Spec-vs-code divergences (flagged; the 2026-09-13 resolutions are marked inline)

1. **Occupancy BCE**: `architecture_v5.md` §4.1 specifies `BCEWithLogits(logits, true_occupancy)`
   for the occupancy decoder. The shipped objective (`03_training_and_objective.md` lineage) has
   **no occupancy reconstruction term**: the decoder is supervised *only* by `L_phys`, so at the
   config default `lambda_phys = 0` (staging B) the occupancy decoder receives **no gradient at
   all** until the physics stage. **Resolved 2026-09-13: the BCE term is implemented** (§4,
   `1c53f90`) — the decoder is supervised on masked pixels from step 0.
2. **Null-goal convention**: MetaDiT's own CFG feeds a constant `0.5`-filled spectrum *through* the
   encoder; this repo zeroes the conditioning (`c_physics`/`a_goal = 0`) and skips the frozen
   encoder on null steps (audit B5). Internally consistent; note for any MetaDiT comparison.
3. **Mask ratio semantics**: the requested curriculum ratio is nominal — block masking does not
   achieve it (min-side clamps inflate blocks, independent placement overlaps them; measured, and
   now logged per bucket as `mask_fraction_achieved_mean`). **Resolved 2026-09-13: the
   random-placement masker is calibrated** (§4, `67beea5`); `sensitivity_masks` remains nominal
   (documented; its achieved coverage is logged the same way).
4. **Evaluator diversity** does not implement §8.3 check 10: `pairwise_spectrum_diversity`
   perturbs the latent (`z_hat`), not the target spectrum; the spec asks for target-spectrum
   perturbation sensitivity. Deferred (no trained checkpoint to validate against).
5. `z_y_normalized` is a declared output boundary that nothing consumes (kept, documented).

### 3.4 Deferred items (need a trained checkpoint or a decision)

- §8.2 diagnostic re-runs on the 192-D encoder: `vicreg_gradient_attribution` and the
  within-bucket spatial-structure probe were legacy-scripted and are retired; they need porting to
  `UnifiedJEPA` before their prior numbers can be compared.
- NN baseline pool is capped at 200 training spectra and is not scenario-stratified (raise or
  parameterise before reporting it as a baseline).
- Target-spectrum perturbation sensitivity check (§8.3 check 10).

---

## 4. Operator decisions — resolved 2026-09-13

1. **Scalar conditioning scale — KEEP RAW (no standardisation).** The three scalars stay in raw
   physical units throughout the network, matching the config's declared representation. No code
   change; revisit only if training curves show h-dependence lagging r-dependence.
2. **Occupancy decoder supervision — ADD THE BCE TERM (implemented, commit `1c53f90`).**
   `L_occ = BCEWithLogits(decoded occupancy, true occupancy)` on masked pixels; config
   `loss.lambda_occ = 1.0` (validated ≥ 0). The decoder is now supervised from step 0, including
   while `lambda_phys = 0` (staging B), instead of receiving no gradient at all.
3. **Masker calibration — CALIBRATE (implemented, commit `9c8b9cb`).** `random_masks` redraws until
   the achieved masked fraction is within ±2 % of the requested ratio, with the closest draw kept
   as a bounded fallback; fixed seeds remain reproducible and the achieved fraction stays logged.
   `sensitivity_masks` (half_sensitivity placement) remains nominal — its placement is
   sensitivity-ranked, and its achieved coverage is reported by the same logging.

---

## 5. Verification status (honest)

- **Local environment:** Python 3.14 / torch 2.14 CPU; the released weights and dataset splits are
  **not staged** locally, so data-dependent tests skip loudly by design.
- **Test suite at the time of writing:** `python -m pytest tests/ -q --tb=line` →
  **274 passed, 22 skipped, 0 failed** (skips: CUDA-only paths and not-staged data/weights).
- **Static audit:** `python scripts/preflight/repo_static_audit.py` → 0 findings.
- **Smokes run:** `train_unified.py --no-train --use-synthetic-smoke` (forward+backward),
  `--max-steps 3` synthetic training run with checkpoint write, evaluator unit paths.
- **NOT verified:** anything requiring real data or a trained checkpoint — no gradient-based
  training has been run for this architecture in this session, and no scientific claim is made
  here. The per-scenario gates (`eval_scenarios.py`, hard stratum) remain to be run per
  `CLOUD_TRAINING.md`.

---

## 6. Next step — Kaggle verification run (5–10 % of the schedule)

1. Push this branch (`work-192d`) to the GitHub repo (the code is local; Kaggle must be able to
   clone it) — or upload the tree as a Kaggle Dataset.
2. Create a Kaggle notebook from `notebooks/cloud_train_runner.ipynb`, set `REPO_URL`/`REPO_REF`,
   attach the staged MetaDiT dataset, GPU accelerator + Internet.
3. Mandatory real-data preflight:
   `python scripts/train/train_unified.py --config configs/unified.yaml --device cuda --preflight`
   (non-zero exit = do not train).
4. Verification-scale run (≈10 % of the 1500-step schedule):
   `python scripts/train/train_unified.py --config configs/unified.yaml --device cuda --max-steps 150`
   then delete `checkpoints/unified/*.pt` (the shortened schedule alters the LR/EMA ramps) or
   deliberately resume from it.
5. Full run + per-scenario evaluation + guidance-gap curve, exactly as in `CLOUD_TRAINING.md` §1.

Local check performed here: `kaggle` CLI is installed but **no Kaggle credentials are configured**
(`~/.kaggle/kaggle.json` absent), so the run could not be launched from this session.
