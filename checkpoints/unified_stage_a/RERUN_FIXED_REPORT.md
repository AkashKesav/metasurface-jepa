# Stage-A fixed rerun — REPORT (not a pass declaration)

Run: `akashkesav/metasurface-jepa-stage-a-joint-target` kernel v2
Code: branch `docs/full-training-audit-pr`, kernel cloned tip `7d81de7`
  (EXPECTED_TIP at author time `db68cf5`; kernel-file-only delta, training code identical).
  Local follow-up `d4eda3a` (docstring, L_raw empty-mask guard, test docstring) landed
  after the kernel launched and is NOT in this run.
Config: `configs/unified_stage_a.yaml` — `joint_target: true`,
  `predictor_type: masked_query`, `lambda_raw=1.0` (UNNORMALIZED),
  `lambda_var=10.0`, `lambda_cov=1.0`, `lambda_inv/scalar/occ/phys=0`,
  `total_steps=1500`, `batch_size=2`, mask 0.5, `S_goal=S_true`.
Hardware: Tesla P100-PCIE-16GB (checkpoint `gpu(ckpt)` field).
Exit: `train_unified` exit code 0. `final.pt` step 1499, artifact final.
  Released spectrum keys (106) absent from state-dict as designed (reloaded from disk).

## Did it run completely? Yes

- `steps_run=1500`, `final_step=1499`, `n_val_points=29` (val every 50, step>0).
- `metrics_history.json`: 29 val points + 12-step `train_loss` curve shape verified locally.
- `wall_clock_seconds=85.4`, `seconds_per_step=0.0569` (compute only; session wall time
  is longer: clone + pip install + 29 validations + 15 atomic 89MB writes).
- Speed question: 85s compute for 1500 steps is GENUINE, not skipped work —
  8.45M params, batch 2 (=3000 samples seen), `lambda_phys=0` (no surrogate backward,
  the expensive path), tiny K=16 cross-attention. Local 12-step CPU synthetic smoke
  gives 0.5s/step; P100 at 0.057s/step is the expected ~9x. The old "fast convergence
  to 1e-5" was collapse symptom (constant target), not a skipped run; the fixed run's
  loss is honest (11→14, see below).

## Did it learn? No — FAIL (3/4 gates), but a different, honest failure

Offline eval on real `val_set.mat` (2 batches x 2, mask 0.5;
`scripts/eval/eval_checkpoint_latents.py`, fresh `latent_eval_fixed.json`):

| metric | old FAIL run | fixed rerun | gate |
| --- | --- | --- | --- |
| `L_total` (train) | 1.27e-05 (fake) | 12.6–15.1 (honest) | — |
| `raw_mse` | 235.7 | **5.35** (44x better) | info |
| `raw_cos_err` | 0.0012 | 0.019 | info |
| `z_hat` concentration | 0.00064 | 0.00112 (1.75x) but **< 0.01** | **FAIL** |
| `scale_ratio_zh_zy` | 0.061 | **0.364** (6x) but **< 0.5** | **FAIL** |
| `joint_target_delta_rel` | 4.66 | **0.092 (< 1)** | **PASS** |
| `target_spec_sensitivity_normalized` | 0.0034 | **8.4e-05 (< 0.01)** | **FAIL (worse)** |
| `joint_gate_tanh` | -0.094 | -0.246 (open) | info |
| `z_y_norm` vs `z_y_geo_norm` | 226 vs 42 (5.4x growth) | 49 vs 54 (0.91x) | info |

In-run curve (metrics_history): `raw_mse` 2.36 (step 50) → 5.29 (step 1450) while
`L_total` rises 11.2 → 14.3; `delta_rel` 0.002 → 0.09; sensitivity stays ~1e-3.

## Reading

1. `norm_delta` fix WORKS: target is geometry-anchored again (growth 0.91x, delta 9%).
2. Loss is honest now (unnormalized L_raw + var/cov): no fake 1e-5.
3. Predictor still near-constant and 2.7x too small; 1500 steps x batch 2 with
   `lambda_var=10` vs `lambda_raw=1` may be projector-dominated — needs a measured
   lambda-balance + longer-run decision, not a guess (Standing Rule 3).
4. Goal coupling got WORSE (8.4e-05): a bounded per-token LayerNorm delta preserves
   geometry but carries almost no spectrum variation. This is the VERDICT §8 caveat
   verbatim: bounding alone keeps geometry predictable from geometry — making the
   target *require* S is a Stage-D design question, do NOT pull forward per Rule 1.
5. Stale `latent_eval.json` (old FAIL numbers, path `.kaggle_stage_a_output/...`) was
   present in kernel v2 output alongside the fresh metrics — it is contamination from
   the previous output version, NOT this run. The authoritative artifact for this run
   is `latent_eval_fixed.json` (evaluated locally from v2 `final.pt`).

## Provenance notes

- Evaluator reports ckpt commit `7d81de7 dirty=True`: correct — ckpt saved at `7d81de7`,
  evaluated under `d4eda3a`. No code drift affecting the run (delta = docstring + guard).
- `missing=106` = released spectrum encoder keys (by design, reloaded from disk).
- Full suite at `db68cf5`: 587 passed, 14 skipped (1 pre-existing Barlow flaky deselected).
  Reviewer follow-ups at `d4eda3a`: 63/63 focused pass. Two independent reviewer agents:
  no blockers.

## Decisions needed (Standing Rule 3 — do not guess)

1. Lambda balance + run length for the next Stage-A attempt (var/cov vs raw scale,
   steps/batch within P100 quota).
2. Whether to pursue goal-requirement mechanisms (Stage-D scope) or accept Stage-A as
   non-collapse-only and move the goal question to its proper milestone.
3. Whether to keep the 59 `.kaggle_*` scratch dirs (now gitignored, left on disk) or
   archive/delete after this report is accepted.
