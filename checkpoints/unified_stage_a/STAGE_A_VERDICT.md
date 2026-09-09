# Stage-A joint-target run — VERDICT: FAIL

Run: `akashkesav/metasurface-jepa-stage-a-joint-target`
Commit: `338f1252957b0c4e1c9812bcb7ed6afc86365fd4` (dirty — data symlinks)
Hardware: Tesla P100-PCIE-16GB, torch 2.5.1+cu124, CUDA 12.4
Config: `configs/unified_stage_a.yaml` — `joint_target: true`,
`predictor_type: masked_query`, all loss weights zero except `lambda_raw: 1.0`,
`total_steps: 1500`, `batch_size: 2`, mask ratio 0.5, `S_goal = S_true`.
Model: 8.45 M parameters, `unified_occ_param_spectrum_jepa_v1_joint-mq`.

## 1. Did it run? Yes — completely.

`final.pt` carries `step: 1499`, `is_epoch_end: True`, `artifact_type: final`.
It did not crash and did not stop early. Same for `latest.pt`.

## 2. Did it learn? No. It collapsed.

Evaluated offline on real `val_set.mat` (`scripts/eval/eval_checkpoint_latents.py`,
2 batches x 2, mask ratio 0.5). "Init" = a freshly constructed model from the
same config, so the comparison is like-for-like.

| metric | init | step 1499 | reading |
|---|---|---|---|
| `L_total` | 0.01227 | **1.27e-05** | looks perfect — but see below |
| `raw_mse` | 6.95 | **235.7** | absolute error got **34x worse** |
| `raw_cos_err` | 1.0 | 0.00122 | direction matches, magnitude does not |
| `z_hat_norm` | 13.856 | **13.860** | predictor output scale never moved |
| `z_y_norm` | 30.53 | **226.52** | target scale grew **7.4x** |
| `z_hat_std_dim / z_hat_norm` | — | **0.00064** | predictor emits ~one vector for every masked token |
| `scale_ratio_zh_zy` | 0.454 | **0.0612** | prediction is 16x smaller than its target |
| `raw_mse` vs `L_total` | — | 235.7 vs 1.3e-5 | the two disagree by 7 orders of magnitude |

The model found the trivial solution: make the target a nearly constant vector
and predict that constant. Cosine distance then goes to zero. Nothing about the
run is learning physics.

## 3. Why `L_total` lied

`jepa_loss()` (`src/losses/jepa_loss.py:53`) returns `cosine_distance(pred,
target)` — scale-free. `L_raw` (`src/losses/unified_losses.py:260`) is
`F.mse_loss(F.normalize(z_hat), F.normalize(z_y))` — **also** scale-free, despite
the name. With `lambda_inv = lambda_var = lambda_cov = 0` in the Stage-A config
there is no scale-sensitive term and no anti-collapse term anywhere in the
objective. A 16x magnitude gap is invisible to every term that is enabled.

## 4. Root cause: unbounded fusion delta

§3 specifies `Z_joint = Z_G + tanh(gate) * delta`, `delta = cross_attn(Z_G, Z_S)`,
with `gate = 0` at init. The zero gate makes the target *bit-identical to
geometry-only at initialisation*, but **nothing bounds `||delta||`**. Measured
directly (probe: `_probe_fusion.py`):

```
tanh(gate)                        =  -0.0944
||Z_G||      (geometry target)    =    41.81
||delta||    (raw cross-attn out) =  2057.11     <-- 49x Z_G
||tanh(gate) * delta||            =   194.10     <-- 4.6x Z_G
||Z_joint||                       =   226.25
||delta(S) - delta(S_shuffled)||  =     8.11  (0.39% of ||delta||)
```

So `delta` is (a) enormous and (b) **essentially independent of the goal
spectrum** (0.39% sensitivity). The fusion learned a large constant offset that
swamps the geometry signal: `Z_joint ≈ Z_G + const`, with `|const| ≈ 4.6 |Z_G|`.
The target's remaining variation is only 0.29% of its norm, so "predict the
constant" drives cosine distance to ~0.

This is the original failure the Joint Target Redesign was written to fix —
the goal has no usable effect — reappearing through a different door: an
additive term with no scale bound, combined with a scale-free objective.

`target_spec_sensitivity_normalized = 0.0034` (0.34% of target norm) confirms
the goal coupling is dead.

## 5. What is *not* wrong

Checked and cleared, so the next attempt does not chase ghosts:

- **No numerical divergence.** max |weight| = 15, total parameter norm = 686,
  no NaN/Inf in any tensor. Weights genuinely moved (cos(init, final) between
  0.16 and 0.47 per module). An earlier "parameters grew 1e8-1e12x" alarm was an
  artifact of measuring `|Δ|/|θ_init|` on zero-initialised parameters.
- **No T1/T2/T3 regression.** `ema_state.total_steps = 1500` (T3 fixed),
  `ema_state.target` present with 75 tensors (T2 fixed), `on_optimizer_step`
  still the only EMA update site (T1 fixed).
- **LR schedule is fine.** `base_lrs = [3e-4, 3e-4]`, decayed to 0 by step 1500
  as intended; `last_epoch = 1500`.
- **Real data.** `use_synthetic: false`; local re-evaluation on `val_set.mat`
  reproduces `L_total ≈ 1.3e-5`, consistent with the run's 8.7e-6.

## 6. Why it finished in ~10 minutes (the "how the hell" question)

The wall clock itself is *not* anomalous. It is a small workload:

- 8.45 M parameters, 192-d, 6 geometry blocks, 256 geometry + 301 spectrum tokens
- batch size 2 → 1500 steps = **3000 samples seen**
- `lambda_phys = 0` → **no surrogate backward pass**, which is by far the most
  expensive part of this pipeline when enabled
- plus git clone, `pip install`, 30 validations, and 15 atomic 89 MB checkpoint
  writes

So ~1500 cheap steps on a P100 is a few minutes of compute. The 10 minutes is
consistent and is not evidence of a skipped run.

The thing that *is* anomalous is the loss value, not the duration: the run
"converged" to 1e-5 within a few hundred steps because it fell into the constant
solution immediately. Fast convergence here is the symptom, not the cause.

Caveat: the run recorded no per-step timing, so the exact split between
setup/stepping/validation is not recoverable. Step timing has now been added to
the training script (see §7).

## 7. Process fixes applied (mechanics, not science)

The reason this failure was invisible until a manual post-mortem:

- **The run's metrics were never persisted.** `validate()` computes
  `raw_mse`, `raw_cos_err`, `z_hat_norm`, `z_y_norm` and more — all printed to
  stdout and discarded. `save_checkpoint` was called with
  `metrics={"L_total": last_loss}` only. Kaggle persists just the
  `checkpoints/` tree, so stdout (which is not downloadable via
  `kaggle kernels output`) took every diagnostic with it.
  → `scripts/train/train_unified.py` now writes
  `checkpoints/<subdir>/metrics_history.json` (every validation point, atomic)
  and `final_metrics.json`, and stores the full validation dict in the
  checkpoint's `metrics`/`best_prediction`.
- **No wall-clock instrumentation** → added `elapsed` to the periodic log and
  `wall_clock_seconds` / `seconds_per_step` to `final_metrics.json`.
- **No offline way to judge a finished checkpoint** → added
  `scripts/eval/eval_checkpoint_latents.py`, which reconstructs the decisive
  diagnostics from weights alone and emits an explicit PASS/FAIL with four
  checks: token-collapse, prediction/target scale mismatch, fusion domination,
  and goal sensitivity. It also scores a random-init baseline for contrast.
- **Config/behaviour mismatch (minor):** the Stage-A config lists
  `eval_mask_ratios: [0.0, 0.5, 1.0]`, but `validate()` only reads
  `curriculum.val_mask_ratio` (default 0.5). Ratio 0.0 is also degenerate —
  `random_masks` returns all-visible, so there are zero masked tokens and the
  masked-only objective is undefined. The evaluator now guards this.

## 8. Remediation — needs an operator decision (AGENTS.md Standing Rule 3)

The fixes below change the objective/architecture, i.e. they are scientific
decisions, not mechanics. Not applied.

1. **Bound the fusion delta** (addresses §4 directly). Minimal change preserving
   §3's init guarantee: `joint = Z_G + tanh(gate) * LayerNorm(delta)`, so
   `||delta||` is O(1) per token and `tanh(gate)` controls a *relative* mixing
   strength. Alternative: convex blend `joint = (1-a)*Z_G + a*LN(delta)` with
   `a = sigmoid(gate)`, which bounds `||Z_joint||` but does **not** keep
   `joint == Z_G` at `gate = 0` unless the gate is re-centred.
2. **Make the objective scale-sensitive.** Either make `L_raw` a genuine
   un-normalized MSE, or add `L_scale = (||z_hat|| - sg(||z_y||))^2`. Without
   this, no fix to the fusion is observable in the loss.
3. **Anti-collapse term.** With `lambda_var = lambda_cov = 0` nothing stops a
   constant predictor. Re-enabling VICReg variance/covariance is the standard
   remedy; the design doc's adaptive ladder (`L_J -> VICReg -> LeJEPA`) already
   anticipates this.
4. **Acceptance gate.** §12 should require, before Stage B:
   `scale_ratio_zh_zy ∈ [0.5, 2]`, `z_hat` concentration > 0.01,
   `joint_target_delta_rel < 1`, and `target_spec_sensitivity_normalized > 0.01`.
   All four are now computed by the evaluator.

Note that (1) alone will *not* restore goal conditioning: with a bounded delta
the target keeps its geometry structure, and the student can still predict it
from geometry while ignoring the spectrum. Closing that loophole (making the
target *unpredictable without* `S`) is the substantive design question and is
left to the operator.
