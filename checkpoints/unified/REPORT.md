# Unified 192-D JEPA — cloud run report (Kaggle)

**CURRENT STATUS: the three scenario gates PASS on the primary statistic; scalar dependence fails
one of its two strata — §15 is the current reading, superseding §4, §11 and §12's gate numbers.**
Full epoch on the complete training split (70,000 steps, λ_phys = 3.32). Primary gate statistic is
the paired per-sample win rate (operator decision `3eb38a4`), with the mean criterion reported
alongside:

| gate | win rate (n=32) | primary | mean criterion | full split (17,488) |
|---|---|---|---|---|
| scenario A (hard stratum) | **0.9688** | pass | true | 0.9939 |
| scenario B | 0.9375 | pass | true | — |
| scenario C | 0.9063 | pass | true | — |
| scalar dependence, one known | 0.4375 | **FAIL** | true | **0.5004** (CI contains 0.5) |
| scalar dependence, two known | 0.5938 | pass | true | **0.5106** (means differ by 0.0001) |

**The scalar rows are noise at n = 32** (binomial SE ≈ 0.088): measured on the full split both
strata sit at chance, so the correct statement is *scalar dependence is not demonstrated in either
stratum* (§16) — not the one-pass/one-fail split the 32-sample batch produced, and not the "True"
the mean criterion reported for both on differences of 0.0006 and 0.0001. The cause is now
localised (§17): the scalar conditioning reaches the scalar encoder's output (summary token moves
11 %) but does **not** propagate — `scalar_pred` moves **0.000145** under scalar perturbation
against **0.246** under spectrum perturbation, a factor of ~1,700. The scalar-summary → predictor
path is effectively dead.

The independent goal probe agrees (real 0.0990 vs shuffled 0.6035, a **6.1×** margin), and the
win rate on the full 17,488-sample split is **0.9939** (§14).

Three readings that matter more than the headline:

- **Switching to the win rate exposed a gate that was passing on the wrong statistic.**
  `scalar_dependence_one_known` passes the mean criterion while losing on the majority of samples
  (0.4375). It had not been reported as failing before. The change surfaced this; it did not cause
  it — the same model was measured both ways (§15.2).
- **Scenario A's mean is one sample.** Median real 0.0807 vs shuffled 0.6529 — ~8× better — while
  one pathological sample (real 25.06 against 0.88) drags the mean to 0.8779. That pathology is a
  **0.15 % tail** (27 of 17,488 samples), scattered with no input property predicting it (§14.2),
  and it remains open.
- **The objective does NOT degrade at scale — that reading was mine and it was wrong** (§12.3).
  The "`L_inv` 1.38 → 8.26, `L_cov` → 872" figure came from validation, which runs the objective
  in `eval()` mode where the projector's BatchNorm uses stale running statistics; measured
  train-vs-eval on the same state the gap is **759×** and **364×**. Training's own numbers at step
  69,990 (`L_inv=0.0069`, `L_cov=2.29`) match the train-mode measurement, and `L_cov` is only
  **3.65 %** of the gradient budget (§13).

The earlier negative readings in §4 were produced by a **2-sample** estimator and are superseded:
the model was content-sensitive and the measurement could not see it (§11.2).

> **Correction (recorded rather than silently edited).** An earlier revision of this file listed
> "generative diversity = 0.0" as evidence against the model. That was wrong: the evaluator's
> `diversity_check` runs with its default `perturbation_scale=0.0`, which is a **determinism**
> check (same input → identical output), and its own docstring says the result "must not be
> presented as genuine generative diversity". The spec's actual probe — *perturb the target
> spectrum slightly and confirm the decoded design moves proportionally*
> (`architecture_v5.md` §8.3) — is **not implemented** (see §7).

The gate is the per-scenario hard-stratum real-vs-shuffled physics-consistency gap
(`architecture_v5.md` §8.3 check 8), reported per scenario and never pooled. The criterion was
never relaxed: only its **sample size** changed (§10.3, audit B27), after the 2-sample version was
shown to be an estimator with almost no power.

This file is the CLOUD_TRAINING.md §3 sync-back record for the unified 192-D cloud session of
2026-09-13.

---

## 1. What ran

| | |
|---|---|
| Platform | Kaggle (kernel, private), Tesla **P100-PCIE-16GB**, Internet ON |
| Code package | Kaggle dataset `anosvol/metasurface-jepa-192d-unified-code` — **v3** for the verification run, **v4** for the full run |
| Commit | `4deab8a44c9e5011c7ab95f350109851abea4e88` (verification, §3) · `330f9419451336475aeaa492de6f35614e1c3b0b` (full run + gate, §4) — branch `work-192d` |
| Config | `configs/unified.yaml`, sha256 `e6cb5880c3ef65f6c3cbce231a96d6684dbfd82cebf1f57e86bd0f2265c7b375` |
| Data (staged, symlinked) | Kaggle dataset `anosvol/metadit-aaai2026-staging` |
| Kernels | `anosvol/metasurface-jepa-192d-verify-run` (v3), `anosvol/metasurface-jepa-192d-full-run` |

**Why the code travels as a dataset:** the authoring machine has no GitHub credentials and
`origin` is the upstream `tegga8/metasurface-jepa` (not pushable), so the committed tree is
packaged with `git archive HEAD` (canonical LF endings) and uploaded privately. `PROVENANCE.txt`
inside the package carries the commit and the config sha256, and the kernel verifies both before
running anything — a stale dataset version cannot silently train different code.

Staged data (sizes in bytes, as observed by the kernel):
`split_data/train_set.mat` 1,250,200,400 · `val_set.mat` 156,273,152 · `test_set.mat` 156,282,088
· `weights/spec_encoder.pth` 62,434,870 · `surrogate_model.bin` 25,419,827 ·
`metadit-small.bin` 148,843,390.

### 1.1 Environment deviation (operational, not a research deviation)

The Kaggle image ships **torch 2.10.0+cu128**, whose build list is
`['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — it **cannot execute on the
assigned P100** (compute capability **sm_60**). The kernel detects this (device capability vs
`torch.cuda.get_arch_list()`) and installs the `requirements.txt` pin
`torch==2.5.1`/`torchvision==0.20.1`, whose cu124 wheels cover `sm_50…sm_90`. Post-install:
`torch 2.5.1+cu124`, arch list `['sm_50','sm_60',…,'sm_90']`. Install took 205 s.

This is exactly the failure recorded for the earlier (unijepa v6) line and is now handled by the
kernel rather than left to chance: the run refuses to fall back to CPU and refuses to train on a
torch that does not match the validated pin.

---

---

## 2. Defects the cloud runs surfaced (all fixed before the evaluated run)

1. **Provenance false alarm (line endings).** The first package pinned a config sha256 computed
   from the **Windows working tree (CRLF)**; the package carries the **committed blob (LF)**.
   The kernel's staleness guard fired and refused to train — correctly, since the two hashes
   genuinely differed. Fix: the kernel hashes CRLF→LF-normalised bytes and the provenance records
   both hashes with the convention spelled out, so the check tracks content rather than the
   authoring checkout's format.
2. **B21 — the mandatory preflight could not run.** The B18 guard in
   `src/assembly.py::decode_geometry` tested the model's **mode** (`assert not self.training`)
   rather than **gradient tracking**, contradicting its own message ("run this path under
   eval/no_grad for diagnostics"). It therefore refused `preflight()`'s hard-assembly
   diagnostics, which legitimately run in train mode, and the preflight exited 1. B18 had added
   the guard without updating its one existing legitimate caller, and nothing caught it
   locally because the preflight needs the real splits and released weights, which the dev
   machine does not stage. Fix (`4deab8a`): predicate is
   `not self.training or not torch.is_grad_enabled()` (a gradient-enabled training forward is
   still refused), plus a `no_grad` wrapper on the preflight's diagnostic block, with regression
   tests for both directions.
3. **B22 — the evaluation aborted on a device mismatch.** `derangement_permutation` forwarded the
   *target* device to `torch.randperm` while using the caller's generator; torch requires them to
   match, so the evaluator's seeded shuffled control (CPU generator + CUDA target, audit B12)
   raised `Expected a 'cuda' device type for generator but found 'cpu'` and killed
   `eval_scenarios.py` before it produced a single number. Fix (`9753265`): draw on the
   generator's own device, then move the permutation to the target. The pre-fix failure is
   CUDA-only, so the regression test that exercises it is `skipif`-gated and skips loudly here.
4. **B23 — the guidance-gap sweep never ran.** `run_guidance_gap_sweep.py` called
   `train.engine.load_checkpoint(..., strict_model=True, strict_objective=False)` — the TRAINING
   resume API, which requires an objective/optimizer/scheduler and has no `strict_model`
   argument — raising `TypeError`. Fix (`330f941`): a testable `load_eval_model()` mirroring the
   evaluator's loader (strict model state + EMA target state).

B21 is recorded in `docs/implementation/unified_jepa/AUDIT_REPORT_192D.md` §2; B22/B23 are in
their commit messages. Every fix is its own commit with a regression test.

---

---

## 3. Verification-scale run (CLOUD_TRAINING.md §1 step 6) — **PASSED**

Command: `train_unified.py --config configs/unified.yaml --device cuda --max-steps 150`
(150 = 10 % of `train.total_steps: 1500`).

**Preflight** (`--preflight`, real data + released weights, GPU): `exit=0` in 24 s.
Shapes and finiteness all OK (`z_hat` [2,256,192] → geometry [2,3,64,64] → surrogate
[2,2,301]); `geometry_invariants_ok`, `known_scalar_precedence_ok`,
`unknown_scalar_precedence_ok` all true. Gradient ownership — the check that matters:

| Component | params receiving gradient |
|---|---|
| student | 362 |
| predictor | 210 |
| occupancy decoder | 14 |
| released spectrum encoder | **0** |
| EM surrogate | **0** |
| `ema` target | **0** |
| `scalar_mlp_ema` target | **0** |

So the frozen set is genuinely frozen (audit B1) and the decoder now receives gradient even with
`lambda_phys = 0` (the added `L_occ`, audit B19/B20 lineage).

**Training loop:** 150 steps, `exit=0`, 23 s. Loss 23.53 → 16.20. Per-term trajectory:
`L_inv` 0.174 → ~0.013 (invariance being learned), `L_var` ~0.5–0.7 (hinge), `L_cov` 0.4 → ~2–3,
`L_scalar` 0–0.35, **`L_phys` = 0 throughout** (stage B: `lambda_phys = 0.0` by config, so the
frozen surrogate is exercised only in the preflight, not in the training loop — expected for
this phase).

**Validation** ran at steps 50 and 100 and reported the **easy and hard strata separately**
(never pooled, audit B6): easy = mask 0.25 + all scalars known; hard = mask 1.0 + all scalars
unknown. Guidance gap is reported on the **hard** stratum: 1.638 → 2.358 (normalised
2.468 → 3.060) — non-zero and moving, i.e. the goal conditioning is influencing predictions.
That is a pipeline observation, **not** a gate result.

**Mask-ratio calibration confirmed end-to-end on real data** (audit B20) — achieved fraction for
each requested bucket across the run:

| requested | achieved (mean) |
|---|---|
| 0.25 | 0.2462 |
| 0.50 | 0.5002 |
| 0.75 | 0.7582 |
| 1.00 | 1.0000 |

Bucket frequencies: 0.25 → 0.233, 0.5 → 0.313, 0.75 → 0.287, 1.0 → 0.167.
Scalar regimes: all_known 0.300, all_unknown 0.353, mixed 0.347.
Report carries `total_steps = 150` and `ema_total_steps = 150` — the EMA schedule is initialised
from the run's length (audit B2) and restored on resume (B3).

**Artifacts:** `checkpoints/unified/final.pt` and `latest.pt`, 162,953,994 bytes each (163 MB),
plus `preflight.log`, `verify_run.log`, `summary.json`, staged `VERDICT.txt`.
Retrieved locally via `kaggle kernels output anosvol/metasurface-jepa-192d-verify-run`.

> **These checkpoints are verification artifacts, not results** (CLOUD_TRAINING.md §1 step 6).
> The shortened schedule changes the LR and EMA ramps, so the full run starts fresh rather than
> resuming from them.

---

---

## 4. Full run + acceptance gate — **the gate fails**

Kernel `anosvol/metasurface-jepa-192d-full-run` (version 2), code commit
`330f9419451336475aeaa492de6f35614e1c3b0b`. All stages exited 0: preflight 24 s, full run
**1500 steps in 152 s**, `eval_scenarios.py --scenario all` 12 s, guidance-gap sweep 3 s.
Checkpoints `final.pt` / `latest.pt`, 162,953,994 bytes each.

### 4.1 The gate (`eval_scenarios.py`, real validation split)

Gate criterion in the evaluator: `gate = real_spectrum_error < shuffled_spectrum_error`
(lower is better). `shuffled` conditions the model on a deranged spectrum, so a model that
actually uses the spectrum must do worse on it.

| scenario | real | null | shuffled | shuffled − real | gate |
|---|---|---|---|---|---|
| **A** — pure inverse design (**hard stratum**: full occupancy mask, all scalars unknown) | 0.6336 | 0.6420 | 0.6332 | **−0.00039** | **FAIL** |
| B — partial parameters | 0.3418 | 0.3455 | 0.3436 | +0.00185 | pass |
| C — retrofit | 0.3124 | 0.3124 | 0.3087 | −0.00365 | **FAIL** |

Scenario B's pass is a technicality: a +0.0019 margin on a 0.34 error (~0.5 %), and the
`real_minus_null` gap there is the same order. The scenario the design's gates apply to is A,
and it fails.

### 4.2 Corroborating diagnostics (same evaluation)

- **Spectrum-sensitivity probe — not implemented (see the correction in the banner and §7).**
  `diversity_A` (`pairwise_spectrum_diversity = 0.0`, `deterministic = true`) is a **determinism**
  check that passed; it is not evidence of anything about the spectrum. The spec's
  perturb-the-target-spectrum probe does not exist in the evaluator, so this gate is currently
  **unmeasured**, not failed.
- **Scalar dependence — zero.** `scalar_dependence_one_known`: real and shuffled are
  **bit-identical** (`0.6261729598045349`). `scalar_dependence_two_known`: `0.3237603` vs
  `0.3237450` (≈1.5e-5). Neither gate passes; the scalars are not influencing the decode.
- **Retrieval baseline — the model loses badly.** `nn_baseline` (L1 nearest real training
  spectrum → its real geometry): mean error `0.0744`, best `0.0486`, versus the model's `0.6336`
  on scenario A — the trivial baseline is **~8.5× better**. No claim of useful inverse design
  survives this comparison.
- **Occupancy head — partially working, partially collapsed.** Scenario A masked-region
  IoU `0.684`, F1 `0.812` (precision `0.729`, recall `0.916`) — the decoder does learn occupancy
  structure, so the failure is specifically in the *conditioning*, not the decoder.
  `collapse_check`: `pred_occupancy_fraction` `0.5012 ± 0.0066` against a true `0.3989` —
  near-constant and biased high (not all-empty, but low variability).
- Guidance-gap sweep (§20.3) ran clean (`exit=0`); see `guidance_gap_sweep.log`.

### 4.3 Reading this honestly (stated as context, not as an excuse)

- **Scale.** 1500 steps × batch 2 = **3,000 samples**, ≈2 % of a single epoch (the train split is
  ≈140 k samples). The run is a pipeline-scale run, not a converged model.
- **Physics was off.** `lambda_phys = 0.0` (staging B) throughout, so nothing in this run
  optimized spectrum consistency directly; the only pathway tying the spectrum to the design was
  `L_inv` through the target latent.
- **The invariance objective had not aligned anything yet.** Validation reported
  `raw_cos_err = 1.0` (predicted and target latents orthogonal) at steps 50 and 100.
- **What this does and does not establish.** It establishes that *this* configuration, at this
  scale, does not pass the gate. It does **not** establish that the architecture cannot — that
  would require a converged run. Both readings are hypotheses at this point.

**Decision point (`AGENTS.md` → If something fails).** The failure is recorded here; the next
step is the operator's call among (a) diagnose and retry within scope — most plausibly train to
convergence and/or activate the physics stage, as separate one-change commits; (b) a scope or
threshold decision; or (c) stop the line. No mechanism is to be added, and no threshold moved,
in order to make this gate pass.

---

---

## 5. Honest verification status

- **Verified:** the real-data pipeline runs end to end on a cloud GPU — data staging, released
  weights, frozen-component gradient ownership, training to the configured schedule, EMA updates,
  stratified validation, calibrated masking, checkpoint write, and **both** evaluators
  (`eval_scenarios.py`, `run_guidance_gap_sweep.py`) against the produced checkpoint.
- **Measured and negative:** the §8 gates this project is judged by. Scenario A (hard stratum)
  fails, generative diversity is exactly zero, scalar dependence is exactly zero, and a trivial
  retrieval baseline beats the model by ~8.5×. See §4. These are results — negative ones — and
  they are the reason this report carries a NEGATIVE banner rather than a claim.
- **Not established:** that the architecture cannot pass these gates. This run is ~2 % of an
  epoch with `lambda_phys = 0`; whether the failure is scale/staging or something deeper is a
  hypothesis, and testing it is the operator's call (§4.3).
- **Local gate for the same commit:** `python -m pytest tests/ -q --tb=line` → **278 passed,
  23 skipped, 0 failed**; `scripts/preflight/repo_static_audit.py` → 0 findings.
  (One skip is a CUDA-gated regression test for B22 — see §2.)

---

## 6. Resume / repo hygiene

- Checkpoint files are **not** committed (`checkpoints/**/*.pt` is gitignored); only this report
  is tracked.
- To reproduce or continue: dataset `anosvol/metasurface-jepa-192d-unified-code` (**v4**;
  commits `4deab8a` = verification run, `330f941` = full run) + `anosvol/metadit-aaai2026-staging`,
  kernels `anosvol/metasurface-jepa-192d-verify-run` (v3) and
  `anosvol/metasurface-jepa-192d-full-run` (v2).
  `--resume checkpoints/unified/latest.pt` works if a future session re-attaches the run's
  output directory; a fresh full run is the clean default.
- **Cost note for planning:** the full 1500-step run took **152 s** on a P100 (the frozen
  surrogate is not in the loop while `lambda_phys = 0`), so longer schedules — the most obvious
  next experiment (§4.3a) — are cheap; the fixed cost per session is the ~205 s torch pin install.
- **Deviation to be aware of:** the evaluated package is pinned at `330f941`, the branch head at
  the time of writing. Rebuild the package from head for any further run so the pin stays exact.

---

---

## 7. Physics-path audit (pre-activation), 2026-09-13

Run before enabling `lambda_phys`, because the spec makes several checks preconditions
(`04 §3` "Do not silently choose STE without the check"; `04 §13` acceptance list; `03 §5`
"Ramp lambda_phys from zero"). Code read: `src/physics/physics_loop.py`,
`src/losses/unified_losses.py`, `src/assembly.py::decode_geometry`,
`scripts/train/train_unified.py`, `configs/unified.yaml`.

### 7.1 Correctly wired (verified in source)

1. **The path matches the spec.** `z_hat` + `scalar_pred` → `decode_geometry` → assembled
   `[B,3,64,64]` → frozen surrogate → normalized SmoothL1 against the true spectrum
   (`physics_loop.physics_loss_from_out`; `architecture_v5.md` §4.3, `04 §4`). One student
   forward, one physics decode, one surrogate forward — no second spectrum predictor (`04 §5`).
2. **Frozen-but-differentiable surrogate.** `load_surrogate` sets `requires_grad_(False)` +
   `.eval()` on the parameters but leaves the forward differentiable w.r.t. geometry, and
   `UnifiedJEPALoss.train()` re-pins the surrogate to eval so `objective.train()` cannot flip its
   38 BatchNorm layers into batch-stat mode (the corruption protocol-v1 recorded). Matches `03 §6`
   / `04 §4`.
3. **Normalisation.** Per-sample **target** std, applied to both prediction and target, with
   `smooth_l1` — the spec's "normalized L1 or SmoothL1", real/imag convention preserved.
4. **Ramp.** `objective.lambda_phys = lambda_phys * min(1, (step+1)/ramp_steps)` when
   `ramp_steps > 0`, recomputed from `step` every iteration — so it is resume-consistent without
   separate schedule state. `03 §5` ("Ramp lambda_phys from zero") is satisfied.
5. **Retention and known-scalar substitution** are applied inside `decode_geometry` before
   assembly, so physics cannot overwrite observed geometry (`04 §6`,
   `architecture_v5.md` §4.1).
6. **Null-step skip** (operator decision recorded in `AGENTS.md`, audit B5) — `goal_mode != "null"`
   gates `L_phys`; the spectrum-free terms still train that branch.
7. **Per-step frozen guard covers the surrogate** (audit B18) — `_assert_no_ema_gradients(model,
   step, objective)` checks `ema`, `scalar_mlp_ema`, the released encoder **and**
   `objective.surrogate`.
8. **Fail-loud loading.** With `lambda_phys > 0` in real mode, a missing/unusable surrogate
   checkpoint raises instead of silently substituting a zero placeholder (audit B9/Fix 3), and
   `_validate_config` refuses `staging.phase` A/B with `lambda_phys > 0` (audit B17).

### 7.2 Broken or missing (must be addressed around activation)

1. **The STE decision has now been verified against the real surrogate — and it is
   load-bearing.** This was the top pre-activation gap; it is closed by the probe kernel
   `anosvol/metasurface-jepa-192d-physics-probe` (commit `6e6427d`, real released
   `surrogate_model.bin`, 6,328,698 params, `requires_grad: false`). Measured on a real
   validation batch (8 samples, hard stratum mask = 1.0):
   - **Soft path (`use_ste=False`) is a silent dead end.** `L_phys` computes to **18.58** — a
     perfectly plausible-looking number — while **`student_params_with_grad = 0`** (encoder 0,
     predictor 0, decoder 0, scalar encoder 0). Every gradient is zero.
   - **STE path (`use_ste=True`) works.** `L_phys = 0.336`, **360 student params with gradients**
     (encoder 72, predictor 210, decoder 14, scalar encoder 17), with **0** on the surrogate and
     **0** on both EMA targets.
   - **Soft occupancy is dramatically out-of-distribution for the surrogate:**
     `spectrum_rel_diff = 0.9599` (96 % relative difference between the soft and hard-forward
     surrogate outputs), `surrogate_out_of_distribution = true`, `ste_recommended = true`.

   So `physics_use_ste: true` is not a preference — with it off, physics would contribute
   *nothing* while reporting a healthy loss. This is precisely the failure `04 §3`'s
   "Do not silently choose STE without the check" exists to prevent, and it also explains the
   18.58-vs-0.336 loss gap: the soft field drives the surrogate into a degenerate regime.
   **Consequence to fix:** nothing currently refuses the combination
   `lambda_phys > 0` **and** `physics_use_ste: false` — a configuration that would silently make
   the physics term a no-op. The B18 guard covers only `hard_forward=True`, not the plain soft
   path. (`use_ste=False, hard_forward=False, training=True` takes the bare `else` branch in
   `decode_geometry` with no assertion.) Recorded as the next defect to fix.
2. **Nothing asserts the physics term actually reaches the student.** The preflight counts student
   parameters that received gradient, but those gradients also come from
   `L_inv/L_var/L_cov/L_occ` — a physics path with a dead Jacobian would pass the ownership check
   unnoticed. The probe's `Q2` measurement above is exactly the missing assertion, and the
   instrument (`surrogate_gradient_test`, and a stricter per-term variant) exists but is never
   called by the pipeline. **Fix: run this as part of the preflight when `lambda_phys > 0`.**
3. **The spec's spectrum-sensitivity probe is not implemented.** `architecture_v5.md` §8.3:
   "Test by perturbing the target spectrum slightly with everything else fixed and confirming the
   decoded design changes proportionally". The evaluator's `diversity_check` defaults to
   `perturbation_scale=0.0`, which is a determinism check (its docstring: must not be presented as
   genuine generative diversity); with `perturbation_scale > 0` it perturbs `z_hat`, not the
   target spectrum. This is the single most diagnostic probe for the failure just measured.
4. **Validation reports `L_phys = 0` while physics is active.** `physics_active` requires
   `model.training`, and `validate()` runs under `model.eval()` + `no_grad()`, so the validation
   JSON will show `L_phys: 0.0` / `L_phys_weighted: 0.0` in every physics-enabled run. The gating
   itself is defensible (in eval mode `decode_geometry` takes the soft path — a different
   quantity), but reporting a hard `0.0` is misleading; it should be reported as not-evaluated.
5. **`PhysicsSpectrumLoss` is inert dead code.** `_enabled` is never set (`enable()` has no
   callers), so the inactive branch always returns a zero tensor — a placeholder that reads like a
   real fallback.
6. **Degenerate-spectrum crash risk: retired by measurement.** The B18 guard raises when any
   sample's spectrum std `< 1e-3`. The census run in the same probe kernel over **20,000 training
   and 17,488 validation samples** found `std_min = 0.4278` (train) / `0.4298` (val) and **zero**
   samples below `1e-3` — a ~430× margin at the observed minimum. The guard will not fire on this
   dataset.

### 7.4 What the negative result most plausibly means (hypothesis, not conclusion)

The predictor *is* goal-sensitive — validation's guidance gap on the hard stratum was 2.47 → 3.06
(normalized) and non-zero. But the **decode** is not: scenario A's real-vs-shuffled errors differ
by 0.0004, and the predicted occupancy fraction is nearly constant (`0.5012 ± 0.0066` against a
true `0.3989`). That pattern — goal-sensitive latent, goal-insensitive output near the dataset
mean — is consistent with **decoder collapse to the mean**, which is exactly what `L_occ` alone
permits (with only BCE supervision, the mean occupancy is a competitive solution) and exactly what
the physics term is supposed to break. It is also consistent with plain under-training (~2 % of an
epoch). Both readings predict that activating physics is the informative next experiment; neither
is established yet.

---

---

## 8. EMA and pipeline wiring audit, 2026-09-13

Requested alongside the physics audit. Source read directly; the two claims marked *(spot-checked)*
were re-verified by me independently of the agent report that produced them.

### 8.1 EMA — correct

1. **Construction.** `EMAEncoder.__init__` deep-copies the source encoder and sets
   `requires_grad_(False)` on every target parameter (`target_encoder.py:17-19`), and
   `UnifiedJEPA.__init__` asserts none is trainable (`assembly.py:214-217`). Both targets
   (`ema` ← `occupancy_encoder`, `scalar_mlp_ema` ← `scalar_encoder`) are built this way
   (`assembly.py:194-205`).
2. **Seeding.** `build_unified_model` explicitly initialises both targets from their students
   (`assembly.py:533-534`) — the 384-D checkpoints are deliberately *not* loaded.
3. **Update maths.** `p_t.lerp_(p_s, 1 - m)` is exactly `m·p_t + (1-m)·p_s`, under
   `@torch.no_grad()` (`target_encoder.py:31-35`) — no gradient can reach the target.
4. **Cadence.** `objective.on_optimizer_step(model, step)` (`train_unified.py:938`) runs *after*
   `optimizer.step()`/`scheduler.step()` (`:936-937`) and updates **both** targets
   (`unified_losses.py:266-269`) — correct EMA semantics (target follows the updated student).
5. **Schedule.** `current_momentum(step) = start + (end-start)·min(1, step/total_steps)`, with
   `set_total_steps` called from the trainer (audit B2). The 0-based loop step is passed, so the
   first update uses m = 0.996 exactly. Recomputed from `step`, so it is resume-consistent.
6. **Target path isolation.** The target forward is inside `with torch.no_grad():`
   (`assembly.py:335`), so `z_y_raw` carries no `grad_fn`; the objective's projector still receives
   gradient from the `p_y` branch because the *projector* is a separate learnable module — the
   canonical VICReg topology, as documented at `unified_losses.py:96-104`.
7. **Target-side FiLM.** `scalar_mlp_ema` is fed the all-known 6-dim true-scalar input
   (`assembly.py:337-342`) — conditioning only, never decoded, never a loss target (§3.6).
8. **Freeze enforcement.** `enforce_frozen_reference_modes()` pins both targets and the released
   encoder to `eval()`, and is called from `train()` **and** at the top of every `forward()`
   (`assembly.py:225-236, 269`).
9. **Round trip.** `collect_ema_state`/`restore_ema_state` carry momentum endpoints, `total_steps`
   and both target weight sets, with a loud warning when a legacy checkpoint has no `target`
   (`engine.py:146-218`); the trainer restores them on resume (audit B3).
10. **Per-step guard.** `_assert_no_ema_gradients` checks `ema`, `scalar_mlp_ema`, the released
    encoder and the surrogate every step (`train_unified.py:347-372`, called `:853, :934`).
    *(Empirically confirmed on the cloud run: preflight reported `ema_params_with_grad: 0`,
    `scalar_mlp_ema_params_with_grad: 0`, `released_params_with_grad: 0`, `surrogate:
    params_with_grad: 0` while 362 student params had gradients.)*

### 8.2 EMA — issues (none currently breaking, all worth knowing)

1. **The targets are saved twice.** `SAVED_EXCLUDES = (".released.",)` (`assembly.py:61`), so the
   EMA targets (registered submodules) travel inside `ckpt["model"]` **and** again inside
   `ema_state`. Restoring is likewise doubled (`load_into_model` + `restore_ema_state`). Redundant
   but complete — the risk is the inverse of a gap: if one path were wrong the other would mask it,
   so a divergence would be hard to notice.
2. **`restore_ema_state` overrides the config's momentum endpoints.** It assigns
   `ema.momentum_start/end` from the checkpoint (`engine.py:187-189`), and the trainer's resume
   path re-applies only the *schedule length* (`total_steps`). Editing `ema_momentum_start/end` in
   the YAML and then resuming therefore has **no effect** — the checkpoint wins. Correct for exact
   resume, surprising for a deliberate schedule change.
3. **`EMAEncoder.update` zips parameter iterators** (`target_encoder.py:34`). `zip` silently stops
   at the shorter sequence, so any future structural divergence between target and student would
   silently skip parameters rather than raise. Safe today (the target is a deep copy made at
   construction), fragile by design.
4. **Not verified empirically: that the target actually *tracks* the student.** Validation showed
   `raw_cos_err = 1.0` (predicted and target latents orthogonal) with `raw_z_y_norm ≈ 30.3` against
   `raw_z_hat_norm ≈ 9.2`. That is consistent with a genuinely lagging/diverged target as well as
   with an undertrained student, and no artefact in the run measures target-vs-student distance.
   **Cheap decisive check:** report ‖target − student‖ / ‖student‖ per encoder at the end of the
   next run. Not yet done.

### 8.3 Pipeline wiring — verdicts

| Stage | Verdict | Key evidence |
|---|---|---|
| Data path (dataset → collate → factorize → masker → scalar masker → forward) | **wired** | `train_unified.py:657,770-771,788-789,905-907,443-445,423-424` |
| Curriculum (mask ratio, scalar regime, goal dropout) + RNG round trip | **wired** | `train_unified.py:427,424,450-451,275-288`; `spectrum_encoder.py:104-110` |
| Goal / CFG — null branch truly zeroes the conditioning | **wired** | `spectrum_encoder.py:104-110` (zeros, and skips the released encoder) |
| Optimizer ownership (EMAs/released/surrogate out, projector in) | **wired** | `train_unified.py:714,721-728`; `assembly.py:54-55,214-217`; `physics_loop.py:72-73` |
| Checkpoint round trip | **wired except `cfg`** | `engine.py:272-297,331-379`; `train_unified.py:805,809-816,966-991` |
| LR schedule (warmup + cosine, driven by `total_steps`, stepped once per optimizer step) | **wired** | `train_unified.py:626,729-731,936-937,380-396` |
| Validation (same factorization/masker conventions, per-stratum) | **wired** | `train_unified.py:461,492-503,522-528,782-792` |
| Eval path (checkpoint → model + EMA, no objective needed) | **wired** | `eval_scenarios.py:37,518-527` |

### 8.4 Pipeline wiring — not wired / half-wired

1. **`cfg_forward` has no consumer.** *(spot-checked)* The function exists and is correct
   (`guidance.py:57`, `cfg_combine = z_null + w·(z_real − z_null)` at `:31,102-103`), but a
   repo-wide grep finds it **only at its definition** plus tests/docs — no call in `scripts/train/`,
   `scripts/eval/` or `scripts/diagnostics/`. So training prepares the unconditional branch
   (goal dropout + null steps) and then **nothing ever uses classifier-free guidance at
   inference**; the guidance weight `w` is never exercised. The evaluator's real/null/shuffled
   comparison does its own two-pass calls and does not combine them.
2. **`ckpt["cfg"]` is saved but never read.** *(spot-checked)* `save_checkpoint` writes it
   (`engine.py:280`) and the schema requires it, but no load path consumes it — resume always uses
   the YAML passed via `--config`. Editing the config between save and resume therefore silently
   diverges from what the checkpoint recorded. This is the only saved-but-not-restored state.
3. **`ScalarMasker.sample`'s `masked_values` is discarded** — `sample_scalar_known` keeps only the
   `known` flags (`train_unified.py:287-288`); value masking is re-derived inside
   `model._build_scalar_input`. Correct result, dead output.
4. **`validate` hardcodes `placement="random"`** (`train_unified.py:501-503,589-591`) regardless of
   `cfg.curriculum.mask_placement`. With `half_sensitivity` configured, training and validation
   would mask differently; currently both are `random`, so it is latent.

---

## 9. Long run (20,000 steps) — the learning verdict, 2026-09-13

Run to separate "too few steps" from "structurally not learning": **identical config, no changes
except `--max-steps 20000`** (13× the previous run). Kernel
`anosvol/metasurface-jepa-192d-long-run` (v5), commit `0b69b23`, all stages `exit=0`
(preflight 26 s, **20,000 steps in 1846 s**, eval 16 s, EMA probe 3 s).

### 9.1 The EMA is working — the earlier suspicion is retired

Measured directly (nothing in the pipeline did this before): relative L2 distance between each EMA
target and its student at step 19,999.

| pair | tensors | rel. distance (mean) | min / max |
|---|---|---|---|
| `ema` (occupancy encoder) | 74 | **0.0009** | 0.0000 / 0.0063 |
| `scalar_mlp_ema` (scalar encoder) | 18 | **0.0013** | 0.0000 / 0.0037 |

Both track to within ~0.1 %. So the `raw_cos_err`/norm-gap observations were *not* a drifting
target; the targets are fine. §8.2 item 4 is closed.

### 9.2 It IS learning — but it plateaus at ~5,000 steps, and the hard gate still fails

Improvements over 20k steps (validation trajectory, 399 validations):

| | step 50 | step ~5,000 | step 19,999 |
|---|---|---|---|
| `raw_mse` (easy) | 6.147 | 4.721 | 4.731 |
| `raw_cos_err` (easy) | 1.0000 | 0.9587 | 0.8588 |
| `proj_mse` (= `L_inv`, easy) | 1.420 | 4.227 | 4.153 |
| `proj_cos_err` (hard) | 0.7732 | 0.7397 | 0.7938 |
| `L_cov` (hard) | 15.08 | 174.00 | 160.30 |

**Every quantity plateaus by ~step 5,000 and then does not move for another 15,000 steps.** So the
1500-step result was not merely undertrained — a 13× longer schedule buys nothing after the first
third.

Real progress *did* happen, which the 1500-step run could not show:

- **The decoder is no longer collapsed to the mean.** `collapse_check` predicted occupancy fraction
  `0.3817 ± 0.1288` against a true `0.3989` (was `0.5012 ± 0.0066`). It now varies per sample.
- **Scenario B gate: PASS** — real `0.0935` vs shuffled `0.3330` (+0.2396).
- **Scenario C gate: PASS** — real `0.1388` vs shuffled `0.3609` (+0.2220).
- **Scalar dependence (one known): PASS** — `0.216466` vs `0.218410`.

**But the gate that matters still fails.** Scenario A (hard stratum: full occupancy mask + all
scalars unknown): real `0.2046` vs shuffled `0.1889` → **−0.0157, gate false** — the true spectrum
is *slightly worse* than a deranged one.

### 9.3 The CFG sweep localises the failure (new measurement)

`cfg_guidance_sweep_A` (audit B26 wired this in; it had no caller before):

| w | 0.0 (pure null) | 0.5 | 1.0 (plain real) | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|---|---|
| spectrum error | 0.7302 | 0.3826 | **0.2046** | 8.6815 | 9.5531 | 9.3941 |

Reading: removing the goal entirely costs a lot (`w=0` → 0.7302 vs `w=1` → 0.2046), so the model
*is* using the conditioning. But the real-vs-shuffled gap is ~0, so what it uses is the **presence
of a goal, not its content**. Extrapolating past `w=1` diverges catastrophically (8.7–9.6), i.e.
the real/null difference is not a meaningful direction to amplify. `w=1` — no guidance at all — is
optimal, so the CFG machinery adds nothing here as trained.

This is a sharper statement of the failure than "the gate is red": the spectrum conditions the
model as a mode switch, not as a target to fit. It also means the projector-absorption hypothesis
(§7.4) is only *part* of the story — the representation improved and the decoder un-collapsed, so
absorption is not total.

---

## 10. Physics ON — activation validated, and a gate-precision finding (2026-09-13)

Run: kernel `anosvol/metasurface-jepa-192d-physics-goal-probe` (v1), commit `f3f1244`
(staging C, `lambda_phys = 0.1` ramped over 500 steps), all stages `exit=0`: preflight 23 s,
**10,000 steps in 1292 s**, eval 14 s, goal-content probe 12 s, EMA probe 3 s.

### 10.1 The activation is confirmed working end to end

- **The B24 physics-alive preflight check passes in the real pipeline**: it reports **362 student
  parameters with gradient from the physics term ALONE**, with `surrogate = 0` and both EMA
  targets `= 0`. The check that took three cloud round-trips to get right now does its job.
- **The ramp behaves as specified**: `L_phys` 0.011 → 0.218 by step 10, `L_phys_w` ramping
  `0 → ~0.004` over the first 500 steps, then holding at `0.1 · L_phys`.
- Cost: **0.129 s/step with physics vs 0.092 s/step without** (+40 %) — the frozen surrogate in the
  loop is cheap.
- EMA still tracking at step 9,999: relative target↔student distance `0.0033` (occupancy) /
  `0.0080` (scalar).

### 10.2 The goal-content probe: the content signal is ALIVE at every stage

Presence = real vs null; content = real vs a deranged (another sample's) spectrum. Hard stratum
(full mask, all scalars unknown), N = 8.

| stage | presence (real vs null) | **content (real vs shuffled)** |
|---|---|---|
| `c_physics` (frozen spectrum encoder) | 1.000 | **1.320** |
| after `c_phys_proj` (linear) | — | **1.141** |
| after `goal_proj` (linear) | — | **1.010** |
| `z_hat` (predictor output) | 0.100 | **0.142** |
| `occupancy_logits` (decoder) | 0.253 | **0.341** |
| `binary_occupancy` | 0.561 | **0.615** |
| fraction of occupancy pixels flipped | 14.1 % | **16.9 %** |
| normalized spectrum error | null `0.6124` | real **`0.2989`** vs shuffled **`0.5561`** |

Two things follow.

1. **The input carries the content** (`c_physics` content ratio 1.32 — the real-vs-shuffled
   difference exceeds the vector norm itself), the linear projections preserve it (1.14 / 1.01),
   and it survives to the deployed binary design. The "presence switch, not a target"
   characterisation from the physics-OFF 20k run (§9.3) **does not hold with physics on**.
2. **Real is ~1.9× better than shuffled** (0.2989 vs 0.5561) on this 8-sample hard-stratum probe —
   the direction the gate requires.

### 10.3 Why the official gate disagrees — it is decided on TWO samples

`scenario_A_rns` on the same checkpoint reports real `0.5734` vs shuffled `0.4474` → gate **false**,
the opposite direction from §10.2. The cause is in the measurement, not the model:

`eval_scenarios._load_val_batch` builds its batch as `b = cfg["train"]["batch_size"]` — **2**. Every
real/null/shuffled comparison, the shuffled control (a derangement of 2 items = one swap), the
scalar-dependence checks and the whole `cfg_guidance_sweep_A` are therefore computed on **two
samples**.

Corroborating evidence that the gate is under-powered rather than informative:

- the CFG sweep is non-monotonic and noisy on this run — `w=0` 0.608, `0.5` 0.434, `1.0` 0.573,
  `2` 0.671, `3` 0.599, `5` 0.405 — whereas with physics off it was a clean monotone blow-up;
- scenario A's gap moved from `−0.0157` (20 k, physics off) to `−0.1260` (10 k, physics on) while B
  (`+0.4444`) and C (`+0.2316`) hold large, stable margins;
- an 8-sample probe on the *same* checkpoint points the other way (§10.2).

**This is a measurement-precision problem, not a licence to declare a pass.** The criterion is
unchanged — `real < shuffled` on the hard stratum, per scenario, never pooled — and a gate decided
on 2 samples cannot settle the question in either direction. It must be evaluated on a proper
sample size before any conclusion is drawn from §10.2. Making the evaluation batch a first-class
parameter (and reporting per-sample errors, not just batch means) is the next change; because it
moves the gate's numbers it is recorded here as an operator decision rather than applied silently.

---

## 11. λ_phys sweep — results (2026-09-13)

Kernel `anosvol/metasurface-jepa-192d-lambda-sweep` (v1), commit `3bd472a`, all stages `exit=0`.
Calibration 11 s, then five arms trained from scratch at 10,000 steps each (935 s for the
physics-off control, 1084–1125 s for the physics arms), scored by the **32-sample** gate
(audit B27) — the first properly-powered gate readings in the project.

### 11.1 Calibration — the grid came from gradient share, not from guessing

At step 0 on a real batch, `L_phys = 0.4417`. Gradient norms delivered **by the physics term
alone** vs **by everything else**:

| module | ‖∇L_phys‖ | ‖∇L_other‖ | λ for 1 % / 10 % / 50 % share |
|---|---|---|---|
| `occupancy_decoder` | **12.395** | 5.327 | physics already dominates here at λ=1 |
| `predictor` | 4.524 | 18.155 | |
| `occupancy_encoder` | 1.588 | 5.041 | |
| `fusion_encoder` | 1.013 | 3.814 | |
| `scalar_decoder` | 0.649 | 1.014 | |
| `scalar_encoder` | 0.340 | 1.179 | |
| **median-derived grid** | | | **0.0335 / 0.3689 / 3.3201** |

The committed `λ = 0.1` sits at ≈ 2.7 % gradient share — small, but not the ~0.03 % the loss
*magnitudes* suggested, so the "very likely inert" prediction in the plan was too pessimistic.
Note the decoder: physics out-gradients every other term there by 2.3× at λ=1, so the decoder is
where the physics signal has its strongest grip.

### 11.2 The gate passes — on every arm, including the physics-off control

Scenario A, hard stratum (full occupancy mask + all scalars unknown), n = 32, gate criterion
`real < shuffled`:

| arm (λ) | real | shuffled | gap | beats fraction | paired σ | gate |
|---|---|---|---|---|---|---|
| **0** (control) | 0.2901 | 0.6021 | +0.3121 | 0.906 | 0.301 | **pass** |
| 0.0335 | 0.4390 | 0.5719 | +0.1329 | 0.719 | 0.301 | **pass** |
| 0.1 (was committed) | 0.3995 | 0.5792 | +0.1797 | 0.781 | 0.311 | **pass** |
| 0.3689 | 0.2235 | 0.6162 | +0.3927 | 0.969 | 0.235 | **pass** |
| **3.3201** | **0.1679** | 0.5687 | **+0.4009** | 0.969 | 0.202 | **pass** |

Two conclusions, both of which change the project's read of its own history:

1. **The earlier "gate fails" readings were a measurement artifact, not a property of the model.**
   The λ=0 control *passes* here with 90.6 % of samples supporting the comparison. The old
   2-sample batch was val samples 0–1, which are a subset of this 32-sample batch — so those two
   particular samples were unrepresentative, exactly the "single swap" failure audit B27 fixed.
   **The model was content-sensitive all along; the gate could not see it.**
2. Retained caveat: B and C also pass, and scalar dependence still does not (§11.4).

### 11.3 Which λ is best — paired across the same 32 samples

Because every arm is evaluated on the identical 32 samples, the arms can be compared pairwise per
sample rather than by their means:

| λ | mean real | vs λ=0 | se | t | verdict |
|---|---|---|---|---|---|
| 0 | 0.2901 | — | — | — | control |
| 0.1 | 0.3995 | +0.1095 | 0.0427 | **+2.57** | **worse than no physics** |
| 0.0335 | 0.4390 | +0.1489 | 0.0430 | **+3.46** | **worse than no physics** |
| 0.3689 | 0.2235 | −0.0665 | 0.0294 | −2.27 | better |
| **3.3201** | **0.1679** | **−0.1222** | 0.0330 | **−3.70** | better |

Head-to-head, λ = 3.3201 beats every other arm significantly (t = −2.48 vs 0.3689, −3.70 vs 0,
−5.70 vs 0.1, −6.70 vs 0.0335). **There is a harmful region: λ ∈ [0.03, 0.1] makes the
hard-stratum error significantly worse than no physics at all**, and the committed 0.1 was in it.
The improvement is still rising at the top of the swept range, so 3.3201 is a lower bound on the
optimum rather than a located maximum — recorded, and the reason the value is treated as
provisional.

**The pre-registered rule is degenerate here** (every arm passes, so "the smallest passing λ"
selects nothing): the quality ordering above is what actually decides, and λ = 3.3201 is taken.

### 11.4 Everything else measured per arm

| arm (λ) | B gate | C gate | scalar 1-known | scalar 2-known | collapse (frac ± σ) | CFG w=1 |
|---|---|---|---|---|---|---|
| 0 | pass | pass | **fail** | pass | 0.4361 ± 0.0881 | 0.290 |
| 0.0335 | pass | pass | **fail** | fail | 0.4357 ± 0.0906 | 0.439 |
| 0.1 | pass | pass | **fail** | fail | 0.4371 ± 0.0832 | 0.400 |
| 0.3689 | pass | pass | **fail** | pass | 0.4335 ± 0.0792 | 0.224 |
| 3.3201 | pass | pass | **fail** | fail | 0.4790 ± **0.0364** | **0.168** |

- **Scalar dependence fails in the one-known stratum for every arm** — physics does not fix it.
  This is now a separate, clearly-isolated open issue.
- **A caution at high λ**: the predicted occupancy fraction's spread drops from 0.088 (λ=0) to
  **0.0364** at λ=3.32, i.e. the decoder becomes more constant as physics grows (fraction 0.479
  against a true 0.399). The best spectrum error coincides with a partial re-collapse of the
  occupancy head — worth watching on the full run.
- **The retrieval baseline is 0.1254** on this 32-sample batch; λ=3.32's 0.1679 is within 34 % of
  it, where the previous 2-sample reading had the model 2.7× worse.
- **CFG stays "no guidance is best"**: `w=1` is optimal in every arm and `w>1` diverges (9–15),
  consistent with the real/null difference not being a meaningful direction.
- The independent goal probe agrees with the gate in every arm (`real_beats_shuffled = true`,
  real 0.18–0.34 vs shuffled 0.55–0.64).

### 11.5 What this changes

- The acceptance gate — the criterion this project has been failing — **passes**, with a properly
  powered test, at the best λ.
- The previously reported negative result is superseded: it was produced by a 2-sample estimator.
  The correct statement is *the model was content-sensitive and the measurement could not see it*.
- λ = 0.1 was a **harmful** value, not merely an inert one; λ = 3.3201 is the best measured.

---

## 12. Full epoch on the complete training split (2026-09-13)

Kernel `anosvol/metasurface-jepa-192d-full-epoch` (v1), commit `096ca6b`, λ_phys = 3.32, staging D.
All stages `exit=0`: preflight 21 s, **70,000 steps in 7,812 s (2 h 10 m)** — one epoch over the
full ~140k-sample training split at batch 2 — eval 16 s, goal probe 10 s, EMA probe 3 s.
This is the first run that is both properly powered (32-sample gate) and full-scale.

### 12.1 The gate passes on all three scenarios

| scenario | real | null | shuffled | gap | gate | beats fraction | paired σ |
|---|---|---|---|---|---|---|---|
| **A** (hard stratum) | 0.8779 | 0.5632 | 1.2459 | +0.3680 | **pass** | 31/32 | 5.7795 |
| **B** (partial params) | 0.1078 | 0.4898 | 1.1825 | +1.0747 | **pass** | 0.938 | 3.6492 |
| **C** (retrofit) | 0.0445 | 0.1119 | 0.1931 | +0.1486 | **pass** | 0.906 | 0.2296 |

Scalar dependence now passes **both** strata for the first time (one-known 0.913926 vs 0.916606;
two-known 0.123611 vs 0.124269 — the second by a margin of 0.0007, so "passes" is technical), and
the goal probe agrees with the gate: real **0.0990** vs shuffled 0.6035 vs null 0.6065 — a
**6.1×** margin on the hard stratum.

### 12.2 Scenario A's mean is one sample — the median is the honest summary

The `paired σ` of 5.78 on a meagre +0.3680 gap is the tell. Per-sample:

| | real | shuffled |
|---|---|---|
| median | **0.0807** | **0.6529** |
| mean | 0.8779 | 1.2459 |
| min / max | 0.0157 / **25.0568** | 0.0507 / 21.2290 |

**A single sample (index 7) scores real = 25.06 against shuffled = 0.88** — a −24.18 paired
difference. Removing it changes the picture completely:

- real mean: **0.8779 → 0.0980**
- gap: **+0.3680 → +1.1597**
- and the apparent anomaly that `null (0.5632) < real (0.8779)` disappears (real 0.098 ≪ null 0.563)

So the model is **good on 31 of 32 samples** (median 8× better than shuffled) with one
pathological failure that dominates the mean. The gate passes either way — but the mean is not a
robust summary of scenario A, and the sample-7 failure is a concrete, isolated thing to diagnose.

This also reconciles the probe-versus-gate discrepancy seen earlier: the probe samples indices
0–7 via `seed=0`, the evaluator samples 0–31 via `seed=42`, and the pathological sample falls in
the evaluator's batch but not the probe's. Neither measurement was wrong; they were different
samples.

### 12.3 Trajectory over the epoch, and a correction to the "objective degrades" reading

| quantity | first | last (step 69,950) |
|---|---|---|
| `raw_mse` (easy) | 5.8113 | **4.4055** |
| `raw_cos_err` (easy) | 0.9994 | **0.8271** |
| `proj_mse` (= `L_inv`, easy) — **see the correction below** | 1.3822 | 8.2556 |
| `L_var` (easy) | 0.6005 | 0.0024 |
| `L_cov` (easy) — **see the correction below** | 9.2381 | 872.18 |
| `z_hat` norm | 9.6067 | 6.3922 |
| `z_y` norm | 30.2278 | 27.8660 |

- **The raw representation improves over the full epoch** — `raw_cos_err` 0.999 → 0.827, better
  than the 20k-step plateau (0.859) — so one epoch does buy something the short runs could not.
- EMA tracking is the best measured yet: **9.2e-05 / 8.6e-05** relative distance (0.009 %).
- Occupancy collapse did **not** worsen at scale: predicted fraction `0.4305 ± 0.0776` against a
  true 0.3989 — better spread than the 10k-step λ=3.32 arm's 0.0364.
- CFG: `w=0` 0.5632, `w=0.5` 0.4164, `w=1` 0.8779, `w≥2` 10.0–16.5. Still "no guidance is best",
  and the non-monotonicity is again the outlier in scenario A, not a change in the model.

> **Correction (recorded rather than silently edited).** This section previously read "the
> objective degrades badly at scale — `L_inv` 1.38 → 8.26, `L_cov` → 872 — the single largest open
> objective defect". **That was wrong, and the whole claim came from reading an eval-mode number
> as if it were the training objective.**
>
> `validate()` runs the objective under `eval()`, where the projector's two `BatchNorm1d` layers
> use their accumulated **running** statistics instead of batch statistics. Measured at the final
> checkpoint on one batch, same state, projector in each mode:
>
> | term | projector `train()` | projector `eval()` | ratio | validation reported |
> |---|---|---|---|---|
> | `L_inv` | 0.011127 | 8.446861 | **759×** | 8.2556 |
> | `L_cov` | 1.989043 | 724.186768 | **364×** | 872.18 |
> | `L_var` | 0.000613 | 0.000386 | 0.6× | 0.0024 |
>
> The eval-mode values reproduce what validation printed, and the training log at step 69,990
> reads `L_inv=0.0069 L_cov=2.2886` — which the train-mode measurement matches. So **training was
> healthy and the trajectory column was measuring a different function.** The mechanism: the one
> shared projector is applied to TWO different distributions (the predictor's output and the EMA
> target's output), and its running statistics average the two, fitting neither — hence a 364–759×
> eval/train gap.
>
> **Why it matters beyond the number**: the projectors' outputs are training-only machinery, so
> nothing deployed is affected. But it means every `L_inv`/`L_cov` figure this project has quoted
> from a validation block is a train-mode quantity reported as an eval-mode one, and cannot be used
> to judge the training objective. `L_cov` in particular is **3.65 %** of the gradient budget at the
> final state (§13), so neither it nor its 872 were ever the thing steering training.
>
> **Open, and separate**: a projector whose `BatchNorm` normalizes each branch by its own batch
> statistics can make the invariance loss small without the raw representations being aligned —
> measured here as `L_inv` (train) `0.011` against `raw_mse` `4.41` / `raw_cos_err` `0.83`. That is
> the projector-absorption mechanism of §7.4 with a concrete candidate cause, and it needs its own
> ablation before anything changes; it is not established by this measurement.

---

## 13. Post-hoc diagnosis of the full-epoch checkpoint (2026-09-13)

Kernel `anosvol/metasurface-jepa-192d-outlier-diag` (v4), read-only, no training. The checkpoint
was recovered from the full-epoch kernel's output and re-published as the private dataset
`anosvol/metasurface-jepa-192d-full-epoch-ckpt` (188 MB), so the trained artifact can be analysed
without paying the 2 h 10 m re-train. Runs in ~4 min (mostly the torch pin install).

### 13.1 Per-term gradient share at the final state — the λ decision input

Measured at step 69,999 on a real batch, each term differentiated ALONE (projector restored from
the checkpoint — `objective_state`, 15 keys, nothing missing):

| term | λ | value | ‖∇L‖ (unweighted) | λ·‖∇L‖ | **share of gradient budget** |
|---|---|---|---|---|---|
| **`L_phys`** | 3.32 | 0.045799 | 4.2971 | 14.2663 | **66.15 %** |
| `L_scalar` | 1.0 | 0.021801 | 2.6081 | 2.6081 | 12.09 % |
| `L_inv` | 25.0 | 0.011127 | 0.0839 | 2.0979 | 9.73 % |
| `L_occ` | 1.0 | 0.257898 | 0.9690 | 0.9690 | 4.49 % |
| `L_var` | 25.0 | 0.000613 | 0.0335 | 0.8381 | 3.89 % |
| `L_cov` | 1.0 | 1.989043 | 0.7867 | 0.7867 | **3.65 %** |

Conclusions, and the decisions they support:

- **`λ_cov` is NOT to be touched.** At 3.65 % of the gradient budget — third-smallest — and with a
  value of `1.99` (not 872) on a real batch at the final state, re-weighting it would be tuning a
  term that barely steers training. The "`λ_cov = 1` vs `λ_var = 25` imbalance" story that the
  earlier reading suggested is dead: the two are 3.65 % and 3.89 %, i.e. equally minor.
- **`λ_phys = 3.32` already dominates the gradient budget at 66 %.** The sweep's trend (higher λ →
  lower hard-stratum error, still rising at the top of the range) still argues for checking above
  3.32, but the *upside is bounded* and the risk is real: invariance is down to 9.7 %, so further
  physics weight starves the representation objectives. Treat an upward extension as a bounded
  check expecting diminishing returns, not the obvious win it looked like from the error curve
  alone.
- **N-dependence of `L_cov` is real but small**: 4.746 → 3.632 → 1.989 as the masked-token count
  goes 126 → 256 → 512 (N/D 0.66 → 2.67). A 2.4× range — it explains part of the eval-mode
  inflation in principle, but nowhere near the measured 364×, which the BatchNorm mode accounts
  for (§12.3).

### 13.2 The scenario-A outlier is NOT a data pathology

Reproducing the evaluator's exact batch (seed 42, 32 samples), the worst sample and the median
sample:

| batch pos | split index | true occ. frac | pred occ. frac | spectrum std | err real | err shuffled | scalars |
|---|---|---|---|---|---|---|---|
| **7** | **7176** | 0.4766 | 0.4486 | 0.4839 | **25.0568** | 0.8818 | [2.79, 0.99, 4.9] |
| 8 | 12136 | 0.4355 | 0.4394 | 0.6068 | 0.2227 | 0.6139 | [2.81, 0.88, 4.97] |
| 3 | 10702 | 0.4141 | 0.3628 | 0.5695 | 0.2222 | 0.7117 | [2.52, 0.94, 4.87] |
| 1 (median) | 8846 | 0.4805 | 0.4848 | 0.5953 | 0.0800 | 0.3727 | [2.85, 0.52, 4.42] |

Its occupancy, predicted occupancy, spectrum std and scalars are all inside the normal range, and
the batch contains **no** empty or near-empty occupancies (`n_occ_frac_zero = 0`,
`n_occ_frac_below_0.01 = 0`, spectrum std range 0.484–0.688). So the failure is **not** degenerate
input: the model maps that particular spectrum to a catastrophic design while a *different*
spectrum on the same sample yields 0.88. It is an instability hole in the learned inverse map —
specific and diagnosable, unlike the aggregate "one sample is bad" reading.

---

## 14. Failure-rate scan over the full validation split (2026-09-13)

Kernel `anosvol/metasurface-jepa-192d-failure-scan` (v1), read-only on the full-epoch checkpoint,
all stages `exit=0`, scan 154 s. Scenario A (hard stratum) evaluated on **all 17,488 validation
samples**, chunked at 256, real and shuffled conditioning — so the gate is measured on 17,488
samples instead of the 32 the evaluator uses.

### 14.1 The distribution

| conditioning | mean | median | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| **real** | **0.1196** | **0.0749** | 0.1973 | 0.3500 | 6.7443 | 30.4993 |
| shuffled | 0.6084 | 0.6023 | 0.9009 | 1.1731 | — | 23.6908 |

- **`real` beats `shuffled` on 99.39 % of the 17,488 samples** (vs 31/32 at n = 32). The paired
  difference is `+0.4889` mean, `+0.5093` median.
- **The model now beats the retrieval baseline on the mean**: `0.1196` against `0.1254`, and on
  the median it is ~8× better than the shuffled control.
- p90 `0.1973` — 90 % of designs are within 0.2 of the target spectrum.

### 14.2 The failure mode is a 0.15 % tail, not a property of the model

| threshold | samples | fraction |
|---|---|---|
| > 0.5 | 34 | 0.1944 % |
| > 1.0 | **27** | **0.1544 %** |
| > 2.0 | 25 | 0.1430 % |
| > 5.0 | 21 | 0.1201 % |
| > 10.0 | 17 | 0.0972 % |

27 catastrophic designs out of 17,488. The worst 20 are **scattered** — split indices 12874,
11053, 5573, 7431, 6897, 11402, 7176, 11199, 10643, 17440, 9774, 366, 5117, 9479, 17387, 14411,
16712, 6805, 318, 9814 — with **no input property that predicts them**: every one of the 20 has an
occupancy fraction in 0.36–0.61 and a spectrum std in 0.43–0.61, i.e. squarely inside the normal
range. There is no degenerate-input regime to exclude and no cluster to characterise.

The signature is distinctive: on those samples the **shuffled** conditioning gives a *normal*
result (0.7–1.4) while the **true** spectrum gives 15–30. So the model maps a small set of specific
spectra into a region where its decoded design blows up — an instability hole in the learned
inverse map, not a data problem and not a general failure.

### 14.3 What this settles

- **"One bad sample out of 32" is now a rate: 0.15 %, unpredicted by the inputs.** The 32-sample
  gate batch simply happened to contain one of the 27 (a ~5 % chance per batch), which is why its
  mean read 0.8779 while the full-split mean is 0.1196. The 2-sample batches used before audit B27
  hit the same tail far more easily.
- **The model is working.** Mean better than the retrieval baseline, median 8× better than the
  control, 99.39 % of samples on the right side of the gate.
- **A mean-based gate is fragile against this tail.** `real < shuffled` on the means can flip on a
  single catastrophic sample, whereas the paired beats-fraction (99.39 %) is stable. Making the
  beats-fraction the primary gate statistic — or reporting both — is a **gate-definition decision
  for the operator**, not a change to make silently; the criterion itself is unchanged and nothing
  here relaxes it.
- The honest statement of quality is therefore: *a working inverse design with a ~0.15 %
  catastrophic-failure tail whose cause is not yet identified.*

---

## 15. The paired win rate as the primary gate — first reading (2026-09-13)

Operator decision recorded in `3eb38a4`: the primary gate statistic is
`real_beats_shuffled_fraction` (the paired per-sample win rate), with the mean criterion kept and
reported as `gate_mean_criterion`. Kernel `anosvol/metasurface-jepa-192d-new-gate` (v1), commit
`3eb38a4`, `exit=0`, ran the authoritative evaluator against the full-epoch checkpoint with
`--samples 32`.

### 15.1 The output

| gate | win rate | `gate` | `gate_mean_criterion` |
|---|---|---|---|
| scenario **A** (hard stratum) | **0.9688** | pass | true |
| scenario **B** | 0.9375 | pass | true |
| scenario **C** | 0.9063 | pass | true |
| scalar dependence, one known | **0.4375** | **FAIL** | true |
| scalar dependence, two known | 0.5938 | pass | true |

### 15.2 Two findings, one of them new

1. **The change exposed a gate that was passing on the wrong statistic.**
   `scalar_dependence_one_known` passes the mean criterion while its win rate is **0.4375** — it
   loses on more samples than it wins, and passed only because its few wins were larger than its
   losses. That is exactly the pathology the switch was made to catch, and the old statistic was
   reporting it as a pass. **Scalar dependence in the one-known stratum is now a failing gate**,
   and it had not been before. This is a finding the change surfaced, not one it caused: the same
   model was measured both ways.
2. **The 32-sample gate now agrees with the 17,488-sample scan.** Scenario A's win rate of 0.9688
   matches the full-split 0.9939 to within the batch-size difference, and is unmoved by the sample
   that had driven the mean to 0.8779 with a paired σ of 5.78. Robustness is the point, and it
   holds.

The two statistics disagree on one-known: the mean says pass, the win rate says fail. Where they
disagree the win rate is the honest one — a conditioning that loses on the majority of samples is
not demonstrating dependence, however large its occasional wins.

### 15.3 Threshold

The implemented threshold is **0.5** (definitional majority), config key
`eval.gate_beats_fraction_min`, validated to [0, 1]. At **0.75** — the value that would make the
gate mean "measurably better" rather than "more often than not" — `scalar_dependence_two_known`
(0.5938) flips to **fail** as well, joining one-known; the three scenario gates pass either way.
That is a one-line change and is the operator's call; it is flagged rather than applied because it
changes a gate's verdict, which is precisely the kind of change that must not be made silently.

---

## 16. Scalar dependence over the full validation split (2026-09-13)

The new primary gate reported `scalar_dependence_one_known` **failing** at a win rate of 0.4375 and
`two_known` passing at 0.5938 — a 32-sample batch, where the binomial standard error around 0.5 is
**0.088**. Both readings are inside one SE of chance, so neither said anything. Kernel
`anosvol/metasurface-jepa-192d-scalar-scan` (v1), commit `3eb38a4`, `exit=0`, scan 285 s: both
scalar gates measured on **all 17,488 validation samples**.

### 16.1 The result — scalar dependence is not demonstrated

| stratum | win rate | 95 % CI | mean real / shuffled | median real / shuffled | mean criterion |
|---|---|---|---|---|---|
| one known | **0.5004** | [0.4930, 0.5078] — **includes 0.5** | 0.1185 / 0.1179 | 0.0733 / 0.0734 | True |
| two known | **0.5106** | [0.5032, 0.5180] | 0.1071 / 0.1072 | 0.0866 / 0.0866 | True |

- **`one_known`: 8,751 wins out of 17,488 — a coin flip.** The CI contains 0.5, and the mean error
  is marginally *worse* under true conditioning (0.1185 vs 0.1179).
- **`two_known`: 8,930 wins out of 17,488 — 1.06 % above chance.** The CI excludes 0.5 by a hair,
  and the means differ by **0.0001** (0.1071 / 0.1072). A statistically detectable effect of
  immeasurable size is not "demonstrating scalar usage".
- The two 32-sample readings were **both noise**: z = −0.71 and +0.94 against the full-split rates.
  The gate split them into one pass and one fail purely by sampling.

**So the honest statement is not "one stratum fails" (§15.2) but "scalar dependence is not
demonstrated in either stratum"** — and the win rate's ability to say that plainly, where the mean
criterion reported `True` for both, is the strongest argument for the switch made in `3eb38a4`.
The old statistic called both strata passing on differences of 0.0006 and 0.0001.

### 16.2 What this does and does not establish

- **Established:** the scalar *conditioning* input, as exercised by this test, does not measurably
  change the deployed design's spectrum error. Measured on 17,488 samples, with a CI that excludes
  any effect larger than ~1 %.
- **Not established: that the model ignores scalars.** The test perturbs only the conditioning
  input (`sv_cond`) while the decode/assembly path keeps the TRUE scalar values for the known
  columns (`scalar_values=sv`). With 1 of 3 columns known, the remaining two flow through
  `scalar_pred`; with 2 of 3 known, only one does — which may be exactly why the two-known stratum
  is marginally the more sensitive one. Whether the cause is "the encoder-side conditioning is
  ignored" or "the test's known-column substitution masks it" is **not distinguishable from this
  measurement** and needs the direct instrument used for the spectrum: perturb the scalar
  conditioning and measure the change in `z_hat` / `scalar_pred` / the decoded geometry per stage,
  as `REPORT.md` §10.2 did for the goal.
- The gate as configured at 32 samples **cannot** decide this: at that size the win rate has a
  ±0.176 (95 %) interval, so everything from 0.32 to 0.68 reads as "consistent with chance".

---

## 17. Where the scalar conditioning dies (2026-09-13)

§16 left two hypotheses for the absent scalar dependence: (a) the encoder-side conditioning is
ignored, or (b) it acts but the known-column substitution masks it before the deployed design.
Kernel `anosvol/metasurface-jepa-192d-scalar-effect` (v2), commit `3eb38a4`, `exit=0`: perturbing
the scalar conditioning and the spectrum separately, N = 64, hard stratum, measuring the change at
every stage with the **spectrum as a calibrated control** (known to act: §10.2).

### 17.1 The effect ratio, stage by stage

Relative change from perturbing the **scalars** vs from perturbing the **spectrum**:

| stage | scalar Δ | spectrum Δ | ratio | one / two / all known |
|---|---|---|---|---|
| `scalar_encoder` FiLM params | 0.0150 | — | — | same across strata |
| `scalar_encoder` summary token | 0.1086 | — | — | same across strata |
| `z_hat` (predictor output) | 0.0075 | 0.2360 | **0.032** | 0.032 / 0.031 / 0.039 |
| `scalar_pred` (absolute) | **0.000145** | 0.2457 | **0.0006** | 0.0006 / 0.0006 / 0.0008 |
| occupancy logits | 0.0028 | 0.3820 | 0.007 | 0.007 / 0.007 / 0.008 |
| assembled geometry | 0.0324 | 0.4252 | 0.076 | 0.076 / 0.076 / 0.085 |
| **spectrum error** | **0.000467** | 0.4715 | **0.001** | 0.001 / 0.0005 / 0.002 |

### 17.2 The finding — hypothesis (a), localised

**The conditioning does reach the scalar encoder's output**: perturbing the scalar input moves the
FiLM parameters and moves the summary token by **11 %** relative — not nothing. **It then fails to
propagate.** By `z_hat` the effect is 3 % of the spectrum's, and at `scalar_pred` it is
**0.000145** in absolute terms on scalars whose range is ~2.5–5, i.e. ~0.006 % — indistinguishable
from zero.

So the signal does not die at the substitution (hypothesis b): it is **already gone before the
substitution happens**. The scalar prediction is essentially independent of the scalar conditioning
input, which is exactly what §16 measured downstream and why the gate reads a coin flip.

The asymmetry is the lead: perturbing the **spectrum** moves `scalar_pred` by **0.246** while
perturbing the **scalars** moves it by **0.000145** — a factor of ~1,700. The scalar-query token
responds to the spectrum and ignores the scalar-summary token. **The scalar-summary → predictor
path is effectively dead**, and that is the specific thing to look at; whether the token is
out-competed in attention, whether the fusion ordering buries it, or whether the FiLM path is the
only one that acts at all is not settled by this measurement.

This is a real architectural finding about the shipped model, not a measurement artifact: it is
reproduced identically across one, two and all known scalars, and the control (the spectrum)
behaves as expected throughout.

---

## 18. The spectrum-sensitivity gate now exists, and passes (2026-09-13)

`architecture_v5.md` §8.3: *"Test by perturbing the target spectrum slightly with everything else
fixed and confirming the decoded design changes proportionally, not just that raw latent variance
looks healthy."* This gate **did not exist** — the evaluator's `diversity_check` runs at
`perturbation_scale=0`, which its own docstring describes as a determinism check and explicitly
says must not be presented as generative diversity, and with a positive scale it perturbs the
latent rather than the target spectrum. Recorded as an open gap since §7; implemented in
`cf8e361` as `spectrum_sensitivity_probe()` and wired into `run_all_scenarios` as
`spectrum_sensitivity_A`.

Kernel `anosvol/metasurface-jepa-192d-sensitivity` (v1), commit `cf8e361`, `exit=0`, on the
full-epoch checkpoint. Perturbation of `scale × per-sample-spectrum-std × fixed noise` applied to
the TARGET spectrum, everything else fixed:

| scale | geometry relative change | occupancy pixels flipped | predicted fraction shift |
|---|---|---|---|
| 0.00 | **0.000000** | **0.000000** | 0.000000 |
| 0.01 | 0.061872 | 0.004143 | 0.001924 |
| 0.05 | 0.144901 | 0.023163 | 0.009174 |
| 0.10 | 0.196073 | 0.041840 | 0.017481 |

`design_moves = True`, `pixels_flipped_monotone_in_scale = True`.

- **Scale 0 moves nothing, exactly** — the probe's own sanity check.
- **The design tracks the target monotonically**, and a 1 % perturbation already moves it
  (0.41 % of pixels flip, 6.2 % relative geometry change), rising to 4.2 % of pixels at 10 %.
- So output-diversity collapse across varying conditions — the Failure Mode 2 that a healthy
  latent-space metric would not catch — **is not occurring in the shipped model.** This is the one
  gate the project had never measured, and it is the second positive result after the scenario
  gates.

The gate blocks on the same run are unchanged (`A` 0.96875 pass, `B` 0.9375 pass, `C` 0.90625
pass, scalar one-known 0.4375 fail, two-known 0.59375 pass), so the new probe is additive: nothing
about the earlier readings moved.

**Remaining spec gaps: none.** Every §8.1/§8.3 check now has an implementation and a measurement.
