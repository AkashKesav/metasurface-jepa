# Unified 192-D JEPA — cloud run report (Kaggle)

**CURRENT STATUS: NEGATIVE RESULT — the acceptance gate FAILS on the hard stratum.**
The full 1500-step run completed and was evaluated per scenario; on scenario A (pure inverse
design = full occupancy mask + all scalars unknown) the model is **no better with the true
spectrum than with a shuffled one** (`real 0.6336` vs `shuffled 0.6332`, gate criterion
`real < shuffled` → **false**), scalar conditioning shows **no measurable effect** (the one-known
stratum's real and shuffled errors are bit-identical), and the trivial L1 nearest-neighbour
retrieval baseline is **~8.5× better** than the model (`0.0744` vs `0.6336`). Full numbers in §4.
The pipeline itself is verified (§3).

> **Correction (recorded rather than silently edited).** An earlier revision of this file listed
> "generative diversity = 0.0" as evidence against the model. That was wrong: the evaluator's
> `diversity_check` runs with its default `perturbation_scale=0.0`, which is a **determinism**
> check (same input → identical output), and its own docstring says the result "must not be
> presented as genuine generative diversity". The spec's actual probe — *perturb the target
> spectrum slightly and confirm the decoded design moves proportionally*
> (`architecture_v5.md` §8.3) — is **not implemented** (see §7). The negative result rests on the
> real-vs-shuffled gate, the scalar-dependence result, the retrieval baseline and the collapse
> check, not on the determinism check.

**This is recorded, not acted on.** Per `AGENTS.md` → *If something fails*, the response to a
failed gate is to record the observed numbers and escalate for a scope decision — never to add
mechanisms or loosen a threshold to make it pass. See §4.3.

The gate is the per-scenario hard-stratum real-vs-shuffled physics-consistency gap
(`architecture_v5.md` §8.3 check 8), reported per scenario and never pooled.

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
(§7.3) is only *part* of the story — the representation improved and the decoder un-collapsed, so
absorption is not total.


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
