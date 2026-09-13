# Unified 192-D JEPA — cloud run report (Kaggle)

**CURRENT STATUS: NEGATIVE RESULT — the acceptance gate FAILS on the hard stratum.**
The full 1500-step run completed and was evaluated per scenario; on scenario A (pure inverse
design = full occupancy mask + all scalars unknown) the model is **no better with the true
spectrum than with a shuffled one** (`real 0.6336` vs `shuffled 0.6332`, gate criterion
`real < shuffled` → **false**), the decoded design is **deterministic under a perturbed target
spectrum** (`diversity_A = 0.0`), scalar conditioning is **exactly zero** in the one-known
stratum (`0.6261729598045349` identical for real and shuffled), and the trivial L1
nearest-neighbour retrieval baseline is **~8.5× better** than the model (`0.0744` vs `0.6336`).
Full numbers in §4. The pipeline itself is verified (§3).

**This is recorded, not acted on.** Per `AGENTS.md` → *If something fails*, the response to a
failed gate is to record the observed numbers and escalate for a scope decision — never to add
mechanisms or loosen a threshold to make it pass. See §4.3 for the decision point.

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

- **Generative diversity — fails outright.** `diversity_A`: `pairwise_spectrum_diversity = 0.0`,
  `deterministic = true`. Perturbing the target spectrum does not move the decoded design at all.
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
