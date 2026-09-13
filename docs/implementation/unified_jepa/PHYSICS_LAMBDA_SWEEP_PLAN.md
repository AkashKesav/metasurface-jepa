# λ_phys sweep — plan (2026-09-13)

Status: **RUNNING** — kernel `anosvol/metasurface-jepa-192d-lambda-sweep` (v1), commit `3bd472a`.

**Stage-0 gate: PASSED.** The goal-content probe on the physics-on checkpoint found the content
signal alive at every stage (`c_physics` content ratio 1.320; after the linear projections
1.141 / 1.010; `z_hat` 0.142 vs 0.100 presence; `occupancy_logits` 0.341 vs 0.253; 16.9 % of
pixels flipped; real 0.2989 vs shuffled 0.5561), so the sweep is the right instrument —
the "presence switch" characterisation came from the physics-OFF run and does not hold with
physics on (`checkpoints/unified/REPORT.md` §10.2).

Two amendments made before starting, both recorded in commits rather than here:

- **Audit B27** (`3bd472a`) gave evaluation its own sample size (`eval.n_samples`, default 32,
  CLI `--samples`, per-sample paired gate statistics). The gate had been decided on
  `train.batch_size` = **2** samples; every arm would otherwise be scored by the same coin flip.
- The `λ` grid is computed by the calibration pass at the head of the sweep kernel (§4), not
  guessed.

## 1. The question the sweep answers

Does the physics term make the spectrum conditioning **content-sensitive** — a target the design
is fitted to — rather than a mere **presence switch**?

That distinction is what the 20k-step physics-off run measured (`checkpoints/unified/REPORT.md`
§9.3): removing the goal entirely costs a lot (CFG `w=0` → 0.7302 vs `w=1` → 0.2046), but
replacing the true spectrum with another sample's costs nothing (scenario A real `0.2046` vs
shuffled `0.1889`, gate **false**). The model uses the *existence* of a goal, not its *content* —
and every learning curve plateaus at ~step 5,000, so more steps of the same objective do not fix
it. `L_phys` is the only term that supervises the decoded geometry against the true spectrum, so
it is the designed lever for exactly this failure.

## 2. Stage 0 — the diagnosis that gates the sweep (running)

`goal_probe.json` from the physics-on kernel measures, at every stage of the forward path, the
PRESENCE effect (real vs null) against the CONTENT effect (real vs a deranged spectrum):

| stage | what a ~0 content ratio there would mean |
|---|---|
| `c_physics` / `a_goal` (frozen spectrum encoder) | the input never separates spectra — **no λ can help; the sweep is void** |
| `c_phys_proj` / `goal_proj` (linear projections) | the projection discards the content |
| `z_hat` (predictor) | the predictor ignores it |
| `occupancy_logits` / `binary_occupancy` (decoder) | the decoder or the 0.5 threshold erases it |
| spectrum error (surrogate) | nothing survives to the deployed design |

**Gate on this before spending sweep compute.** If the content ratio is already ~0 at
`c_physics`, the sweep is the wrong instrument and the finding must be escalated (the frozen
encoder or its read-out is the bottleneck), not swept over.

## 3. Harness

One kernel, arms run sequentially, each arm trained **from scratch** (fresh model, same seed) so
λ is the only variable:

- same config, same `--max-steps 10000` (the plateau is reached by ~5,000, so 10k is comfortably
  inside the regime that matters), same evaluation batch and seeds, same instruments per arm.
- per-arm results are written to `/kaggle/working/results/` **immediately**, so an arm that
  finishes is safe even if a later arm dies with the session.
- the existing `scripts/diagnostics/protocol_v1/step7_lambda_sweep.py` is a **local 100-step CPU
  tool on 16 synthetic samples**; it measures gradient scale well and is reused for the
  calibration in §4, but it cannot answer §1 and is not the harness for this sweep.

Instruments per arm (all already built and committed):

1. `eval_scenarios.py --scenario all` — the A/B/C gates and `cfg_guidance_sweep_A` (B26)
2. the goal-content probe — presence vs content at every stage (§2)
3. `collapse_check` — predicted occupancy fraction mean and std
4. the EMA target↔student distance probe
5. physics gradient norms per module (decoder / encoder / predictor / predictor blocks)
6. `L_phys` trajectory over the run, with the ramp made visible

## 4. Choosing the λ grid — calibrate, do not guess

The protocol's nominal range is `{0.01, 0.1, 1, 10}`, but loss *magnitudes* in this objective do
not bracket it usefully: at the 20k plateau the weighted terms are `λ_cov·L_cov ≈ 146` and
`λ_inv·L_inv ≈ 104`, against an unweighted `L_phys ≈ 0.34`. On magnitude alone `λ = 0.1`
contributes ~0.03 %, which is very likely inert — exactly the silent-no-op shape this project has
been eliminating.

So: **one calibration pass, no training.** Port the `step7` gradient instrument to the cloud, run
it at step 0 on a real batch, and record `‖∇_θ L_phys‖` and `‖∇_θ L_other‖` for
θ ∈ {occupancy_decoder, occupancy_encoder, predictor, fusion_encoder}. Then set the grid so that
λ·‖∇L_phys‖ covers a **1 % / 10 % / 50 %** share of the total gradient norm per module.

Arms, in this order (so partial results stay interpretable):

| arm | λ | why |
|---|---|---|
| 1 | `0` | in-harness control — no surrogate in the loop, cheapest, and the baseline every other arm is compared against |
| 2 | `0.1` | the value currently committed (activation commit `f3f1244`); must be tested as written |
| 3 | λ(10 % gradient share) | the first value with a plausible chance of mattering |
| 4 | λ(50 % gradient share) | upper end — does more physics help, or destabilise? |
| 5 | λ(1 % gradient share) | only if arms 1–4 finish with session time left |

## 5. Pre-registered decision rule

Written down **before** the numbers arrive, so the outcome cannot be reinterpreted afterwards.

- **Primary (the gate):** scenario A, hard stratum, `real < shuffled`
  (`architecture_v5.md` §8.3 check 8). Nothing pooled; B and C reported alongside but not
  substituted for A.
- **Secondary (the mechanism):** the content ratio at `occupancy_logits` and `z_hat` must rise
  clearly above the λ=0 arm (which sits at ~0.01–0.02 relative). A gate pass with the mechanism
  still flat would be coincidence, not understanding.
- **Tertiary (sanity):** the input content ratio at `c_physics` must be non-trivial (§2).
- **Non-degradation:** `L_var` must not collapse and `collapse_check`'s predicted occupancy-fraction
  std must stay ≳ the λ=0 arm's; the EMA distance must stay ≲ 0.01.

Outcomes and what each means:

1. **≥1 λ passes the primary gate and the mechanism moves** → report the *smallest* passing λ.
   Adopting it is a separate one-line commit.
2. **Mechanism moves monotonically with λ but the gate stays red** → physics works but the
   objective is unbalanced (weights, or the decoder's 0.5 threshold, or the plateau is an
   optimisation limit). Escalate with the curve; do not keep turning the knob.
3. **Nothing moves at any λ, including 50 % share** → physics is not the lever for this failure.
   Stop the sweep and escalate; the §2 stage table says where to look next.
4. **Physics destabilises** (`L_cov` runaway, occupancy collapse, divergence) → record the λ
   where it starts and treat it as the ceiling.

## 6. Guardrails (AGENTS.md)

- One variable per arm: λ only. No mechanism added, no threshold moved, no gate relaxed.
- The gate is reported per scenario and never pooled, and a red gate is recorded as red.
- Every arm's numbers are kept, including failures — a sweep that reports only its best arm is
  not a sweep.
- Arms are separate from the production config: adopting a value is its own commit with the
  evidence attached.

## 7. Cost and order of operations

- per arm: 10,000 steps. Physics-off was 1,846 s; with the surrogate in the loop expect
  ~1.5–3× that, so ~45–90 min per physics arm, ~15 min for the control.
- 4 arms ≈ 2.5–5 h GPU, comfortably inside one Kaggle session's soft cap
  (`SOFT_TIME_LIMIT_S` = 10.5 h, run budget 65 % of it), plus the fixed ~4 min torch pin install.
- order: Stage 0 → §5 gate → calibration (§4) → arms 1–4 → record in
  `checkpoints/unified/REPORT.md` with the pre-registered rule alongside the outcome.

## 8. What is explicitly *not* in this plan

- No changes to the objective's terms or topology (the projector-absorption hypothesis in
  `checkpoints/unified/REPORT.md` §7.3 is a separate question with its own evidence requirement).
- No new loss terms, schedules, or curricula.
- No redefinition of the gate, the strata, or the metric definitions.
