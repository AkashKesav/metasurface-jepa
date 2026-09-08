# Audit — Previous Agent's Work (2026-09-08, session cut off ~22:40)

Scope: full working-tree audit after the previous coding-agent session died mid-work.
Evidence base: git status/log/diff, `.workbuddy-ai/memory/2026-09-08.md`, direct source
verification of every fix, targeted test run (79/79 unified tests pass).

---

## 1. Repo context in one paragraph

This repo implements the Goal-Conditioned Physics JEPA project (milestones A–I per
`AGENTS.md`). The AGENTS milestone structure describes the original 384-d
`GoalConditionedJEPA` (Milestone B line), but the **active code line since late August is
the 192-d `UnifiedJEPA`** (`scripts/train/train_unified.py`, `configs/unified.yaml`,
`src/assembly.py`) — occupancy + 3 explicit scalars, FiLM conditioning, two EMA targets.
The two generations coexist; `src/train/engine.py` is still Milestone-B-shaped. All real
training happens on Kaggle; results land in ~57 untracked `.kaggle_*` scratch dirs
(`checkpoints/` itself contains only `.md` reports, zero `.pt` files).

## 2. What the previous agent did today (2026-09-08) — four sessions, last one cut off

### Session 1 — read-only full codebase review (evening)
Recorded in memory. Key verified findings:
- All trained checkpoints live outside version control (~57 `.kaggle_*` dirs).
- Dead/latent bugs: `src/reference/direct_masked_generator.py:114` (5-vs-6 unpack,
  guaranteed ValueError, no importers), `src/runtime/reproducibility.py:67` (`fork_rng`
  never seeds — silent no-op), `goal_mode="shuffled"` advertised but silently behaves as
  `"real"` in `SpectrumPath.forward` and is rejected by `validate_goal_mode`,
  `scripts/preflight/repo_static_audit.py` fails (67 findings) because its allowlist
  predates `train_unified.py`.
- Test suite at that time: 535 passed / 14 skipped.
- Doc drift: `design_doc.md` and `AGENTS.md` describe the dead 384-d architecture; "192"
  appears zero times in either.

### Session 2 — goal-fix plan retired, Joint Target Redesign adopted (22:09–22:14)
Operator decision, executed as:
- `docs/UNIFIED_ARCHITECTURE_FIX_GOAL.md` — DISABLED/SUPERSEDED banner added
  (**uncommitted**).
- `docs/JOINT_TARGET_REDESIGN.md` — NEW 462-line active spec, **untracked** (not yet
  committed). Zero code written against it.
- `AGENTS.md` — dated Standing-Rule-9 operator override recorded (**uncommitted**).
- Verified the disabled code paths (`direct_goal_route` flag, `--lambda-goal`/
  `--goal-margin` in stage4) are already off by default → no code change needed for the
  retirement itself.

### Session 3 — training-path audit, "model trains too quickly" (22:20–22:31)
Root-caused the suspicious fast convergence. Four bugs found:
- **T1 (critical):** `stage4_goal_utility.py` training loop never calls
  `objective.on_optimizer_step` → **both EMA target encoders frozen for the whole run**
  in every September kernel (all September kernels run stage4, not train_unified).
- **T2 (critical):** stage4 loads MetaDiT weights into the student only, after
  build-time EMA resync → EMA target is a **random-init copy** (violates ξ(0)=θ(0)).
- **T3 (moderate):** `EMAEncoder.total_steps` defaults to 1 → momentum pinned at 0.999
  from step 1; the documented 0.996→0.999 ramp never happens (affects train_unified too).
- **T4 (design):** projector has 2×BatchNorm1d → train loss computed in per-batch
  standardized space, val loss in running-stats space → 17× train/val gap; VICReg
  anti-collapse term trivially satisfied in train mode.
- Gauge bug: `pred_eff_rank_frac` computed on a (B=2, 192) matrix → only 2 singular
  values → mathematically floored at exactly 0.500000, dead gauge for the entire run.
- Hard evidence from `.kaggle_sweep_final/results/lambda_0.2/REPORT.json`:
  `raw_cos_err` ≈ 1.0 the whole run (predicted latent orthogonal to target), projected
  error drops only because the projector does all the alignment.
- **Verdict: every September goal-conditioning result (sweeps, goal-margin v1–v4,
  direct-goal v1/v2, matched-weight run, the v2 "PASS") trained under T1+T2 and is
  scientifically invalid as evidence.**

### Session 4 — fixes started, NOT recorded, NOT committed (≈22:31 → death)
The uncommitted working-tree diffs are the beginning of the fix work. The agent died
before writing a memory note, committing, running the full suite, or regenerating kernels.

## 3. Exact uncommitted changes (all verified against source)

| File | Change | Status |
|---|---|---|
| `scripts/eval/stage4_goal_utility.py` (+16) | **T1 fix**: `objective.on_optimizer_step(model, step)` after `optimizer.step()` (mirrors `train_unified.py:815`). **T2 fix**: resync `model.ema.target` + `model.scalar_mlp_ema.target` after `_init_geometry_from_metadit` (mirrors `build_unified_model` L707-708). **T3 fix**: `model.set_total_steps(args.total_steps)`. | Verified: API names exist, signatures match, py_compile OK |
| `scripts/train/train_unified.py` (+5) | **T3 fix**: `model.set_total_steps(total_steps)` before training loop. | Verified: `total_steps` in scope at L537/575 |
| `src/diagnostics/representation_health.py` (+9) | Docstring **CONTRACT note only** for the eff_ranks floor bug. The caller (`token_space_stats` L158, still `eff_ranks(X.mean(dim=1))`) is **NOT fixed** — gauge remains dead at batch 2. | Partial — needs caller fix |
| `AGENTS.md` (+20) | 2026-09-08 operator override entry (Standing Rule 9). | Matches redesign adoption |
| `docs/UNIFIED_ARCHITECTURE_FIX_GOAL.md` (+25) | SUPERSEDED banner, plan disabled for provenance. | Complete as intended |
| `docs/JOINT_TARGET_REDESIGN.md` (new, untracked) | The go-forward spec (see §4 below). | Complete as intended |
| `.commandcode/taste/taste.md` | Agent-framework preference auto-update. | Not project code |

Test status with fixes in place: `tests/test_unified_losses.py`,
`test_unified_model_phase2.py`, `test_unified_data_contract.py` → **79/79 pass**.

Not fixed / not done when the agent died:
- T4 (projector BatchNorm) — untouched; needs an operator design decision.
- eff_ranks production caller — still computes on batch-mean; gauge still dead at B=2.
- No commit, no memory note, no full-suite run, no kernel regeneration.
- v3 direct-goal kernel (pushed 09-07 12:26): local results dirs are **empty** — results
  were never pulled; remote status unknown (kaggle CLI broken under Python 3.14 in this
  environment; WSL blocked by sandbox — check kaggle.com manually). Given T1+T2, v3's
  results would be invalid anyway.

## 4. What is supposed to happen — the adopted go-forward plan

`docs/JOINT_TARGET_REDESIGN.md` (ACTIVE, adopted 2026-09-08, no code written yet):

- **Core change:** JEPA target becomes `Z_joint = J(Z_G, Z_S)` — geometry↔spectrum
  cross-attention with a zero-initialized tanh gate (`joint = geo + tanh(gate)·delta`),
  instead of a geometry-only EMA target. Rationale: a geometry-only target can only
  *reward* spectrum use, never *require* it — the structural root of four failed
  goal-conditioning attempts.
- **Stage schedule (§9):** A joint-target-only → B decoder → C physics ramp →
  D conditionality (real/null/shuffled) → E hard masking → F zero-context →
  G stochasticity. One stage at a time.
- **Do-not-add-yet list (§13):** sparse routing, reciprocal attention, stochastic
  latent, flow/diffusion, aux embedding loss, frequency weighting, spectrum-encoder
  fine-tuning — each requires a measured failure first.
- **Implementation order (§14):** `JointTargetFusion` → `MaskedQueryPredictor` →
  shape/gradient tests → JEPA-only training → decoder supervision → physics validation →
  surrogate loop → hard-stratum eval → goal-sensitivity → physics sweep →
  raw-vs-projected ablation.
- **Anti-cheating rules (§10):** student never sees true masked geometry / teacher
  latent / surrogate output on true geometry; decoder gets no raw spectrum tokens;
  occupancy stream stays occupancy-only (no 3-channel broadcast — leaks scalars);
  spectrum encoder stays frozen.

## 5. Pending operator decisions (explicitly left open by the previous agent)

1. Delete vs. keep-inert the disabled goal-route code (`src/predictor/goal_residual.py`,
   assembly wiring, `configs/unified.yaml:25`, `tests/test_goal_residual_route.py`,
   stage4 ranking flags) — currently off by default.
2. The ~29 goal-related `.kaggle_*` scratch dirs (and root junk
   `_fetch_log*.py`, `_splice_kernel.py`, `poll_v4.py`, `_s4*.bin`, `fix_indent*.py`)
   — clean up or keep. Nothing will be deleted without explicit confirmation.
3. T4 remedy: projector BatchNorm (remove BN / use eval-mode stats / LayerNorm) —
   changes loss semantics, needs a decision.
4. eff_ranks caller fix: pass token-level `z[mask]` (N_tokens, D) instead of
   batch-mean (B, D).
5. v3 kernel: pull or abandon (invalid under T1+T2 regardless).

## 6. Recommended next steps (cheapest first)

1. Commit today's work in two commits: (a) redesign adoption
   (AGENTS.md + FIX_GOAL banner + JOINT_TARGET_REDESIGN.md), (b) the EMA bug fixes
   (stage4 + train_unified + representation_health contract note).
2. Fix the eff_ranks caller (small, mechanical).
3. Decide items in §5 (operator calls).
4. Start Joint Target Redesign Stage A per §14 order — `JointTargetFusion` module +
   forward-only shape/gradient tests, local dev-only (no cloud spend until the stage-A
   gate exists).
5. Do NOT re-run any stage4/goal kernel before the fixes are committed and the redesign
   Stage-A code exists — every such run would burn quota on the broken EMA loop.
