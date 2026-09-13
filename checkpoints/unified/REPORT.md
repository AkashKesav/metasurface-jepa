# Unified 192-D JEPA — cloud run report (Kaggle)

**CURRENT STATUS: PIPELINE VERIFIED — verification-scale run PASSED on real data + released
weights + GPU. The full run and the acceptance gate are recorded in §4 when they land. No
scientific claim is made by this report.** The gate is the per-scenario hard-stratum
real-vs-shuffled physics-consistency gap (`architecture_v5.md` §8.3 check 8), which requires a
completed training run and is reported per scenario, never pooled.

This file is the CLOUD_TRAINING.md §3 sync-back record for the first unified 192-D cloud session
(2026-09-13).

---

## 1. What ran

| | |
|---|---|
| Platform | Kaggle (kernel, private), Tesla **P100-PCIE-16GB**, Internet ON |
| Code package | Kaggle dataset `anosvol/metasurface-jepa-192d-unified-code` v3 |
| Commit | `4deab8a44c9e5011c7ab95f350109851abea4e88` (branch `work-192d`) |
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

## 2. Defects the first cloud run surfaced (both fixed before the passing run)

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
   the guard without updating its one existing legitimate caller, and nothing caught it locally
   because the preflight needs the real splits and released weights, which the dev machine does
   not stage. Fix (`4deab8a`): predicate is `not self.training or not torch.is_grad_enabled()`
   (a gradient-enabled training forward is still refused), plus a `no_grad` wrapper on the
   preflight's diagnostic block, with regression tests for both directions.

Recorded in `docs/implementation/unified_jepa/AUDIT_REPORT_192D.md` (rows for the guard) —
B21 is in §2 of that report.

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

## 4. Full run + acceptance gate

Recorded here once the `anosvol/metasurface-jepa-192d-full-run` kernel completes:

- full run: `train_unified.py --config configs/unified.yaml --device cuda` (1500 steps)
- gate: `eval_scenarios.py --scenario all --device cuda` — read **A / B / C separately**, and
  the **hard stratum** real-vs-shuffled physics-consistency gap; never a pooled number
- `run_guidance_gap_sweep.py` mask-ratio curve (§20.3)

---

## 5. Honest verification status

- **Verified:** the real-data pipeline runs end to end on a cloud GPU — data staging, released
  weights, frozen-component gradient ownership, 150 training steps, EMA updates, stratified
  validation, calibrated masking, checkpoint write/read path.
- **Not verified:** anything scientific. No claim is made that the representation is useful, that
  physics consistency beats the shuffled control, that scalars are used *correctly*, or that the
  decoder avoids collapse. Those are the §8 gates and they need the full run plus
  `eval_scenarios.py`. A completed run is not a result (AGENTS.md Standing Rule 8).
- **Local gate for the same commit:** `python -m pytest tests/ -q --tb=line` → **275 passed,
  22 skipped, 0 failed**; `scripts/preflight/repo_static_audit.py` → 0 findings.

## 6. Resume / repo hygiene

- Checkpoint files are **not** committed (`checkpoints/**/*.pt` is gitignored); only this report
  is tracked.
- To reproduce or continue: dataset `anosvol/metasurface-jepa-192d-unified-code` (v3) +
  `anosvol/metadit-aaai2026-staging`, kernel `anosvol/metasurface-jepa-192d-full-run`.
  `--resume checkpoints/unified/latest.pt` works if a future session re-attaches the run's
  output directory; a fresh full run is the clean default.
- **Deviation to be aware of:** the code package is pinned at `4deab8a`, while the branch head is
  `191e5fc`. The difference is documentation only (`AUDIT_REPORT_192D.md` B21 row + test count);
  no code differs. Future packages should be rebuilt from head to keep the pin exact.
