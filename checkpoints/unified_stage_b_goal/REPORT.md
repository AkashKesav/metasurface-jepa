# Stage-A Run B (full-train + goal margin 0.01) — VERDICT: REJECTED

Run: `akashkesav/metasurface-jepa-stage-a-goal-ab` (70k steps, P100)
Code: validate()-shuf fix + guidance broadcast fix; config
  `configs/unified_stage_b_goal.yaml` (Run-A base + lambda_goal=2.0/margin=0.01).

## Outcome: 70000/70000, all exits 0, skip guard held (10/10/0)

## Gates vs Run A

| gate | bar | Run A | Run B | delta |
| --- | --- | --- | --- | --- |
| raw_mse | info | 0.0202 | **0.0162** | better |
| scale | [0.5,2] | 0.987 | **1.005** | both pass |
| delta_rel | <1 | 0.137 | **0.104** | both pass |
| concentration | >0.01 | 0.00192 | 0.00273 (+42%) | both FAIL |
| sens_norm | >0.01 | 0.00592 | 0.00675 (+14%) | both FAIL |

## Why rejected (decisive, not marginal)

1. **The hinge never bound**: `L_goal = 0.0` at ALL 139 validations. Random-init
   student variation already exceeds margin 0.01, so the term was vacuous the
   whole run — Run B ≈ Run A + noise. The +14% sens delta is not attributable
   to the mechanism.
2. **Val reporting destabilized**: val `L_total` hit 1862-4930 late via `L_cov`
   while train loss stayed healthy (~2). Diagnosis: projector-BatchNorm
   running-stats staleness in eval (train uses batch stats). Cosmetic for the
   model (raw-space eval is BN-free: raw_mse 0.016 proves health; inference
   never touches the projector), but it wrecks val readability and must be
   addressed before any Stage-B work that trusts projected-space eval.
3. Student goal-gap rose 0.011→0.032 on its own (5x the teacher's 0.006):
   the student already differentiates goals more than the near-closed teacher
   gate (tanh=-0.07) requires. Forcing student variation against a dead target
   cannot raise teacher sensitivity — predicts the Run-C design (bind the
   hinge above the noise floor so co-adaptation via L_raw can drag the gate).
