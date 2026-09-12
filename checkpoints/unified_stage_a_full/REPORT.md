# Stage-A Run A (full-train, no goal) — REPORT

Run: `akashkesav/metasurface-jepa-stage-a-full-train` (70k steps, P100)
Code: `4ea3b30` (skip-guard) + kernel hardening; config `configs/unified_stage_a_full.yaml`
  (lambda_raw=3.0, lambda_var=5.0, lambda_cov=1.0, lambda_goal=0.0).
Data: full train set, 139906 samples, batch 2 → 70000 steps ≈ 1.0 epoch.

## Outcome: 70000/70000 steps, exit via manifest (COMPLETE)

- `final_step=69999`, `n_val_points=139`, `wall_clock=3652s` (61 min),
  `final_train_loss=1.972` (from 20.6).
- Skip guard fired as designed: 10 train batches + 0 val skipped (degenerate
  h/r samples that killed the previous attempt at step ~6900). No crash.

## Gates (VERDICT §8.4, offline eval on real val, 2x2, mask 0.5)

| gate | bar | Run A | verdict |
| --- | --- | --- | --- |
| raw_mse | info | 0.0202 (was 5.35) | — |
| raw_cos_err | info | 0.0073 (was 0.019) | direction solved |
| scale_ratio | [0.5, 2] | **0.987** (was 0.364) | PASS |
| delta_rel | < 1 | **0.137** (was 0.092) | PASS |
| concentration | > 0.01 | 0.00192 (was 0.00112) | FAIL |
| sens_normalized | > 0.01 | 0.00592 (was 8.4e-05) | FAIL |

## Reading

1. Rebalance + full data worked: scale 0.36→0.99, raw_mse 5.35→0.02.
   Direction AND magnitude now match.
2. Sensitivity improved 70x (8.4e-05→0.0059) from data scale alone but stays
   ~1.7x short of the gate. This is the measured failure motivating Run B.
3. Concentration gate is suspect as formulated: the predictor scores 0.0019
   vs 0.0018 AT INIT — it never diversified because it faithfully matches a
   target whose own concentration fell during training (0.021→0.0059).
   Punishing the student for matching the teacher measures the wrong thing.
   NOT chased tonight (would mean optimizing a gauge, not physics); flagged
   for operator recalibration of the gate, not the model.
4. Projected space aligned anyway (proj_cos 0.0116) with lambda_inv=0.

## Decision (pre-registered A/B rule)

Run A evaluated → launch Run B (identical + lambda_goal=2.0/margin=0.01).
Keep the goal term ONLY if sens_normalized passes 0.01 with no regression on
scale/concentration/delta/raw_mse vs this report.
