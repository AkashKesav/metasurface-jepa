# Unified JEPA Architecture-Fix Goal

## Objective

Make the unified model reliably target-conditioned for inverse design.

The repaired baseline now responds to shuffled spectrum goals, but its null-goal
result is slightly better than its real-goal result. The current system is
therefore spectrum-responsive but not yet reliably spectrum-useful.

Final matched-weight evidence:

```text
physics(real)      = 0.492732
physics(null)      = 0.485704
physics(shuffled)  = 0.518057
geometry sensitivity(real/shuffled) = 0.0111547
```

The required production direction is:

```text
physics(real) < physics(null)
physics(real) < physics(shuffled)
```

## Audit conclusion

The main problem is an objective/architecture conflict:

```text
JEPA target = geometry-only EMA latent
student     = geometry + requested spectrum latent
```

The dominant JEPA/VICReg objective therefore encourages the predicted latent to
remain spectrum-independent, while the physics objective expects the decoded
geometry to change with the requested spectrum. The observed raw/projected
alignment confirms projector absorption:

```text
raw latent cosine       = -0.06618
projected latent cosine =  0.997437
```

The goal tokens are also diluted inside a 273-token fusion sequence, and the
global physics condition is formed by mean-pooling all spectrum locations.

## Controlled architecture experiment

Keep unchanged:

- released MetaDiT spectrum encoder;
- occupancy factorization and block masking;
- EMA target encoder;
- frozen MetaDiT surrogate;
- scalar bounds and geometry assembly;
- dataset, seed, validation IDs, and checkpoint protocol;
- selected `lambda_phys=0.2` baseline configuration.

Add one controlled goal route:

```text
masked geometry -> base predictor -> z_base
spectrum       -> direct masked-query cross-attention -> z_goal
z_final = z_base + alpha * z_goal
z_final -> geometry decoder -> surrogate
```

Apply JEPA/VICReg supervision primarily to `z_base`. Apply physics and goal
utility supervision to `z_final`. The goal route must reach masked occupancy
queries directly rather than relying only on attention over the fused
256-occupancy + 16-goal + 1-scalar sequence.

The gate `alpha` must be explicit and logged. It may be initialized small, but
it must remain trainable and its value/statistics must be reported.

## Required comparisons

Run the same hard stratum for:

1. current repaired architecture;
2. direct goal-route architecture;
3. real goal;
4. null goal;
5. shuffled goal.

Report:

- real/null/shuffled physics error;
- real-vs-null and real-vs-shuffled geometry sensitivity;
- occupancy IoU/F1 and occupancy fractions;
- scalar MAE and out-of-range fraction;
- raw `z_base`, raw `z_final`, and projected alignment;
- `c_physics` and `a_goal` variation;
- goal-route gradient norms and alpha statistics;
- per-loss raw/weighted values and gradient norms;
- checkpoint manifest and SHA-256.

## Acceptance gates

The architecture fix is useful only if, on the same hard-stratum protocol:

```text
physics(real) < physics(null)
physics(real) < physics(shuffled)
geometry(real) differs from geometry(null/shuffled)
```

It must also preserve non-degenerate occupancy reconstruction, valid scalar
ranges, frozen-surrogate/EMA gradient ownership, and recoverable checkpoints.

Do not replace the released spectrum encoder unless the direct goal-route
experiment fails and diagnostics separately demonstrate that the released
encoder representation is the bottleneck.

## Execution order

### Stage A — Local implementation and tests

- add the smallest direct masked-query goal route;
- keep the old route available behind a configuration switch;
- add shape, gradient, null/shuffle, and checkpoint-resume tests;
- run the focused local test suite only.

### Stage B — Cloud ablation

- use the existing WSL Kaggle workflow;
- run baseline and architecture-fix configurations with identical data and seed;
- persist JSON metrics and checkpoint shards;
- verify terminal kernel status and checkpoint hash.

### Stage C — Decision

- if the real goal beats both null and shuffled, update the production architecture
  report;
- if it fails, inspect goal-route gradients, alpha saturation, and raw latent
  alignment before considering a second architectural change;
- do not increase model size or replace the released encoder before this decision.

## Current status

- P0 objective and validation fixes: complete.
- Physics-weight sweep: complete.
- Raw/projected latent diagnostic: complete.
- Matched `lambda_phys=0.2` goal-margin run: complete and checkpoint verified.
- Current matched run: no cloud job is running now.
- Next active work: implement and test the controlled direct goal-route ablation.

