# Unified JEPA final-fix audit — interim report

This report records the repaired-objective evidence collected from Kaggle. It is
not a production-pass declaration: the goal-utility gate and durable checkpoint
recovery are still under evaluation.

## Code and tests

- P0 occupancy BCE reconstruction is active as `lambda_occ`, masked over unknown
  pixels with full-image IoU/F1 diagnostics.
- Physics evaluation is controlled by `compute_physics`, so eval-mode validation
  can execute the frozen surrogate.
- Scalar predictions use configured physical bounds.
- Joint regime counts, raw/projected latent diagnostics, scalar range diagnostics,
  and per-term reporting are wired into the unified trainer.
- Focused repaired-objective tests: `69 passed, 4 skipped`.
- Real-data strictness tests: `17 passed, 1 skipped`.
- Full suite: `523 passed, 23 skipped, 1 unrelated Barlow collapse failure`.

## Repaired fair comparison

Kaggle kernel: `akashkesav/metasurface-jepa-repaired-baseline-physics`.
The run used the same branch/config/dataset/seed and differed only in
`lambda_phys` (`0.0` versus `0.8`). Both used the hard stratum: 100% occupancy
mask and all scalars unknown.

| metric | baseline | physics-fixed |
|---|---:|---:|
| real physics error | 0.551891 | 0.473670 |
| shuffled physics error | 0.549029 | 0.473389 |
| occupancy IoU | 0.715008 | 0.674600 |
| occupancy F1 | 0.832058 | 0.804403 |
| real-vs-shuffled geometry sensitivity | 0.024161 | 0.001522 |
| scalar normalized MAE | 0.230782 | 0.242866 |
| out-of-range fraction | 0 | 0 |

The physics term improves average physics error, but the physics-fixed model
does not pass spectrum identity utility: real error is slightly worse than
shuffled error.

## Repaired Stage-4 utility run

Kaggle kernel: `akashkesav/metasurface-jepa-stage-4-repaired-cuda-v3`, 1500
steps, `lambda_phys=1.0`, seed 42. Final hard-stratum values:

- real/null/shuffled physics error: `0.469024 / 0.468380 / 0.467548`;
- real-vs-shuffled geometry sensitivity: `0.001444`;
- occupancy IoU/F1: `0.661534 / 0.794799`;
- scalar normalized MAE: `0.246086`;
- scalar out-of-range fraction: `0`;
- `c_physics` and `a_goal` cross-sample standard deviations: `0.090918` and
  `0.061819`.

Gate D: **FAIL** (`physics_real` is not below `physics_shuffled`).

## Running follow-up experiments

- `akashkesav/metasurface-jepa-goal-margin-experiment-v1`: exploratory
  real-vs-deranged goal-margin loss, `lambda_goal=2.0`, margin `0.01`.
- `akashkesav/metasurface-jepa-repaired-physics-sweep-v1`: repaired
  `lambda_phys` sweep over `{0.1, 0.2, 0.4, 0.8}`.

The goal-margin run completed with `lambda_phys=0.8`, exploratory
`lambda_goal=2.0`, margin `0.01`, and seed 42. Final hard-stratum values:

- real/null/shuffled physics error: `0.462472 / 0.474009 / 0.483200`;
- real-vs-shuffled geometry sensitivity: `0.006236`;
- occupancy IoU/F1: `0.661449 / 0.795012`;
- scalar normalized MAE: `0.239241`, out-of-range fraction `0`;
- `c_physics`/`a_goal` cross-sample standard deviations: `0.093942` / `0.115311`.

This exploratory result passes Gate D (`physics_real < physics_shuffled`) and
shows target-dependent decoded geometry. It is not yet a production selection:
the lambda sweep and recoverable checkpoint artifact remain outstanding.

These must finish and produce recoverable artifacts before selecting a
production weight or declaring the spectrum-conditioned gate passed.

The goal-margin v2 rerun produced a five-part checkpoint whose reconstructed
SHA-256 matched its manifest (`685d81e8bc096ee5576b72445484f47ea69fa1578d59b0ad7686d28055ec0fb3`).
The selected-λ rerun likewise matched its manifest
(`4b99b5f152f0db6d74797d3f5e41f649e432d3028d0fa265322e7c711569b7d3`).

The repaired sweep completed from commit `c32a7cf`. Final hard-stratum results:

| `lambda_phys` | real physics | shuffled physics | geometry sensitivity | occupancy IoU |
|---:|---:|---:|---:|---:|
| 0.1 | 0.499490 | 0.504335 | 0.012037 | 0.696715 |
| 0.2 | 0.469160 | 0.478342 | 0.013855 | 0.690452 |
| 0.4 | 0.495172 | 0.495261 | 0.001727 | 0.688401 |
| 0.8 | 0.471492 | 0.470995 | 0.001846 | 0.647405 |

`lambda_phys=0.2` is the selected exploratory point because it has the lowest
real physics error and passes real-vs-shuffled. A same-setting rerun from the
diagnostic/checkpoint-sharding commit is active as
`akashkesav/metasurface-jepa-repaired-selected-lambda-v1`.

The selected-λ rerun completed from commit `2605572` with a durable checkpoint
manifest. Final step-1500 metrics were real/null/shuffled physics
`0.476237 / 0.485890 / 0.481992`, occupancy IoU/F1 `0.697957 / 0.820264`,
and real-vs-shuffled geometry sensitivity `0.010273`. Scalar normalized MAE
was `0.236129` with zero out-of-range fraction. Goal-path gradient norm mean
was `0.1310` (maximum `36.1642`).

The raw/projected diagnostic was consequential: raw latent MSE/cosine were
`6.0598 / -0.0343`, while projected MSE/cosine were `0.00436 / 0.99724`.
This is the measured basis for the raw-alignment ablation now running at
`lambda_raw=0.1`.

The raw-alignment ablation completed with `lambda_phys=0.2` and
`lambda_raw=0.1`. Final hard-stratum values were physics error `0.481385`,
occupancy IoU/F1 `0.690780 / 0.815304`, geometry sensitivity `0.009340`,
scalar normalized MAE `0.236065`, and zero scalar violations. Raw latent
MSE/cosine improved only to `5.94996 / 0.00351`; projected MSE/cosine stayed
at `0.00423 / 0.99732`. It did not improve the selected physics result, so
`lambda_raw` remains an opt-in ablation rather than production configuration.
Its goal-path gradient norm mean was `0.1084` (maximum `18.5349`).

The final goal-margin rerun completed from commit `b8926b9` with all requested
diagnostics and a verified five-part checkpoint. With `lambda_phys=0.8`,
`lambda_goal=2.0`, and margin `0.01`, final values were:

- physics real/null/shuffled: `0.442827 / 0.469963 / 0.484754`;
- real-vs-shuffled geometry sensitivity: `0.013780`;
- occupancy IoU/F1: `0.662521 / 0.795747`;
- scalar normalized MAE: `0.234905`, out-of-range fraction `0`;
- raw latent MSE/cosine: `6.2292 / -0.0579`;
- projected latent MSE/cosine: `0.00405 / 0.99749`;
- `c_physics`/`a_goal` cross-sample standard deviations:
  `0.094129 / 0.104366`;
- goal-path gradient norm mean/max: `0.1333 / 19.6669`.

Gate D passes on this exploratory goal-margin run. Its checkpoint reassembled
to `188,426,071` bytes with SHA-256
`d6e8289725e6b3460ffd0d47892ca24cea9009409b157a0b3cda3a8075c4fd6d`.

The final metadata-compliance kernel
`akashkesav/metasurface-jepa-goal-margin-experiment-v4` completed from commit
`4e5ee0b`. It records validation IDs `[0, 1, ..., 15]`, the same hard-stratum
Gate-D pass, and the following final values: real/null/shuffled physics
`0.473821 / 0.480945 / 0.479728`, geometry sensitivity `0.005129`, occupancy
IoU/F1 `0.674159 / 0.804085`, scalar normalized MAE `0.241196`, raw/projected
cosine `-0.05356 / 0.99756`, and goal-path gradient mean/max
`0.1540 / 77.4082`. Its emitted checkpoint manifest specifies
`188,426,071` bytes and SHA-256
`d650b73d0956039e2a1eb2e7d6e35598f91e0578732b4f8b93499e799fa4251a`.

The matched-weight run combining the selected `lambda_phys=0.2` with the
goal-margin objective completed on WSL Kaggle as
`anosvol/metasurface-jepa-goal-margin-selected-v2`, from commit `b784c65`.
It used dataset `anosvol/metadit-aaai2026-staging`, seed `42`, 1500 steps,
`lambda_goal=2.0`, and margin `0.01`. On the fixed hard stratum
(`100_percent_occupancy_mask_all_scalars_unknown`, validation IDs 0--15), the
final step-1500 values were:

- physics real/null/shuffled: `0.492732 / 0.485704 / 0.518057`;
- real-vs-shuffled geometry sensitivity: `0.0111547`;
- occupancy IoU/F1: `0.690314 / 0.815067`;
- predicted/true occupancy fraction: `0.499176 / 0.422974`;
- scalar normalized MAE: `0.237963`, scalar out-of-range fraction `0`;
- raw latent MSE/cosine: `6.18184 / -0.06618`;
- projected latent MSE/cosine: `0.004019 / 0.997437`;
- `c_physics`/`a_goal` cross-sample standard deviations:
  `0.091998 / 0.085290`;
- goal-path gradient norm mean/max: `4.7147 / 6838.37`.

This satisfies the measured real-vs-shuffled direction and decoded-geometry
sensitivity gate for this run. The checkpoint was downloaded as five shards,
reassembled to `188,426,071` bytes, and matched the emitted SHA-256:
`2faa5ed7c3268a0faef9784a4ea650a4e340b308c0e3deefb7dea818910aec53`.

## Provenance

- repaired comparison commit: `9a108c9`;
- goal-margin code commit: `acf6998`;
- sweep launcher commit: `c32a7cf`;
- Kaggle dataset: `anosvol/metadit-aaai2026-staging`;
- final Kaggle kernel: `anosvol/metasurface-jepa-goal-margin-selected-v2`;
- final checkpoint SHA-256: `2faa5ed7c3268a0faef9784a4ea650a4e340b308c0e3deefb7dea818910aec53`;
- seed: `42`;
- hard-stratum validation batches: `8`;
- one invalid training sample was skipped and recorded by the evaluator.
