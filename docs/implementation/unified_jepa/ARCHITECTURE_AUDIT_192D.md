# Unified 192-D JEPA — architecture audit

Scope: the **unified occupancy–parameter–spectrum JEPA** as it stands after the 2026-09-13
retirement + audit sessions — every component, its wiring, and the evidence behind each claim.
`architecture_v5.md` remains the design authority; this document records what the code actually
does, what has been measured, and what is still open. Defect history: `AUDIT_REPORT_192D.md`
(B1–B18, prior session) and this document (B19–B27). Run records:
`checkpoints/unified/REPORT.md`; the physics sweep: `PHYSICS_LAMBDA_SWEEP_PLAN.md`.

**Evidence grades used throughout.** `[code]` = verified by reading the source (file:line).
`[measured]` = observed in a real run (local gate, cloud run, or probe kernel — named).
`[unverified]` = believed but not yet demonstrated; listed in §7 rather than asserted.

---

## 1. What the architecture is

Single-channel occupancy + explicit scalar parameters + a conditioned spectrum, at 192-D
throughout, with the 384-D leg confined to the frozen MetaDiT boundary.

```
occupancy [B,1,64,64] ──┐
scalar [B,3] + known ──►│ ScalarEncoder (FiLM params + summary token)
                        │        │
                        ▼        ▼
              OccupancyEncoder ─► z_x [B,256,192]
                        │
spectrum [B,2,301] ─► SpectrumPath (FROZEN released encoder + trainable goal read-out)
                        │  c_physics [B,384]      a_goal [B,16,384]
                        ▼
   FusionEncoder(z_x, a_goal↓192, scalar_summary) ─► fused [B,273,192]
                        ▼
   GCLCT predictor(queries=masked-token/mask-token + scalar query, kv=fused, cond=c_physics↓192)
                        ▼
                 z_hat [B,257,192] ─┬─► occ tokens [B,256,192] ─► OccupancyDecoder ─► logits [B,1,64,64]
                                    └─► scalar query [B,192] ────► ScalarDecoder ─► scalar_pred [B,3]

EMA targets (frozen):  ema(occupancy, ALL tokens) + scalar_mlp_ema(all-known scalars) ─► z_y_raw [B,256,192]

physics: decode_geometry(z_hat, scalar_pred | known subst., retention) ─► [B,3,64,64]
         ─► FROZEN MetaDiT surrogate ─► Ŝ [B,2,301] ─► normalized SmoothL1 vs true spectrum
```

### 1.1 Components and their weight counts `[code]`

Built locally from the shipped config (CPU, `build_unified_model`):

| module | params | trainable |
|---|---|---|
| `occupancy_encoder` | 2,668,608 | 2,668,608 |
| `scalar_encoder` | 339,392 | 339,392 |
| `spectrum_path` | 4,991,040 | 464,896 |
| `fusion_encoder` | 962,368 | 962,368 |
| `predictor` (GCLCT) | 6,630,272 | 6,630,272 |
| `occupancy_decoder` | 262,785 | 262,785 |
| `scalar_decoder` | 37,251 | 37,251 |
| `ema` (target) | 2,668,608 | **0** |
| `scalar_mlp_ema` (target) | 339,392 | **0** |
| **total** | **18,899,716** | ~11.37 M |

`spectrum_path`'s trainable 464,896 is the goal read-out (`goal_queries`, `proj_g`, `proj_goal`,
attention projections); the released encoder inside it is frozen `[code]` and excluded from saved
checkpoints (`SAVED_EXCLUDES = (".released.",)`, `assembly.py:61`).

### 1.2 Shipped configuration `[code]`

`hidden 192` · `geo_depth 6` · `predictor_depth 8` · `goal_tokens 16` · `n_film_blocks 6` ·
`spec_dim 256` · EMA `0.996 → 0.999` · mask curriculum `{0.25, 0.5, 0.75, 1.0}` with probs
`{0.25, 0.35, 0.25, 0.15}` (0.0 deliberately excluded — the masked-token objective is undefined
with no masked tokens) · scalar regimes `all_known/all_unknown/mixed` · `guidance_dropout 0.1` ·
loss weights `λ_inv 25, λ_var 25, λ_cov 1, λ_scalar 1, λ_occ 1, λ_phys 0.1 (ramped 500)`,
`scalar_loss_type l1` · staging `phase C`, `physics_use_ste true` · `eval.n_samples 32` ·
`train.batch_size 2`, `total_steps 1500`, `warmup 100`, `lr 3e-4`, `seed 42`.

### 1.3 Shape flow `[measured]`

A real forward (B=2) returns `z_x (2,256,192)`, `z_hat (2,256,192)`, `scalar_pred (2,3)`,
`z_y_raw (2,256,192)`, `c_physics (2,384)`, `a_goal (2,16,384)`, loss mask `(2,256)`.

---

## 2. Audit — data path

| item | status | evidence |
|---|---|---|
| `MetaDiTDataset` → `collate_batch` → `factorize_geometry` → `BlockMasker` / `ScalarMasker` → forward | **wired** | `train_unified.py:657,770-771,788-789,905-907,443-445,423-424` `[code]` |
| `data.num_workers` reaches both loaders; `collate_fn` is passed | **wired** | audit B17 wired it after it had been ignored `[code]` |
| Device moves at every hop (occupancy, scalars, spectrum, mask) | **wired** | `[code]`; the evaluator also asserts mask/occ device agreement |
| Mask-ratio calibration: requested vs achieved | **measured** | requested 0.25/0.50/0.75/1.00 → achieved **0.2462 / 0.5002 / 0.7582 / 1.0000** over 150 real steps `[measured: verify run]` |
| Fractional coverage before the B20 fix | **measured negative** | requested 0.5 delivered **0.399** `[measured]` |
| `ScalarMasker.sample`'s `masked_values` output | **dead** | only the `known` flags are used (`train_unified.py:287-288`); value-masking is re-derived in `_build_scalar_input` |

## 3. Audit — encoders

| component | status | evidence |
|---|---|---|
| `OccupancyEncoder` — patch-4 ViT over 16×16 tokens, FiLM per block | **wired** | `n_film_blocks == geo_depth` asserted at construction (audit B18) `[code]` |
| Shared blocks (`Attention`, `CrossAttention`, `TransformerBlock`, `get_2d_sincos_pos_embed`) extracted to `encoders/blocks.py` at retirement | **wired** | three live importers re-pointed `[code]` |
| `ScalarEncoder` — 6-dim `[value, known-flag]×3` input | **wired** | missingness is an explicit flag, never a sentinel (Phase 1 MD §3) `[code]` |
| `SpectrumPath` — frozen released encoder + trainable goal read-out | **wired** | `[code]`; the released encoder is excluded from saves and re-loaded from disk |
| Released encoder frozen at attach time | **fixed + measured** | audit B1: `set_spectrum_path` previously attached it with `requires_grad=True`; the cloud preflight now reports **`released_params_with_grad: 0`** `[measured]` |
| Null-goal branch truly zeroes the conditioning | **wired** | `spectrum_encoder.py:104-110` returns zeros **and skips the released encoder** `[code]` — not a relabel |
| Frozen encoder's input carries per-sample content | **measured positive** | real-vs-deranged `c_physics` ratio **1.320** over 64 samples `[measured: goal probe]` |

## 4. Audit — fusion, predictor, decoders

| component | status | evidence |
|---|---|---|
| `FusionEncoder` — 256 occ + 16 goal (384→192 projected) + 1 scalar summary = 273 tokens, asserted | **wired** | shape assertion in `assembly.py`; `goal_proj` preserves content (ratio 1.010) `[measured]` |
| `GCLCT` predictor — mask-token/visible queries + scalar query, cross-attends fused, conditioned on `c_physics` (384→192) | **wired** | `c_phys_proj` preserves content (ratio 1.141) `[measured]` |
| `need_attn` | **fixed** | audit B16: was accepted and silently ignored; now returns `out["attn_weights"]` |
| `OccupancyDecoder` FiLM conditioning | **fixed** | audit B10: 6-dim `[value, known-flag]×3`, so a known and a predicted scalar of equal magnitude are distinguishable |
| `ScalarDecoder` init | **documented** | zero weight + dataset-mean bias (not Kaiming); the mean bias is what keeps the physical-unit L1 small `[code]` |
| Decode-time scalar rule (true where known, predicted where unknown) identical in training and inference | **wired** | `_effective_scalars`, used by both the decode FiLM and assembly (§4.1) `[code]` |
| Predictor output actually reaches the decoded design | **measured positive** | shuffling the spectrum flips **16.9 %** of occupancy pixels and moves the logits by 0.341 relative `[measured: goal probe, physics on]` |

## 5. Audit — EMA targets

Ten points verified `[code]`, two of them also `[measured]`:

1. deep-copy construction with `requires_grad_(False)` on every target parameter, asserted at
   model construction; 2. seeded from the students at build; 3. `lerp_` EMS maths under
   `@torch.no_grad()`; 4. updated after `optimizer.step()`, once per optimizer step, with the
   0-based step so the first update uses m = 0.996 exactly; 5. schedule recomputed from `step`,
   hence resume-consistent, with `set_total_steps` called by the trainer (audit B2);
   6. target forward under `no_grad` — the shared projector still trains from the `p_y` branch
   (canonical VICReg topology); 7. target-side FiLM fed the all-known 6-dim input, never decoded,
   never a loss target; 8. `eval()` pinning called from `train()` **and** every `forward()`;
   9. `ema_state` round trip with a loud warning for legacy checkpoints (audit B3);
   10. per-step frozen guard covering `ema`, `scalar_mlp_ema`, the released encoder **and** the
   surrogate (audit B18).

**Measured, target actually tracks the student** — relative L2 distance, the thing nothing in the
pipeline previously reported:

| run | `ema` (74 tensors) | `scalar_mlp_ema` (18 tensors) |
|---|---|---|
| 20k steps, physics off (`[measured]`) | 0.0009 | 0.0013 |
| 10k steps, physics on (`[measured]`) | 0.0033 | 0.0080 |

**Residual issues (none breaking):** the targets are serialised twice (`ckpt["model"]` *and*
`ema_state`) so one path could mask a fault in the other; `restore_ema_state` overrides the
config's momentum endpoints on resume; `EMAEncoder.update` zips parameter iterators, so any future
structural divergence would silently skip rather than raise.

## 6. Audit — objective, physics, trainer, evaluation

### 6.1 Objective

| item | status | evidence |
|---|---|---|
| `L = λ_inv·L_inv + λ_var·L_var + λ_cov·L_cov + λ_scalar·L_scalar + λ_occ·L_occ + λ_phys·L_phys` | **wired** | `unified_losses.py:222-225` `[code]` |
| VICReg terms canonical (hinge-mean variance with eps inside the sqrt, unbiased stats, `(N-1)` covariance divisor, branch aggregation ½ var / sum cov) | **verified** | maths audit §3.1 `[code]` |
| Occupancy BCE on **masked pixels only** | **wired** | operator decision 2026-09-13; visible pixels are retained at assembly, so the decoder infers only what was hidden |
| Projector owned by the objective, shared by both branches | **wired** | spec §17: no `model.proj` `[code]` |
| `L_cov` is far under-weighted relative to `λ_var` | **measured** | `L_cov` grows 12.9 → 145.8 (easy) / 15.1 → 160.3 (hard) over 20k steps at `λ_cov = 1` against `λ_var = 25` — recorded, not tuned |
| Null (goal-dropped) steps skip `L_phys` | **decided** | operator decision 2026-09-13 (audit B5); the spectrum-free terms still train that branch |
| `cfg_forward` had no caller anywhere in the pipeline | **fixed** | audit B26: now swept by the evaluator as `cfg_guidance_sweep_A` over `w ∈ {0, 0.5, 1, 2, 3, 5}` |
| `PhysicsSpectrumLoss` | **inert dead code** | `_enabled` is never set; the inactive branch always returns zero |

### 6.2 Physics

| item | status | evidence |
|---|---|---|
| Path: `z_hat` + scalars → decode (retention + known-scalar substitution) → assemble → frozen surrogate → per-sample-target-std-normalized SmoothL1 | **wired** | matches `architecture_v5.md` §4.3 / `04 §4-5` `[code]` |
| Surrogate frozen but differentiable through the input | **wired + measured** | `requires_grad_(False)` + `eval()`, never wrapped in `no_grad`; objective re-pins it to eval so `objective.train()` cannot flip its 38 BatchNorm layers `[code]` |
| STE is **load-bearing**, not a preference | **measured** | `use_ste=False` → `L_phys` **18.58** with **0** student params carrying gradient; `use_ste=True` → **360** params (encoder 72, predictor 210, decoder 14, scalar encoder 17), surrogate 0, EMA 0. Soft field is **96 %** out of distribution (`spectrum_rel_diff` 0.9599) against the real surrogate `[measured: probe kernel]` |
| `λ_phys > 0` with STE off is refused | **fixed** | audit B24 (config guard + preflight assertion) |
| Ramp from zero | **wired + measured** | `λ_phys · min(1, (step+1)/500)`, recomputed from `step`; `L_phys_w` ramps 0 → ~0.004 over the first 500 steps `[measured]` |
| Physics term reaches the student | **measured** | preflight reports **362** student params with gradient from the physics term **alone** `[measured: physics-on kernel]` |
| Degenerate-spectrum guard | **measured unreachable** | min per-sample spectrum std **0.4278** over 20k train / 17,488 val samples vs the `1e-3` threshold — 0 samples below `[measured]` |
| Cost | **measured** | 0.092 s/step physics-off · **0.129 s/step** physics-on (+40 %) |

### 6.3 Trainer

| item | status | evidence |
|---|---|---|
| Optimizer owns exactly the trainable students + the objective projector; EMA / released encoder / surrogate excluded | **wired + measured** | `requires_grad` filter; cloud preflight ownership table shows 0 grads on all frozen sets `[measured]` |
| LR warmup + cosine driven by `total_steps`, stepped once per optimizer step | **wired** | `[code]` |
| Checkpoint round trip: model, objective, optimizer (+ shape fingerprint), scheduler, RNG streams, EMA, masker RNG, curriculum RNG, step counters | **wired** | audits B3/B4 `[code]` |
| `ckpt["cfg"]` | **saved-only** | written and schema-required, never read on resume — editing the YAML between save and resume silently diverges from the recorded config |
| Validation reports the easy and hard strata separately, never pooled | **wired + measured** | audit B6; every cloud run reported both strata `[measured]` |
| Validation's `L_phys` | **misleading** | physics is gated on `model.training`, so validation reports `L_phys = 0.0` in every physics-enabled run |
| Resume equivalence, frozen-gradient ownership, curriculum/RNG determinism | **covered by the local suite** | 286 passed / 23 skipped / 0 failed `[measured]` |

### 6.4 Evaluation

| item | status | evidence |
|---|---|---|
| Scenarios A/B/C evaluated separately, never pooled | **wired** | `eval_scenarios.py` `[code]` |
| Hard-stratum gate = `real < shuffled` per scenario | **wired** | `real_null_shuffled` `[code]` |
| **Gate sample size** | **fixed** | audit B27: it was `train.batch_size` = **2**, so every comparison, the shuffled control (a 2-item derangement = one swap), the scalar checks and the whole guidance sweep were decided by two samples. Evaluation now has its own knob (`eval.n_samples`, default 32, CLI `--samples`) and reports `n_samples`, `real_beats_shuffled_fraction`, the paired-difference mean/std and the per-sample errors |
| Occupancy metrics use the model's **raw sigmoid** | **fixed** | audit B7: they previously used the assembled/retained occupancy, making visible-region IoU identically 1.0 |
| Shuffled control reproducible | **fixed** | audit B12: explicit seeded derangement |
| Derangement device mismatch | **fixed** | audit B22: drawing on the target device with a CPU generator raised and aborted the whole evaluation |
| Guidance-gap maths | **fixed** | audit B14: per-sample L2 / per-sample σ, shared with `cfg_forward` so they cannot diverge |
| Retrieval baseline present | **wired + measured** | nearest real training spectrum → its real geometry: mean 0.0744, best 0.0486 `[measured]` |
| Spectrum-sensitivity probe the spec requires (perturb the target spectrum, design must move proportionally) | **not implemented** | the evaluator's `diversity_check` at its default is a *determinism* check — its own docstring says it must not be presented as generative diversity |

---

## 7. What is not verified

- **No gate has passed.** Every gate reading to date is either red or produced by a
  measurement that was too coarse to settle anything. The trap to avoid is reading the
  encouraging probe (real 0.2989 vs shuffled 0.5561 on 8 samples) as a pass while the official
  gate on the same checkpoint says the opposite.
- Physics at `λ = 0.1` has been trained for 10k steps only, on the 2-sample gate; the calibrated
  sweep is the first run scored by a properly powered gate.
- Whether the physics term makes the conditioning content-sensitive **at the gate**, as opposed
  to in the probe.
- The projector-absorption hypothesis (`REPORT.md` §7.4) — the projected space moved while the
  raw space stayed near-orthogonal for 20k steps — is consistent with the observations but has
  not been isolated by an ablation.
- `ckpt["cfg"]` divergence on a config-edited resume; validation's `L_phys = 0` reporting;
  `PhysicsSpectrumLoss` dead code. All recorded, none fixed.
- Everything on the cloud has run on a P100 with a pinned torch 2.5.1; no other GPU or torch
  version has been exercised.

## 8. Honest status

The architecture is **coherent, wired end to end, and instrumented** — every stage from data
loading to gate evaluation has been read line by line and exercised on real data with the
released weights, and the frozen set is provably frozen (measured zero gradients), the EMA
provably tracks (0.1 %), the masking provably hits its targets (±2 %), and the physics term
provably reaches the student (362 params). What it has **not** done is pass its own acceptance
gate: on the hard stratum the model is currently no better with the true spectrum than with
another sample's. The most recent evidence says the conditioning path itself is healthy and the
failure lies in the scale of the physics weight and/or the objectivity of the measurement —
which is exactly what the running sweep and the B27 fix are for. No scientific claim is made by
this document; the gate is the gate.
