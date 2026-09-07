# Unified JEPA — Original Present Architecture

## Scope

This document describes the original repaired architecture identified as
`unified_occ_param_spectrum_jepa_v1`. It deliberately excludes the later
experimental additions: the direct goal residual, real-vs-null ranking loss,
and detached comparison branches.

The task is spectrum-conditioned inverse design:

```text
incomplete geometry + scalar observations + requested spectrum
                           ↓
                    completed geometry
                           ↓
                  simulated target spectrum
```

The model must reconstruct valid geometry and change that geometry in response
to the requested target spectrum.

## 1. Semantic data representation

The released MetaDiT geometry has shape `[3,64,64]` and uses:

```text
channel 0 = occupancy × (r_atom / 5)
channel 1 = occupancy × h_atom
channel 2 = occupancy × (l_lattice / 3)
```

The unified model factorizes this into:

```text
occupancy: [1,64,64]
scalars:   [l_lattice, h_atom, r_atom]
```

The scalar order is fixed everywhere. Approximate valid ranges are:

```text
l_lattice: [2.5, 3.0]
h_atom:    [0.5, 1.0]
r_atom:    [3.5, 5.0]
```

Every scalar also has a known/unknown flag. Unknown values are zeroed before
the scalar encoder. Known values are hard-retained during final assembly.

## 2. Occupancy masking

The `64×64` occupancy image is represented on a `16×16` token grid with patch
size 4. Training masks one to four axis-aligned rectangular blocks using the
ratios:

```text
25%, 50%, 75%, 100%
```

The mask convention is:

```text
1 = visible token
0 = masked token
```

Masked positions receive learned mask tokens and positional information.
Visible occupancy pixels are hard-retained during geometry assembly.

The critical evaluation regime is:

```text
100% occupancy masked + all scalars unknown
```

## 3. Original end-to-end data flow

```text
occupancy + scalar values/flags
              ↓
       masked occupancy encoder
              ↓
       geometry tokens z_x
              │
              ├───────────────┐
              ↓               ↓
       scalar encoder    released spectrum encoder
              ↓               ↓
       scalar summary    local spectrum tokens
              │               │
              │        c_physics + a_goal
              └───────┬───────┘
                      ↓
               fusion transformer
                      ↓
                 GCLCT predictor
                      ↓
                    z_hat
                      ↓
                geometry decoder
                      ↓
               assembled geometry
                      ↓
              frozen EM surrogate
                      ↓
               predicted spectrum
```

The EMA target path is separate:

```text
complete true geometry → EMA encoder → z_y
```

The EMA target receives geometry but not the requested spectrum.

## 4. Geometry and scalar encoders

The masked occupancy encoder maps the input into 256 geometry tokens:

```text
[B,1,64,64] → patch embedding → [B,256,192] = z_x
```

The scalar encoder receives the value and availability flag for each scalar:

```text
[l_value, l_known,
 h_value, h_known,
 r_value, r_known]
```

It produces FiLM parameters for scalar-conditioned processing and one scalar
summary token for the fusion/predictor path.

## 5. Released spectrum path

The released MetaDiT spectrum encoder remains frozen:

```text
S [B,2,301] → released encoder → A_local [B,301,256]
```

The original spectrum path produces two outputs.

### Global physics condition

The local spectrum tokens are mean-pooled over frequency and projected:

```text
A_local.mean(frequency) → c_physics [B,384]
```

`c_physics` FiLM-modulates every GCLCT predictor block.

### Structured goal tokens

Sixteen learned queries cross-attend over the 301 local spectrum tokens:

```text
learned queries + A_local → a_goal [B,16,384]
```

These tokens enter both the fusion sequence and the predictor key/value
sequence.

### Null diagnostic

The null goal mode replaces both spectrum representations with zeros:

```text
c_physics = 0
a_goal = 0
```

This tests whether the model uses the requested spectrum or falls back to a
geometry prior. The released spectrum encoder itself was not redesigned.

## 6. Fusion and GCLCT predictor

The fusion transformer concatenates:

```text
256 occupancy tokens + 16 goal tokens + 1 scalar token = 273 tokens
```

The GCLCT predictor receives:

```text
queries:
  256 occupancy completion queries + 1 scalar query

key/value:
  256 fused occupancy tokens + 16 goal tokens

global condition:
  c_physics
```

Each predictor block contains:

```text
affine-less LayerNorm → FiLM(c_physics) → self-attention
affine-less LayerNorm → FiLM(c_physics) → cross-attention
affine-less LayerNorm → FiLM(c_physics) → MLP
```

The output is:

```text
z_hat [B,257,192]
```

The first 256 tokens predict occupancy latents. The final token predicts the
scalar summary.

## 7. EMA target and JEPA/VICReg objective

The complete true occupancy is passed through a frozen EMA copy of the
occupancy encoder:

```text
complete occupancy → EMA encoder → z_y_raw [B,256,192]
```

The EMA scalar copy supplies target-side FiLM conditioning. It is not a second
scalar latent target.

The objective owns a shared projector:

```text
z_hat → projector → p_hat
z_y   → projector → p_y
```

The representation losses are:

```text
L_inv = masked prediction invariance
L_var = variance regularization
L_cov = covariance regularization
```

The repaired weights were approximately:

```text
lambda_inv = 25
lambda_var = 25
lambda_cov = 1
```

The decoder consumes raw `z_hat`, while the dominant representation objective
is measured in projected space. This produced the diagnostic:

```text
raw latent cosine       = -0.06618
projected latent cosine =  0.997437
```

This is the projector-absorption risk: projected alignment can look excellent
while the raw decoder latent remains poorly aligned.

## 8. Scalar and occupancy decoders

The scalar-summary token is decoded into the three physical parameters. Each
output is bounded with:

```text
prediction = lo + (hi - lo) × sigmoid(raw)
```

This keeps predictions in the configured physical ranges.

The 256 occupancy prediction tokens are reshaped into a `16×16` feature map,
upsampled to `64×64`, and decoded into occupancy logits. The decoder is FiLM
conditioned by effective scalars:

```text
known scalar   → true scalar
unknown scalar → predicted scalar
```

The active occupancy reconstruction term is:

```text
L_occ = BCEWithLogits(occupancy_logits, true_occupancy)
```

It is averaged over unknown/masked pixels and reported with full-image:

```text
occupancy IoU
occupancy F1
predicted occupancy fraction
true occupancy fraction
```

## 9. Geometry assembly and physics surrogate

The decoded outputs are assembled back to the MetaDiT convention:

```text
occupancy × (r_atom / 5)
occupancy × h_atom
occupancy × (l_lattice / 3)
```

Visible occupancy pixels and known scalar values are retained before the
surrogate.

The assembled geometry is passed through the frozen MetaDiT forward surrogate:

```text
geometry [B,3,64,64] → frozen surrogate → spectrum [B,2,301]
```

The surrogate stays in evaluation mode and receives no parameter gradients.
Gradients with respect to the geometry input remain enabled. The normalized
physics loss is:

```text
L_phys = normalized SmoothL1(spectrum_pred, spectrum_target)
```

## 10. Repaired original objective

The repaired objective is:

```text
L = lambda_inv    × L_inv
  + lambda_var    × L_var
  + lambda_cov    × L_cov
  + lambda_scalar × L_scalar
  + lambda_occ    × L_occ
  + lambda_phys   × L_phys
```

The important repairs were:

### Occupancy supervision

`L_occ` was added so the `lambda_phys=0` baseline also trains its occupancy
decoder. Previously it had no active direct occupancy reconstruction signal.

### Validation physics

Physics computation was separated from model training mode. Validation can now
run the real surrogate path while the model is in `eval()` mode:

```python
model.eval()
objective(..., compute_physics=True)
```

### Scalar validity

Scalar outputs are bounded and range violations are reported explicitly.

### Reporting and reproducibility

The pipeline records joint mask/scalar regimes, raw/projected diagnostics,
per-loss values, goal-path gradients, validation sample IDs, checkpoint
manifests, and SHA-256 hashes.

## 11. Repaired original-architecture result

The repaired original architecture used a selected physics weight near
`lambda_phys=0.2`, seed 42, and the hard stratum of 100% occupancy masking and
all scalars unknown.

Final metrics were:

```text
physics real           = 0.492732
physics null           = 0.485704
physics shuffled       = 0.518057
occupancy IoU          = 0.690314
occupancy F1           = 0.815067
scalar normalized MAE = 0.237963
scalar out-of-range    = 0
geometry sensitivity   = 0.0111547
c_physics variation    = 0.091998
a_goal variation       = 0.085290
```

Lower physics error is better. The result means:

- geometry reconstruction is non-degenerate;
- scalar predictions remain valid;
- the spectrum path has measurable variation;
- real goals outperform shuffled goals;
- real goals do not outperform null goals;
- target conditioning is measurable but not yet reliable.

This is the repaired baseline, not the final spectrum-conditioned designer.

## 12. Later experiments excluded from this architecture

After the original architecture, three separate experiments were tested.

### Direct goal residual

```text
z_base + direct spectrum residual = z_final
```

This gave spectrum information a shorter path to masked occupancy tokens. It
increased goal sensitivity, but its first version still had a spectrum-aware
base predictor and was therefore not a fully separated base/goal design.

### Real-vs-null ranking

A ranking term required real physics error to be lower than null and shuffled
errors. The first version could satisfy the ordering by making null outputs
worse, so it was not accepted as a production architecture.

### Detached comparison branches

The next version detached null and shuffled comparison errors so the ranking
loss could not improve merely by damaging those comparison outputs. This is
still an experiment and is not part of the original architecture described in
this document.

## 13. Strengths of the original design

The original architecture has solid foundations:

- semantic geometry factorization;
- explicit scalar missingness;
- rectangular block masking;
- released spectrum encoder reuse;
- EMA geometry target;
- frozen but differentiable physics surrogate;
- hard retention of known geometry and scalar values;
- bounded scalar outputs;
- active occupancy reconstruction;
- joint curriculum logging;
- frozen-reference gradient checks;
- recoverable checkpoint manifests.

These foundations should remain intact during future goal-conditioning work.

## 14. Main limitation

The main limitation is not the raw data factorization. Geometry and spectrum
are already represented as different semantic inputs.

The limitation is that they are mixed before the main prediction objective is
resolved:

```text
geometry tokens + goal tokens + scalar token
              ↓
          shared predictor
```

At the same time, JEPA targets a geometry-only latent. Therefore the
spectrum-conditioned student is trained toward a target that contains no
spectrum information.

This explains the combination of:

```text
excellent projected JEPA alignment
weak raw decoder-latent alignment
nonzero spectrum sensitivity
weak real-goal advantage over null
```

## 15. One-page summary

```text
DATA
  MetaDiT geometry → occupancy + [l_lattice,h_atom,r_atom]
  target spectrum  → frozen released spectrum encoder

CONTEXT
  block-masked occupancy → 256 geometry tokens
  scalar values/flags    → FiLM + scalar summary

GOAL
  spectrum tokens → c_physics + 16 a_goal tokens

PREDICTOR
  geometry + goal + scalar tokens → fusion → GCLCT → z_hat

TARGET
  complete geometry → EMA encoder → z_y

DECODER
  z_hat → occupancy logits + scalar predictions
       → known-value retention + geometry assembly
       → frozen MetaDiT surrogate
```

The original architecture is correctly connected at the data, decoder, and
physics boundaries. Its unresolved scientific issue is target-specific geometry
selection: the requested spectrum must improve decoded geometry rather than
merely alter the latent representation.

