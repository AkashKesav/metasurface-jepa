# Joint Target Redesign — Spectrum-Coupled JEPA Target

**Status:** ACTIVE — adopted 2026-09-08 by operator decision.
**Supersedes:** [`docs/UNIFIED_ARCHITECTURE_FIX_GOAL.md`](UNIFIED_ARCHITECTURE_FIX_GOAL.md) (disabled same date).
**Scope:** design specification only. No code has been written against this yet.

---

## 1. Objective

The current model predicts a geometry-only EMA target while receiving the target spectrum as
an input. That permits the predictor to ignore the requested spectrum; the previous
hard-stratum result showed essentially zero real-vs-null goal gain.

The redesign changes the JEPA target itself:

```text
Z_joint = J(Z_G, Z_S)
```

where geometry and spectrum interact through explicit cross-attention.

Intended pipeline:

```text
masked geometry + target spectrum
            ↓
      joint predictor
            ↓
      Z_pred_joint
            ↓
      geometry decoder
            ↓
         G_hat
            ↓
    frozen EM surrogate
            ↓
         S_hat
```

The teacher constructs:

```text
complete geometry + corresponding spectrum
            ↓
      joint target fusion
            ↓
      Z_target_joint
```

with stop-gradient and EMA geometry weights.

The redesign creates the desired mechanism; it does not by itself prove that the latent
actually uses physics. That remains an experimental acceptance criterion.

---

## 2. Non-negotiable architecture

### Student inputs

```text
occupancy          [B,1,64,64]
occupancy mask     [B,16,16]   # 1 = visible, 0 = masked
scalar values      [B,3]
scalar known flags [B,3]
target spectrum    [B,2,301]
optional noise     [B,64]      # later phase only
```

The occupancy stream must remain occupancy-only. Do not reconstruct the original
three-channel broadcast geometry before masking, because that can leak scalar information
through visible occupied pixels.

### Geometry representation

```text
4×4 patch embedding
[B,1,64,64] → [B,256,192]
```

Retain spatial indexing: `256 = 16 × 16`.

### Physics representation

Reuse the released MetaDiT spectrum encoder initially:

```text
A_local : [B,301,256]
c_phys  : [B,384]
a_goal  : [B,16,384]
```

Project `a_goal` to 192-D for the joint predictor.

Do not replace or fine-tune the released spectrum encoder in the first experiment; it has
already been independently validated in the existing project.

---

## 3. Joint target encoder

This is the main scientific change.

```text
G_true
  ↓
EMA geometry encoder
  ↓
Z_G_target [B,256,192]
  ↓
Q = geometry tokens
K,V = physics tokens
  ↓
cross-attention
  ↓
Z_target_joint [B,256,192]
```

Use:

```text
delta = cross_attn(query=geometry_tokens, key=physics_tokens, value=physics_tokens)
joint = geometry_tokens + tanh(gate) * delta
```

Initialize `gate = 0`. This gives stable initialization while allowing the teacher to learn a
physics-conditioned structural representation.

Target path:

```python
with torch.no_grad():
    z_g = ema_geometry_encoder(G_true, full_scalars)
    a_s = frozen_spectrum_encoder(S_true)
    z_target_joint = joint_target_fusion(z_g, a_s)
```

Teacher properties: EMA geometry weights, frozen spectrum encoder, eval mode, no gradient.

Target-side scalar FiLM must use the existing EMA scalar-conditioning copy.

### Important caveat

A joint target does not automatically guarantee a physics-aware representation. The target
can still remain mostly geometry-driven. Therefore the following must be measured:

- target sensitivity to spectrum
- student sensitivity to spectrum
- student-vs-target alignment

Do not claim "physics-aware latent" from architecture alone.

---

## 4. Student masked-query predictor

Only masked spatial locations are predicted. For every masked token:

```text
q_i = mask_token + position_i + scalar_summary
```

Then:

```text
Q = masked structural queries
K,V = visible geometry tokens + goal tokens
```

The predictor performs:

```text
masked queries
      ↓
self-attention
      ↓
cross-attention(Q = masked queries, K,V = visible geometry + physics goal)
      ↓
FFN
      ↓
predicted masked latent tokens
```

Scatter the predicted tokens back to the original spatial positions to form
`Z_pred_joint [B,256,192]`. For visible positions, preserve the context representation
unchanged in the first implementation.

The mask convention must remain **1 = visible, 0 = masked**, and query index ↔ spatial token
index must be preserved exactly.

The existing architecture's masked-token approach and GCLCT pathway support this design
direction.

---

## 5. Physics routing

Use two spectrum interfaces.

### Fine goal tokens

```text
A_local [301 tokens] → 16 learned queries → a_goal [16,192]
```

Use them as K/V for masked-query cross-attention.

### Global physics condition

```text
c_phys [384] → projection → AdaLN / FiLM
```

Use this only for global predictor modulation.

Do not concatenate raw spectrum tokens directly into the decoder. The decoder should consume
the joint latent; otherwise it can bypass JEPA.

Dense goal attention comes first. Do not introduce top-k sparse routing until dense
conditioning is demonstrated to matter.

---

## 6. Decoder

Input: `Z_pred_joint [B,256,192]`
Output: `occupancy_logits [B,1,64,64]`, `scalar_pred [B,3]`

Known inputs are hard-retained:

```text
visible occupancy → exact original occupancy
masked occupancy  → decoder output
known scalar      → exact original scalar
unknown scalar    → decoder output
```

### Scalar output constraint

The current audited implementation does not reliably provide bounded scalar outputs, so the
redesign must not claim that it already does. Use explicit bounds:

```text
l_lattice ∈ [2.5, 3.0]
h_atom    ∈ [0.5, 1.0]
r_atom    ∈ [3.5, 5.0]
```

with

```python
pred = lo + (hi - lo) * torch.sigmoid(raw)
```

Log predicted min/max and out-of-range fraction.

---

## 7. Losses

The first implementation should be deliberately small.

```text
L = λ_J·L_JEPA + λ_occ·L_occ + λ_sc·L_scalar + λ_phys·L_phys + λ_goal·L_goal
```

but activate terms in stages.

### 7.1 JEPA

Masked positions only:

```text
(1/|M|) Σ_{i∈M} D(ẑ_i, z_target_i)
```

Use normalized Huber or cosine initially.

### 7.2 Occupancy reconstruction

Mandatory, because the previous unified implementation omitted it:

```text
L_occ = BCEWithLogits(occ_logits, occ_true)
```

Prefer masked-pixel supervision; report full-image IoU/F1 separately.

This makes the `lambda_phys = 0` baseline a valid decoder baseline.

### 7.3 Scalar reconstruction

Huber/L1 on unknown scalar positions only.

### 7.4 Physics loss

```text
Z_pred_joint → decoder → G_hat → MetaDiT assembly → frozen differentiable surrogate → S_hat
L_phys = D_S(S_hat, S_goal)
```

Keep the current validated spectrum normalization until the new baseline is reproduced.

### 7.5 Goal-sensitivity loss

Do not enable initially. Once the base pipeline works, add a hard-stratum
real-vs-shuffled constraint:

```text
L_goal = max(0, m + E_real − E_shuffle)
```

Only apply this where counterfactual target swapping is physically meaningful.

---

## 8. Critical validation fix

The existing repository has a real validation bug: `validate()` calls `model.eval()` while
the old objective conditioned physics computation on `model.training`, which could make
validation physics loss fall back to zero.

The new API must separate *training/eval mode* from *whether physics metrics are computed*.
Use `objective(..., compute_physics=True)` in validation even while the model is in eval mode.

The frozen surrogate remains `eval()` with `requires_grad=False`, but must remain
differentiable with respect to generated geometry.

---

## 9. Training schedule

Do not implement the whole system simultaneously.

| Stage | Mask / scalars | Active terms | Goal |
|---|---|---|---|
| **A — Joint target only** | 50% block mask, scalars known, `S_goal = S_true` | `L_JEPA` | student can predict joint target latent |
| **B — Geometry decoder** | same | `+ L_occ`, `+ L_scalar` | joint latent decodes into useful geometry |
| **C — Physics** | same | `+ L_phys`, gradual ramp from zero | decoded geometry matches target spectrum |
| **D — Conditionality** | hard stratum | evaluate real / null / shuffled | only then add `L_goal` or CFG |
| **E — Hard masking** | 25% → 50% → 75% → 100%, retaining lower-mask examples | as above | mask robustness |
| **F — Zero-context** | 100% occupancy mask + required scalar-unknown cases | as above | pure inverse design |
| **G — Stochasticity** | only after deterministic conditioning works | — | — |

---

## 10. Anti-cheating rules

The student must never receive:

- true masked geometry
- complete target geometry
- teacher latent
- surrogate output from true geometry

The decoder must not receive raw spectrum tokens.

The surrogate must have frozen parameters with a differentiable input path.

The teacher must be EMA geometry, EMA scalar conditioning, stop-gradient, eval mode.

The real target spectrum is a legitimate input because spectrum conditioning is the task;
the true geometry must remain inaccessible to the student.

---

## 11. Required diagnostics

Every serious run must report the hard stratum separately: **100% occupancy mask, all
scalars unknown**.

**Geometry** — occupancy IoU, occupancy F1, predicted occupancy fraction, scalar MAE, scalar
min/max, out-of-range fraction.

**Physics** — real-goal error, null-goal error, shuffled-goal error, real-vs-null gap,
real-vs-shuffled gap. The key criterion is:

```text
real-goal physics error < shuffled-goal physics error
```

Null-goal sensitivity alone is insufficient.

**Conditioning** — measure `|Z(S_a) − Z(S_b)|` with fixed context and `|Z(G_a) − Z(G_b)|`
with fixed goal; also `c_phys` variance, `a_goal` variance, goal-related gradient norms.
Attention maps are diagnostic only; they are not proof of goal usage.

**Representation** — report both raw `z_hat` vs `z_target` and projected `p_hat` vs
`p_target`, because the prior architecture supervised projected features while decoding raw
latent features. This is a known projector-absorption risk.

---

## 12. Minimum acceptance criteria

**Correctness** — `L_occ` is active; occupancy decoder gets gradients; validation computes
real physics in eval mode; surrogate parameters get no gradients; EMA target gets no
gradients.

**Geometry** — occupancy metrics are non-degenerate; scalar outputs stay inside configured
ranges.

**Conditioning** — real < shuffled physics error, and the decoded geometry changes when a
valid target spectrum changes.

**Fair comparison** — baseline and physics runs must use the same decoder, `L_occ`, mask
curriculum, scalar curriculum, validation set, and checkpoint provenance.

**Reproducibility** — persist git commit, config, dataset identifier, checkpoint, seed, and
validation sample IDs. Do not rely on `/kaggle/working` as the only checkpoint location.

---

## 13. Do not add yet

```text
sparse top-k spectral routing
reciprocal two-way attention
stochastic latent
latent diffusion / flow
joint-spectrum auxiliary embedding loss
frequency weighting / resonance losses
fine-tuning of the released spectrum encoder
```

Each should require a measured failure or demonstrated need.

---

## 14. Recommended implementation order

1. `JointTargetFusion`
2. `MaskedQueryPredictor`
3. forward-only shape/gradient tests
4. JEPA-only training
5. occupancy + scalar decoder supervision
6. correct physics-enabled validation
7. frozen-surrogate physics loop
8. real/null/shuffled hard-stratum evaluation
9. goal-sensitivity experiment
10. physics-weight sweep
11. raw-vs-projected latent ablation
12. only then consider stronger architecture/routing

---

## 15. Final research hypothesis

> A JEPA whose target explicitly couples geometry with electromagnetic behavior will produce
> a more useful conditional structural latent than geometry-only JEPA, improving masked
> completion and target-spectrum adherence without turning the system into a direct
> spectrum-to-geometry shortcut.

The claim is successful only when the experiments demonstrate all three:

```text
correct geometry + correct target physics + actual dependence on target spectrum
```

The architecture is the mechanism to test that hypothesis — not evidence that the hypothesis
is already true.
