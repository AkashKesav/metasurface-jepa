"""Phase 3 — Unified JEPA loss components (architecture_v5.md §3-§5), Phase 4
MD §4 (physics loss integration).

Combines:
- L_inv  : MSE on masked occupancy tokens (via shared VICReg projector)
- L_var  : VICReg variance penalty
- L_cov  : VICReg covariance penalty
- L_scalar: L1 regression on UNKNOWN scalar positions only
- L_occ  : BCE occupancy reconstruction on UNKNOWN pixels only
- L_phys : physics-response loss through frozen MetaDiT surrogate (Phase 4 MD §4)

Full objective:
    L = lambda_inv  * L_inv
      + lambda_var  * L_var
      + lambda_cov  * L_cov
      + lambda_scalar * L_scalar
      + lambda_occ * L_occ
      + lambda_phys * L_phys

Per Phase 3 MD §5: staged training starts with lambda_phys=0 (physics loss
disabled) until the no-physics architecture is numerically stable. Phase 4
MD §4.1 ramps lambda_phys from 0 over lambda_phys_ramp_steps.

The projector is owned by the objective (spec §17: no model.proj). On
optimizer-step it drives BOTH EMA updates (occupancy + scalar_mlp).
"""

import torch
from torch import nn
import torch.nn.functional as F

from losses.jepa_loss import jepa_loss, ProjectionMLP
from losses.vicreg import vicreg_branch_terms
from losses.objective_modules import VICRegProjector


class OccupancyTokenLoss(nn.Module):
    """JEPA / invariance loss on masked occupancy tokens (standalone).

    Computes MSE between predicted and target latents on masked positions,
    optionally through a projector. Returns (scalar_loss, per_sample).
    """

    def __init__(self, hidden=192, use_proj=True):
        super().__init__()
        self.projector = ProjectionMLP(hidden=hidden) if use_proj else None

    def forward(self, z_hat, z_y_raw, mask):
        return jepa_loss(z_hat, z_y_raw, mask, proj=self.projector)


class ScalarPredictionLoss(nn.Module):
    """L1 regression on scalar parameters at positions marked unknown.

    Only positions where scalar_known is False contribute. Known positions
    are excluded to avoid double-supervision through the known/unknown flag
    (Phase 1 MD §3: missingness is explicit).
    """

    def __init__(self, loss_type="l1"):
        super().__init__()
        assert loss_type in ("l1", "huber"), f"loss_type {loss_type!r} not supported"
        self.loss_type = loss_type

    def forward(self, scalar_pred, scalar_values, scalar_known):
        unknown = ~scalar_known  # (B, 3) bool
        if self.loss_type == "huber":
            err = F.huber_loss(scalar_pred, scalar_values, reduction="none")
        else:
            err = (scalar_pred - scalar_values).abs()
        err = err * unknown.float()
        n_unknown = unknown.sum().clamp(min=1)
        return err.sum() / n_unknown


class PhysicsSpectrumLoss(nn.Module):
    """Physics-response loss (placeholder — disabled in Phase 3).

    When enabled (lambda_phys > 0), the predicted geometry is decoded and
    passed through the frozen MetaDiT EM surrogate to compute spectrum
    error against the target. Currently returns zero; the training loop
    gates activation via lambda_phys > 0.

    Per Phase 3 MD §6: when physics loss is active, the MetaDiT forward
    MUST remain differentiable w.r.t. geometry input — this class should
    only be enabled after the no-physics architecture is numerically
    stable.
    """

    def __init__(self):
        super().__init__()
        self._enabled = False

    def forward(self, spectrum_pred, spectrum_target):
        if not self._enabled:
            return torch.zeros((), device=spectrum_pred.device)
        return F.mse_loss(spectrum_pred, spectrum_target)

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


def _pixel_mask_from_token_mask(mask, target):
    """Expand a 16x16 visible-token mask to the occupancy image resolution.

    ``mask`` uses the model convention 1=visible and 0=masked.  The returned
    tensor has the same spatial size as ``target`` and contains 1 for pixels
    that are eligible for masked reconstruction (the complement of visible
    pixels).  A ``None`` mask means every pixel is eligible.
    """
    if mask is None:
        return torch.ones_like(target, dtype=target.dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(
            f"token mask must be [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}"
        )
    visible = F.interpolate(
        mask.to(dtype=target.dtype), size=target.shape[-2:], mode="nearest"
    )
    return (1.0 - visible).clamp(0.0, 1.0)


def occupancy_reconstruction_loss(occupancy_logits, occupancy_target, mask=None):
    """BCE reconstruction averaged over masked/unknown occupancy pixels.

    The denominator is clamped so the unmasked 0% reference condition remains
    a valid, finite diagnostic instead of dividing by zero.
    """
    target = occupancy_target.to(dtype=occupancy_logits.dtype)
    pixel_mask = _pixel_mask_from_token_mask(mask, target)
    bce = F.binary_cross_entropy_with_logits(occupancy_logits, target, reduction="none")
    return (bce * pixel_mask).sum() / pixel_mask.sum().clamp(min=1.0)


def occupancy_reconstruction_metrics(occupancy_logits, occupancy_target):
    """Full-pixel occupancy diagnostics using a 0.5 probability threshold."""
    pred = occupancy_logits >= 0.0
    target = occupancy_target >= 0.5
    tp = (pred & target).sum().to(dtype=occupancy_logits.dtype)
    fp = (pred & ~target).sum().to(dtype=occupancy_logits.dtype)
    fn = (~pred & target).sum().to(dtype=occupancy_logits.dtype)
    iou = tp / (tp + fp + fn).clamp(min=1.0)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp(min=1.0)
    return {
        "occupancy_iou": iou,
        "occupancy_f1": f1,
        "pred_occupancy_fraction": pred.to(dtype=occupancy_logits.dtype).mean(),
        "true_occupancy_fraction": target.to(dtype=occupancy_logits.dtype).mean(),
    }


class UnifiedJEPALoss(nn.Module):
    """Combined JEPA + VICReg + scalar + (optional) physics objective.

    Architecture: unified_occ_param_spectrum_jepa_v1 (192-D throughout).
    Owns its projector (spec §17: no model.proj). Shares the projector
    between JEPA (invariance) and VICReg (variance + covariance), matching
    the canonical VICReg topology (Bardes et al. 2021) adapted to token-level
    masked geometry.

    Projector gradient ownership (Fix 12, explicitly documented): the
    objective-owned projector is trained from BOTH branches — p_hat =
    projector(z_hat) and p_y = projector(z_y) both flow gradients into the
    projector. This matches the Milestone-B VICRegObjective and canonical
    VICReg: the gradient stops at the EMA target encoder because z_y_raw is
    already detached (stop-grad at the EMA boundary, architecture_v5.md §3.6);
    the projector itself is a shared learnable head. The target branch is NOT
    wrapped in torch.no_grad() because that would freeze the projector's
    target-side updates, diverging from the tested Milestone-B behavior.

    on_optimizer_step updates BOTH EMA targets (occupancy + scalar_mlp)
    per Phase 2 §6.
    """

    name = "unified_jepa"
    term_names = (
        "L_inv",
        "L_var",
        "L_cov",
        "L_scalar",
        "L_occ",
        "L_phys",
        "L_raw",
        "L_goal",
    )

    def __init__(
        self,
        hidden=192,
        lambda_inv=25.0,
        lambda_var=25.0,
        lambda_cov=1.0,
        lambda_scalar=1.0,
        lambda_occ=1.0,
        lambda_phys=0.0,
        lambda_raw=0.0,
        lambda_goal=0.0,
        goal_margin=0.01,
        gamma=1.0,
        eps=1e-4,
        scalar_loss_type="l1",
        surrogate=None,
        physics_use_ste=True,
    ):
        super().__init__()
        self.projector = VICRegProjector(
            input_dim=hidden,
            hidden_dim=hidden,
            output_dim=hidden,
        )
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_scalar = lambda_scalar
        self.lambda_occ = lambda_occ
        self.lambda_phys = lambda_phys
        self.lambda_raw = lambda_raw
        # Goal-requirement margin loss (2026-09-12 operator A/B; default OFF).
        # Hinge on the student's real-vs-shuffled sensitivity:
        #   L_goal = relu(margin - mean||z_hat(S) - z_hat(S_shuf)||[mask])
        # encouraging the prediction to move with the goal spectrum by at least
        # `margin`. Hyperparameters reuse the retired plan's recorded values
        # (lambda_goal=2.0, margin=0.01). The shuffled spectrum is supplied by
        # the caller (training_step deranges deterministically per step); when
        # lambda_goal>0 and no shuffled spectrum is given, fail loudly.
        self.lambda_goal = lambda_goal
        self.goal_margin = goal_margin
        self.gamma = gamma
        self.eps = eps
        self.surrogate = surrogate  # frozen MetaDiT EM surrogate (Phase 4)
        # Phase 4 MD §3: STE choice is DOCUMENTED, not silent. The frozen
        # surrogate's ReLU6 activations have a zero Jacobian on soft occupancy
        # fields (verified by soft_hard_occupancy_test), so STE is the
        # empirical default; set physics_use_ste=False only after re-running
        # that check on a surrogate that accepts soft input.
        self.physics_use_ste = physics_use_ste

        # Fix 6 (spec §8): self.occupancy_loss removed — it was constructed
        # but never called; forward() computes the projected JEPA/VICReg
        # terms inline via the shared objective-owned projector. Keeping an
        # unused module here would be dead, misleading code.
        self.scalar_loss = ScalarPredictionLoss(loss_type=scalar_loss_type)
        self.physics_loss = PhysicsSpectrumLoss()

    def forward(
        self,
        model,
        occupancy,
        scalar_values,
        scalar_known,
        spectrum,
        mask,
        goal_mode="real",
        compute_physics=None,
        physics_hard_forward=False,
        spectrum_shuf=None,
    ):
        """Evaluate the objective.

        ``compute_physics`` controls whether the frozen surrogate is evaluated;
        it is deliberately independent of ``model.training`` so validation can
        compute the real physics term while the model is in eval mode.  The
        default retains the inexpensive historical behavior: physics is
        computed automatically during training only when it is weighted on.

        ``spectrum_shuf`` is the deranged-goal control for the opt-in goal
        margin loss (``lambda_goal > 0``). Required in that case; ignored
        (no second forward) when the goal term is off.
        """
        if compute_physics is None:
            compute_physics = bool(model.training and self.lambda_phys > 0)
        out = model(
            occupancy,
            scalar_values,
            scalar_known,
            spectrum,
            mask,
            goal_mode=goal_mode,
        )
        mask_bool = out["mask"]
        z_hat = out["z_hat"]
        # The direct goal route is intentionally outside the geometry-only
        # JEPA/VICReg representation target. Physics decodes z_hat, while
        # representation losses supervise the base latent when available.
        z_hat_repr = out.get("z_hat_base", z_hat)
        # Joint Target Redesign (docs/JOINT_TARGET_REDESIGN.md §3): when the
        # model exposes a joint target (Z_joint = J(Z_G, Z_S)) the objective
        # supervises against IT — the geometry-only z_y_raw remains a
        # diagnostic (target-side gate-closed reference). Models built before
        # the redesign expose only z_y_raw and behave exactly as before.
        z_y = out.get("z_y_joint", out["z_y_raw"])

        # Projected space (shared projector, single forward per branch)
        p_hat_full = self.projector(z_hat_repr)
        p_y_full = self.projector(z_y)
        p_hat = p_hat_full[mask_bool]
        p_y = p_y_full[mask_bool]

        # JEPA (invariance) + VICReg (var + cov) on masked tokens
        L_inv, L_var, L_cov = vicreg_branch_terms(
            p_hat, p_y, gamma=self.gamma, eps=self.eps
        )

        L_inv_w = self.lambda_inv * L_inv
        L_var_w = self.lambda_var * L_var
        L_cov_w = self.lambda_cov * L_cov

        # Scale-sensitive unnormalized MSE on raw latents (remediation for
        # Stage-A scale-free collapse documented in STAGE_A_VERDICT.md §8).
        # Unnormalized MSE enforces both direction and magnitude matching,
        # preventing the 16x magnitude mismatch previously masked by F.normalize.
        if not bool(mask_bool.any()):
            raise ValueError(
                "UnifiedJEPALoss: mask contains no masked tokens — the "
                "masked-token objective is undefined at 0% masking."
            )
        L_raw = F.mse_loss(z_hat_repr[mask_bool], z_y[mask_bool])
        L_raw_w = self.lambda_raw * L_raw

        # Scalar L1 on unknown positions
        L_scalar = self.scalar_loss(out["scalar_pred"], scalar_values, scalar_known)

        # Occupancy BCE is the direct supervision for the decoded occupancy.
        # It is restricted to hidden pixels so visible-pixel retention remains
        # an inference contract rather than a shortcut in the loss.
        L_occ = occupancy_reconstruction_loss(
            out["occupancy_logits"], occupancy, mask=mask
        )
        L_occ_w = self.lambda_occ * L_occ
        occ_metrics = occupancy_reconstruction_metrics(
            out["occupancy_logits"], occupancy
        )

        # Physics loss: decode geometry → surrogate → spectrum error (Phase 4 MD §4).
        # Reuses the ALREADY-COMPUTED out (z_hat/scalar_pred) via
        # physics_loop.physics_loss_from_out — exactly one student forward per
        # step, one physics decode, one surrogate forward (Fix 11). Delegates
        # to the single authoritative physics implementation.
        if compute_physics and self.lambda_phys > 0 and self.surrogate is not None:
            from physics.physics_loop import physics_loss_from_out

            L_phys, _, _ = physics_loss_from_out(
                model,
                out,
                self.surrogate,
                occupancy,
                scalar_values,
                scalar_known,
                spectrum,
                mask,
                loss_type="smooth_l1",
                use_ste=self.physics_use_ste,
                normalize=True,
                hard_forward=physics_hard_forward,
            )
        else:
            L_phys = self.physics_loss(out.get("spectrum_target", spectrum), spectrum)

        # Goal-requirement margin loss (opt-in, default OFF). Second student
        # forward under the deranged goal; hinge encourages the masked-token
        # prediction to move with the goal by at least `goal_margin`.
        if self.lambda_goal > 0:
            if spectrum_shuf is None:
                raise ValueError(
                    "UnifiedJEPALoss: lambda_goal>0 requires spectrum_shuf "
                    "(deranged goal control) — refusing to train a goal term "
                    "with no counterfactual."
                )
            out_shuf = model(
                occupancy,
                scalar_values,
                scalar_known,
                spectrum_shuf,
                mask,
                goal_mode="real",
                with_target=False,
            )
            z_hat_shuf = out_shuf.get("z_hat_base", out_shuf["z_hat"])
            sens = (
                (z_hat_repr[mask_bool] - z_hat_shuf[mask_bool])
                .norm(dim=-1)
                .mean()
            )
            L_goal = F.relu(
                torch.as_tensor(self.goal_margin, device=sens.device)
                - sens
            )
        else:
            L_goal = torch.zeros((), device=z_hat_repr.device)
        L_goal_w = self.lambda_goal * L_goal

        total = (
            L_inv_w
            + L_var_w
            + L_cov_w
            + self.lambda_scalar * L_scalar
            + L_occ_w
            + self.lambda_phys * L_phys
            + L_raw_w
            + L_goal_w
        )

        out["loss_components"] = {
            "L_inv": float(L_inv.detach()),
            "L_var": float(L_var.detach()),
            "L_cov": float(L_cov.detach()),
            "L_scalar": float(L_scalar.detach()),
            "L_phys": float(L_phys.detach()),
            "L_occ": float(L_occ.detach()),
            "L_raw": float(L_raw.detach()),
            "L_goal": float(L_goal.detach()),
            "L_inv_weighted": float(L_inv_w.detach()),
            "L_var_weighted": float(L_var_w.detach()),
            "L_cov_weighted": float(L_cov_w.detach()),
            "L_occ_weighted": float(L_occ_w.detach()),
            "L_phys_weighted": float((self.lambda_phys * L_phys).detach()),
            "L_raw_weighted": float(L_raw_w.detach()),
            "L_goal_weighted": float(L_goal_w.detach()),
            "L_total": float(total.detach()),
        }
        out["occupancy_metrics"] = {
            k: float(v.detach()) for k, v in occ_metrics.items()
        }
        out["physics_computed"] = bool(
            compute_physics and self.lambda_phys > 0 and self.surrogate is not None
        )
        return {
            "total_loss": total,
            "components": out["loss_components"],
            "out": out,
            "projector_inputs": {"z_hat": z_hat_repr, "z_y": z_y},
            "projector_outputs": {"p_hat": p_hat_full, "p_y": p_y_full},
        }

    def train(self, mode=True):
        """Override: the frozen MetaDiT surrogate registered as self.surrogate
        must stay in EVAL mode regardless of the objective's training mode.

        Diagnostic-protocol finding: objective.train() recursively put the
        surrogate's 38 BatchNorm2d layers into train mode, so every training
        physics forward normalized by BATCH statistics (batch size 2) and
        mutated BN running stats — the physics loss trained against a
        corrupted surrogate, and surrogate outputs for identical geometry
        differed before/after the first train-mode call (measured: spectrum
        error 19.58 eval vs 0.59 train-BN for the same geometry). Eval-mode
        BN remains differentiable w.r.t. its input, so the physics gradient
        path is unchanged.
        """
        super().train(mode)
        if self.surrogate is not None:
            self.surrogate.eval()
        return self

    def on_optimizer_step(self, model, step):
        """Update both EMA targets after optimizer step (Phase 2 §6)."""
        model.ema.update(model.occupancy_encoder, step)
        model.scalar_mlp_ema.update(model.scalar_encoder, step)
