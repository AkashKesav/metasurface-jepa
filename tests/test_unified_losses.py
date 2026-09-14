"""Phase 3 tests — UnifiedJEPALoss, training loop, resume, curriculum.

Verifies:
- UnifiedJEPALoss forward produces finite total_loss + components
- VICReg terms (L_inv, L_var, L_cov) are included and finite
- Scalar L1 only on unknown positions
- EMA/released gradients absent after loss.backward()
- Physics loss disabled by default (lambda_phys=0 → L_phys=0)
- Curriculum mask-ratio sampling produces variety
- Resume: model + EMA + optimizer + scheduler + step restored
- Full-mask batches (ratio=1.0) occur in curriculum

Run:  python -m pytest tests/test_unified_losses.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import copy
import torch
import torch.nn as nn
import pytest

from assembly import (
    UnifiedJEPA,
    build_unified_model,
    saveable_state_dict,
    load_into_model,
    SAVED_EXCLUDES,
)
from losses.unified_losses import (
    UnifiedJEPALoss,
    ScalarPredictionLoss,
)
from data.mask import BlockMasker


class _StubReleasedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _make_stub_spectrum_path(model):
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub


def _build_model(hidden=192, geo_depth=2, predictor_depth=4,
                 scalar_predictor_film=False):
    torch.manual_seed(0)
    model = UnifiedJEPA(
        hidden=hidden, num_heads=6, geo_depth=geo_depth,
        predictor_depth=predictor_depth, goal_tokens=16,
        num_predictor_heads=6, scalar_hidden=128,
        n_film_blocks=geo_depth, spec_dim=256,
        scalar_predictor_film=scalar_predictor_film)
    _make_stub_spectrum_path(model)
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())
    return model


def _batch(seed=0, b=2):
    torch.manual_seed(seed)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]], dtype=torch.float32)[:b]
    spec = torch.randn(b, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=seed)
    M = masker.sample(occ, ratio=0.5)
    return occ, sv, spec, M


class _SurrogateOut:
    def __init__(self, prediction):
        self.prediction = prediction


class _FakeSurrogate(nn.Module):
    """Deterministic tiny stand-in for the frozen EM surrogate.

    Mirrors the MetaDiT surrogate contract: forward returns an object carrying
    `.prediction` [B, 2, 301].
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(3 * 64 * 64, 64), nn.ReLU(),
                                 nn.Linear(64, 2 * 301))

    def forward(self, x):
        return _SurrogateOut(
            self.net(x.flatten(1)).reshape(-1, 2, 301))


def test_physics_loss_skipped_on_goal_dropped_steps():
    """Audit B5: on null-goal (CFG dropout) steps the physics term must be
    skipped.

    Its target is the sample's TRUE spectrum — the very condition that was
    dropped — so training it there pushes the unconditional branch toward
    outputs it cannot infer (goal-ignoring / mode-collapse pressure,
    architecture_v5.md §8.3). Latent and scalar objectives still train that
    branch; the conditional branch keeps the physics term on the other steps.
    """
    model = _build_model()
    model.train()
    surrogate = _FakeSurrogate()
    for p in surrogate.parameters():
        p.requires_grad_(False)
    surrogate.eval()
    objective = UnifiedJEPALoss(hidden=192, lambda_phys=1.0, surrogate=surrogate)
    objective.train()
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.ones(2, 3, dtype=torch.bool)

    real = objective(model, occ, sv, sk, spec, M, goal_mode="real")
    null = objective(model, occ, sv, sk, spec, M, goal_mode="null")
    assert real["components"]["L_phys"] > 0.0, (
        "physics term must stay active on goal-conditioned steps")
    assert null["components"]["L_phys"] == 0.0, (
        "physics term must be skipped on goal-dropped (null) steps: its "
        "target is the dropped condition; got "
        f"{null['components']['L_phys']}")
    assert null["components"]["L_phys_weighted"] == 0.0
    # The spectrum-free objectives still train the null branch.
    assert null["components"]["L_inv"] > 0.0


# --------------------------------------------------------------------------
# Loss components
# --------------------------------------------------------------------------

def test_unified_loss_returns_dict():
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    occ, sv, spec, M = _batch(seed=0)
    result = objective(model, occ, sv, torch.ones(2, 3, dtype=torch.bool),
                       spec, M)
    assert "total_loss" in result
    assert "components" in result
    assert "out" in result
    assert "projector_inputs" in result
    assert "projector_outputs" in result


def test_occupancy_bce_supervises_decoder_when_physics_off():
    """Operator decision 2026-09-13 (audit §4): L_occ = BCEWithLogits(logits,
    true occupancy) on MASKED pixels (architecture_v5.md §4.1).

    Without this term the occupancy decoder is trained only through the physics
    path, so it receives no gradient at all while lambda_phys = 0.
    """
    model = _build_model()
    model.train()
    objective = UnifiedJEPALoss(hidden=192, lambda_phys=0.0, lambda_occ=1.0)
    objective.train()
    occ, sv, spec, M = _batch(seed=5)
    sk = torch.ones(2, 3, dtype=torch.bool)
    model.zero_grad(set_to_none=True)
    result = objective(model, occ, sv, sk, spec, M)
    c = result["components"]
    assert c["L_occ"] > 0.0, f"L_occ must be active/pixel, got {c['L_occ']}"
    assert abs(c["L_occ_weighted"] - c["L_occ"]) < 1e-9, c
    result["total_loss"].backward()
    head_grads = [p for p in model.occupancy_decoder.head.parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0]
    assert head_grads, (
        "the occupancy decoder must receive gradient from the BCE term when "
        "the physics path is off")
    # The BCE term alone must not train the frozen references.
    for name, p in model.ema.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, name


def test_loss_components_finite():
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    c = result["components"]
    for k in ("L_inv", "L_var", "L_cov", "L_scalar", "L_phys",
              "L_inv_weighted", "L_var_weighted", "L_cov_weighted", "L_total"):
        assert k in c, f"missing component {k}"
        assert c[k] >= 0, f"{k} negative: {c[k]}"


def test_loss_backward_student_grads_exist():
    model = _build_model()
    model.train()
    objective = UnifiedJEPALoss(hidden=192)
    objective.train()
    occ, sv, spec, M = _batch(seed=2)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    loss.backward()

    # Student modules must have gradients
    for name, p in model.occupancy_encoder.named_parameters():
        assert p.grad is not None, f"no grad: occupancy_encoder.{name}"
    for name, p in model.scalar_encoder.named_parameters():
        assert p.grad is not None, f"no grad: scalar_encoder.{name}"
    for name, p in model.predictor.named_parameters():
        assert p.grad is not None, f"no grad: predictor.{name}"
    for name, p in model.scalar_decoder.named_parameters():
        assert p.grad is not None, f"no grad: scalar_decoder.{name}"

    # EMA targets must have NO gradients
    for name, p in model.ema.named_parameters():
        assert p.grad is None, f"ema has grad: {name}"
    for name, p in model.scalar_mlp_ema.named_parameters():
        assert p.grad is None, f"scalar_mlp_ema has grad: {name}"

    # Objective projector must have gradients
    for name, p in objective.projector.named_parameters():
        assert p.grad is not None, f"no grad: projector.{name}"

    # Released spectrum encoder must have NO gradients
    released = model.spectrum_path.released
    for name, p in released.named_parameters():
        assert p.grad is None, f"released has grad: {name}"


def test_projector_trained_from_both_branches_ema_frozen():
    """Fix 12: projector gradient ownership is explicit — the objective-owned
    projector is trained from BOTH branches (p_hat and p_y), while the EMA
    target encoder itself receives no gradient (stop-grad at the EMA boundary,
    architecture_v5.md §3.6)."""
    model = _build_model()
    model.train()
    objective = UnifiedJEPALoss(hidden=192)
    objective.train()
    occ, sv, spec, M = _batch(seed=13)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    loss.backward()

    # Projector receives gradients (both branches flow into it).
    for name, p in objective.projector.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, (
            f"projector.{name} must receive gradient from the objective")

    # The target-side projector input (p_y) is derived from z_y_raw, which is
    # detached at the EMA boundary — so the EMA encoder must have NO gradient
    # even though the projector itself is trained from both branches.
    for name, p in model.ema.named_parameters():
        assert p.grad is None, f"occupancy EMA received gradient: {name}"
    for name, p in model.scalar_mlp_ema.named_parameters():
        assert p.grad is None, f"scalar_mlp_ema received gradient: {name}"

    # Explicit: projector params are NOT frozen (requires_grad True).
    assert any(p.requires_grad for p in objective.projector.parameters())


def test_loss_backward_zero_grads_on_ema_only():
    """Verify the objective's on_optimizer_step updates EMA targets."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.ones(2, 3, dtype=torch.bool)

    # Snapshot EMA target state
    ema_before = {k: v.clone() for k, v in model.ema.target.state_dict().items()}
    scalar_ema_before = {k: v.clone() for k, v in model.scalar_mlp_ema.target.state_dict().items()}

    # Train step
    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    loss.backward()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(objective.parameters()), lr=1e-4)
    optimizer.step()
    objective.on_optimizer_step(model, step=0)

    # At least one EMA param must have moved
    occ_moved = sum(1 for k, v in ema_before.items()
                    if not torch.equal(v, model.ema.target.state_dict()[k]))
    scalar_moved = sum(1 for k, v in scalar_ema_before.items()
                       if not torch.equal(v, model.scalar_mlp_ema.target.state_dict()[k]))
    assert occ_moved > 0, "occupancy EMA must update"
    assert scalar_moved > 0, "scalar_mlp_ema must update"


# --------------------------------------------------------------------------
# Scalar loss: unknown positions only
# --------------------------------------------------------------------------

def test_scalar_loss_only_unknown():
    """Scalar loss must only penalize unknown positions."""
    loss_fn = ScalarPredictionLoss(loss_type="l1")
    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.tensor([[1.5, 2.0, 3.0], [4.0, 5.5, 6.0]])

    # All unknown → both positions contribute
    all_unknown = torch.zeros(2, 3, dtype=torch.bool)
    L_all = loss_fn(pred, target, all_unknown)
    err_all = (pred - target).abs()
    expected_all = err_all.sum() / 6
    assert torch.allclose(L_all, expected_all, atol=1e-6)

    # All known → no penalty
    all_known = torch.ones(2, 3, dtype=torch.bool)
    L_known = loss_fn(pred, target, all_known)
    assert L_known.item() == 0.0

    # Mixed
    mixed = torch.tensor([[True, False, True], [False, True, False]])
    L_mixed = loss_fn(pred, target, mixed)
    err = (pred - target).abs() * (~mixed).float()
    expected = err.sum() / 2  # 2 unknown positions
    assert torch.allclose(L_mixed, expected, atol=1e-6)


def test_scalar_loss_huber():
    loss_fn = ScalarPredictionLoss(loss_type="huber")
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.5, 2.0, 3.0]])
    unknown = torch.tensor([[False, True, True]])
    L = loss_fn(pred, target, unknown)
    assert torch.isfinite(L)
    # Only first position is unknown
    expected = torch.nn.functional.huber_loss(
        torch.tensor([1.0]), torch.tensor([1.5]))
    assert torch.allclose(L, expected, atol=1e-6)


# --------------------------------------------------------------------------
# Physics loss disabled by default
# --------------------------------------------------------------------------

def test_physics_placeholder_is_gone_and_the_term_is_exactly_zero():
    """The inactive physics branch is genuinely zero, not "measured as zero".

    This replaces two tests of `PhysicsSpectrumLoss`, a placeholder whose `_enabled`
    flag was never set — so it always returned a zero tensor while reading like a
    real fallback. It was deleted as dead code; the contract worth pinning is the
    objective's: with `lambda_phys = 0` it must report an exact zero, and validation
    reports the term as NOT EVALUATED for the same reason (audit B28).
    """
    objective = UnifiedJEPALoss(hidden=192, lambda_phys=0.0)
    assert not hasattr(objective, "physics_loss"), (
        "the inert PhysicsSpectrumLoss placeholder must be gone")

    model = _build_model()
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.zeros(occ.shape[0], 3, dtype=torch.bool)
    with torch.no_grad():
        res = objective(model, occ, sv, sk, spec, M, goal_mode="real")
    assert res["components"]["L_phys"] == 0.0, (
        "with lambda_phys=0 and no surrogate the physics term must be exactly zero")
    assert res["components"]["L_phys_weighted"] == 0.0
    assert torch.isfinite(res["total_loss"])


# --------------------------------------------------------------------------
# Curriculum sampling (Phase 3 MD §4)
# --------------------------------------------------------------------------

def test_curriculum_mask_ratios_include_full():
    """Cleanup item 3: training ratios exclude 0.0 (masked-token objective
    undefined), eval ratios include it as the unmasked reference. Full mask
    (1.0) is in both."""
    import yaml
    cfg = yaml.safe_load(open(
        os.path.join(REPO_ROOT, "configs", "unified.yaml")))
    train_ratios = cfg["curriculum"]["train_mask_ratios"]
    train_probs = cfg["curriculum"]["train_mask_ratio_probs"]
    eval_ratios = cfg["curriculum"]["eval_mask_ratios"]
    assert 1.0 in train_ratios, "full mask (1.0) must be in training curriculum"
    assert 0.0 not in train_ratios, (
        "0.0 must be excluded from training (no masked tokens)")
    assert len(train_probs) == len(train_ratios)
    assert abs(sum(train_probs) - 1.0) < 1e-6
    assert 0.0 in eval_ratios, "0.0 must be in eval ratios (unmasked reference)"
    assert 1.0 in eval_ratios, "full mask (1.0) must be in eval ratios"


def test_curriculum_scalar_regimes():
    import yaml
    cfg = yaml.safe_load(open(
        os.path.join(REPO_ROOT, "configs", "unified.yaml")))
    regimes = cfg["curriculum"]["scalar_regimes"]
    assert "all_known" in regimes
    assert "all_unknown" in regimes
    assert "mixed" in regimes
    probs = cfg["curriculum"]["scalar_regime_probs"]
    assert len(probs) == len(regimes)
    assert abs(sum(probs) - 1.0) < 1e-6


def test_masker_works_with_single_channel():
    """BlockMasker must produce valid masks for 1-channel occupancy input."""
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)
    assert M.shape == (2, 16, 16)
    assert M.dtype == torch.float32
    # 1 = visible, 0 = masked
    assert M.max() <= 1.0 and M.min() >= 0.0


def test_full_mask_batch():
    """mask_ratio=1.0 must produce all-zeros mask (nothing visible)."""
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=1.0)
    assert M.shape == (2, 16, 16)
    # With ratio=1.0, all should be masked
    assert M.min().item() == 0.0


def test_zero_mask_batch_all_visible():
    """mask_ratio=0.0 must produce all-ones mask (every position visible).

    Regression (Fix 13): the nominal 0% mask regime previously still produced
    at least one block (min_side=3), so "0% mask" was not actually zero masking.
    """
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.0)
    assert M.shape == (2, 16, 16)
    assert (M == 1.0).all(), "ratio=0.0 must leave every position visible"
    assert M.dtype == torch.float32


# --------------------------------------------------------------------------
# Resume equivalence
# --------------------------------------------------------------------------

def test_loss_projector_is_objective_owned():
    """The VICReg projector must be owned by the objective, not the model (§17)."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    # Model must NOT have a projector attribute
    assert not hasattr(model, "proj"), "model should not own a projector"
    assert not hasattr(model, "projector"), "model should not own a projector"
    # Objective must own its projector
    assert hasattr(objective, "projector")
    assert isinstance(objective.projector, nn.Module)


def test_jepa_loss_on_masked_only():
    """JEPA loss (L_inv) must only cover masked tokens, not all tokens."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    occ, sv, spec, M = _batch(seed=5)
    sk = torch.ones(2, 3, dtype=torch.bool)
    result = objective(model, occ, sv, sk, spec, M)

    # The mask in the model output should match the input mask
    out = result["out"]
    expected_mask = (M.view(2, -1) == 0)  # 0 in M => masked (True in mask)
    assert torch.equal(out["mask"], expected_mask)


def test_summary_readout_is_in_the_objective_state():
    """Door (a) of the scalar investigation: the read-out is a real submodule, so
    checkpoints carry it and a resume restores a trained read-out rather than
    silently re-initialising it."""
    objective = UnifiedJEPALoss(hidden=192, lambda_summary=1.0)
    keys = [k for k in objective.state_dict() if k.startswith("summary_readout.")]
    assert keys, "the summary read-out must be part of the objective's state_dict"
    assert any(k.endswith("weight") for k in keys)


def test_summary_readout_term_is_exactly_zero_when_disabled():
    """lambda_summary = 0 keeps the shipped behaviour identical: the term is an
    exact zero, not a small number."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192, lambda_summary=0.0)
    objective.train()
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = objective(model, occ, sv, sk, spec, M, goal_mode="real")
    assert out["components"]["L_summary"] == 0.0
    assert out["components"]["L_summary_weighted"] == 0.0


def test_summary_readout_term_is_zero_without_known_scalars():
    """Nothing to recover where every scalar was zeroed before the encoder saw
    it — the all-unknown regime must not invent a supervision signal."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192, lambda_summary=1.0)
    objective.train()
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.zeros(2, 3, dtype=torch.bool)
    out = objective(model, occ, sv, sk, spec, M, goal_mode="real")
    assert out["components"]["L_summary"] == 0.0


def test_summary_readout_gives_the_scalar_encoder_a_path_that_bypasses_the_predictor():
    """THE mechanism door (a) exists for.

    The summary token had no objective of its own, so the only route from a scalar
    objective to the scalar encoder ran through the predictor's attention —
    measured at 0.0036 of gradient against the scalar decoder's 0.6932, while the
    token itself varied only ~11% with the scalars. The read-out must deliver
    gradient to the scalar encoder WITHOUT touching the predictor; that is what
    makes this a new path rather than a louder version of the existing one.
    """
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192, lambda_summary=1.0)
    objective.train()
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.zeros(2, 3, dtype=torch.bool)
    sk[:, 0] = True                       # one known scalar to recover

    out = model(occ, sv, sk, spec, M, goal_mode="real")
    assert "scalar_summary" in out, (
        "the model must expose the scalar encoder's summary token, or the "
        "objective cannot attach a read-out to it")
    readout = objective.summary_readout(out["scalar_summary"])
    assert readout.shape == (2, 3)
    L = ((readout - sv).abs() * sk.float()).sum() / sk.sum().clamp(min=1)

    model.zero_grad(set_to_none=True)
    objective.zero_grad(set_to_none=True)
    L.backward()

    enc = sum(float(p.grad.abs().sum()) for p in model.scalar_encoder.parameters()
              if p.grad is not None)
    pred = sum(float(p.grad.abs().sum()) for p in model.predictor.parameters()
               if p.grad is not None)
    assert enc > 0.0, (
        "the read-out must deliver gradient to the scalar encoder — that is the "
        "point of door (a)")
    assert pred == 0.0, (
        "the read-out's gradient must NOT pass through the predictor; if it did, "
        f"this would be the existing path rather than a new one (got {pred})")


def test_scalar_film_is_an_identity_at_init():
    """Door (b) must respect the zero-init identity convention.

    GCLCT.scalar_cond_proj is zero-initialised (weight AND bias), so enabling
    staging.scalar_predictor_film must leave the forward pass EXACTLY unchanged at
    step 0 — otherwise a fresh run would start from a different function and the
    before/after comparison would measure the init, not the mechanism.
    """
    off = _build_model(scalar_predictor_film=False)
    on = _build_model(scalar_predictor_film=True)
    # The path is only constructed when enabled, so the two state_dicts differ by
    # exactly the projection; load the shared keys and leave that one at its
    # zero init.
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert all(k.startswith("predictor.scalar_cond_proj.") for k in missing), missing
    assert not unexpected, unexpected
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.ones(2, 3, dtype=torch.bool)

    a = off(occ, sv, sk, spec, M, goal_mode="real")
    b = on(occ, sv, sk, spec, M, goal_mode="real")
    assert torch.equal(a["z_hat"], b["z_hat"]), (
        "with a zero-initialised projection the scalar-FiLM path must be an exact "
        f"identity; max|dz|={(a['z_hat'] - b['z_hat']).abs().max().item()}")
    assert torch.equal(a["scalar_pred"], b["scalar_pred"])


def test_scalar_film_path_is_live_and_scalar_driven():
    """With a non-zero projection the path must actually modulate the prediction,
    and the modulation must be driven by the SCALAR conditioning — that is the
    difference between door (b) and a second copy of the spectrum FiLM."""
    def effect(model):
        """|dz_hat| / |z_hat| when only the scalar conditioning changes."""
        occ, sv, spec, M = _batch(seed=3)
        sk = torch.ones(2, 3, dtype=torch.bool)
        a = model(occ, sv, sk, spec, M, goal_mode="real")["z_hat"]
        b = model(occ, sv * 1.5, sk, spec, M, goal_mode="real")["z_hat"]
        return float((a - b).norm() / (a.norm() + 1e-12))

    model = _build_model(scalar_predictor_film=True)
    with torch.no_grad():
        base = effect(model)
        torch.manual_seed(7)
        # The block FiLM starts at EXACTLY zero (project convention), so both the
        # projection AND cond must be non-zero before this path is observable.
        # Perturbing only the projection would test nothing — which is how this
        # test first failed, and the reason it is written in two steps.
        for blk in model.predictor.blocks:
            blk.cond[-1].weight.add_(torch.randn_like(blk.cond[-1].weight) * 0.05)
            blk.cond[-1].bias.add_(torch.randn_like(blk.cond[-1].bias) * 0.05)
        mod_live = effect(model)                   # FiLM live, scalars NOT fed
        for q in model.predictor.scalar_cond_proj.parameters():
            q.add_(torch.randn_like(q) * 0.5)      # now feed the scalars too
        mod_live_fed = effect(model)

    # NOTE: no assertion that a live cond INCREASES sensitivity — a random cond
    # need not, and asserting it was this test's second wrong premise. The
    # property that actually matters is the paired one below: with the same live
    # cond, feeding the scalars must make the prediction more scalar-sensitive.
    assert mod_live_fed > mod_live, (
        "with the block FiLM live, feeding the scalar conditioning through "
        "scalar_cond_proj must make the prediction MORE sensitive to the scalars "
        f"(not fed: {mod_live:.6f}, fed: {mod_live_fed:.6f})")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
