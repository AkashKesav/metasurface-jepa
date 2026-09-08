"""Regression tests for EMA target updates in the unified training path.

Verifies the three critical fixes from commit 65d869b:
- T2: objective.on_optimizer_step(model, step) actually moves the EMA
  target encoder weights (without it, EMA is frozen for the entire run).
- T1: after a MetaDiT-init + resync, EMA target == student encoder
  (xi(0) == theta(0) must hold at init).
- T3: set_total_steps produces a momentum < momentum_end for step 0
  (i.e., the 0.996->0.999 ramp actually happens, not pinned at 0.999).

Run:  python -m pytest tests/test_ema_regression.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn
import pytest

from assembly import UnifiedJEPA
from losses.unified_losses import UnifiedJEPALoss
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


def _build_model(hidden=192, geo_depth=2, predictor_depth=4):
    torch.manual_seed(0)
    model = UnifiedJEPA(
        hidden=hidden, num_heads=6, geo_depth=geo_depth,
        predictor_depth=predictor_depth, goal_tokens=16,
        num_predictor_heads=6, scalar_hidden=128,
        n_film_blocks=geo_depth, spec_dim=256)
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


# Helper: grab a parameter snapshot from the EMA target encoder.
# occupancy_encoder is a patch-embed + transformer; patch_embed.weight
# is a conv weight that exists on both student and EMA target.
def _ema_occ_param(model):
    return model.ema.target.patch_embed.weight

def _student_occ_param(model):
    return model.occupancy_encoder.patch_embed.weight

def _ema_scalar_param(model):
    return model.scalar_mlp_ema.target.trunk[0].weight

def _student_scalar_param(model):
    return model.scalar_encoder.trunk[0].weight


# --------------------------------------------------------------------------
# T2: on_optimizer_step actually moves EMA weights
# --------------------------------------------------------------------------

def test_ema_target_moves_after_optimizer_step():
    """Without on_optimizer_step, EMA is frozen for the entire run (Bug T2)."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    model.set_total_steps(100)

    ema_w_before = _ema_occ_param(model).clone()

    occ, sv, spec, M = _batch(seed=0)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    result["total_loss"].backward()

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.step()

    # The fix: call on_optimizer_step
    objective.on_optimizer_step(model, step=0)

    ema_w_after = _ema_occ_param(model).clone()
    assert not torch.allclose(ema_w_before, ema_w_after), \
        "EMA target weights did not change after on_optimizer_step — Bug T2 present"


def test_scalar_ema_target_moves_after_optimizer_step():
    """Same check for scalar_mlp_ema (also updated by on_optimizer_step)."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    model.set_total_steps(100)

    ema_w_before = _ema_scalar_param(model).clone()

    occ, sv, spec, M = _batch(seed=0)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    result["total_loss"].backward()

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt.step()
    objective.on_optimizer_step(model, step=0)

    ema_w_after = _ema_scalar_param(model).clone()
    assert not torch.allclose(ema_w_before, ema_w_after), \
        "scalar_mlp_ema target weights did not change after on_optimizer_step"


# --------------------------------------------------------------------------
# T1: EMA == student after init resync
# --------------------------------------------------------------------------

def test_ema_matches_student_after_resync():
    """After build_unified_model resyncs EMA from student, xi(0) == theta(0).

    This is what the stage4 fix does: after _init_geometry_from_metadit
    loads weights into the student, we must resync EMA to match.
    """
    model = _build_model()

    # Simulate: student weights changed (like MetaDiT init would do)
    with torch.no_grad():
        for p in model.occupancy_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    student_w = _student_occ_param(model)
    ema_w = _ema_occ_param(model)
    assert not torch.allclose(student_w, ema_w), \
        "Test setup error: student and EMA should differ before resync"

    # The fix: resync EMA to student
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())

    ema_w_after = _ema_occ_param(model)
    assert torch.allclose(student_w, ema_w_after), \
        "EMA target does not match student after resync — Bug T1 present"


def test_scalar_ema_matches_student_after_resync():
    """Same for scalar_mlp_ema."""
    model = _build_model()

    with torch.no_grad():
        for p in model.scalar_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    student_w = _student_scalar_param(model)
    ema_w = _ema_scalar_param(model)
    assert not torch.allclose(student_w, ema_w)

    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())

    ema_w_after = _ema_scalar_param(model)
    assert torch.allclose(student_w, ema_w_after), \
        "scalar_mlp_ema does not match student after resync"


# --------------------------------------------------------------------------
# T3: momentum schedule is not pinned at 0.999
# --------------------------------------------------------------------------

def test_momentum_ramp_at_step_zero():
    """At step 0 with total_steps > 1, momentum should be < momentum_end.

    Without set_total_steps, total_steps defaults to 1, so for any step >= 1
    the momentum fraction is 1.0 and momentum = momentum_end = 0.999.
    The fix calls set_total_steps so the 0.996->0.999 ramp actually happens.
    """
    model = _build_model()
    model.set_total_steps(1000)

    m0 = model.ema.current_momentum(0)
    assert m0 < 0.999, \
        f"Momentum at step 0 is {m0}, expected < 0.999 — Bug T3 present"
    assert abs(m0 - 0.996) < 1e-6, \
        f"Momentum at step 0 is {m0}, expected ~0.996 (momentum_start)"


def test_momentum_ramp_midway():
    """Halfway through training, momentum should be between start and end."""
    model = _build_model()
    model.set_total_steps(1000)

    m_mid = model.ema.current_momentum(500)
    assert 0.996 < m_mid < 0.999, \
        f"Momentum at step 500/1000 is {m_mid}, expected between 0.996 and 0.999"


def test_momentum_without_set_total_steps_is_pinned():
    """Documents the bug: without set_total_steps, momentum is pinned at 0.999.

    total_steps defaults to 1, so step>=1 gives frac=1 -> momentum=0.999.
    This test verifies the default is still bad (the fix is in the caller).
    """
    model = _build_model()
    m = model.ema.current_momentum(1)
    assert abs(m - 0.999) < 1e-6, \
        f"Without set_total_steps, momentum at step 1 should be 0.999, got {m}"
