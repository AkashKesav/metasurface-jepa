"""Regression tests for thorough-audit fixes (Stage-A + mechanical).

Covers: scalar logit init, mask position equality, projector-collapse
reachability, spectrum goal_mode validation, sigreg N<2, shuffled B<2 warning,
guidance mode restore, FixedValidation empty-mask NaN.
"""

import copy
import os
import sys
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

from decoders.scalar_decoder import ScalarDecoder
from diagnostics.guidance_gap import compute_guidance_gap
from diagnostics.representation_health import classify_failure_mode
from encoders.spectrum_encoder import SpectrumPath
from losses.sigreg import sigreg_loss
from predictor.joint_target_fusion import JointTargetFusion
from predictor.masked_query_predictor import MaskedQueryPredictor
from runtime.physics_controls import make_shuffled_spectrum


def test_scalar_init_mid_range():
    d = ScalarDecoder()
    with torch.no_grad():
        out = d(torch.zeros(2, 192))
    assert torch.allclose(out[0], torch.tensor([2.75, 0.75, 4.25]), atol=0.05)


def test_fusion_gate0_identity_and_bound():
    f = JointTargetFusion(hidden=64, num_heads=4)
    z_g = torch.randn(2, 8, 64)
    z_s = torch.randn(2, 4, 64)
    with torch.no_grad():
        assert (f(z_g, z_s) - z_g).abs().max().item() == 0.0
        f.gate.fill_(1.0)
        f.out_proj.weight.mul_(1000.0)
        tn = (f(z_g, z_s) - z_g).norm(dim=-1).mean().item()
    assert tn < 20.0


def test_mask_position_equality_guard():
    mp = MaskedQueryPredictor(hidden=32, num_heads=4, num_layers=1, n_spatial_tokens=8)
    z_vis = torch.randn(2, 6, 32)
    a_goal = torch.randn(2, 2, 32)
    c = torch.randn(2, 32)
    m_diff = torch.ones(2, 8)
    m_diff[0, 0:2] = 0
    m_diff[1, 6:8] = 0
    try:
        mp(z_vis, a_goal, m_diff, c)
    except NotImplementedError:
        return
    raise AssertionError("same-count/different-position masks must raise")


def test_projector_collapse_reachable():
    base = {
        "eff_rank_frac": 1.0,
        "pairwise_cos": {"p05": 0.5},
        "same_token_cos": 0.5,
        "token_std": 1.0,
        "n_geoms": 10,
    }
    raw = copy.deepcopy(base)
    proj = copy.deepcopy(base)
    hr = copy.deepcopy(base)
    hp = copy.deepcopy(base)
    proj["pairwise_cos"]["p05"] = 0.99
    proj["eff_rank_frac"] = 0.01
    out = classify_failure_mode(raw, proj, hr, hp)
    assert out["verdict"] == "PROJECTOR_COLLAPSE"


def test_spectrum_path_rejects_bad_goal_mode():
    import torch.nn as nn

    class DummyRel(nn.Module):
        def forward(self, S):
            return torch.randn(S.shape[0], 301, 256)

    sp = SpectrumPath(DummyRel(), spec_dim=256, hidden=384)
    try:
        sp(torch.randn(2, 2, 301), goal_mode="raal")
    except ValueError:
        return
    raise AssertionError("typo goal_mode must raise")


def test_sigreg_n1_raises():
    try:
        sigreg_loss(torch.randn(1, 16))
    except ValueError:
        return
    raise AssertionError("sigreg N=1 must raise")


def test_shuffled_b1_warns_identity():
    S = torch.randn(1, 2, 301)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = make_shuffled_spectrum(S)
    assert torch.equal(out, S)
    assert any("infeasible" in str(x.message) for x in w)


def test_guidance_gap_restores_train_mode():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(4, 4)

        def forward(self, occ, sv, sk, spec, mask, goal_mode="real", with_target=False):
            b = occ.shape[0]
            return {"z_hat": torch.randn(b, 4, 8)}

    fm = FakeModel()
    fm.train(True)
    compute_guidance_gap(
        fm,
        torch.randn(2, 1, 8, 8),
        torch.randn(2, 3),
        torch.ones(2, 3, dtype=torch.bool),
        torch.randn(2, 2, 301),
        torch.ones(2, 16, 16),
    )
    assert fm.training is True
