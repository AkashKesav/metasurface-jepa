"""Regression tests for thorough-audit fixes (Stage-A + mechanical).

Covers: scalar logit init, mask position equality, projector-collapse
reachability, spectrum goal_mode validation, sigreg N<2, shuffled B<2 warning,
and guidance mode restore. (FixedValidation empty-mask NaN is covered by
convention-match with eval_checkpoint_latents.py; no dedicated test here.)
"""

import copy
import os
import sys
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn

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


def _tiny_goal_model():
    # Local imports: module-level `assembly` import is stripped by the repo
    # linter (unresolvable first-party at lint time), so import here.
    from assembly import UnifiedJEPA
    from data.mask import BlockMasker

    torch.manual_seed(0)
    model = UnifiedJEPA(hidden=192, num_heads=6, geo_depth=2, predictor_depth=2)

    class _StubRel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))

        def forward(self, S):
            return self.net(S.transpose(1, 2))

    stub = _StubRel()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub  # type: ignore[assignment]  # test stub
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[2.75, 0.75, 4.25], [2.75, 0.75, 4.25]])
    sk = torch.ones(2, 3, dtype=torch.bool)
    spec = torch.randn(2, 2, 301)
    masker = BlockMasker(
        placement="random", grid=16, min_side=3, k_range=(1, 4), seed=0
    )
    M = masker.sample(occ, ratio=0.5)
    return model, occ, sv, sk, spec, M


def test_goal_term_off_by_default():
    from losses.unified_losses import UnifiedJEPALoss

    obj = UnifiedJEPALoss(hidden=192)
    assert obj.lambda_goal == 0.0
    model, occ, sv, sk, spec, M = _tiny_goal_model()
    model.eval()
    with torch.no_grad():
        r = obj(model, occ, sv, sk, spec, M, goal_mode="real", compute_physics=False)
    assert r["components"]["L_goal"] == 0.0
    assert r["components"]["L_goal_weighted"] == 0.0


def test_goal_term_requires_shuffled():
    from losses.unified_losses import UnifiedJEPALoss

    obj = UnifiedJEPALoss(hidden=192, lambda_goal=2.0, goal_margin=0.01)
    model, occ, sv, sk, spec, M = _tiny_goal_model()
    model.eval()
    try:
        with torch.no_grad():
            obj(model, occ, sv, sk, spec, M, goal_mode="real", compute_physics=False)
    except ValueError:
        return
    raise AssertionError("lambda_goal>0 without spectrum_shuf must raise")


def test_goal_term_hinge_behavior():
    from losses.unified_losses import UnifiedJEPALoss

    obj = UnifiedJEPALoss(hidden=192, lambda_goal=2.0, goal_margin=0.01)
    model, occ, sv, sk, spec, M = _tiny_goal_model()
    model.eval()
    with torch.no_grad():
        # Identical control -> zero sensitivity -> hinge pays full margin.
        r_same = obj(
            model,
            occ,
            sv,
            sk,
            spec,
            M,
            goal_mode="real",
            compute_physics=False,
            spectrum_shuf=spec.clone(),
        )
        assert abs(r_same["components"]["L_goal"] - 0.01) < 1e-6
        assert abs(r_same["components"]["L_goal_weighted"] - 0.02) < 1e-6
        # Deranged control -> bounded hinge in [0, margin].
        shuf = spec[torch.tensor([1, 0])]
        r_shuf = obj(
            model,
            occ,
            sv,
            sk,
            spec,
            M,
            goal_mode="real",
            compute_physics=False,
            spectrum_shuf=shuf,
        )
        g = r_shuf["components"]["L_goal"]
        assert 0.0 <= g <= 0.01
        assert abs(r_shuf["components"]["L_goal_weighted"] - 2.0 * g) < 1e-9
        # Weighted goal participates in the total.
        assert r_shuf["components"]["L_total"] > 0.0


def test_repo_text_files_utf8():
    # Regression guard for the 2026-09-12 Run-A Kaggle outage: two configs
    # generated via locale-encoded open() carried a cp1252 em-dash (0x97)
    # that decoded fine on Windows but crashed yaml.safe_load on Linux,
    # killing three cloud runs with zero logs. All first-party text files
    # must be strict UTF-8.
    import io
    roots = ['configs', 'src', 'scripts', 'tests']
    exts = ('.py', '.yaml', '.yml', '.md', '.json')
    bad = []
    for root in roots:
        for dirpath, dirnames, filenames in __import__('os').walk(root):
            dirnames[:] = [d for d in dirnames if d != '__pycache__']
            for fn in filenames:
                if fn.endswith(exts):
                    p = __import__('os').path.join(dirpath, fn)
                    try:
                        io.open(p, 'rb').read().decode('utf-8')
                    except UnicodeDecodeError:
                        bad.append(p)
    assert not bad, f'non-UTF8 text files: {bad}'


def test_factorize_or_skip_degenerate():
    # A batch with an occupied pixel but zero r/h scalars violates the
    # factorize invariant (Run-A cloud crash at step ~6900). The helper must
    # convert ONLY that AssertionError into _SkippedBatch; real bugs re-raise.
    import sys, os
    sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts', 'train'))
    from train_unified import _factorize_or_skip, _SkippedBatch
    import torch
    G = torch.zeros(2, 3, 64, 64)
    G[0, 1, 10, 10] = 0.5  # occupied via ch1, but ch0 amax = 0 -> r = 0
    G[0, 2, :, :] = 2.75 / 3.0
    G[1, 0, 20, 20] = 4.25 / 5.0
    G[1, 1, 20, 20] = 0.75
    G[1, 2, :, :] = 2.75 / 3.0
    try:
        _factorize_or_skip(G)
    except _SkippedBatch:
        return
    raise AssertionError('degenerate batch must raise _SkippedBatch')


def test_factorize_or_skip_healthy_passes():
    import sys, os
    sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts', 'train'))
    from train_unified import _factorize_or_skip, _SkippedBatch
    import torch
    G = torch.zeros(2, 3, 64, 64)
    G[:, 0, 10, 10] = 4.25 / 5.0
    G[:, 1, 10, 10] = 0.75
    G[:, 2, :, :] = 2.75 / 3.0
    occ, sv = _factorize_or_skip(G)
    assert occ.shape == (2, 1, 64, 64)
    assert sv.shape == (2, 3)
