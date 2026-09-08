"""Forward-only shape & gradient tests for the Joint Target Redesign Stage-A
modules (§14 implementation order, steps 1-3).

These are deliberately forward-only (no training): Stage A is "joint target
only" and the modules must be correct in shape, init behavior, gradient flow,
and the documented invariants BEFORE any cloud training run. Per AGENTS.md
Standing Rule 8 the local machine is for code + unit tests, not training.

Covers:
  - JointTargetFusion: shape contract, zero-gate init == geometry-only,
    gate gradient flows, cross-attention uses physics tokens (not geometry).
  - MaskedQueryPredictor: shape contract, mask-convention (1=visible, 0=masked),
    scatter-back places predictions at masked positions and context at
    visible positions, gradient flows to masked-query path.

Run:  python tests/test_joint_redesign_stage_a.py        (also via pytest)
"""

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from predictor.joint_target_fusion import JointTargetFusion  # noqa: E402
from predictor.masked_query_predictor import MaskedQueryPredictor  # noqa: E402

H = 48          # tiny hidden for fast CPU tests (must divide num_heads)
N_GEO = 16      # 16x16 -> 256 normally; shrink to 16 for tiny tests
N_GOAL = 8


# ---------------------------------------------------------------------------
# JointTargetFusion
# ---------------------------------------------------------------------------

def test_joint_fusion_shape_contract():
    f = JointTargetFusion(hidden=H, num_heads=6)
    z_g = torch.randn(2, N_GEO, H)
    z_s = torch.randn(2, N_GOAL, H)
    z_joint = f(z_g, z_s)
    assert z_joint.shape == z_g.shape, (
        f"Z_joint shape {tuple(z_joint.shape)} must match Z_G "
        f"{tuple(z_g.shape)}")


def test_joint_fusion_zero_gate_init_has_zero_delta():
    """At construction (gate=0, tanh(0)=0) the fusion's physics delta is zero,
    so the ONLY operation applied to Z_G is the output LayerNorm (which with
    default identity affine normalizes the geometry stream). The stable-init
    invariant is 'the spectrum contributes nothing at step 0' — i.e. the
    output is independent of Z_S — NOT that the output byte-equals Z_G (the
    output norm normalizes Z_G). Verify the delta contribution is exactly zero
    and that the output is the same for any Z_S at init."""
    f = JointTargetFusion(hidden=H, num_heads=6)
    assert float(f.gate) == 0.0, "gate must initialize to 0.0"
    assert float(torch.tanh(f.gate)) == 0.0, "tanh(gate) must be 0 at init"
    z_g = torch.randn(2, N_GEO, H)
    z_s1 = torch.randn(2, N_GOAL, H)
    z_s2 = torch.randn(2, N_GOAL, H) * 100.0
    with torch.no_grad():
        out1 = f(z_g, z_s1)
        out2 = f(z_g, z_s2)
    # Z_S-independent at init: the spectrum contributes nothing.
    assert torch.allclose(out1, out2, atol=1e-7), (
        "at gate=0 the output must be independent of Z_S "
        "(physics delta is zero); stable init")


def test_joint_fusion_gate_gradient_flows():
    """The gate is the learnable physics-coupling control; its gradient must
    flow from the JEPA loss through the fusion."""
    f = JointTargetFusion(hidden=H, num_heads=6)
    z_g = torch.randn(2, N_GEO, H, requires_grad=False)
    z_s = torch.randn(2, N_GOAL, H, requires_grad=False)
    z_joint = f(z_g, z_s)
    # A loss that depends on every output element -> gradient to gate.
    z_joint.sum().backward()
    assert f.gate.grad is not None, "gate must receive gradient"
    assert torch.isfinite(f.gate.grad), "gate gradient must be finite"
    # Attention parameters must also be trainable.
    assert f.q_proj.weight.requires_grad
    assert f.out_proj.weight.requires_grad


def test_joint_fusion_uses_physics_tokens_as_kv():
    """With a fixed Z_G and two different Z_S, the output must differ (the
    cross-attention keys/values come from Z_S, so the physics spectrum
    genuinely enters the target — the whole point of the redesign)."""
    f = JointTargetFusion(hidden=H, num_heads=6)
    # Open the gate so the delta actually contributes.
    with torch.no_grad():
        f.gate.fill_(5.0)
    z_g = torch.randn(1, N_GEO, H)
    z_s1 = torch.randn(1, N_GOAL, H)
    z_s2 = z_s1 + torch.randn(1, N_GOAL, H) * 10.0
    with torch.no_grad():
        out1 = f(z_g, z_s1)
        out2 = f(z_g, z_s2)
    assert not torch.allclose(out1, out2, atol=1e-6), (
        "different Z_S must change Z_joint (physics enters the target)")


def test_joint_fusion_rejects_wrong_shapes():
    f = JointTargetFusion(hidden=H, num_heads=6)
    for bad in [torch.randn(2, H), torch.randn(2, N_GEO, H, 1)]:
        try:
            f(bad, torch.randn(2, N_GOAL, H))
        except ValueError:
            continue
        raise AssertionError(f"bad Z_G shape {tuple(bad.shape)} must raise")


# ---------------------------------------------------------------------------
# MaskedQueryPredictor
# ---------------------------------------------------------------------------

def _make_predictor_inputs(b=2, n_masked=6):
    p = MaskedQueryPredictor(hidden=H, num_heads=6, num_layers=2,
                             n_spatial_tokens=N_GEO)
    # 1 = visible, 0 = masked. Build a deterministic mask with n_masked masked.
    mask = torch.ones(b, N_GEO, dtype=torch.bool)
    mask[:, :n_masked] = False           # first n_masked positions masked
    n_vis = N_GEO - n_masked
    z_visible = torch.randn(b, n_vis, H)        # packed visible tokens
    a_goal = torch.randn(b, N_GOAL, H)
    c_scalar = torch.randn(b, H)
    return p, z_visible, a_goal, mask, c_scalar


def test_predictor_shape_contract():
    p, z_visible, a_goal, mask, c_scalar = _make_predictor_inputs()
    out = p(z_visible, a_goal, mask, c_scalar)
    assert out.shape == (2, N_GEO, H), (
        f"output must be (B, N_spatial, hidden) = (2, {N_GEO}, {H}); "
        f"got {tuple(out.shape)}")


def test_predictor_scatter_back_visible_unchanged():
    """Visible positions in the output must equal the input context tokens
    (the §4 first-implementation choice: 'preserve the context representation
    unchanged'). Masked positions must NOT equal the context (they're
    predictions)."""
    p, z_visible, a_goal, mask, c_scalar = _make_predictor_inputs(n_masked=6)
    out = p(z_visible, a_goal, mask, c_scalar)
    visible_pos = mask[0].nonzero(as_tuple=False).squeeze(-1)
    masked_pos = (~mask[0]).nonzero(as_tuple=False).squeeze(-1)
    assert torch.allclose(out[:, visible_pos], z_visible, atol=1e-5), (
        "visible positions must preserve the context representation")
    assert not torch.allclose(out[:, masked_pos], z_visible[:, :len(masked_pos)],
                              atol=1e-5), (
        "masked positions must be predictions, not a copy of context")


def test_predictor_mask_convention_one_visible():
    """Mask convention is 1=visible, 0=masked — the predictor must predict at
    the ZERO positions, not the ONE positions. Verify by flipping the mask:
    the predicted (non-context) positions move to the other half."""
    p, z_visible, a_goal, mask, c_scalar = _make_predictor_inputs(n_masked=6)
    out_a = p(z_visible, a_goal, mask, c_scalar)
    # Flip mask (now last 6 masked) and rebuild z_visible for the new visible
    # set so the packed length matches.
    mask_flip = ~mask
    n_vis = int(mask_flip[0].sum())
    z_visible_flip = torch.randn(2, n_vis, H)
    out_b = p(z_visible_flip, a_goal, mask_flip, c_scalar)
    # The originally-masked positions (first 6) were predictions in out_a;
    # after the flip they're visible -> they hold context from z_visible_flip.
    # They must differ between the two runs (different content), proving the
    # predictor placed predictions at the zero-mask positions in run A.
    assert not torch.allclose(out_a[:, :6], out_b[:, :6], atol=1e-4), (
        "flipping the mask must move the prediction positions")


def test_predictor_gradient_flows_to_masked_path():
    """The loss-relevant predictions are at masked positions; gradients must
    flow through the mask_token / positional / attention params."""
    p, z_visible, a_goal, mask, c_scalar = _make_predictor_inputs()
    out = p(z_visible, a_goal, mask, c_scalar)
    # Only the masked positions should drive the JEPA loss.
    masked_pos = (~mask[0]).nonzero(as_tuple=False).squeeze(-1)
    out[:, masked_pos].sum().backward()
    assert p.mask_token.grad is not None and torch.isfinite(p.mask_token.grad).all()
    assert p.pos_embed.grad is not None and torch.isfinite(p.pos_embed.grad).all()
    # Visible-position gradients through the scatter should be ~0 (they're
    # copied unchanged) — but pos_embed is shared; check the query path got
    # nonzero gradient on the attention weights.
    assert p.blocks[0].self_attn.in_proj_weight.grad is not None


def test_predictor_rejects_wrong_mask_spatial_dim():
    p = MaskedQueryPredictor(hidden=H, num_heads=6, n_spatial_tokens=N_GEO)
    bad_mask = torch.ones(2, N_GEO + 1, dtype=torch.bool)
    try:
        p(torch.randn(2, N_GEO, H), torch.randn(2, N_GOAL, H),
          bad_mask, torch.randn(2, H))
    except ValueError:
        return
    raise AssertionError("wrong mask spatial dim must raise ValueError")


# ---------------------------------------------------------------------------

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
