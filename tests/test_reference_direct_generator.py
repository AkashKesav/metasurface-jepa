"""Smoke/regression tests for the REFERENCE-ONLY direct masked generator
(Baseline 2, design doc §10.1).

History: this module had zero tests and a guaranteed ValueError in
DirectMaskedGenerator.forward — it unpacked 5 values from
_JEPAForwardMixin._encode, which returns 6
(z_hat, z_x, mask, weights, c_physics, a_goal). The bug went unnoticed
because nothing imported the module. These tests exist so the reference
baseline stays runnable for the Milestone-B comparison / ablation table.

Reference-only: this module is NOT part of the active research pipeline
(see the module docstring in src/reference/direct_masked_generator.py).

Run:  python -m pytest tests/test_reference_direct_generator.py -v
"""

import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from reference.direct_masked_generator import DirectMaskedGenerator  # noqa: E402

H = 48  # tiny hidden for fast CPU tests (divisible by num_heads=6)


class _StubReleasedEncoder(nn.Module):
    """Minimal stand-in for the frozen released spectrum encoder.

    S: (B, 2, 301) -> (B, 301, 256) — same I/O contract as ReleasedSpectrumEncoder.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(2, 256)

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _build():
    torch.manual_seed(0)
    m = DirectMaskedGenerator(
        hidden=H, num_heads=6, geo_depth=1,
        predictor_depth=1, goal_tokens=4,
        num_predictor_heads=6,
    )
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    m.spectrum_path.released = stub
    return m


def _batch(b=2):
    torch.manual_seed(1)
    G = (torch.rand(b, 3, 64, 64) > 0.5).float()
    S = torch.randn(b, 2, 301)
    M = torch.ones(b, 16, 16)
    M[:, :4, :4] = 0  # 4x4 block masked -> 16 masked tokens (>=1 required)
    return G, S, M


def test_forward_shapes_and_contract():
    """forward must unpack _encode's 6-tuple correctly and return
    g_hat (B,3,64,64), z_latent (B,256,3*patch^2), mask (B,256)."""
    m = _build()
    G, S, M = _batch()
    out = m(G, S, M)
    assert out["g_hat"].shape == (2, 3, 64, 64), (
        f"g_hat must be (2,3,64,64), got {tuple(out['g_hat'].shape)}")
    assert out["z_latent"].shape == (2, 256, 3 * 4 * 4), (
        f"z_latent must be pixel tokens (2,256,48), got {tuple(out['z_latent'].shape)}")
    assert out["mask"].shape == (2, 256), (
        f"mask must be (2,256), got {tuple(out['mask'].shape)}")
    # 16 masked tokens per sample (the 4x4 block)
    assert out["mask"].sum(dim=1).tolist() == [16, 16]


def test_forward_null_goal_mode():
    """goal_mode='null' must route through SpectrumPath's null branch
    (zero physics/goal) without crashing."""
    m = _build()
    G, S, M = _batch()
    out = m(G, S, M, goal_mode="null")
    assert out["g_hat"].shape == (2, 3, 64, 64)
    assert torch.isfinite(out["g_hat"]).all()


def test_loss_finite_and_gradients_flow():
    """Masked-pixel L1 loss must be finite and reach the pixel-headed
    predictor (this baseline has no JEPA objective by construction)."""
    m = _build()
    G, S, M = _batch()
    loss, out = m.loss(G, S, M)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    assert m.predictor.head.weight.grad is not None, (
        "pixel head must receive gradient from the masked-pixel L1 loss")
    assert torch.isfinite(m.predictor.head.weight.grad).all()


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
