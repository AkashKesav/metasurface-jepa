"""Test BlockMasker RNG state save/restore for exact resume (hardening spec §2/4)."""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
from torch import nn
from data.mask import BlockMasker


def test_masker_rng_state_save_restore():
    """Masker RNG state must be saved in checkpoint and restored for exact resume."""
    device = torch.device("cpu")

    # Create masker with known seed
    masker = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 4), seed=12345)

    # Sample some masks to advance RNG
    G = torch.randn(2, 3, 64, 64)
    masks_before = []
    for _ in range(3):
        masks_before.append(masker.sample(G, 0.5).clone())

    # Save RNG state
    rng_state = masker.get_rng_state()

    # Sample more masks
    masks_after = []
    for _ in range(2):
        masks_after.append(masker.sample(G, 0.5).clone())

    # Create new masker with same seed
    masker2 = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 4), seed=12345)

    # Restore RNG state
    masker2.set_rng_state(rng_state)

    # Sample same number of masks as after restore
    masks_restored = []
    for _ in range(2):
        masks_restored.append(masker2.sample(G, 0.5).clone())

    # Masks after restore must match masks after save
    for i, (m1, m2) in enumerate(zip(masks_after, masks_restored)):
        assert torch.equal(m1, m2), f"Mask {i} differs after RNG restore"

    print("PASS: test_masker_rng_state_save_restore")


def test_masker_half_sensitivity_rng():
    """Half-sensitivity placement uses RNG for random/sensitivity choice."""
    device = torch.device("cpu")

    class _DummySurrogate(nn.Module):
        def forward(self, x):
            # Return a tensor that depends on x so gradients can flow
            return type("obj", (object,), {"prediction": (x.mean(dim=[1,2,3], keepdim=True) * torch.randn(x.shape[0], 2, 301)).sum() * 0 + torch.randn(x.shape[0], 2, 301)})()

    surrogate = _DummySurrogate().eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    masker = BlockMasker(placement="half_sensitivity", grid=16, min_side=3, k_range=(1, 4), seed=999)
    G = torch.randn(2, 3, 64, 64)

    # Sample masks to advance RNG
    masks_before = [masker.sample(G, 0.5, surrogate).clone() for _ in range(3)]

    # Save state
    rng_state = masker.get_rng_state()

    # Sample more
    masks_after = [masker.sample(G, 0.5, surrogate).clone() for _ in range(2)]

    # New masker, restore state
    masker2 = BlockMasker(placement="half_sensitivity", grid=16, min_side=3, k_range=(1, 4), seed=999)
    masker2.set_rng_state(rng_state)

    # Sample same number
    masks_restored = [masker2.sample(G, 0.5, surrogate).clone() for _ in range(2)]

    # Must match
    for i, (m1, m2) in enumerate(zip(masks_after, masks_restored)):
        assert torch.equal(m1, m2), f"Half-sensitivity mask {i} differs after RNG restore"

    print("PASS: test_masker_half_sensitivity_rng")


def test_mask_ratio_calibration_hits_requested_coverage():
    """Operator decision 2026-09-13 (audit B20): block-mask coverage is
    calibrated toward the requested ratio.

    A single uncorrected draw drifts badly — the min-side clamp inflates small
    blocks and independently placed blocks overlap, so the achieved masked
    fraction is neither the requested ratio nor predictable from it.
    """
    occ = torch.rand(8, 1, 64, 64)
    for ratio in (0.25, 0.5, 0.75):
        masker = BlockMasker(placement="random", grid=16, min_side=3,
                             k_range=(1, 4), seed=11)
        achieved = [float((masker.sample(occ, ratio) < 0.5).float().mean().item())
                    for _ in range(10)]
        mean_achieved = sum(achieved) / len(achieved)
        assert abs(mean_achieved - ratio) <= 0.03, (
            f"requested {ratio}, achieved {mean_achieved:.3f} — coverage must "
            "be calibrated to the requested ratio")

    # Calibration must not break seed determinism (same seed -> same masks).
    a = BlockMasker(placement="random", seed=11).sample(occ, 0.5)
    b = BlockMasker(placement="random", seed=11).sample(occ, 0.5)
    assert torch.equal(a, b), "calibrated sampling must stay deterministic"


if __name__ == "__main__":
    test_masker_rng_state_save_restore()
    test_masker_half_sensitivity_rng()
    test_mask_ratio_calibration_hits_requested_coverage()