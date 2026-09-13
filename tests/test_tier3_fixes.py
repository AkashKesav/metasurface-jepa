"""Regression tests for BUGLOG Tier 3 rows #17, #20, #21 (2026-08-18).

#17 CUDA RNG state is saved/restored with checkpoints; CPU-only environments
    skip the CUDA part safely. (The full checkpoint save/restore round-trip
    lives in tests/test_checkpoint_resume.py.)
#20 jepa_loss raises on a zero-token mask instead of silently falling back to
    the full-token mean (masked-only objective is undefined).
#21 classify_health returns explicit UNAVAILABLE for n_geoms < 2 instead of
    letting NaN flow into a HEALTHY/COLLAPSED verdict.

The rows that exercised the retired Milestone-B validation machinery (#13
prediction-health projection, #14 healthy references, #18
IntervalLossAccumulator, #19 partition invariance) and the two static source
checks against the deleted `train_milestone_b.py` were removed with that
machinery in the 2026-09-13 legacy retirement.

Run:  python tests/test_tier3_fixes.py   (pytest-collectable)
"""

import math
import os
import random
import sys

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from diagnostics.representation_health import (  # noqa: E402
    classify_health, token_space_stats,
)
from losses.jepa_loss import jepa_loss  # noqa: E402
from train.engine import (  # noqa: E402
    collect_rng_state, restore_rng_state,
)


class _ScaleProj(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = float(k)

    def forward(self, x):
        if self.k == 1.0:
            return x
        return x * self.k


# --------------------------------------------------------------------------
# #17 — CUDA RNG state in checkpoint save/resume
# --------------------------------------------------------------------------

def test_collect_restore_rng_cpu_roundtrip():
    """State captured after draw N must make the next draw equal draw N+1 of the
    original stream (checkpoint semantics: save after a step, restore before the
    next step). torch + numpy + python RNG all roundtrip via the engine functions."""
    torch.manual_seed(7)
    draw1 = torch.randn(3).clone()
    state = collect_rng_state()                # captured after draw1
    draw2 = torch.randn(3).clone()             # the draw restore must replay
    _ = torch.randn(3)                         # advance the stream
    restore_rng_state(state)
    assert torch.equal(torch.randn(3), draw2), \
        "torch RNG must replay the exact next draw after restore"

    np.random.seed(11)
    n1 = np.random.rand(3).copy()
    state = collect_rng_state()                # captured after n1
    n2 = np.random.rand(3).copy()
    _ = np.random.rand(3)
    restore_rng_state(state)
    assert np.array_equal(np.random.rand(3), n2), "numpy RNG must roundtrip"

    random.seed(5)
    p1 = [random.random() for _ in range(3)]
    state = collect_rng_state()                # captured after p1
    p2 = [random.random() for _ in range(3)]
    _ = [random.random() for _ in range(3)]
    restore_rng_state(state)
    replay = [random.random() for _ in range(3)]
    assert [round(v, 12) for v in replay] == [round(v, 12) for v in p2], \
        "python RNG must roundtrip"


def test_cuda_rng_state_roundtrip_when_available():
    if not torch.cuda.is_available():
        print("SKIP (no CUDA) test_cuda_rng_state_roundtrip_when_available")
        return
    torch.cuda.manual_seed(11)
    c1 = [s.clone() for s in torch.cuda.get_rng_state_all()]
    _ = torch.randn(3, device="cuda")
    restore_rng_state({"torch_cuda_rng": c1})
    assert all(torch.equal(a, b) for a, b in
               zip(torch.cuda.get_rng_state_all(), c1))


def test_cuda_state_restore_skipped_safely_on_cpu():
    """A checkpoint saved on GPU (torch_cuda_rng present) restored on a CPU-only
    machine must skip the CUDA part, not error (and vice versa: None is fine)."""
    state = collect_rng_state()
    restore_rng_state(state)                 # None (CPU) or valid list (CUDA)
    fake_gpu_state = {"torch_rng": torch.get_rng_state(),
                      "numpy_rng": __import__("numpy").random.get_state(),
                      "python_rng": __import__("random").getstate(),
                      "torch_cuda_rng": [torch.zeros(1, dtype=torch.uint8)]}
    restore_rng_state(fake_gpu_state)        # must not raise on CPU


# --------------------------------------------------------------------------
# #20 — explicit zero-mask policy
# --------------------------------------------------------------------------

def test_zero_mask_raises_value_error():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)      # no masked tokens at all
    raised = False
    try:
        jepa_loss(pred, target, mask)
    except ValueError as e:
        raised = True
        assert "no masked tokens" in str(e)
    assert raised, "zero-mask input must raise, not silently fall back to d.mean()"
    # and with proj too
    try:
        jepa_loss(pred, target, mask, proj=_ScaleProj(2.0))
    except ValueError:
        pass
    else:
        raise AssertionError("zero-mask with proj must raise too")


def test_nonzero_mask_still_works():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 0] = True
    loss, per = jepa_loss(pred, target, mask)
    assert math.isfinite(loss.item()) and per.shape == (2,)


# --------------------------------------------------------------------------
# #21 — n_samples < 2 -> explicit UNAVAILABLE, never a NaN verdict
# --------------------------------------------------------------------------

def test_classify_health_unavailable_for_one_sample():
    raw1 = token_space_stats(torch.randn(1, 4, 3))     # n_geoms = 1
    healthy = token_space_stats(torch.randn(4, 4, 3))  # n_geoms = 4
    status, signals = classify_health(raw1, healthy, healthy, healthy)
    assert status == "UNAVAILABLE"
    assert not status in ("HEALTHY", "COLLAPSED"), (
        "a one-sample health diagnostic must never produce HEALTHY/COLLAPSED")
    assert "n_geoms=1" in signals.get("reason", "")
    assert signals["votes"] == 0


def test_classify_health_unavailable_for_healthy_side_too():
    raw_ok = token_space_stats(torch.randn(4, 4, 3))
    small = token_space_stats(torch.randn(1, 4, 3))
    status, _ = classify_health(raw_ok, raw_ok, small, raw_ok)
    assert status == "UNAVAILABLE"


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
