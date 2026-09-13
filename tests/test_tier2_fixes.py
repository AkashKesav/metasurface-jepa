"""Regression tests for BUGLOG Tier 2 row #11 (2026-08-18).

#11 load_into_model must refuse silent non-strict loads: strict key matching by
    default, with released-filtered keys excluded on BOTH sides.

The other Tier 2 rows (#7-#9: healthy_references device contract, exact n_samples
in fixed_validation_from_loader, single-batch mask std) exercised the retired
Milestone-B validation machinery and were removed with it in the 2026-09-13 legacy
retirement. The resume-integrity contract they shared lives in
tests/test_checkpoint_resume.py (engine save/load strictness).

Run:  python tests/test_tier2_fixes.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

from assembly import load_into_model, saveable_state_dict  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self, with_extra=False):
        super().__init__()
        self.l = nn.Linear(3, 3)
        if with_extra:
            self.extra = nn.Parameter(torch.zeros(1))


class _WithReleased(nn.Module):
    """Frozen released-style submodule must be filtered on save and load."""

    def __init__(self):
        super().__init__()
        self.sub = nn.Module()
        self.sub.released_enc = nn.Linear(2, 2)
        self.l = nn.Linear(3, 3)


def test_load_roundtrip_strict_ok():
    m = _Tiny()
    sd = saveable_state_dict(m)
    m2 = _Tiny()
    load_into_model(m2, sd, "cpu")          # strict by default -> must succeed
    assert torch.equal(m2.l.weight, m.l.weight)


def test_load_raises_on_key_mismatch():
    m = _Tiny()
    sd = saveable_state_dict(m)
    m3 = _Tiny(with_extra=True)             # model has a param the ckpt lacks
    try:
        load_into_model(m3, sd, "cpu")
    except RuntimeError as e:
        assert "missing" in str(e) or "unexpected" in str(e)
        return
    raise AssertionError("load_into_model must raise on key mismatch, not load "
                         "silently with strict=False")


def test_load_filters_released_keys_before_strict():
    m = _WithReleased()
    sd = saveable_state_dict(m)
    assert not any(".released." in k for k in sd)
    m2 = _WithReleased()
    load_into_model(m2, sd, "cpu")          # released filter -> no mismatch


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
