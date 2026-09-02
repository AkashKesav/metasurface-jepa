"""Unit tests for collapse_trend() — the effective-rank trajectory early warning.

Motivation: both real Kaggle screening phases collapsed (EXPERIMENT_LOG.md
Phase 0/1) and the documented collapse signature (arXiv:2607.23531, abstract
verified 2026-09-02) shows effective-rank degeneration PRECEDES the point where
the per-validation vote thresholds fire. collapse_trend() makes that precursor
visible; it is log-only and must never change classify_health verdicts.

Run:  python tests/test_collapse_trend.py     (also collectable by pytest)
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from diagnostics.representation_health import collapse_trend  # noqa: E402


def test_insufficient_history_returns_neutral():
    t = collapse_trend([1.0, 0.9])
    assert t["trend"] == "insufficient_history"
    assert t["early_warning"] is False
    assert t["n"] == 2


def test_declining_rank_flags_early_warning():
    """Rank falling >25% across the window -> early_warning, regardless of any
    single validation's vote-based status."""
    t = collapse_trend([0.80, 0.60, 0.45])
    assert t["trend"] == "declining"
    assert t["early_warning"] is True
    assert abs(t["rel_change"] - (0.45 - 0.80) / 0.80) < 1e-9


def test_stable_and_rising_do_not_warn():
    assert collapse_trend([0.80, 0.78, 0.79])["trend"] == "stable"
    assert collapse_trend([0.80, 0.78, 0.79])["early_warning"] is False
    t_up = collapse_trend([0.50, 0.70, 0.90])
    assert t_up["trend"] == "rising"
    assert t_up["early_warning"] is False


def test_nan_entries_are_skipped():
    """NaN markers from n_geoms < 2 batches (Bug #21 contract) must not poison
    the trajectory or crash the math.isnan filter."""
    t = collapse_trend([float("nan"), 0.80, float("nan"), 0.60, 0.45])
    assert t["n"] == 3
    assert t["trend"] == "declining"
    assert t["early_warning"] is True


def test_none_entries_are_skipped():
    t = collapse_trend([None, 0.80, None, 0.60, 0.45])
    assert t["n"] == 3
    assert t["early_warning"] is True


def test_decline_frac_boundary_is_exclusive():
    """Exactly -25% is NOT a warning (strict threshold); -26% is."""
    assert collapse_trend([1.0, 0.875, 0.75])["trend"] == "stable"
    t = collapse_trend([1.0, 0.87, 0.74])
    assert t["trend"] == "declining"
    assert t["early_warning"] is True


def test_near_zero_first_value_does_not_explode():
    """A near-zero first value must not divide by zero (eps guard) and stays
    finite; relative change from ~0 to a larger value reads as rising."""
    t = collapse_trend([1e-12, 0.5, 1.0])
    assert math.isfinite(t["rel_change"])
    assert t["trend"] == "rising"


def test_trend_is_log_only_no_classification_side_effects():
    """Contract: the function returns a plain dict of readouts and cannot feed
    classify_health — verify the dict schema has no status-like key."""
    t = collapse_trend([0.8, 0.6, 0.4])
    assert set(t) == {"trend", "n", "early_warning", "first", "last",
                      "rel_change"}
    assert "status" not in t and "votes" not in t


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
