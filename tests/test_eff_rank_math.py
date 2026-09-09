"""Unit tests for the effective-rank math correction (audit items 2 and 3).

eff_rank_unnorm / eff_rank_frac previously returned the Shannon entropy H and the
normalized entropy H/log(D) while being NAMED effective rank. The effective rank
(Roy–Vetterli) is exp(H) — in #-of-dims units — and the fraction is exp(H)/D.

Run:  python tests/test_eff_rank_math.py        (also collectable by pytest)
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

from diagnostics.representation_health import (  # noqa: E402
    COLLAPSED_ANCHOR, eff_ranks, goal_token_stats, token_eff_ranks,
    token_space_stats,
)


def _spectrum_matrix(singular_values):
    """X (B, D) with exact singular values via SVD construction."""
    B, D = 64, 16
    x0 = torch.randn(B, D)
    U, _, Vt = torch.linalg.svd(x0)
    return U[:, :D] @ torch.diag(torch.tensor(singular_values, dtype=torch.float32)) @ Vt


def test_uniform_spectrum_is_full_effective_rank():
    """Uniform eigenvalues -> H = log(D) -> exp(H) = D, fraction 1.0 (float32
    SVD round-trip noise accounted for)."""
    X = _spectrum_matrix([1.0] * 16)
    s = eff_ranks(X)
    assert abs(s["eff_rank_unnorm"] - 16.0) < 0.1, f"got {s['eff_rank_unnorm']:.6f}"
    assert abs(s["eff_rank_frac"] - 1.0) < 0.01, f"got {s['eff_rank_frac']:.6f}"


def test_rank_one_spectrum_has_effective_rank_one():
    """One dominant singular value -> H ~ 0 -> exp(H) ~ 1."""
    X = _spectrum_matrix([1.0] + [0.0] * 15)
    s = eff_ranks(X)
    assert abs(s["eff_rank_unnorm"] - 1.0) < 0.01, f"got {s['eff_rank_unnorm']:.6f}"
    assert abs(s["eff_rank_frac"] - 1.0 / 16.0) < 0.001, f"got {s['eff_rank_frac']:.6f}"


def test_median_spectrum_between_1_and_d():
    """Effective rank must lie strictly between 1 and D for a spread spectrum."""
    X = _spectrum_matrix([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625] + [0.0] * 8)
    s = eff_ranks(X)
    assert 1.0 < s["eff_rank_unnorm"] < 16.0, f"got {s['eff_rank_unnorm']:.6f}"


def test_entropy_scale_would_fail_these_bounds():
    """Guard against regression to the entropy scale: exp(H) for a spread spectrum
    is strictly larger than H itself, and equals D for uniform (H only equals log D)."""
    X = _spectrum_matrix([1.0] * 16)
    s = eff_ranks(X)
    assert s["eff_rank_unnorm"] > math.log(16.0) + 10.0  # exp(log 16) = 16 >> log 16


def test_goal_token_effective_rank_uses_exp_scale():
    """goal_token_stats: well-spread goal tokens per sample -> exp(H) near the number
    of (token) dimensions; identical tokens -> 0 (degenerate catch, unchanged)."""
    B, G, D = 4, 16, 16
    u = torch.linalg.svd(torch.randn(B, G, D)).U          # orthonormal rows per sample
    r = goal_token_stats(u)["goal_token_effective_rank"]
    assert r > 12.0, f"got {r:.4f} (entropy scale could never exceed {math.log(G):.3f})"
    assert r <= 16.5, f"got {r:.4f}"

    same = u[:, :1, :].expand(B, G, D).contiguous()
    r_same = goal_token_stats(same)["goal_token_effective_rank"]
    assert r_same == 0.0


def test_collapsed_anchor_on_exp_scale():
    """The collapsed anchor must have moved to the exp(H) scale (item 2 fix)."""
    assert abs(COLLAPSED_ANCHOR["eff_rank_unnorm"] - math.exp(2.5986)) < 1e-3
    assert abs(COLLAPSED_ANCHOR["eff_rank_frac"] - math.exp(2.5986) / 384.0) < 1e-5


def test_two_row_pooled_input_is_refused_as_degenerate():
    """Regression for the September pred_eff_rank_frac = 0.5000000 false-stable
    failure: a (B=2, D) matrix has centered rank <= 1, so eff_rank_frac is
    deterministically 1/min(2,D) regardless of data. eff_ranks must REFUSE this
    with NaN rather than emit a data-blind number that misclassifies as stable.
    """
    for D in (3, 8, 192):
        s_random = eff_ranks(torch.randn(2, D))
        s_identical = eff_ranks(torch.ones(2, D))   # both rows identical -> rank 1
        for k in ("eff_rank_unnorm", "eff_rank_frac", "participation", "top_eig_frac"):
            assert math.isnan(s_random[k]), f"B=2 D={D} {k} must be NaN, got {s_random[k]}"
            assert math.isnan(s_identical[k]), f"B=2 D={D} identical {k} must be NaN"


def test_three_row_input_still_measured():
    """n=3 is NOT provably degenerate (centered rank <= 2), so the guard refuses
    only n<3 — a 3-row matrix still gets a real number (no over-refusal)."""
    s = eff_ranks(torch.randn(3, 8))
    assert math.isfinite(s["eff_rank_unnorm"]) and math.isfinite(s["eff_rank_frac"])


def test_token_eff_ranks_has_real_dynamic_range_at_small_batch():
    """The CONTRACT-recommended token-level gauge on flattened (N_tokens, D):
    even at validation batch B=2 with T masked tokens, N_tokens = 2*T is large,
    so the gauge has real dynamic range (a collapsed latent reads low, a
    spread latent reads high) — the gauge the pooled (B=2, D) path could never
    provide. Keys are token_eff_rank_-namespaced to coexist with pooled stats.
    """
    D, T = 16, 64
    collapsed = torch.zeros(2, T, D) + torch.randn(1, D)   # all tokens identical
    spread = torch.randn(2, T, D)
    sc = token_eff_ranks(collapsed.reshape(-1, D))
    ss = token_eff_ranks(spread.reshape(-1, D))
    for k in ("token_eff_rank_unnorm", "token_eff_rank_frac"):
        assert sc[k] < ss[k], f"{k}: collapsed {sc[k]} should be < spread {ss[k]}"
    assert sc["token_eff_rank_unnorm"] < 2.0, "collapsed token gauge should be ~1"


def test_token_space_stats_carries_both_pooled_and_token_gauges():
    """token_space_stats on a (B=2, T, D) token latent: pooled eff_rank is NaN
    (degeneracy guard), but the token_eff_rank gauge is a real number — so a
    small-batch validation run never loses its rank signal entirely."""
    s = token_space_stats(torch.randn(2, 64, 16))
    assert math.isnan(s["eff_rank_frac"]), "pooled eff_rank_frac must be NaN at B=2"
    assert math.isfinite(s["token_eff_rank_frac"]), (
        "token_eff_rank_frac must be a real number even at small batch")
    assert s["token_eff_rank_frac"] > 0.0


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