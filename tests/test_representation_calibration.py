"""B9 — Representation calibration smoke tests.

Covers the pure-math diagnostics shared by the unified evaluators:

1. grouped_view is a pure view (no recomputation) of an existing token_space_stats dict.
2. mean-pooled uses X.mean(dim=1) — calling with (N, T, D) and pre-pooled (N, D) gives
   the same result for mean_pooled keys.
3. same-token fixed positions: same_token_cos is invariant to column permutation
   (relabeling spatial tokens) of a (N, T, D) tensor.
4. geometry_linear_probes are finite and deterministic (same inputs -> identical R^2).

The two VICReg-gradient-attribution cases and the random-calibration-encoder RNG
case exercised the retired 384-D `build_model` / `losses.objectives` /
`representation_calibration.py` path; they were removed with it in the 2026-09-13
legacy retirement. Gradient-ownership coverage for the live path lives in
tests/test_unified_model_phase2.py and tests/test_phase5_contracts.py.
"""

import os
import sys

import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def test_grouped_view_is_pure_view():
    from diagnostics.representation_health import token_space_stats, grouped_view
    X = torch.randn(64, 16, 128)
    flat = token_space_stats(X)
    gv = grouped_view(flat)
    # mean_pooled keys must equal the expected set
    assert set(gv.keys()) == {"mean_pooled", "token_level", "n_geoms"}, gv.keys()
    assert set(gv["mean_pooled"].keys()) == {
        "pairwise_cos", "eff_rank_unnorm", "eff_rank_frac",
        "participation", "top_eig_frac"}
    assert set(gv["token_level"].keys()) == {
        "token_var", "token_std", "same_token_cos"}
    assert gv["n_geoms"] == 64
    # values are aliases, not copies — same object
    assert gv["token_level"]["token_std"] is flat["token_std"]
    assert gv["mean_pooled"]["eff_rank_frac"] is flat["eff_rank_frac"]


def test_mean_pooled_matches_manual_pooling():
    """token_space_stats' mean_pooled keys (pairwise_cos, eff_rank, etc.) must
    equal those computed on X.mean(dim=1) directly — the pooling is a step
    inside token_space_stats, not a property of its (N, T, D) input."""
    from diagnostics.representation_health import (
        token_space_stats, pairwise_cos_stats, eff_ranks, same_token_cos)
    X = torch.randn(32, 16, 128)
    s = token_space_stats(X)
    pooled = X.mean(dim=1)  # (32, 128)
    p_pooled = pairwise_cos_stats(pooled)
    e_pooled = eff_ranks(pooled)
    assert abs(s["eff_rank_frac"] - e_pooled["eff_rank_frac"]) < 1e-6
    assert abs(s["eff_rank_unnorm"] - e_pooled["eff_rank_unnorm"]) < 1e-6
    assert abs(s["participation"] - e_pooled["participation"]) < 1e-6
    for pk in ("mean", "p05", "min"):
        assert abs(s["pairwise_cos"][pk] - p_pooled[pk]) < 1e-6


def test_same_token_cos_column_permutation_invariant():
    """same_token_cos is a cross-sample statistic at each spatial position — it
    should be invariant to relabeling the spatial positions (column permutation
    over the T axis)."""
    from diagnostics.representation_health import same_token_cos
    X = torch.randn(8, 16, 64)
    base = same_token_cos(X)
    perm = torch.randperm(16)
    permuted = X[:, perm, :]
    permuted_val = same_token_cos(permuted)
    # Both should be the same value (average over all token positions is invariant)
    # But the per-position values change; only the mean is invariant
    assert abs(base - permuted_val) < 1e-6, (base, permuted_val)


def test_geometry_linear_probes_finite_and_deterministic():
    from diagnostics.representation_probes import geometry_linear_probes
    N, T, D = 64, 16, 128
    X = torch.randn(N, T, D)
    params = np.random.RandomState(42).randn(N, 3).astype(np.float64)
    r1 = geometry_linear_probes(X, params, ridge_lambda=1.0, seed=7)
    r2 = geometry_linear_probes(X, params, ridge_lambda=1.0, seed=7)
    for key in ("l_lattice_r2", "h_atom_r2", "r_atom_r2", "mean_r2"):
        assert key in r1, f"missing key: {key}"
        assert isinstance(r1[key], float), f"{key} not float: {type(r1[key])}"
        assert r1[key] == r1[key], f"{key} is NaN"  # NaN check (v != v)
        assert r1[key] == r2[key], f"{key} not deterministic"


if __name__ == "__main__":
    test_grouped_view_is_pure_view()
    test_mean_pooled_matches_manual_pooling()
    test_same_token_cos_column_permutation_invariant()
    test_geometry_linear_probes_finite_and_deterministic()
    print("PASS: all B9 representation calibration tests")
