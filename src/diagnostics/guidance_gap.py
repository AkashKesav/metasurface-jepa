"""Normalized classifier-free guidance diagnostic (§20.3, Phase 4 MD §3.5.1).

Computes the normalized guidance gap (audit B14 — the spec's per-sample L2
form, not a mean-absolute difference with a global std):

    normalized_gap_i = ||P_i(Z_x, A_goal) - P_i(Z_x, A_∅)||_2 / std(P_i(Z_x, A_goal))
    normalized_gap   = mean_i normalized_gap_i

This is a single-forward-pair diagnostic: one forward with real goal,
one with null goal, measure the normalized difference. A small or
constant gap across very different targets indicates Failure Mode 2
(predictor ignores the spectrum — §13).

When run across a curriculum sweep of mask ratios (20/40/60/80/100%),
the curve should rise with mask ratio — at 0% mask the goal is
redundant (context is complete), at 100% mask the goal is the only
input and collapse → zero gap is a severe failure. The sweep reports the
all-known and all-unknown scalar strata separately: the all-unknown stratum is
the one where spectrum dependence is the gate (architecture_v5.md §8.3).
"""

import torch


def normalized_gap_stats(z_real, z_null):
    """Per-sample L2 gap + scale-normalized form (shared by the diagnostic and
    the CFG inference path, so the two definitions cannot diverge — audit B14).

        gap_i  = ||z_real_i - z_null_i||_2                (over all latent dims)
        norm_i = gap_i / std(z_real_i)                    (same latent dims)

    Args:
        z_real, z_null: matching tensors, any leading batch dim (B, ...).

    Returns dict: guidance_gap, normalized_guidance_gap, z_real_std, z_null_std
    (batch means of the per-sample quantities).
    """
    diff = (z_real - z_null).flatten(1)                   # (B, N)
    gap = diff.norm(dim=1)                                # (B,)
    std_real = z_real.flatten(1).std(dim=1, unbiased=False)
    std_null = z_null.flatten(1).std(dim=1, unbiased=False)
    return {
        "guidance_gap": float(gap.mean().item()),
        "normalized_guidance_gap": float(
            (gap / std_real.clamp(min=1e-6)).mean().item()),
        "z_real_std": float(std_real.mean().item()),
        "z_null_std": float(std_null.mean().item()),
    }


@torch.no_grad()
def compute_guidance_gap(model, occ, sv, sk, spec, mask, device="cpu"):
    """Single (occ, spec, mask) → normalized guidance gap scalar.

    Args:
        model: UnifiedJEPA (eval mode).
        occ:  [B,1,64,64] occupancy.
        sv:   [B,3] scalar values.
        sk:   [B,3] bool known flags.
        spec: [B,2,301] spectrum.
        mask: [B,16,16] visibility mask.
        device: target device.

    Returns:
        dict with the normalized_gap_stats keys plus:
            gap_form: the exact formula used (audit B14 provenance).
    """
    model.eval()

    out_real = model(occ, sv, sk, spec, mask,
                     goal_mode="real", with_target=False)
    out_null = model(occ, sv, sk, spec, mask,
                     goal_mode="null", with_target=False)

    stats = normalized_gap_stats(out_real["z_hat"], out_null["z_hat"])
    stats["gap_form"] = ("mean_i ||z_real_i - z_null_i||_2 / "
                         "std_i(z_real_i)")
    return stats


@torch.no_grad()
def guidance_gap_sweep(model, occ, sv, spec, masker, ratios, device="cpu",
                       scalar_strata=None):
    """Normalized guidance gap across mask-ratio buckets, PER SCALAR STRATUM.

    Audit B14: the old sweep fixed all scalars known, so it could never probe
    the strat/regime that matters (all scalars unknown + full mask).

    Args:
        model:     UnifiedJEPA (eval mode).
        occ:       [B,1,64,64] base occupancy.
        sv:        [B,3] scalar values.
        spec:      [B,2,301] spectrum.
        masker:    BlockMasker instance.
        ratios:    list of mask ratios (e.g. [0.2, 0.4, 0.6, 0.8, 1.0]).
        device:    target device.
        scalar_strata: {name: all_known_bool}; defaults to all-known and
                       all-unknown.

    Returns:
        {stratum_name: {ratio: normalized_guidance_gap}}
    """
    if scalar_strata is None:
        scalar_strata = {"all_known": True, "all_unknown": False}
    out = {}
    for name, all_known in scalar_strata.items():
        sk = torch.full((occ.shape[0], 3), bool(all_known),
                        dtype=torch.bool, device=device)
        curve = {}
        for ratio in ratios:
            # Fix (CUDA mask bug): masker.sample returns CPU tensors — move to
            # the active device before the model forward.
            M = masker.sample(occ, ratio).to(device)
            gap_info = compute_guidance_gap(model, occ, sv, sk, spec, M, device)
            curve[ratio] = gap_info["normalized_guidance_gap"]
        out[name] = curve
    return out
