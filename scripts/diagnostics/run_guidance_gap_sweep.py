#!/usr/bin/env python
"""Run guidance-gap sweep across mask ratios (§20.3, Phase 4 MD §3.5.1).

Produces the normalized guidance-gap curve:
    ||P(Z_x, A_goal) - P(Z_x, A_∅)|| / sigma(Z_x)

across 20/40/60/80/100% masking buckets. A curve flat near zero, especially
at low mask ratios, indicates Failure Mode 2 (predictor ignores the spectrum).

Usage:
    python scripts/diagnostics/run_guidance_gap_sweep.py \
        --config configs/unified.yaml \
        --checkpoint checkpoints/unified/latest.pt
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
from data.mask import BlockMasker
from assembly import build_unified_model


def load_eval_model(model, ckpt_path, device):
    """Load a training checkpoint into an eval model.

    Mirrors the authoritative evaluator (``eval_scenarios._load_eval``): the model
    state dict strictly, plus the EMA target state the guidance gap depends on
    (the predictor is FiLM-conditioned by ``scalar_mlp_ema``).

    ``train.engine.load_checkpoint`` is the TRAINING resume path — it requires an
    objective/optimizer/scheduler and has no ``strict_model`` argument, so calling
    it here raised
      TypeError: load_checkpoint() got an unexpected keyword argument 'strict_model'
    and the §20.3 sweep never ran at all (audit B23).
    """
    from assembly import load_into_model
    from train.engine import restore_ema_state

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_into_model(model, ckpt["model"], device=device, strict=True)
    restore_ema_state(model, ckpt.get("ema_state", {}))
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Guidance gap sweep (§20.3)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    spec_path = cfg["weights"].get("spectrum")
    model = build_unified_model(cfg, spec_path, device=device)
    model.eval()

    if args.checkpoint and os.path.exists(args.checkpoint):
        load_eval_model(model, args.checkpoint, device)
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Synthetic test data
    torch.manual_seed(42)
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float().to(device)
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[2.5, 0.8, 4.0], [2.8, 1.0, 4.2]]).to(device)
    spec = torch.randn(2, 2, 301).to(device)

    # Cleanup item 3: prefer eval_mask_ratios (the evaluation-condition list;
    # falls back to the legacy mask_ratios key). The guidance-gap sweep is
    # defined over MASKED buckets (20/40/60/80/100%), so 0.0 — the unmasked
    # reference — is filtered out here regardless of which key supplies the
    # list.
    ratios = cfg.get("curriculum", {}).get(
        "eval_mask_ratios",
        cfg.get("curriculum", {}).get("mask_ratios",
                                      [0.2, 0.4, 0.6, 0.8, 1.0]))
    if 0.0 in ratios:
        ratios = [r for r in ratios if r > 0.0]

    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)

    from diagnostics.guidance_gap import guidance_gap_sweep
    results = guidance_gap_sweep(model, occ, sv, spec, masker, ratios,
                                  device=device)

    print("\n=== Guidance Gap Sweep (§20.3) ===")
    for stratum, curve in results.items():
        for ratio in sorted(curve.keys()):
            print(f"  [{stratum:>11}] mask {ratio:.0%}: "
                  f"normalized_gap = {curve[ratio]:.6f}")

    print(f"\n{json.dumps(results, indent=2)}")
    print("\nInterpretation:")
    print("  - Gap should increase with mask ratio")
    print("  - The all-unknown stratum (full mask + no scalars) is the gate;")
    print("    a flat/near-zero gap there = Failure Mode 2")


if __name__ == "__main__":
    main()
