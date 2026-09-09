"""Offline latent-health evaluation of a saved JEPA checkpoint.

WHY THIS EXISTS
---------------
Cloud runs (Kaggle) persist only the `checkpoints/` tree. The training script's
per-step metrics go to stdout, which is NOT downloadable via `kaggle kernels
output`. So after a run finishes, the only recoverable evidence of what it
learned is the checkpoint itself. This script reconstructs the decisive
diagnostics from the weights alone, on real data, so a run can be judged
PASS/FAIL without re-running it.

It answers one question above all: **is a near-zero training loss genuine
prediction, or collapse?** Collapse is the documented historical failure mode of
this repo (frozen-EMA / fast-predictor runs that "converge" while the raw latent
goes orthogonal). The tell is a tiny loss coexisting with:
  - near-zero per-dimension std of z_hat or z_y          (constant representation)
  - raw_cos_err ~= 1.0                                   (orthogonal, not matched)
  - target_spec_sensitivity == 0                         (goal has no effect)

Usage
-----
    python scripts/eval/eval_checkpoint_latents.py \
        --checkpoint checkpoints/unified_stage_a/final.pt \
        --split val --batches 2 --device cpu

Optional: --init-baseline also scores a freshly initialised model with the same
diagnostics. If the trained checkpoint and random init score identically, the
run learned nothing (or the metric is degenerate) — that comparison is the
cheapest possible sanity check and should be run for every new objective.
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch.utils.data import DataLoader

from assembly import build_unified_model
from data.dataset import MetaDiTDataset, collate_batch
from data.factorize import factorize_geometry
from data.mask import BlockMasker
from losses.unified_losses import UnifiedJEPALoss
from runtime.physics_controls import make_shuffled_spectrum
from train.engine import restore_ema_state


def _sample_mask(masker, occ, ratio, surrogate=None):
    """Stage-A broadcast contract: one mask shared across the batch.

    Mirrors scripts/train/train_unified.py::_sample_mask. BlockMasker's
    random_masks draws a per-sample block placement, which the
    MaskedQueryPredictor cannot consume.
    """
    b = occ.shape[0]
    if b == 1:
        return masker.sample(occ, ratio, surrogate)
    m1 = masker.sample(occ[:1], ratio, surrogate)
    return m1.repeat(b, 1, 1)


def _collapse_stats(x):
    """VICReg-style representation-health numbers for a [N, D] tensor.

    std_dim  : mean per-dimension std across the batch (=0 means that dimension
               is constant => collapse on that axis).
    norm     : mean vector norm (=0 means the vector is identically zero).
    """
    x = x.float()
    return {
        "std_dim": float(x.std(dim=0).mean()),
        "norm": float(x.norm(dim=-1).mean()),
    }


@torch.no_grad()
def evaluate(model, objective, batches, mask_ratio, device, grid=16):
    model.eval()
    objective.eval()
    masker = BlockMasker(placement="random", grid=grid, min_side=3,
                         k_range=(1, 4), seed=12345)
    acc = {}
    n = 0

    def add(k, v):
        acc.setdefault(k, []).append(float(v))

    for occ, sv, spec in batches:
        B = occ.shape[0]
        sk = torch.ones(B, 3, dtype=torch.bool, device=device)
        if getattr(model, "requires_broadcast_mask", False):
            M = _sample_mask(masker, occ, mask_ratio).to(device)
        else:
            M = masker.sample(occ, mask_ratio).to(device)

        # Degenerate guard (checked BEFORE the forward pass, because
        # objective() itself raises on an empty token population):
        # mask ratio 0.0 makes every token visible, so there are no masked
        # positions to predict and VICReg correctly refuses to emit statistics
        # for 0 rows. Record NaN and continue rather than killing the sweep.
        n_masked = int((M < 0.5).sum())
        if n_masked == 0:
            for k in ("L_total", "raw_mse", "raw_cos_err", "z_hat_std_dim",
                      "z_hat_norm", "z_y_std_dim", "z_y_norm", "proj_mse",
                      "proj_cos_err", "joint_target_delta_rel"):
                add(k, float("nan"))
            n += 1
            acc.setdefault("_degenerate", []).append(1)
            continue

        result = objective(model, occ, sv, sk, spec, M, goal_mode="real",
                           compute_physics=False)
        out = result["out"]
        mb = out["mask"]

        z_hat, z_y = out["z_hat"], out.get("z_y_joint", out["z_y_raw"])
        zh, zy = z_hat[mb], z_y[mb]

        add("raw_mse", torch.nn.functional.mse_loss(zh, zy))
        add("raw_cos_err", (1 - torch.nn.functional.cosine_similarity(
            zh, zy, dim=-1).clamp(min=0)).mean())
        for name, t in (("z_hat", zh), ("z_y", zy)):
            s = _collapse_stats(t)
            add(f"{name}_std_dim", s["std_dim"])
            add(f"{name}_norm", s["norm"])

        # Projected space (the space L_inv actually trains in)
        ph = result["projector_outputs"]["p_hat"][mb]
        py = result["projector_outputs"]["p_y"][mb]
        add("proj_mse", torch.nn.functional.mse_loss(ph, py))
        add("proj_cos_err", (1 - torch.nn.functional.cosine_similarity(
            ph, py, dim=-1).clamp(min=0)).mean())

        c = result["components"]
        for k in ("L_total", "L_inv", "L_var", "L_cov", "L_scalar", "L_occ",
                  "L_phys"):
            if k in c:
                add(k, c[k])

        add("z_y_geo_norm", out["z_y_raw"][mb].norm(dim=-1).mean())

        # How far did the joint fusion move the target away from geometry-only?
        # §3: joint = Z_G + tanh(gate)*delta. If ||delta|| >> ||Z_G|| the target
        # is dominated by the spectrum branch; with a scale-free (cosine) loss
        # nothing penalises that, so this ratio must be watched explicitly.
        if "z_y_joint" in out and "z_y_raw" in out:
            geo = out["z_y_raw"][mb]
            delta = (out["z_y_joint"][mb] - geo).norm(dim=-1).mean()
            scale = geo.norm(dim=-1).mean().clamp(min=1e-8)
            add("joint_target_delta_rel", delta / scale)
        n += 1

    degenerate = len(acc.pop("_degenerate", []))
    res = {k: float(np.mean(v)) for k, v in acc.items() if not k.startswith("_")}

    # Scale agreement: a cosine-only objective can report ~0 loss while the
    # prediction is orders of magnitude smaller than its target. This ratio is
    # the single cheapest detector of that failure.
    if res.get("z_y_norm"):
        res["scale_ratio_zh_zy"] = res["z_hat_norm"] / max(res["z_y_norm"], 1e-12)
    if res.get("z_y_geo_norm"):
        res["target_norm_growth_vs_geo"] = (
            res["z_y_norm"] / max(res["z_y_geo_norm"], 1e-12))
    res["z_hat_concentration"] = (
        res["z_hat_std_dim"] / max(res["z_hat_norm"], 1e-12))
    res["z_y_concentration"] = (
        res["z_y_std_dim"] / max(res["z_y_norm"], 1e-12))

    if degenerate:
        res["note"] = (f"{degenerate}/{n} batch(es) had an empty masked "
                       f"population (all-visible mask); metrics are NaN there")

    # --- §11 conditioning diagnostic: does the target actually move with the goal?
    try:
        occ_v, sv_v, spec_v = batches[0]
        B = occ_v.shape[0]
        if B < 2:
            res["target_spec_sensitivity"] = float("nan")
            res["target_spec_sensitivity_normalized"] = float("nan")
            res["sensitivity_note"] = "B<2: shuffled control infeasible"
        else:
            sk_v = torch.ones(B, 3, dtype=torch.bool, device=device)
            if getattr(model, "requires_broadcast_mask", False):
                Mv = _sample_mask(masker, occ_v, mask_ratio).to(device)
            else:
                Mv = masker.sample(occ_v, mask_ratio).to(device)
            spec_shuf = make_shuffled_spectrum(spec_v)
            o_real = model(occ_v, sv_v, sk_v, spec_v, Mv, goal_mode="real")
            o_shuf = model(occ_v, sv_v, sk_v, spec_shuf, Mv, goal_mode="real")
            zr = o_real.get("z_y_joint", o_real["z_y_raw"])
            zs = o_shuf.get("z_y_joint", o_shuf["z_y_raw"])
            sens = (zr - zs).norm(dim=-1).mean()
            scale = zr.norm(dim=-1).mean().clamp(min=1e-8)
            res["target_spec_sensitivity"] = float(sens)
            res["target_spec_sensitivity_normalized"] = float(sens / scale)
    except Exception as exc:  # diagnostics must never kill the run
        res["target_spec_sensitivity"] = float("nan")
        res["target_spec_sensitivity_normalized"] = float("nan")
        res["sensitivity_error"] = f"{type(exc).__name__}: {exc}"

    fusion = getattr(model, "joint_target_fusion", None)
    if fusion is not None:
        res["joint_gate_tanh"] = float(torch.tanh(fusion.gate).detach())
    res["n_batches"] = n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--mask-ratios", default="0.0,0.5,1.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--init-baseline", action="store_true",
                    help="also score a freshly initialised model for comparison")
    ap.add_argument("--out", default=None,
                    help="write the JSON report here (default: alongside ckpt)")
    args = ap.parse_args()

    device = torch.device(args.device)
    t0 = time.time()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    print(f"checkpoint : {args.checkpoint}")
    print(f"commit     : {ck.get('git_commit')} (dirty={ck.get('git_dirty')})")
    print(f"step       : {ck.get('step')}  artifact={ck.get('artifact_type')}")
    print(f"gpu(ckpt)  : {(ck.get('env_versions') or {}).get('gpu')}")
    print(f"arch       : {cfg.get('_architecture_id')}")
    print(f"joint_target={cfg.get('joint_target')} "
          f"predictor={cfg.get('predictor_type')}")

    spec_path = cfg["weights"]["spectrum"]
    if not os.path.exists(spec_path):
        raise RuntimeError(f"spectrum weights missing at {spec_path}")
    model = build_unified_model(cfg, spec_path, device=device)
    model.set_total_steps(cfg["train"].get("total_steps", 1))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"state_dict : missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing[:5]={missing[:5]}")
    if unexpected:
        print(f"  unexpected[:5]={unexpected[:5]}")
    restore_ema_state(model, ck.get("ema_state"))

    loss_cfg = cfg.get("loss", {})
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"],
        lambda_inv=loss_cfg.get("lambda_inv", 0.0),
        lambda_var=loss_cfg.get("lambda_var", 0.0),
        lambda_cov=loss_cfg.get("lambda_cov", 0.0),
        lambda_scalar=loss_cfg.get("lambda_scalar", 0.0),
        lambda_occ=loss_cfg.get("lambda_occ", 0.0),
        lambda_phys=0.0,
        lambda_raw=loss_cfg.get("lambda_raw", 0.0),
        gamma=loss_cfg.get("gamma", 1.0),
        eps=loss_cfg.get("eps", 1e-4),
        surrogate=None,
        physics_use_ste=False,
    ).to(device)

    # --- real data ---
    split_path = cfg["data"]["val_split" if args.split == "val" else
                             ("test_split" if args.split == "test" else "train_split")]
    if not os.path.exists(split_path) and args.split == "test":
        split_path = cfg["data"]["val_split"]
    ds = MetaDiTDataset(split_path,
                        max_samples=args.batches * args.batch_size,
                        seed=cfg["train"].get("seed", 42) + 1000)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    batches = []
    for G, S in loader:
        occ, sv = factorize_geometry(G)
        batches.append((occ.to(device), sv.to(device), S.to(device)))
    print(f"data       : {split_path} ({len(batches)} batches x {args.batch_size})")

    ratios = [float(x) for x in args.mask_ratios.split(",")]
    report = {"checkpoint": args.checkpoint, "step": ck.get("step"),
              "split": args.split, "by_mask_ratio": {}}
    for r in ratios:
        report["by_mask_ratio"][f"r{r:g}"] = evaluate(
            model, objective, batches, r, device)

    if args.init_baseline:
        init_model = build_unified_model(cfg, spec_path, device=device)
        init_model.set_total_steps(cfg["train"].get("total_steps", 1))
        report["init_baseline"] = {
            f"r{r:g}": evaluate(init_model, objective, batches, r, device)
            for r in ratios
        }

    print("\n" + "=" * 78)
    hdr = f"{'metric':<34}" + "".join(f"{'r'+format(r,'g'):>14}" for r in ratios)
    print(hdr)
    print("-" * 78)
    keys = ["L_total", "raw_mse", "raw_cos_err",
            "z_hat_std_dim", "z_hat_norm", "z_hat_concentration",
            "z_y_std_dim", "z_y_norm", "z_y_geo_norm", "z_y_concentration",
            "scale_ratio_zh_zy", "target_norm_growth_vs_geo",
            "proj_mse", "proj_cos_err",
            "joint_target_delta_rel", "target_spec_sensitivity",
            "target_spec_sensitivity_normalized", "joint_gate_tanh"]
    for k in keys:
        row = f"{k:<34}"
        any_present = False
        for r in ratios:
            v = report["by_mask_ratio"][f"r{r:g}"].get(k)
            if v is None:
                row += f"{'-':>14}"
            else:
                any_present = True
                row += f"{v:>14.6g}"
        if any_present:
            print(row)
    print("=" * 78)

    # --- verdict ---
    r_main = f"r{ratios[min(1, len(ratios)-1)]:g}"
    m = report["by_mask_ratio"][r_main]
    notes = []
    failed = []

    conc = m.get("z_hat_concentration")
    if conc is not None and not np.isnan(conc):
        if conc < 0.01:
            failed.append(
                f"COLLAPSE: predictor emits ~one vector for every masked token "
                f"(std/norm={conc:.3g} < 0.01)")
        else:
            notes.append(f"predictor varies across tokens (std/norm={conc:.3g})")

    sr = m.get("scale_ratio_zh_zy")
    if sr is not None and not np.isnan(sr):
        if not (0.5 <= sr <= 2.0):
            failed.append(
                f"SCALE MISMATCH: ||z_hat||/||z_y||={sr:.3g} — invisible to a "
                f"cosine-only loss, so L_total is not evidence of learning")
        else:
            notes.append(f"prediction/target magnitudes agree (ratio={sr:.3g})")

    rmse = m.get("raw_mse")
    if rmse is not None and not np.isnan(rmse) and rmse > 1.0:
        notes.append(f"raw_mse={rmse:.4g} (absolute latent error)")

    jr = m.get("joint_target_delta_rel")
    if jr is not None and not np.isnan(jr):
        if jr > 1.0:
            failed.append(
                f"FUSION DOMINATES: ||Z_joint - Z_geo|| / ||Z_geo|| = {jr:.3g} "
                f"> 1 — the target is no longer geometry-anchored")
        else:
            notes.append(f"fusion is a bounded perturbation of Z_geo ({jr:.3g})")

    s = m.get("target_spec_sensitivity_normalized")
    if s is not None and not np.isnan(s):
        # 1% of the target norm is the smallest coupling that can plausibly
        # steer an inverse-design search.
        if s < 0.01:
            failed.append(
                f"GOAL DEAD: shuffling the goal spectrum moves the target by "
                f"{s:.3g} of its norm (< 1%) — conditioning is not usable")
        else:
            notes.append(f"goal coupling ACTIVE (norm sensitivity={s:.3g})")

    g = m.get("joint_gate_tanh")
    if g is not None:
        notes.append(f"fusion gate tanh={g:.4g} "
                     f"({'open' if abs(g) > 1e-3 else 'closed at init'})")

    print("\nVERDICT:", "FAIL" if failed else "PASS")
    for n in failed:
        print("  [x]", n)
    for n in notes:
        print("  [ ]", n)
    if not failed and not notes:
        print("  - inconclusive")

    report["verdict"] = "FAIL" if failed else "PASS"
    report["verdict_failures"] = failed
    report["verdict_notes"] = notes
    report["eval_seconds"] = round(time.time() - t0, 1)

    out_path = args.out or os.path.join(
        os.path.dirname(args.checkpoint), "latent_eval.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
