#!/usr/bin/env python3
"""Stage 4 — Real vs shuffled goal utility (plan Gate D).

Trains the unified model at lambda_phys=1.0 (best from the Stage-5 sweep) and,
at fixed intervals, evaluates the SAME held-out geometry + SAME mask under
    real spectrum
    null spectrum
    shuffled spectrum (derangement — no sample keeps its own spectrum)

using the unified model's forward under each goal_mode. Reports
    L_real/L_null/L_shuffled  = masked-token JEPA cosine loss
    gap_null/gap_shuffled     = L_null/shuffled - L_real
    sensitivity_*             = mean ||z_hat(real) - z_hat(alt)|| on masked tokens

Gate D: gap_shuffled > 0 means the real spectrum helps vs a shuffled one
(model responds to target identity, not just "a spectrum is present").

Persists:
    results/stage4/goal_utility_metrics.json
    results/stage4/latest.pt   (reproducibility)

Run (Kaggle):
    python scripts/eval/stage4_goal_utility.py --config configs/unified.yaml \\
        --data-root /kaggle/input/metadit-aaai2026-staging/metadit \\
        --lambda-phys 1.0 --total-steps 1500 --eval-every 250
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from assembly import build_unified_model, set_spectrum_path
from data.dataset import MetaDiTDataset, collate_batch
from data.factorize import factorize_geometry
from data.mask import BlockMasker
from encoders.occupancy_encoder import OccupancyEncoder
from losses.unified_losses import UnifiedJEPALoss, occupancy_reconstruction_metrics
from physics.physics_loop import load_surrogate, physics_loss_from_out
from runtime.reproducibility import set_seed
from runtime.device import resolve_device
from train.engine import save_checkpoint, load_checkpoint, collect_ema_state
from train.train_unified import (
    _build_scalar_masker_bank,
    RegimeLogger,
    training_step,
    build_scheduler,
    _assert_no_ema_gradients,
)
try:
    import encoders.occ_init  # noqa: F401 (may not exist)
except ImportError:
    pass


def _init_geometry_from_metadit(model, path):
    """Best-effort released-MetaDiT init for the unified occupancy encoder.
    Missing file or shape mismatch => keep random init (the eval measures
    whether TRAINING makes the model spectrum-dependent, not the init)."""
    import os, torch
    if not os.path.exists(path):
        print(f"[stage4] metadit weights not found at {path}; using random init",
              flush=True)
        return
    try:
        sd = torch.load(path, map_location="cpu")
        enc = getattr(model, "occupancy_encoder", None)
        if enc is None or not hasattr(enc, "init_from_metadit"):
            # Fall back to a direct partial load if API differs.
            missing, unexpected = enc.load_state_dict(sd, strict=False) if enc else (None, None)
            print(f"[stage4] occupancy_encoder partial load missing={list(missing)[:4]} "
                  f"unexpected={list(unexpected)[:4]}", flush=True)
            return
        enc.init_from_metadit(sd, blocks_to_take=6)
        print(f"[stage4] initialized occupancy_encoder from {path}", flush=True)
    except Exception as e:
        print(f"[stage4] metadit init skipped ({type(e).__name__}: {e}); "
              "using random init", flush=True)


def masked_cosine_loss(z_hat, z_y, mask):
    d = 1.0 - torch.nn.functional.cosine_similarity(
        torch.nn.functional.normalize(z_hat, dim=-1),
        torch.nn.functional.normalize(z_y, dim=-1), dim=-1).clamp(min=0)
    return d[mask].mean().item()


def persist_checkpoint_shards(path, shard_bytes=40 * 1024 * 1024):
    """Make a large checkpoint recoverable through Kaggle output artifacts.

    Kaggle may list an oversized single output file without its payload. The
    manifest and bounded shards are independently downloadable and can be
    concatenated in lexical order to reconstruct ``path`` exactly.
    """
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    parts = []
    with path.open("rb") as src:
        index = 0
        while True:
            chunk = src.read(shard_bytes)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            part = path.with_name(f"{path.name}.part{index:04d}")
            with part.open("wb") as dst:
                dst.write(chunk)
            parts.append(part.name)
            index += 1
    manifest = {
        "source": path.name,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "part_bytes": shard_bytes,
        "parts": parts,
        "reassemble": "concatenate parts in listed order and verify sha256",
    }
    path.with_name(f"{path.name}.manifest.json").write_text(
        json.dumps(manifest, indent=2))
    return manifest


@torch.no_grad()
def eval_goal_utility(model, surrogate, batches, device, generator, step,
                      objective=None):
    """Real/null/shuffled goal utility on the unified model (Gate D).

    Each batch is (occ, sv, spec, mask). Shuffling permutes the spectrum across
    the batch (derangement so no sample keeps its own spectrum).
    """
    model.eval()
    agg = {k: [] for k in
           ["L_real", "L_null", "L_shuffled", "gap_null", "gap_shuffled",
            "sensitivity_null", "sensitivity_shuffled",
            "physics_real", "physics_null", "physics_shuffled",
            "geometry_sensitivity_null", "geometry_sensitivity_shuffled",
            "occupancy_iou", "occupancy_f1", "pred_occupancy_fraction",
            "true_occupancy_fraction", "scalar_out_of_range_fraction",
            "scalar_mae", "scalar_normalized_mae", "scalar_pred_min",
            "scalar_pred_max",
            "c_physics_cross_sample_std", "a_goal_cross_sample_std"]}
    for key in ("raw_z_mse", "raw_z_cosine", "raw_z_hat_norm",
                "raw_z_y_norm", "projected_mse", "projected_cosine",
                "projected_pred_norm", "projected_target_norm"):
        agg[key] = []
    for occ, sv, spec, M in batches:
        occ = occ.to(device); sv = sv.to(device)
        spec = spec.to(device); M = M.to(device)
        B = occ.shape[0]
        sk = torch.zeros(B, 3, dtype=torch.bool, device=device)  # all unknown = hard
        # Derangement: no sample keeps its own spectrum.
        perm = torch.randperm(B, generator=generator) if B > 1 else torch.arange(B)
        if B > 1:
            while (perm == torch.arange(B)).any():
                perm = torch.randperm(B, generator=generator)
        spec_shuf = spec[perm]

        out_r = model(occ, sv, sk, spec, M, goal_mode="real")
        out_n = model(occ, sv, sk, spec, M, goal_mode="null")
        out_s = model(occ, sv, sk, spec_shuf, M, goal_mode="real")

        mask = out_r["mask"]
        z_r, z_y = out_r["z_hat"], out_r["z_y_raw"]
        z_n, z_s = out_n["z_hat"], out_s["z_hat"]
        L_real = masked_cosine_loss(z_r, z_y, mask)
        L_null = masked_cosine_loss(z_n, z_y, mask)
        L_shuf = masked_cosine_loss(z_s, z_y, mask)
        raw_diff = (z_r - z_y)[mask]
        raw_pred = z_r[mask]
        raw_target = z_y[mask]
        agg["raw_z_mse"].append(float(raw_diff.square().mean()))
        agg["raw_z_cosine"].append(float(torch.nn.functional.cosine_similarity(
            raw_pred, raw_target, dim=-1).mean()))
        agg["raw_z_hat_norm"].append(float(raw_pred.norm(dim=-1).mean()))
        agg["raw_z_y_norm"].append(float(raw_target.norm(dim=-1).mean()))
        if objective is not None:
            proj_pred = objective.projector(z_r)[mask]
            proj_target = objective.projector(z_y)[mask]
            agg["projected_mse"].append(float((proj_pred - proj_target).square().mean()))
            agg["projected_cosine"].append(float(torch.nn.functional.cosine_similarity(
                proj_pred, proj_target, dim=-1).mean()))
            agg["projected_pred_norm"].append(float(proj_pred.norm(dim=-1).mean()))
            agg["projected_target_norm"].append(float(proj_target.norm(dim=-1).mean()))
        def decode_and_error(out, target_spec):
            geometry, _ = model.decode_geometry(
                out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M,
                scalar_known=sk, scalar_values=sv, hard_forward=True)
            prediction = surrogate(geometry).prediction
            scale = target_spec.std(dim=-1, keepdim=True).clamp_min(1e-4)
            error = ((prediction - target_spec).abs() / scale).mean().item()
            return geometry, error
        g_r, p_r = decode_and_error(out_r, spec)
        g_n, p_n = decode_and_error(out_n, spec)
        g_s, p_s = decode_and_error(out_s, spec)
        agg["L_real"].append(L_real)
        agg["L_null"].append(L_null)
        agg["L_shuffled"].append(L_shuf)
        agg["gap_null"].append(L_null - L_real)
        agg["gap_shuffled"].append(L_shuf - L_real)
        agg["sensitivity_null"].append(
            (z_r - z_n).norm(dim=-1)[mask].mean().item())
        agg["sensitivity_shuffled"].append(
            (z_r - z_s).norm(dim=-1)[mask].mean().item())
        agg["physics_real"].append(p_r)
        agg["physics_null"].append(p_n)
        agg["physics_shuffled"].append(p_s)
        agg["geometry_sensitivity_null"].append((g_r - g_n).abs().mean().item())
        agg["geometry_sensitivity_shuffled"].append((g_r - g_s).abs().mean().item())
        om = occupancy_reconstruction_metrics(out_r["occupancy_logits"], occ)
        for key in ("occupancy_iou", "occupancy_f1", "pred_occupancy_fraction", "true_occupancy_fraction"):
            agg[key].append(float(om[key]))
        bounds = model.scalar_decoder.bounds
        agg["scalar_out_of_range_fraction"].append(float(
            ((out_r["scalar_pred"] < bounds[:, 0]) |
             (out_r["scalar_pred"] > bounds[:, 1])).float().mean()))
        scalar_error = (out_r["scalar_pred"] - sv).abs()
        scalar_scale = (bounds[:, 1] - bounds[:, 0]).clamp_min(1e-6)
        agg["scalar_mae"].append(float(scalar_error.mean()))
        agg["scalar_normalized_mae"].append(float((scalar_error / scalar_scale).mean()))
        agg["scalar_pred_min"].append(float(out_r["scalar_pred"].min()))
        agg["scalar_pred_max"].append(float(out_r["scalar_pred"].max()))
        agg["c_physics_cross_sample_std"].append(float(out_r["c_physics"].std(dim=0).mean()))
        agg["a_goal_cross_sample_std"].append(float(out_r["a_goal"].std(dim=0).mean()))
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    out["step"] = step
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "unified.yaml"))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument("--lambda-goal", type=float, default=0.0,
                    help="exploratory real-vs-shuffled margin-loss weight")
    ap.add_argument("--goal-margin", type=float, default=0.01,
                    help="exploratory normalized-physics margin")
    ap.add_argument("--total-steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "stage4"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device or cfg["train"].get("device", "cpu"))
    set_seed(args.seed)
    cfg["loss"]["lambda_phys"] = args.lambda_phys

    data_root = Path(args.data_root)
    train_split = str(data_root / "split_data" / "train_set.mat")
    val_split = str(data_root / "split_data" / "val_set.mat")

    model = build_unified_model(cfg, spec_weights=str(data_root / "weights" / "spec_encoder.pth"),
                                device=device)
    set_spectrum_path(model, str(data_root / "weights" / "spec_encoder.pth"), device=device)
    _init_geometry_from_metadit(model, str(data_root / "weights" / "metadit-small.bin"))
    surrogate = load_surrogate(str(data_root / "weights" / "surrogate_model.bin"), device=device)
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"],
        lambda_inv=cfg["loss"]["lambda_inv"],
        lambda_var=cfg["loss"]["lambda_var"],
        lambda_cov=cfg["loss"]["lambda_cov"],
        lambda_scalar=cfg["loss"]["lambda_scalar"],
        lambda_occ=cfg["loss"]["lambda_occ"],
        lambda_phys=cfg["loss"]["lambda_phys"],
        gamma=cfg["loss"].get("gamma", 1.0),
        eps=cfg["loss"].get("eps", 1e-4),
        surrogate=surrogate,
        physics_use_ste=cfg["staging"].get("physics_use_ste", True),
    )
    objective.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad] + \
                [p for p in objective.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["wd"])
    scheduler = build_scheduler(
        optimizer, cfg["train"]["lr"], cfg["train"]["warmup_steps"],
        args.total_steps)

    ds = MetaDiTDataset(train_split, max_samples=cfg["data"].get("max_train_samples", 0),
                        seed=args.seed)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=True, num_workers=0, collate_fn=collate_batch)

    vds = MetaDiTDataset(val_split, max_samples=16, seed=args.seed + 1000)
    vloader = DataLoader(vds, batch_size=cfg["train"]["batch_size"],
                         shuffle=False, num_workers=0, collate_fn=collate_batch)

    masker = BlockMasker(
        grid=16, min_side=3, k_range=(1, 4),
        placement=cfg["curriculum"].get("mask_placement", "random"),
        seed=args.seed)
    scalar_bank = _build_scalar_masker_bank(cfg, seed=args.seed)
    regime_logger = RegimeLogger(cfg)

    # Fixed hard-stratum val batches: 100% mask, all scalars unknown.
    # Each batch carries (occ, sv, spec, mask) so eval can permute spectrum.
    fixed_batches = []
    invalid_val_samples = 0
    for G, S in vloader:
        valid = ((G[:, 0].amax(dim=(1, 2)) > 0) &
                 (G[:, 1].amax(dim=(1, 2)) > 0))
        invalid_val_samples += int((~valid).sum().item())
        if not valid.any():
            continue
        G, S = G[valid], S[valid]
        G, S = G.to(device), S.to(device)
        occ, sv = factorize_geometry(G)
        M = torch.zeros(G.shape[0], 16, 16, device=device)  # all masked
        fixed_batches.append((occ, sv, S, M))
    if not fixed_batches:
        raise RuntimeError("No val batches for goal-utility eval")

    rng = torch.Generator().manual_seed(args.seed)
    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, objective, optimizer,
                               scheduler, device)
        start_step = ckpt.get("step", -1) + 1

    eval_history = []
    model.train()
    objective.train()
    data_iter = iter(loader)
    t0 = time.time()

    invalid_train_samples = 0
    goal_grad_history = []

    def get_batch():
        nonlocal data_iter
        nonlocal invalid_train_samples
        while True:
            try:
                G, S = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                G, S = next(data_iter)
            valid = ((G[:, 0].amax(dim=(1, 2)) > 0) &
                     (G[:, 1].amax(dim=(1, 2)) > 0))
            invalid_train_samples += int((~valid).sum().item())
            if valid.any():
                G, S = G[valid], S[valid]
                occ, sv = factorize_geometry(G)
                return occ.to(device), sv.to(device), S.to(device)

    for step in range(start_step, args.total_steps):
        optimizer.zero_grad(set_to_none=True)
        occ, sv, spec = get_batch()
        result, M, sk = training_step(
            model, objective, occ, sv, spec, cfg, device, step,
            masker, rng, regime_logger, surrogate=surrogate,
            scalar_masker_bank=scalar_bank)
        loss = result["total_loss"]
        goal_term = loss.new_zeros(())
        if args.lambda_goal > 0 and spec.shape[0] > 1:
            # Derangement for the training batch: the requested target is
            # changed while geometry, mask, and scalar-known state stay fixed.
            goal_spec = torch.roll(spec, shifts=1, dims=0)
            out_shuf = model(occ, sv, sk, goal_spec, M, goal_mode="real")
            p_real, _, _ = physics_loss_from_out(
                model, result["out"], surrogate, occ, sv, sk, spec, M,
                loss_type="smooth_l1", use_ste=True, normalize=True)
            p_shuf, _, _ = physics_loss_from_out(
                model, out_shuf, surrogate, occ, sv, sk, spec, M,
                loss_type="smooth_l1", use_ste=True, normalize=True)
            goal_term = torch.relu(args.goal_margin + p_real - p_shuf)
            loss = loss + args.lambda_goal * goal_term
        loss.backward()
        goal_grad_sq = 0.0
        for name, parameter in model.named_parameters():
            if name.startswith("spectrum_path.") and parameter.grad is not None:
                goal_grad_sq += float(parameter.grad.detach().square().sum())
        goal_grad_norm = goal_grad_sq ** 0.5
        goal_grad_history.append(goal_grad_norm)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] +
            [p for p in objective.parameters() if p.requires_grad],
            cfg["train"].get("clip_grad_norm", 1.0))
        optimizer.step()
        scheduler.step()
        _assert_no_ema_gradients(model, step)

        if (step + 1) % cfg["train"].get("log_every_steps", 10) == 0:
            print(f"[step {step+1}/{args.total_steps}] "
                  f"L_total={result['components']['L_total']:.3f} "
                  f"L_phys={result['components'].get('L_phys', 0):.3f} "
                  f"L_goal={float(goal_term.detach()):.4f} "
                  f"goal_grad={goal_grad_norm:.3e}")

        if (step + 1) % args.eval_every == 0:
            m = eval_goal_utility(model, surrogate, fixed_batches, device, rng,
                                  step + 1, objective=objective)
            eval_history.append(m)
            print(f"  [goal-utility @ step {step+1}] "
                  f"L_real={m['L_real']:.4f} "
                  f"L_shuffled={m['L_shuffled']:.4f} "
                  f"gap_shuffled={m['gap_shuffled']:+.4f}  "
                  f"(gap>0: real helps vs shuffled)")
            with open(out_dir / "goal_utility_metrics.json", "w") as f:
                json.dump(eval_history, f, indent=2)
            save_checkpoint(
                str(out_dir / "latest.pt"), model, objective, optimizer, scheduler,
                cfg, global_step=step, is_epoch_end=False,
                metrics=m, ema_state=collect_ema_state(model),
                masker_rng_state=masker.get_rng_state(), device=device,
                artifact_type="latest", extra={"eval_history": eval_history})

    final = eval_goal_utility(model, surrogate, fixed_batches, device, rng,
                              args.total_steps, objective=objective)
    eval_history.append(final)

    ckpt_path = out_dir / "latest.pt"
    save_checkpoint(
        str(ckpt_path), model, objective, optimizer, scheduler, cfg,
        global_step=args.total_steps - 1, is_epoch_end=True,
        metrics=final, ema_state=collect_ema_state(model),
        masker_rng_state=masker.get_rng_state(), device=device,
        artifact_type="final", extra={"eval_history": eval_history})
    checkpoint_manifest = persist_checkpoint_shards(ckpt_path)
    with open(out_dir / "goal_utility_metrics.json", "w") as f:
        json.dump(eval_history, f, indent=2)
    config_sha = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    commit = "unknown"
    try:
        commit = os.popen(f"git -C {REPO_ROOT} rev-parse HEAD").read().strip()
    except Exception:
        pass
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump({
            "architecture_id": "unified_occ_param_spectrum_jepa_v1",
            "git_commit": commit,
            "config_sha256": config_sha,
            "dataset_root": str(data_root),
            "seed": args.seed,
            "lambda_goal": args.lambda_goal,
            "goal_margin": args.goal_margin,
            "invalid_val_samples_skipped": invalid_val_samples,
            "invalid_train_samples_skipped": invalid_train_samples,
            "total_steps": args.total_steps,
            "validation_stratum": "100_percent_occupancy_mask_all_scalars_unknown",
            "validation_batch_count": len(fixed_batches),
            "regime_report": regime_logger.report(),
            "goal_path_grad_norm_mean": float(np.mean(goal_grad_history)),
            "goal_path_grad_norm_max": float(np.max(goal_grad_history)),
            "checkpoint": str(out_dir / "latest.pt"),
            "checkpoint_manifest": checkpoint_manifest,
        }, f, indent=2)

    gate_d = (final["physics_real"] < final["physics_shuffled"] and
              final["geometry_sensitivity_shuffled"] > 0)
    print("\n=== Stage 4 Gate D verdict ===")
    print(f"  L_real             = {final['L_real']:.4f}")
    print(f"  L_shuffled         = {final['L_shuffled']:.4f}")
    print(f"  gap_shuffled       = {final['gap_shuffled']:+.4f}")
    print(f"  sensitivity_shuf   = {final['sensitivity_shuffled']:.4f}")
    print(f"  Gate D (real<shuf) : {'PASS' if gate_d else 'FAIL'}")
    print(f"  metrics -> {out_dir / 'goal_utility_metrics.json'}")
    print(f"  checkpoint -> {ckpt_path}")
    print(f"  wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
