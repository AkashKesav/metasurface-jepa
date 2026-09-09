"""A finished run must leave behind the evidence needed to judge it.

Context (Stage-A, 2026-09-09): the Kaggle run's loss landed at L_total ~1e-5,
which reads as a perfect result. It was actually a collapse. The run was
unjudgeable afterwards for two reasons, both fixed here:

1. Cloud runs persist only the `checkpoints/` tree; stdout is NOT downloadable
   via `kaggle kernels output`. `validate()` computed raw_mse, raw_cos_err and
   latent norms and printed them — and every one was lost, because
   `save_checkpoint` was handed `metrics={"L_total": last_loss}` only.
2. Nothing recorded how fast the run actually went, so "it finished in
   10 minutes" could not be split into setup vs stepping vs validation.

These tests lock both guarantees in.
"""

import json
import os
import shutil
import sys
import tempfile

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "train"))


def _load_cfg():
    with open(os.path.join(REPO_ROOT, "configs", "unified.yaml")) as f:
        return yaml.safe_load(f)


def _run_tiny_train(cfg, tmpdir):
    """Run a few synthetic steps, capturing stdout the way a cloud log would."""
    from train_unified import train

    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    cfg["data"]["val_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    cfg["weights"]["spectrum"] = os.path.join(tmpdir, "dummy_spec.pth")
    cfg["data"]["use_synthetic"] = True
    cfg["train"]["total_steps"] = 4
    cfg["train"]["grad_accum"] = 1
    cfg["train"]["ckpt_every_steps"] = 1000
    cfg["train"]["log_every_steps"] = 1
    cfg["train"]["val_every_steps"] = 2
    cfg["train"]["guidance_dropout"] = 0.0
    cfg["curriculum"]["train_mask_ratios"] = [0.5]
    cfg["curriculum"]["train_mask_ratio_probs"] = [1.0]
    cfg["curriculum"]["scalar_regimes"] = ["all_known"]
    cfg["curriculum"]["scalar_regime_probs"] = [1.0]
    cfg["loss"]["lambda_phys"] = 0.0
    return train(cfg, no_train=False, device="cpu", use_synthetic_smoke=True)


def test_validation_metrics_are_persisted_not_just_printed():
    """Every validation point must reach disk with the raw-latent diagnostics.

    The discriminator is `raw_cos_err` / `raw_z_y_norm`: these were computed
    and printed before, but never written, which is exactly why the Stage-A
    collapse had to be reconstructed by hand from the weights afterwards.
    """
    from train_unified import train  # noqa: F401  (import path sanity)

    cfg = _load_cfg()
    subdir = "test_metrics_persistence"
    cfg["checkpoint_subdir"] = subdir
    metrics_dir = os.path.join(REPO_ROOT, "checkpoints", subdir)
    shutil.rmtree(metrics_dir, ignore_errors=True)

    try:
        with tempfile.TemporaryDirectory() as td:
            _run_tiny_train(cfg, td)

        hist_path = os.path.join(metrics_dir, "metrics_history.json")
        assert os.path.exists(hist_path), (
            "validation metrics were not persisted to metrics_history.json")

        with open(hist_path) as f:
            hist = json.load(f)

        assert hist["val"], "no validation points recorded"
        assert hist["train_loss"], "no training-loss points recorded"

        point = hist["val"][0]
        assert "step" in point
        # The scale/collapse diagnostics that distinguish learning from
        # collapse. raw_cos_err alone is NOT enough — it read ~0 during the
        # Stage-A collapse.
        for key in ("raw_mse", "raw_cos_err", "raw_z_hat_norm",
                    "raw_z_y_norm", "scale_ratio_zh_zy"):
            assert key in point, (
                f"{key} missing from the persisted validation point; without "
                f"it a finished run cannot be judged")
    finally:
        shutil.rmtree(metrics_dir, ignore_errors=True)


def test_final_metrics_records_wall_clock():
    """'How long did it take' must be answerable from the output directory."""
    cfg = _load_cfg()
    subdir = "test_metrics_persistence"
    cfg["checkpoint_subdir"] = subdir
    metrics_dir = os.path.join(REPO_ROOT, "checkpoints", subdir)
    shutil.rmtree(metrics_dir, ignore_errors=True)

    try:
        with tempfile.TemporaryDirectory() as td:
            _run_tiny_train(cfg, td)

        path = os.path.join(metrics_dir, "final_metrics.json")
        assert os.path.exists(path), "final_metrics.json was not written"
        with open(path) as f:
            final = json.load(f)

        assert final["wall_clock_seconds"] > 0
        assert final["seconds_per_step"] > 0
        assert final["steps_run"] == 4
        assert final["n_val_points"] >= 1
    finally:
        shutil.rmtree(metrics_dir, ignore_errors=True)


def test_checkpoint_carries_full_validation_metrics():
    """best_prediction must hold more than L_total.

    A checkpoint whose only metric is a scale-free loss is not evidence of
    anything — that is precisely the state Stage-A's final.pt was found in.
    """
    import torch
    from train_unified import train  # noqa: F401

    cfg = _load_cfg()
    subdir = "test_metrics_persistence"
    cfg["checkpoint_subdir"] = subdir
    metrics_dir = os.path.join(REPO_ROOT, "checkpoints", subdir)
    shutil.rmtree(metrics_dir, ignore_errors=True)

    try:
        with tempfile.TemporaryDirectory() as td:
            _run_tiny_train(cfg, td)

        ckpt_path = os.path.join(metrics_dir, "final.pt")
        assert os.path.exists(ckpt_path)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        bp = ck.get("best_prediction") or {}
        metrics = bp.get("metrics") or {}
        assert "raw_cos_err" in metrics, (
            "checkpoint metrics contain only the scale-free loss; the "
            "raw-latent diagnostics were dropped")
    finally:
        shutil.rmtree(metrics_dir, ignore_errors=True)


def _load_verdict_fn():
    """Load verdict_from_metrics from the script without making scripts/ a pkg."""
    import importlib.util

    path = os.path.join(REPO_ROOT, "scripts", "eval",
                        "eval_checkpoint_latents.py")
    spec = importlib.util.spec_from_file_location("_ckpt_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verdict_from_metrics


# The exact numbers measured on the Stage-A final checkpoint (r=0.5, val).
STAGE_A_METRICS = {
    "L_total": 1.27379e-05,
    "raw_mse": 235.672,
    "raw_cos_err": 0.00122283,
    "z_hat_norm": 13.8599,
    "z_y_norm": 226.521,
    "z_hat_concentration": 0.000638179,
    "scale_ratio_zh_zy": 0.0611858,
    "joint_target_delta_rel": 4.65706,
    "target_spec_sensitivity_normalized": 0.00338377,
    "joint_gate_tanh": -0.0943533,
}


def test_evaluator_fails_the_stage_a_signature():
    """The offline gate must FAIL on the numbers Stage-A actually produced.

    This is the regression that matters: Stage-A was reported as a working run
    because L_total was 1e-5. Any future change that weakens the gate to
    "loss went down" should break here.
    """
    verdict_from_metrics = _load_verdict_fn()
    failed, _notes = verdict_from_metrics(dict(STAGE_A_METRICS))

    kinds = " | ".join(failed)
    assert "COLLAPSE" in kinds, f"collapse not detected: {failed}"
    assert "SCALE MISMATCH" in kinds, f"scale mismatch not detected: {failed}"
    assert "FUSION DOMINATES" in kinds, f"fusion not detected: {failed}"
    assert "GOAL DEAD" in kinds, f"dead goal not detected: {failed}"


def test_evaluator_passes_a_healthy_signature():
    """A genuinely healthy run must not be failed by the gate.

    Guards against the opposite error — a gate so strict that no real run can
    ever pass it, which would just train us to ignore it.
    """
    verdict_from_metrics = _load_verdict_fn()
    healthy = {
        "raw_mse": 0.42,
        "z_hat_norm": 47.0,
        "z_y_norm": 50.0,
        "z_hat_concentration": 0.28,
        "scale_ratio_zh_zy": 0.94,
        "joint_target_delta_rel": 0.31,
        "target_spec_sensitivity_normalized": 0.19,
        "joint_gate_tanh": 0.42,
    }
    failed, _notes = verdict_from_metrics(healthy)
    assert not failed, f"healthy run wrongly failed: {failed}"
