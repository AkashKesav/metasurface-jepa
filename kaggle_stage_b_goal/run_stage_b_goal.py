"""Kaggle kernel: Joint Target Redesign Stage A RUN B (full-train + goal).

Operator A/B authorized 2026-09-12 (AGENTS.md Standing Rule 9).
Run B scope: identical schedule to Run A plus goal margin loss.
set (total_steps=70000 ~= 1 epoch at batch 2), rebalanced weights
(lambda_raw=3.0, lambda_var=5.0, lambda_cov=1.0) PLUS goal margin loss (lambda_goal=2.0, margin=0.01).
Config: configs/unified_stage_b_goal.yaml (checkpoint_subdir unified_stage_b_goal).

Kernel flow:
  1. Clone the docs/full-training-audit-pr branch at the pinned commit.
  2. Install deps (requirements.txt pins torch 2.5.1 + torchvision 0.20.1).
  3. Symlink the staging dataset's split_data/ + weights/ into data/metadit/.
  4. Run train_unified.py with the Run-B config on cuda.
  5. Run eval_checkpoint_latents.py on final.pt (with init baseline) so the
     output carries a fresh PASS/FAIL verdict (no stale eval files).
  6. Copy final/latest checkpoints + metrics + fresh eval to /kaggle/working/results.

AGENTS.md Standing Rule 8: cloud GPU; local machine is code + unit tests only.
"""

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path("/kaggle/working/metasurface-jepa")
RESULTS = Path("/kaggle/working/results")
RESULTS.mkdir(parents=True, exist_ok=True)


def _fatal_hook(exc_type, exc, tb):
    # Persist ANY uncaught exception: Kaggle's log is unreachable from the
    # API, so without this a crash leaves zero evidence (the v1 mystery:
    # COMPLETE status with only .kernel_commit and no markers).
    try:
        (RESULTS / "_FATAL_.txt").write_text(
            "".join(traceback.format_exception(exc_type, exc, tb))[-8000:]
        )
    finally:
        sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _fatal_hook

_stage_n = 0


def mark(name, text=""):
    # Stage-marker files: written to RESULTS as the script progresses, so a
    # mid-run ERROR still leaves a trail showing exactly how far it got.
    global _stage_n
    _stage_n += 1
    (RESULTS / f"{_stage_n:02d}_{name}.txt").write_text(text)
    print(f"[stage {_stage_n:02d} {name}] {text[:200]}", flush=True)


BRANCH = "docs/full-training-audit-pr"
FORK_URL = "https://github.com/AkashKesav/metasurface-jepa.git"
# Pin the exact code commit this kernel was authored against. Updated at push
# time to the commit carrying configs/unified_stage_b_goal.yaml.
EXPECTED_TIP = "c7c473ffd0edc16f4b8a2192845e3c4ad6a2f698"

# 1. Clone the pinned branch.
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        f"--branch={BRANCH}",
        FORK_URL,
        str(REPO),
    ],
    check=True,
)
# Prune VCS metadata: unneeded for the run and halves the output bundle
# (which has been truncating downloads).
shutil.rmtree(REPO / ".git", ignore_errors=True)

tip = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
print(f"cloned branch tip: {tip}", flush=True)
print(f"expected tip:      {EXPECTED_TIP}", flush=True)
(REPO / ".kernel_commit").write_text(
    f"cloned={tip}\nexpected_at_author_time={EXPECTED_TIP}\n"
)
mark("clone_done", f"tip={tip} expected={EXPECTED_TIP}")

# 2. Install deps — skipped when the image already carries the pinned torch
# (saves ~5 min and removes the biggest setup failure surface; verified and
# recorded in the marker either way).
_have = subprocess.run(
    [
        "python",
        "-c",
        "import torch; print(torch.__version__, torch.cuda.is_available())",
    ],
    capture_output=True,
    text=True,
).stdout.strip()
if _have.startswith("2.5.1"):
    _torch_ver = _have + " (pip skipped, pinned version present)"
else:
    subprocess.run(
        ["python", "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")],
        check=True,
    )
    _torch_ver = subprocess.run(
        [
            "python",
            "-c",
            "import torch; print(torch.__version__, torch.cuda.is_available())",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
mark("deps_done", f"torch={_torch_ver}")

# 3. Locate the staging dataset and symlink it in.
data_root = None
for root, _, _ in os.walk("/kaggle/input"):
    p = Path(root)
    if (p / "split_data" / "train_set.mat").exists() and (
        p / "weights" / "surrogate_model.bin"
    ).exists():
        data_root = p
        break
if data_root is None:
    raise RuntimeError("MetaDiT staging dataset not found under /kaggle/input")
print(f"staging dataset: {data_root}", flush=True)

metadit_dir = REPO / "data" / "metadit"
metadit_dir.mkdir(parents=True, exist_ok=True)
for name in ("split_data", "weights"):
    link = metadit_dir / name
    target = data_root / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    print(f"symlinked {link} -> {target}", flush=True)
mark("data_done", f"data_root={data_root}")

cfg_path = REPO / "configs" / "unified_stage_b_goal.yaml"
print("config device line:", flush=True)
for line in cfg_path.read_text().splitlines():
    if line.strip().startswith("device:"):
        print(f"  {line}", flush=True)
        break

# 4. Fail-fast preflight on the Run-A config (model build, one real sample
# through forward+physics+backward, gradient ownership incl. fusion grads).
# Runs BEFORE the 70k-step job so config/code errors surface in minutes with
# a clear cause instead of an opaque mid-run ERROR with no artifacts.
run_env = dict(os.environ)
run_env["PYTHONUNBUFFERED"] = "1"
# Captured (not inherited): the full preflight transcript persists to
# RESULTS/preflight.log whether it passes or fails.
preflight_proc = subprocess.run(
    [
        "python",
        str(REPO / "scripts" / "train" / "train_unified.py"),
        "--config",
        str(cfg_path),
        "--device",
        "cuda",
        "--preflight",
    ],
    cwd=str(REPO),
    env=run_env,
    capture_output=True,
    text=True,
)
(RESULTS / "preflight.log").write_text(
    (preflight_proc.stdout or "") + "\n--- STDERR ---\n" + (preflight_proc.stderr or "")
)
print(preflight_proc.stdout[-3000:], flush=True)
print(f"preflight exit code: {preflight_proc.returncode}", flush=True)
mark(
    "preflight_done" if preflight_proc.returncode == 0 else "preflight_FAILED",
    f"exit={preflight_proc.returncode} tail={(preflight_proc.stdout or '')[-500:]}",
)
if preflight_proc.returncode != 0:
    raise RuntimeError("preflight FAILED — refusing to start the full run")

# 5. Run the training (don't check=True: capture partial results on failure).
# stdout+stderr tee'd incrementally to RESULTS/train.log so progress survives
# even if the run dies before its first checkpoint (PYTHONUNBUFFERED=1 above
# keeps the child stream unbuffered).
train_log = open(RESULTS / "train.log", "w")
proc = subprocess.run(
    [
        "python",
        str(REPO / "scripts" / "train" / "train_unified.py"),
        "--config",
        str(cfg_path),
        "--device",
        "cuda",
    ],
    cwd=str(REPO),
    env=run_env,
    stdout=train_log,
    stderr=subprocess.STDOUT,
)
train_log.close()
print(f"train_unified exit code: {proc.returncode}", flush=True)

# 6. Fresh offline eval of the produced final.pt (kills the stale-eval class
# of confusion: this latent_eval.json is generated in-run, from this checkpoint).
ckpt_dir = REPO / "checkpoints" / "unified_stage_b_goal"
final_pt = ckpt_dir / "final.pt"
eval_proc = None
if final_pt.exists():
    eval_proc = subprocess.run(
        [
            "python",
            str(REPO / "scripts" / "eval" / "eval_checkpoint_latents.py"),
            "--checkpoint",
            str(final_pt),
            "--split",
            "val",
            "--batches",
            "2",
            "--mask-ratios",
            "0.5",
            "--device",
            "cuda",
            "--init-baseline",
            "--out",
            str(ckpt_dir / "latent_eval.json"),
        ],
        cwd=str(REPO),
        env=run_env,
    )
    print(f"eval exit code: {eval_proc.returncode}", flush=True)
else:
    print("no final.pt produced; skipping eval", flush=True)

# 6. Copy artifacts to results/.
if ckpt_dir.exists():
    for f in ckpt_dir.glob("*.pt"):
        shutil.copy2(f, RESULTS / f.name)
    for f in ckpt_dir.glob("*.json"):
        shutil.copy2(f, RESULTS / f.name)

manifest = {
    "branch": BRANCH,
    "cloned_tip": tip,
    "expected_tip_at_author_time": EXPECTED_TIP,
    "config": "configs/unified_stage_b_goal.yaml",
    "stage": "A-run-B-goal",
    "active_loss_terms": [
        "L_JEPA raw unnormalized (lambda_raw=3.0)",
        "VICReg var (lambda_var=5.0)",
        "VICReg cov (lambda_cov=1.0)",
        "goal margin (lambda_goal=2.0, margin=0.01)",
    ],
    "schedule": "total_steps=70000 (~1 epoch, batch 2), warmup=2000, val_every=500, ckpt_every=2000",
    "gates": "scale_ratio in [0.5,2], concentration>0.01, delta_rel<1, sens_norm>0.01",
    "train_unified_exit_code": proc.returncode,
    "preflight_exit_code": preflight_proc.returncode,
    "eval_exit_code": eval_proc.returncode if eval_proc is not None else None,
    "results_files": sorted(p.name for p in RESULTS.iterdir()),
}
(RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
print("=== RUN MANIFEST ===", flush=True)
print(json.dumps(manifest, indent=2), flush=True)
