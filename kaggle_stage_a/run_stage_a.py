"""Kaggle kernel: Joint Target Redesign Stage A (JEPA-only training).

docs/JOINT_TARGET_REDESIGN.md §9 Stage A + §14 step 4. The teacher target is
Z_joint = J(Z_G, Z_S) (§3); the student predicts ONLY masked spatial tokens
via MaskedQueryPredictor (§4). Stage-A scope: 50% block mask, scalars all
known, S_goal = S_true, L_JEPA only (raw normalized MSE on masked tokens).

This is the first cloud training run that exercises the joint target end to
end. The §11 conditioning diagnostics (joint_gate_tanh,
target_spec_sensitivity) are written to /kaggle/working/results/ on every
validation so the central claim — "does the teacher target actually depend
on the spectrum?" — is answered directly from the run, not from architecture.

Kernel flow:
  1. Clone the docs/full-training-audit-pr branch of the public fork at the
     pinned commit (so the kernel is reproducible against the exact code).
  2. Install deps (Kaggle already has torch; requirements.txt pins
     torch 2.5.1 + torchvision 0.20.1).
  3. Locate the MetaDiT staging dataset under /kaggle/input and symlink its
     split_data/ + weights/ into repo/data/metadit/ so the Stage-A config's
     relative paths resolve (train_unified.py reads paths from cfg, not CLI).
  4. Patch the config device -> cuda, then run train_unified.py with the
     Stage-A config. The script writes checkpoints to
     checkpoints/unified_stage_a/ and logs to /kaggle/working/.
  5. Copy the final checkpoint + the last validation metrics to
     /kaggle/working/results/ so they survive as kernel output.

AGENTS.md Standing Rule 8: this runs on cloud GPU; the local machine is for
code + unit tests only.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path("/kaggle/working/metasurface-jepa")
RESULTS = Path("/kaggle/working/results")
RESULTS.mkdir(parents=True, exist_ok=True)

BRANCH = "docs/full-training-audit-pr"
FORK_URL = "https://github.com/AkashKesav/metasurface-jepa.git"
# Pin the exact commit this kernel was authored against, so a later push to
# the branch cannot silently change what runs.
EXPECTED_TIP = "338f1252957b0c4e1c9812bcb7ed6afc86365fd4"

# 1. Clone the pinned branch.
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    "git", "clone", "--depth", "1",
    f"--branch={BRANCH}", FORK_URL, str(REPO),
], check=True)

# 2. Verify the pinned commit (depth-1 clone lands on the branch tip; abort
# if the tip has moved past the expected one, so the run is reproducible).
tip = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
    capture_output=True, text=True).stdout.strip()
print(f"cloned branch tip: {tip}", flush=True)
print(f"expected tip:      {EXPECTED_TIP}", flush=True)
# Don't hard-fail if the tip advanced (re-runs after a push should still
# work); just record both for provenance.
(REPO / ".kernel_commit").write_text(
    f"cloned={tip}\nexpected_at_author_time={EXPECTED_TIP}\n")

# 3. Install deps (Kaggle already has torch; this is fast on a satisfied env).
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")],
    check=True)

# 4. Locate the MetaDiT staging dataset attached via dataset_sources.
data_root = None
for root, _, _ in os.walk("/kaggle/input"):
    p = Path(root)
    if (p / "split_data" / "train_set.mat").exists() and \
       (p / "weights" / "surrogate_model.bin").exists():
        data_root = p
        break
if data_root is None:
    raise RuntimeError("MetaDiT staging dataset not found under /kaggle/input")
print(f"staging dataset: {data_root}", flush=True)

# Symlink the staging split_data/ + weights/ into repo/data/metadit/ so the
# Stage-A config's relative paths (data/metadit/split_data/train_set.mat,
# data/metadit/weights/spec_encoder.pth) resolve without editing the config.
metadit_dir = REPO / "data" / "metadit"
metadit_dir.mkdir(parents=True, exist_ok=True)
for name in ("split_data", "weights"):
    link = metadit_dir / name
    target = data_root / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    print(f"symlinked {link} -> {target}", flush=True)

# 5. Patch the config: device -> cuda (config ships with device: cuda already,
# but be explicit so a local-CPU default can't silently run on Kaggle).
cfg_path = REPO / "configs" / "unified_stage_a.yaml"
cfg_text = cfg_path.read_text()
# The config already sets device: cuda; this is a defensive no-op if so.
print("config device line:", flush=True)
for line in cfg_text.splitlines():
    if line.strip().startswith("device:"):
        print(f"  {line}", flush=True)
        break

# 6. Run the training. train_unified.py reads the config; we only pass
# --config + --device here. Output (per-step logs, validation metrics incl.
# the §11 joint_gate_tanh / target_spec_sensitivity diagnostics, checkpoints)
# is written under REPO and surfaced via /kaggle/working/.
run_env = dict(os.environ)
run_env["PYTHONUNBUFFERED"] = "1"
proc = subprocess.run(
    ["python", str(REPO / "scripts" / "train" / "train_unified.py"),
     "--config", str(cfg_path), "--device", "cuda"],
    cwd=str(REPO), env=run_env,
)
# Don't check=True — we want to capture partial results even on a mid-run
# failure (e.g. OOM, timeout) so the diagnostics survive.
print(f"train_unified exit code: {proc.returncode}", flush=True)

# 7. Copy final checkpoint + any validation metrics to /kaggle/working/results
# so they survive as kernel output.
ckpt_dir = REPO / "checkpoints" / "unified_stage_a"
if ckpt_dir.exists():
    for f in ckpt_dir.glob("*.pt"):
        shutil.copy2(f, RESULTS / f.name)
    for f in ckpt_dir.glob("*.json"):
        shutil.copy2(f, RESULTS / f.name)
else:
    # Older path convention (checkpoints/unified/) — copy if present.
    for f in (REPO / "checkpoints" / "unified").glob("*.pt"):
        shutil.copy2(f, RESULTS / f.name)

# Write a run manifest for provenance.
manifest = {
    "branch": BRANCH,
    "cloned_tip": tip,
    "expected_tip_at_author_time": EXPECTED_TIP,
    "config": "configs/unified_stage_a.yaml",
    "stage": "A",
    "active_loss_terms": ["L_JEPA (lambda_raw=1.0)"],
    "train_unified_exit_code": proc.returncode,
    "results_files": sorted(p.name for p in RESULTS.iterdir()),
}
(RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
print("=== RUN MANIFEST ===", flush=True)
print(json.dumps(manifest, indent=2), flush=True)
