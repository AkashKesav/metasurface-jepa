import os
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/metasurface-jepa")
subprocess.run([
    "git", "clone", "--depth", "1", "--branch", "fix/unified-final",
    "https://github.com/AkashKesav/metasurface-jepa.git", str(repo)
], check=True)
subprocess.run(["python", "-m", "pip", "install", "-q", "-r",
                str(repo / "requirements.txt")], check=True)

data_root = None
for root, _, _ in os.walk("/kaggle/input"):
    p = Path(root)
    if (p / "split_data" / "train_set.mat").exists() and \
       (p / "weights" / "surrogate_model.bin").exists():
        data_root = p
        break
if data_root is None:
    raise RuntimeError("MetaDiT staging dataset not found under /kaggle/input")

subprocess.run([
    "python", str(repo / "scripts/eval/stage4_goal_utility.py"),
    "--config", str(repo / "configs/unified.yaml"),
    "--data-root", str(data_root), "--total-steps", "1500",
    "--eval-every", "250", "--device", "cuda",
    "--lambda-phys", "0.8", "--lambda-goal", "2.0",
    "--goal-margin", "0.01", "--seed", "42",
    "--output-dir", "/kaggle/working/results/goal_margin",
], cwd=repo, check=True)
