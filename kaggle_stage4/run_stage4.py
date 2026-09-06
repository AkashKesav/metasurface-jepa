import os
import subprocess
from pathlib import Path

REPO = Path("/kaggle/working/metasurface-jepa")
subprocess.run([
    "git", "clone", "--depth", "1", "--branch", "fix/unified-final",
    "https://github.com/AkashKesav/metasurface-jepa.git", str(REPO)
], check=True)
subprocess.run(["python", "-m", "pip", "install", "-q", "-r",
                str(REPO / "requirements.txt")], check=True)

data_root = None
for root, dirs, files in os.walk("/kaggle/input"):
    root_path = Path(root)
    if (root_path / "split_data" / "train_set.mat").exists() and \
       (root_path / "weights" / "surrogate_model.bin").exists():
        data_root = root_path
        break
if data_root is None:
    raise RuntimeError("MetaDiT staging dataset not found under /kaggle/input")

print(f"DATA_ROOT={data_root}", flush=True)
cmd = [
    "python", str(REPO / "scripts/eval/stage4_goal_utility.py"),
    "--config", str(REPO / "configs/unified.yaml"),
    "--data-root", str(data_root / "metadit" if (data_root / "metadit").exists() else data_root),
    "--lambda-phys", "1.0", "--total-steps", "1500", "--eval-every", "250",
    "--output-dir", "/kaggle/working/results/stage4",
]
subprocess.run(cmd, cwd=REPO, check=True)
