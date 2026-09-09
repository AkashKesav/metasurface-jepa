import json
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

script = repo / "scripts/eval/stage4_goal_utility.py"
root = Path("/kaggle/working/results/sweep")
root.mkdir(parents=True, exist_ok=True)
summary = {}
for lam in (0.1, 0.2, 0.4, 0.8):
    name = f"lambda_{lam:g}"
    subprocess.run([
        "python", str(script), "--config", str(repo / "configs/unified.yaml"),
        "--data-root", str(data_root), "--total-steps", "1500",
        "--eval-every", "500", "--device", "cuda",
        "--lambda-phys", str(lam), "--seed", "42",
        "--output-dir", str(root / name),
    ], cwd=repo, check=True)
    history = json.loads((root / name / "goal_utility_metrics.json").read_text())
    summary[name] = history[-1]
(root / "SWEEP.json").write_text(json.dumps({
    "architecture_id": "unified_occ_param_spectrum_jepa_v1",
    "lambdas": [0.1, 0.2, 0.4, 0.8],
    "seed": 42,
    "provenance": "same repaired branch/config/dataset/seed; lambda_phys only differs",
    "results": summary,
}, indent=2))
print(json.dumps(summary, indent=2), flush=True)
