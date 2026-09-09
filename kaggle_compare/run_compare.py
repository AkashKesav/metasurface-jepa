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
base = ["python", str(script), "--config", str(repo / "configs/unified.yaml"),
        "--data-root", str(data_root), "--total-steps", "1500",
        "--eval-every", "250", "--device", "cuda"]
for name, lam in (("baseline", "0.0"), ("physics_fixed", "0.8")):
    subprocess.run(base + ["--lambda-phys", lam, "--seed", "42",
                           "--output-dir", f"/kaggle/working/results/{name}"],
                   cwd=repo, check=True)

result = {"architecture_id": "unified_occ_param_spectrum_jepa_v1",
          "data_root": str(data_root),
          "baseline_lambda_phys": 0.0,
          "physics_fixed_lambda_phys": 0.8,
          "provenance": "same branch/config/dataset/seed; lambda_phys only differs"}
for name in ("baseline", "physics_fixed"):
    path = Path(f"/kaggle/working/results/{name}/goal_utility_metrics.json")
    history = json.loads(path.read_text())
    result[name] = history[-1]
Path("/kaggle/working/results/COMPARISON.json").write_text(
    json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
