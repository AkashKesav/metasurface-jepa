"""Kaggle kernel: run the metasurface-jepa test suite on the cloud GPU env.

Clones the docs/full-training-audit-pr branch of the public fork, installs
deps, and runs pytest -q. Saves the summary + a per-test breakdown to
/kaggle/working/results so it can be pulled via kaggle CLI output download.

This is a PREFLIGHT validation run — not training (AGENTS.md Standing Rule 8:
local machine for code+tests, cloud GPU for runs that need a GPU; here we
validate the suite passes in the Kaggle environment before any training
spend). Expected: ~547 passed, ~14-30 skipped (data-dependent tests skip
cleanly when weights/splits are absent on the fresh Kaggle clone).
"""
import os
import subprocess
from pathlib import Path

REPO = Path("/kaggle/working/metasurface-jepa")
RESULTS = Path("/kaggle/working/results")
RESULTS.mkdir(parents=True, exist_ok=True)

BRANCH = "docs/full-training-audit-pr"
FORK_URL = "https://github.com/AkashKesav/metasurface-jepa.git"

# 1. Clone the exact branch we just pushed (public fork — anonymous clone OK).
subprocess.run([
    "git", "clone", "--depth", "1", "--branch", BRANCH, FORK_URL, str(REPO)
], check=True)

# 2. Pin the commit we expect (6b7de73 = the Kaggle-fix tip).
expected_tip = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
).stdout.strip()
print(f"cloned branch tip: {expected_tip}", flush=True)

# 3. Install deps (Kaggle already has torch; requirements.txt pins
# torch 2.5.1 + torchvision 0.20.1 — pip will see satisfied deps quickly).
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")],
    check=True,
)

# 4. Run the full suite. Invoke with NO positional path so pytest.ini's
# `testpaths = tests` applies — otherwise pytest would also try to collect
# the vendored external/lejepa/tests/ subtree, whose `lejepa` import isn't
# installed in the Kaggle env (10 collection errors, exit 2). Use -p
# no:cacheprovider for a clean run, and capture a junit XML for structured
# per-test results.
xml_path = RESULTS / "junit.xml"
cmd = [
    "python", "-m", "pytest",
    "-q",
    "-p", "no:cacheprovider",
    "--junitxml", str(xml_path),
    "-o", "python_files=tests/test_*.py",
]
# Run from the repo root so pytest.ini is discovered + conftest/sys.path
# injection applies.
proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)

# 5. Persist raw output for offline inspection.
(RESULTS / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
(RESULTS / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
print("=== STDOUT (tail) ===", flush=True)
print(proc.stdout[-4000:], flush=True)
if proc.stderr.strip():
    print("=== STDERR (tail) ===", flush=True)
    print(proc.stderr[-2000:], flush=True)
print(f"=== exit code: {proc.returncode} ===", flush=True)
print(f"=== junit: {xml_path} ===", flush=True)

# Surface the pytest summary line (e.g. "563 passed, 14 skipped in 209.33s").
for line in proc.stdout.splitlines()[::-1][:5]:
    if "passed" in line or "failed" in line or "error" in line:
        print(f"SUMMARY: {line.strip()}", flush=True)
        break
