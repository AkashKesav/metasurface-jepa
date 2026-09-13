# CLOUD_TRAINING.md — Kaggle / Colab Training Runbook (unified 192-D path)

This is the **canonical** cloud-training workflow referenced by `AGENTS.md`. Any work whose
task involves gradient-based training points here instead of re-deriving its own cloud setup.
The local machine (RTX 3050, 4GB VRAM / 16GB RAM) is dev-only — see `AGENTS.md` Standing Rule 5.

The single training entry point is a standalone CLI:

```bash
python scripts/train/train_unified.py --config configs/unified.yaml [--resume <path>] \
    [--device cuda] [--max-steps N] [--preflight] [--no-train] [--use-synthetic-smoke]
```

so it can be invoked identically from either platform below with no notebook-specific glue.

> The legacy `milestone_b_preflight.py` / `train_milestone_b.py` workflow was retired with the
> 384-D path on 2026-09-13 (`AGENTS.md` → Dated operator overrides).

---

## 0. One-time setup (per platform, not per session)

### Repository
Push the repo to GitHub (private is fine; both platforms support token auth). Keep the branch
you train from fixed and record the commit in the run report — the architecture is pinned by
commit, not by "latest".

### Dataset + released weights staging
Everything comes from the MetaDiT release (**dataset and released weights, not re-downloaded per
session**). The trainer expects this layout under `data/metadit/`:

```text
data/metadit/
  split_data/train_set.mat
  split_data/val_set.mat
  weights/spec_encoder.pth        # released spectrum encoder
  weights/metadit-small.bin       # released DiT (reference only)
  weights/surrogate_model.bin     # frozen EM surrogate (physics path)
```

- **Kaggle**: create a private Kaggle Dataset from your downloaded `data/metadit/` folder, then
  attach it to the notebook (Add Data). It appears under `/kaggle/input/<dataset-name>/`.
- **Colab**: upload `data/metadit/` once to a fixed Drive folder and mount it each session.

Do **not** commit dataset or weight files to the repo.

### Dependency contract
`requirements.txt` pins the tested combination (PyTorch 2.5.1 / Torchvision 0.20.1). Install with
`pip install -r requirements.txt`; do not change the pins without re-running the full test suite
and the preflight below.

---

## 1. Kaggle workflow (per session)

1. New Notebook → Settings → Accelerator **GPU T4 x2 / P100**; turn **Internet ON** (clone/install).
2. Attach your staged dataset (Add Data).
3. Clone the pinned repo state and install:
   ```python
   !git clone https://github.com/<you>/<repo>.git
   %cd <repo>
   !git checkout <pinned-commit-or-branch>
   !pip install -r requirements.txt
   ```
4. Stage the dataset (symlink is preferred; a real directory is never clobbered):
   ```python
   !ln -s /kaggle/input/<dataset-name>/metadit data/metadit
   !ls data/metadit/split_data data/metadit/weights
   ```
5. **Preflight (mandatory, real data, end-to-end)**:
   ```python
   !python scripts/train/train_unified.py --config configs/unified.yaml --device cuda --preflight
   ```
   This loads real splits and released weights, runs a forward/backward with the shared
   ownership checks (student modules get gradients; both EMA targets, the released spectrum
   encoder, and the frozen surrogate receive **none**), and exits non-zero on any failure.
   **Non-zero exit = do not start training.**
6. **Verification-scale run (5–10 % of the schedule)** — proves the full real-data pipeline
   (data loading → training step → EMA updates → checkpoint write) before committing GPU-hours:
   ```python
   !python scripts/train/train_unified.py --config configs/unified.yaml --device cuda --max-steps 150
   ```
   (`--max-steps` overrides `train.total_steps`; with `total_steps: 1500` in the config, 150
   steps ≈ 10 %. Checkpoints land in `checkpoints/unified/`.)
   Inspect the printed per-step losses and the final JSON. These checkpoints are **verification
   artifacts, not results** — delete them before the full run (or deliberately resume from them;
   the shortened schedule affects the LR/EMA ramp, so a fresh full run is the clean default):
   ```python
   !rm -f checkpoints/unified/*.pt
   ```
7. **Full run** (resume whenever a checkpoint from a previous session exists):
   ```python
   !python scripts/train/train_unified.py --config configs/unified.yaml --device cuda
   # or:
   !python scripts/train/train_unified.py --config configs/unified.yaml --device cuda \
       --resume checkpoints/unified/latest.pt
   ```
   Training writes `checkpoints/unified/latest.pt` frequently and `final.pt` at the end
   (atomic writes; `checkpoints/**/*.pt` is gitignored). Keep the working directory (or a
   symlinked persistent folder) stable so `--resume` finds them.
8. **Evaluate** (after any training): the authoritative per-scenario evaluator —
   `A` pure inverse design (full occupancy mask + all scalars unknown — the hard stratum),
   `B` partial-parameter conditioning, `C` retrofit; never pooled:
   ```python
   !python scripts/eval/eval_scenarios.py --config configs/unified.yaml \
       --checkpoint checkpoints/unified/latest.pt --device cuda
   ```
   And the §20.3 guidance-gap curve across mask-ratio buckets:
   ```python
   !python scripts/diagnostics/run_guidance_gap_sweep.py --config configs/unified.yaml \
       --checkpoint checkpoints/unified/latest.pt --device cuda
   ```
   The acceptance gate is the **hard stratum** real-vs-shuffled physics-consistency gap
   (`architecture_v5.md` §8.3 check 8) — never report a pooled gap.
9. **Before the session ends** (Kaggle sessions cap at ~9–12 h; quota ~30 GPU-h/week):
   save a notebook version ("Save & Run All") so `/kaggle/working/` is snapshotted, and/or push
   results back (`git add -f checkpoints/unified/*.pt` is wrong — push the **report** and keep
   checkpoint files on Kaggle output/Drive; see §3).

---

## 2. Colab workflow (per session)

1. Runtime → Change runtime type → **GPU** (T4 free tier; A100/Pro T4 if available).
2. Mount Drive — checkpoints and dataset live there (Colab local disk is wiped each session):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Clone + install (code does not need to persist on Drive):
   ```python
   !git clone https://github.com/<you>/<repo>.git
   %cd <repo>
   !git checkout <pinned-commit-or-branch>
   !pip install -r requirements.txt
   ```
4. Stage data + checkpoints on Drive:
   ```python
   !ln -s /content/drive/MyDrive/<project>/data/metadit data/metadit
   !mkdir -p /content/drive/MyDrive/<project>/checkpoints/unified
   !ln -s /content/drive/MyDrive/<project>/checkpoints checkpoints
   ```
5. Preflight + verification run + full run, exactly as in §1 steps 5–7 (same commands; `--device cuda`).
6. Because `checkpoints/` is symlinked to Drive, checkpoints survive disconnects — but confirm
   files are actually landing on Drive after the first checkpoint.
7. Evaluate as in §1 step 8.

**Watch out for:** free-tier idle timeouts and ~12 h hard caps — resume from
`checkpoints/unified/latest.pt` in the next session instead of restarting.

---

## 3. Sync-back checklist (before closing every cloud session)

- [ ] `checkpoints/unified/latest.pt` (and `final.pt` if the run completed) persisted somewhere
      durable — Kaggle output / Drive — not only the ephemeral session disk.
- [ ] `checkpoints/unified/REPORT.md` updated: platform/GPU, commit, steps run, loss/regime
      observations, whether the verification-scale run passed, resume path, deviations.
- [ ] Report + code changes pulled back locally (`git pull`) before the next coding session —
      the next session assumes the report is current.
- [ ] If the run is unfinished: note the exact `--resume` path and remaining steps in the report.

---

## 4. Post-training verification (what must be recorded)

1. `scripts/eval/eval_scenarios.py` output per scenario (A/B/C separately), including:
   - hard-stratum real/null/shuffled comparison (the gate: real must beat shuffled),
   - occupancy IoU/F1 on the occupied class and predicted occupancy-fraction variability,
   - scalar MAE on unknown positions (and known-position diagnostics as labelled),
   - generative-diversity numbers, and the NN-retrieval baseline.
2. `run_guidance_gap_sweep.py` curve (§20.3).
3. Anything that failed, with observed numbers — see `AGENTS.md` "If something fails": never
   loosen a gate silently; escalate to the operator.

No scientific claim is valid from a session that did not run the training; a coding session
that reads cloud artifacts must verify them (checkpoint schema, step counts, config hash)
before trusting them.
