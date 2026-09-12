# Kaggle scratch archive — 2026-09-12

63 untracked scratch entries (7.9 GB) consolidated into
`kaggle_scratch_archive_2026-09-12.tar.gz` (6.0 GB) to de-clutter the workspace.
The tarball is gitignored (never commit it); this manifest is the restore index.

## Why

~2 years of pulled Kaggle outputs, sharded checkpoints, per-run metrics/logs and
dataset-staging copies accumulated at the repo root. Authoritative records
(`checkpoints/*/REPORT.md`, `RERUN_FIXED_REPORT.md`, configs) are tracked in git;
these dirs were reproducible-by-rerun caches and pre-fix evidence. Operator
approved archiving on 2026-09-12.

## Kept live (NOT archived)

- `.kaggle_stage_a_fixed/` (353 MB) — current fixed Stage-A rerun
  (`final.pt`, `metrics_history.json`, `latent_eval_fixed.json`). Promote or
  archive it only after its report is accepted.
- `kaggle_*/` kernel source dirs (tracked: `kernel-metadata.json` + `run_*.py`).
- `data/metadit/` (184 MB, live dataset + released weights — never scratch).
- `kaggle_test_suite/kernel-metadata.json` (untracked kernel source; `git add` it
  if the test-suite kernel is wanted).

## Archived (63 entries, see tarball listing) — CORRECTION 2026-09-12

`.metadit_stage_20260903/` was NOT scratch: `data/metadit/split_data` is a
symlink into `.metadit_stage_20260903/metasurface-jepa/data/metadit/split_data`,
so deleting it dangled the live dataset link (caught by a local real-data run
refusing to start). The `split_data/` subtree (1.5 GB: train/val/test .mat)
was restored from this tarball back to its original location; the tarball copy
is now backup-only. Never delete `.metadit_stage_20260903/` again without
re-homing that symlink first.

Pulled outputs (`.kaggle_compare_*`, `.kaggle_goal*`, `.kaggle_stage4*`,
`.kaggle_sweep_*`, `.kaggle_unified_*`, `.kaggle_selected_*`, `.kaggle_raw_*`,
`.kaggle_corrected_*`, `.kaggle_direct_goal*`, `.kaggle_live_*`,
`.kaggle_metasurface-jepa-*`, `.kaggle_result_*`, `.kaggle_checkpoints_eval`,
`.kaggle_probe_20260903`, `.kaggle_output_20260903`, `.kaggle_pull_20260903*`,
`.kaggle_test_suite_output`, loose `.kaggle_*.txt` logs), staging copies
(`kaggle_anosvol_dataset/` 1.7 GB, `.metadit_stage_20260903/` 1.7 GB),
`.tmp_stage4_pull/`, `p2/`.

## Verification (before deletion)

- `tar -tzf` lists 2709 entries, exit 0.
- Spot-extract + `diff`/`cmp` identical: `.kaggle_sweep_logs.txt` (small) and
  `.kaggle_sweep_final/results/lambda_0.2/final.pt` (180 MB checkpoint).
- Originals deleted only after both identity checks passed.

## Restore

```bash
tar -tzf kaggle_scratch_archive_2026-09-12.tar.gz | head      # inspect
tar -xzf kaggle_scratch_archive_2026-09-12.tar.gz -C /tmp/restore <path>  # single entry
tar -xzf kaggle_scratch_archive_2026-09-12.tar.gz            # full restore
```
