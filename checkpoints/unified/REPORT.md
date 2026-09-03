# Unified JEPA Kaggle comparison

Date: 2026-09-03  
Commit: `d1b8a36`  
Hardware: Kaggle Tesla P100 16 GB  
Preflight: PASS  

Both runs used the same seed, data, 1,500 optimizer steps, validation suite,
and frozen MetaDiT surrogate. The baseline used `lambda_phys=0`; the fixed
run used `lambda_phys=0.1` with a 200-step ramp.

| Run | Final loss | Hard spectrum MAE (all) | Hard spectrum MAE (50% mask) | Hard spectrum MAE (100% mask) |
|---|---:|---:|---:|---:|
| Baseline | 12.8869 | 0.3568 | 0.4313 | 0.4970 |
| Physics fixed, 0.1 | 12.8740 | 0.4062 | 0.5699 | 0.3416 |
| Physics sweep, 0.2 | 12.9204 | 0.3940 | 0.5527 | 0.3168 |
| Physics sweep, 0.4 | 12.9292 | 0.4190 | 0.5514 | 0.4676 |
| Physics sweep, 0.8 | 13.0875 | 0.2060 | 0.2672 | 0.2155 |
| Physics sweep, 1.0 | 13.0673 | 0.2149 | 0.2822 | 0.2257 |

The first sweep favors `lambda_phys=0.8` for surrogate spectrum accuracy,
with `1.0` close behind. This is a single-seed screening result, not a final
claim; it should be confirmed with another seed before selecting the production
weight.

The new validation suite and isolated output directories worked. One remaining
diagnostic limitation is that effective-rank estimation with the current
two-sample validation batch saturates at approximately 0.5, so its collapse
trend should not be used as a hard gate until the validation subset is larger.

Raw Kaggle JSON artifacts are retained locally under `.kaggle_compare_results/`
and `.kaggle_sweep_results/`.
