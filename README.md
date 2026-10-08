# IMT2024031 - Revised L1/L2 polynomial regression submission

The submitted predictions use Lasso (L1) for both problems. Ridge (L2) is the comparison; Elastic Net is not used.

| Dataset | Degree | Lasso alpha | Pooled OOF MSE | Pooled OOF R2 |
|---|---:|---:|---:|---:|
| var1 | 5 | 0.007880462815669913 | 0.3278774124501606 | 0.9657434527694847 |
| var2 | 11 | 0.00045 | 0.23504296873727482 | 0.9946747184238399 |

## Validation interpretation
Five shuffled folds (seed 42) are used. StandardScaler is fitted within each training fold. These folds informed model selection, so the scores are tuning-CV results, not independent nested-CV or hidden-test performance. R2 is not classification accuracy. Final models train on all 1,000 training rows.

## Reproduce final predictions and figures
Run from this folder:
```
python3 -m pip install -r requirements.txt
python3 code/train_final.py
python3 code/plots.py
python3 code/verify_submission.py
```
Training overwrites the corresponding generated predictions and OOF tables. Verification refits in memory and compares existing files without replacing them. Its numerical tolerances accommodate minor library/platform differences. Tested dependency versions are listed in requirements-tested.txt.

## Search evidence

The audit was interrupted for the submission deadline. See results/search_coverage.csv for completed/pending degrees. Pending results are blank, never invented. Rerun code/search_audit.py to finish pending degrees using saved checkpoints. Completed degrees: {'var1': [1, 2, 3, 4, 5, 6], 'var2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}

The original archive did not include full per-degree search logs. This revision includes a newly computed independent audit, not fabricated historical results:

- `results/search_audit_all_candidates.csv`: completed degrees/alphas, scores and convergence diagnostics.
- `results/var1_per_degree_search_audit.csv` and `var2_per_degree_search_audit.csv`: best converged candidate within each family and degree.
- `results/search_var*_degree*.csv`: individual degree checkpoints.
- `results/selected_degree_l1_l2_comparison.csv`: verified original selected-degree comparison.

To reproduce the independent audit (existing per-degree checkpoint files are reused; delete them first for a fresh run):
```
python3 code/search_audit.py
```
The audit uses warm-start Lasso paths, tol=1e-4 and max_iter=10000 (2000 for var1 degrees 7-10 and var2 degrees 15-20), with up to six CPU workers. Ridge uses an SVD solution. Non-converged Lasso candidates are flagged and excluded from per-degree minima. This is a finite numerical search; it does not certify a global optimum. Audit scores may differ slightly from cold-start Lasso. The verified submitted configurations are preserved.

To run the original cold-start exhaustive search (max_iter=100000; potentially slow):
```
python3 code/model_selection_l1_l2.py
```
It writes `results/IMT2024031_var1_l1_l2_per_degree.csv` and `results/IMT2024031_var2_l1_l2_per_degree.csv`. Neither search automatically changes train_final.py's CONFIG.

## Required deliverables
- `IMT2024031_assignment_report.pdf`
- `predictions/IMT2024031_pred_var1.csv`
- `predictions/IMT2024031_pred_var2.csv`
- GitHub repository with the included source code and supporting files.

Repository: https://github.com/Heer-create-lgtm/ml_assignment_1
Upload these revised files to the repository before submitting its link.

## Changes in this revision
Added independent search evidence and convergence flags; clarified selection optimism; implemented real prediction/OOF verification; made summary plots read CSV metrics; corrected report typography; removed bytecode files. Submitted test predictions and model configurations are unchanged.
