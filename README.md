# IMT2024031 - Final Polynomial Regression Submission

## Final selected models

| Problem | Regularizer | Degree | Alpha | 5-fold OOF MSE | 5-fold OOF R2 | Non-zero coefficients |
|---|---|---:|---:|---:|---:|---:|
| var1 | Lasso (L1) | 5 | 0.00788046 | 0.327877 | 0.965743 | 149 / 461 |
| var2 | Lasso (L1) | 11 | 0.00045000 | 0.235043 | 0.994675 | 163 / 363 |

The final configurations were selected through polynomial-degree and regularization-strength
cross-validation comparing L1 and L2 regularization. The final fixed models were then
re-evaluated with 5-fold out-of-fold predictions and refit on all 1000 training samples.

## Final prediction files

- `predictions/IMT2024031_pred_var1.csv`
- `predictions/IMT2024031_pred_var2.csv`

Each file contains exactly one column named `y` and 1000 predictions.

## Reproducibility

Training/test data are included under `data/`.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Reproduce the final models and prediction files:

```bash
python3 code/train_final.py
```

Generate the diagnostic figures:

```bash
python3 code/plots.py
```

The expensive degree-by-degree L1/L2 selection experiment is also included:

```bash
python3 code/model_selection_l1_l2.py
```

That selection script can take substantially longer for the high-degree models.

## Final diagnostics

The plots in `plots/` include:
- predicted vs actual out-of-fold predictions
- coefficient magnitude/rank plots
- out-of-fold residual plots
- final model metric summary

## Verification

Selected-degree L1/L2 checks (same five-fold shuffled splits and fold-wise standardization):
- var1, degree 5: L1 MSE = 0.327877; L2 MSE = 0.513036 -> L1 wins.
- var2, degree 11: L1 MSE = 0.235043; L2 MSE = 0.236987 -> L1 wins.

A detailed CSV is included at `results/selected_degree_l1_l2_comparison.csv`.

The submitted prediction CSVs were checked against the final training pipeline:
- var1 maximum absolute prediction difference: approximately `2.1e-14`
- var2 maximum absolute prediction difference: approximately `1.2e-13`

These are floating-point roundoff differences, so the supplied prediction CSVs match the final model outputs.

## Repository

GitHub repository:
https://github.com/Heer-create-lgtm/ml_assignment_1
