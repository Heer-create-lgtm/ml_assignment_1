#!/usr/bin/env python3
"""
Model-selection experiment used for IMT2024031.

For each problem and each allowed polynomial degree, compare:
  - L1 regularization (Lasso)
  - L2 regularization (Ridge)

Regularization strength is selected by 5-fold CV on log-spaced alpha grids; the exact final var2 L1 alpha (0.00045) is also included for reproducibility.
The script writes per-degree comparison CSVs.

This is a CPU/scikit-learn implementation. It is intentionally a separate
selection script from train_final.py, which reproduces the final selected
models and predictions without rerunning the expensive selection sweep.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold

ROLL = "IMT2024031"
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
L1_ALPHAS = np.unique(np.r_[np.logspace(-5, 1, 30), 0.00045])
L2_ALPHAS = np.logspace(-6, 3, 30)

def evaluate(problem, max_degree):
    df = pd.read_csv(DATA / f"{ROLL}_train_{problem}.csv")
    X = df.drop(columns=["y"]).to_numpy(float)
    y = df["y"].to_numpy(float)
    kf = KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for degree in range(1, max_degree + 1):
        t0 = time.time()
        XP = PolynomialFeatures(degree, include_bias=False).fit_transform(X)
        l1_err = np.zeros(len(L1_ALPHAS))
        l2_err = np.zeros(len(L2_ALPHAS))
        for tr, va in kf.split(XP):
            sc = StandardScaler()
            Xtr = sc.fit_transform(XP[tr])
            Xva = sc.transform(XP[va])
            # L1 path
            order = np.argsort(L1_ALPHAS)[::-1]
            lasso = Lasso(
                alpha=float(L1_ALPHAS[order[0]]),
                max_iter=100000,
                tol=1e-4,
                selection="cyclic",
                random_state=RANDOM_STATE,
                warm_start=True,
            )
            for j in order:
                lasso.alpha = float(L1_ALPHAS[j])
                lasso.fit(Xtr, y[tr])
                p = lasso.predict(Xva)
                l1_err[j] += np.mean((y[va] - p) ** 2)

            # L2 path using the closed-form SVD solution.
            y_center = y[tr] - y[tr].mean()
            U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
            Uty = U.T @ y_center
            coef = Vt.T @ (
                (Uty[:, None] * s[:, None])
                / (s[:, None] ** 2 + L2_ALPHAS[None, :])
            )
            pred = Xva @ coef + y[tr].mean()
            l2_err += np.mean((y[va, None] - pred) ** 2, axis=0)

        l1_mse = l1_err / N_FOLDS
        l2_mse = l2_err / N_FOLDS
        i1 = int(np.argmin(l1_mse))
        i2 = int(np.argmin(l2_mse))
        rows.append({
            "problem": problem,
            "degree": degree,
            "terms": int(XP.shape[1]),
            "l1_alpha": float(L1_ALPHAS[i1]),
            "l1_cv_mse": float(l1_mse[i1]),
            "l2_alpha": float(L2_ALPHAS[i2]),
            "l2_cv_mse": float(l2_mse[i2]),
            "winner": "L1 (Lasso)" if l1_mse[i1] <= l2_mse[i2] else "L2 (Ridge)",
            "best_cv_mse": float(min(l1_mse[i1], l2_mse[i2])),
            "seconds": float(time.time() - t0),
        })
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"{ROLL}_{problem}_l1_l2_per_degree.csv", index=False)
    return out

if __name__ == "__main__":
    evaluate("var1", 10)
    evaluate("var2", 20)
