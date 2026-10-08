#!/usr/bin/env python3
"""
IMT2024031 - Final selected L1 models.

Based on CV calculations on the supplied training CSVs:

var1:
    Lasso (L1), degree 5, alpha = 0.0078804628

var2:
    Lasso (L1), degree 11, alpha = 0.00045

The selected models are refit on ALL training rows and used to
generate the required 1000-row test prediction CSVs.

CPU + scikit-learn only.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


ROLL = "IMT2024031"

# Adjust these only if your CSVs are somewhere else.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PRED_DIR = ROOT_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# Selected models from the CV study
# ============================================================

CONFIG = {
    "var1": {
        "degree": 5,
        "alpha": 0.007880462815669913,
    },
    "var2": {
        "degree": 11,
        "alpha": 0.00045,
    },
}


def load_problem(problem):
    train = pd.read_csv(
        DATA_DIR / f"{ROLL}_train_{problem}.csv"
    )

    test = pd.read_csv(
        DATA_DIR / f"{ROLL}_test_{problem}.csv"
    )

    features = [
        c for c in train.columns
        if c != "y"
    ]

    X = train[features].to_numpy(
        dtype=np.float64
    )

    y = train["y"].to_numpy(
        dtype=np.float64
    )

    X_test = test[features].to_numpy(
        dtype=np.float64
    )

    return X, y, X_test


def make_model(degree, alpha):
    return Pipeline([
        (
            "poly",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),
        (
            "scale",
            StandardScaler()
        ),
        (
            "lasso",
            Lasso(
                alpha=alpha,
                max_iter=100000,
                tol=1e-4,
                selection="cyclic",
                random_state=RANDOM_STATE
            )
        )
    ])


def run(problem):
    cfg = CONFIG[problem]

    X, y, X_test = load_problem(problem)

    model = make_model(
        cfg["degree"],
        cfg["alpha"]
    )

    # 5-fold OOF sanity check for the selected fixed model.
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=1
    )

    cv_mse = mean_squared_error(
        y,
        oof
    )

    cv_r2 = r2_score(
        y,
        oof
    )

    # Final fit on ALL training data.
    model.fit(X, y)

    test_pred = model.predict(
        X_test
    )

    train_pred = model.predict(
        X
    )

    train_mse = mean_squared_error(
        y,
        train_pred
    )

    train_r2 = r2_score(
        y,
        train_pred
    )

    poly = model.named_steps["poly"]
    lasso = model.named_steps["lasso"]

    terms = len(
        lasso.coef_
    )

    nonzero = int(
        np.count_nonzero(
            np.abs(lasso.coef_) > 1e-12
        )
    )

    pred_path = (
        PRED_DIR /
        f"{ROLL}_pred_{problem}.csv"
    )

    pd.DataFrame({
        "y": test_pred
    }).to_csv(
        pred_path,
        index=False
    )

    print("\n" + "=" * 70)
    print(problem)
    print("=" * 70)
    print("Model          : Lasso (L1)")
    print(f"Degree         : {cfg['degree']}")
    print(f"Alpha          : {cfg['alpha']:.12g}")
    print(f"Terms          : {terms}")
    print(f"Non-zero       : {nonzero}/{terms}")
    print(f"5-fold OOF MSE  : {cv_mse:.10f}")
    print(f"5-fold OOF R2   : {cv_r2:.10f}")
    print(f"Train MSE      : {train_mse:.10f}")
    print(f"Train R2       : {train_r2:.10f}")
    print(f"Predictions    : {pred_path}")
    print("=" * 70)

    return {
        "problem": problem,
        "model": "Lasso (L1)",
        "degree": cfg["degree"],
        "alpha": cfg["alpha"],
        "terms": terms,
        "nonzero": nonzero,
        "cv_mse": cv_mse,
        "cv_r2": cv_r2,
        "train_mse": train_mse,
        "train_r2": train_r2,
    }


results = [
    run("var1"),
    run("var2"),
]

summary = pd.DataFrame(results)

summary_path = (
    PRED_DIR /
    f"{ROLL}_selected_model_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\nFINAL SUMMARY")
print(summary.to_string(index=False))
print(f"\nSaved: {summary_path}")
