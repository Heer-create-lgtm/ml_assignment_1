#!/usr/bin/env python3
"""Reproduce final IMT2024031 predictions using L1 regularization (Lasso)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRED = ROOT / "predictions"
RESULTS = ROOT / "results"
PRED.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
ROLL = "IMT2024031"
SEED = 42
CONFIG = {
    "var1": {"degree": 5, "alpha": 0.007880462815669913},
    "var2": {"degree": 11, "alpha": 0.00045},
}

def make_model(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=alpha, max_iter=100000, tol=1e-4,
                         selection="cyclic", random_state=SEED)),
    ])

def run(problem):
    train = pd.read_csv(DATA / f"{ROLL}_train_{problem}.csv")
    test = pd.read_csv(DATA / f"{ROLL}_test_{problem}.csv")
    features = [c for c in train.columns if c != "y"]
    X = train[features].to_numpy(float)
    y = train["y"].to_numpy(float)
    X_test = test[features].to_numpy(float)
    cfg = CONFIG[problem]
    model = make_model(cfg["degree"], cfg["alpha"])
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    model.fit(X, y)
    pred = model.predict(X_test)
    fit_pred = model.predict(X)
    beta = model.named_steps["model"].coef_
    oof_df = pd.DataFrame({"y_true": y, "y_oof": oof, "residual": y-oof})
    oof_df.to_csv(RESULTS / f"{problem}_oof_predictions.csv", index=False)
    pd.DataFrame({"y": pred}).to_csv(PRED / f"{ROLL}_pred_{problem}.csv", index=False)
    return {
        "problem": problem,
        "model": "Lasso (L1)",
        "degree": cfg["degree"],
        "alpha": cfg["alpha"],
        "terms": len(beta),
        "nonzero": int(np.count_nonzero(np.abs(beta) > 1e-12)),
        "cv_mse": mean_squared_error(y, oof),
        "cv_r2": r2_score(y, oof),
        "train_mse": mean_squared_error(y, fit_pred),
        "train_r2": r2_score(y, fit_pred),
    }

if __name__ == "__main__":
    summary = pd.DataFrame([run("var1"), run("var2")])
    summary.to_csv(PRED / f"{ROLL}_selected_model_summary.csv", index=False)
    print(summary.to_string(index=False))
