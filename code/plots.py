#!/usr/bin/env python3
"""Generate final diagnostic plots for IMT2024031."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
PLOTS = BASE / "plots"
RESULTS = BASE / "results"
PLOTS.mkdir(exist_ok=True)

CONFIG = {
    "var1": {"degree": 5, "alpha": 0.007880462815669913},
    "var2": {"degree": 11, "alpha": 0.00045},
}

def make_model(degree, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=100000, tol=1e-4,
                        selection="cyclic", random_state=42)),
    ])

for problem, cfg in CONFIG.items():
    train = pd.read_csv(DATA / f"IMT2024031_train_{problem}.csv")
    X = train.drop(columns=["y"]).to_numpy(float)
    y = train["y"].to_numpy(float)
    model = make_model(cfg["degree"], cfg["alpha"])
    cv = KFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    model.fit(X, y)
    beta = model.named_steps["lasso"].coef_

    mse = mean_squared_error(y, oof)
    r2 = r2_score(y, oof)
    residual = y - oof

    pd.DataFrame({
        "y_true": y, "y_oof": oof, "residual": residual
    }).to_csv(RESULTS / f"{problem}_oof_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, oof, s=8, alpha=0.5)
    lo, hi = min(y.min(), oof.min()), max(y.max(), oof.max())
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1,
            label="perfect prediction")
    ax.set(xlabel="true y", ylabel="predicted y (out-of-fold)")
    ax.set_title(f"{problem}: Lasso degree {cfg['degree']}\n"
                 f"CV MSE = {mse:.3f}, R² = {r2:.3f}", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{problem}_pred_vs_actual.png", dpi=160)
    plt.close(fig)

    mags = np.sort(np.abs(beta))[::-1]
    mags = mags[mags > 1e-12]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(mags) + 1), mags, marker=".")
    ax.set(xlabel=f"coefficient rank ({len(mags)} non-zero of {len(beta)})",
           ylabel="|β| (standardized scale, log)")
    ax.set_title(f"{problem}: coefficient sizes")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{problem}_coefficient_sizes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(oof, residual, s=8, alpha=0.5)
    ax.axhline(0, color="red", linewidth=1)
    ax.set(xlabel="out-of-fold prediction",
           ylabel="residual (true - predicted)")
    ax.set_title(f"{problem}: OOF residuals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{problem}_residuals.png", dpi=160)
    plt.close(fig)

print("Final plots generated.")
