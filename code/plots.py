#!/usr/bin/env python3
"""Create final L1/L2-only diagnostic plots."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def pred_vs_actual(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=12, alpha=0.55)
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction")
    ax.set(xlabel="true y", ylabel="predicted y (out-of-fold)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def residuals(y_true, y_pred, title, path):
    r = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, r, s=12, alpha=0.55)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set(xlabel="out-of-fold prediction", ylabel="residual (true - predicted)")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def coefficient_sizes(coef, title, path):
    mags = np.sort(np.abs(coef))[::-1]
    mags = mags[mags > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(np.arange(1, len(mags)+1), mags, marker=".", linewidth=1)
    ax.set(xlabel=f"coefficient rank ({len(mags)} non-zero of {len(coef)})",
           ylabel="|beta| (standardized scale, log)", title=title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def regularizer_comparison(path):
    source = pd.read_csv(RESULTS / "selected_degree_l1_l2_comparison.csv")
    df = pd.DataFrame([{"problem": row.problem, "regularizer": family,
                        "cv_mse": getattr(row, col)}
                       for row in source.itertuples()
                       for family, col in [("L1 (Lasso)", "L1_cv_mse"), ("L2 (Ridge)", "L2_cv_mse")]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, problem in zip(axes, ["var1", "var2"]):
        sub = df[df.problem == problem]
        bars = ax.bar(sub.regularizer, sub.cv_mse)
        ax.set_title(f"{problem}: selected degree")
        ax.set_ylabel("5-fold OOF MSE")
        ax.tick_params(axis="x", rotation=15)
        for b, v in zip(bars, sub.cv_mse):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def final_metrics(path):
    df = pd.read_csv(ROOT / "predictions" / "IMT2024031_selected_model_summary.csv").rename(columns={"cv_mse": "mse", "cv_r2": "r2"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(df.problem, df.mse)
    axes[0].set_title("5-fold OOF MSE"); axes[0].set_ylabel("MSE"); axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(df.problem, df.r2)
    axes[1].set_title("5-fold OOF R2"); axes[1].set_ylabel("R2"); axes[1].set_ylim(0, 1.02); axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    for problem in ["var1", "var2"]:
        df = pd.read_csv(RESULTS / f"{problem}_oof_predictions.csv")
        pred_vs_actual(df.y_true.to_numpy(), df.y_oof.to_numpy(), f"{problem}: predicted vs actual", PLOTS / f"{problem}_pred_vs_actual.png")
        residuals(df.y_true.to_numpy(), df.y_oof.to_numpy(), f"{problem}: OOF residuals", PLOTS / f"{problem}_residuals.png")
    # coefficients are loaded from final fitted model via a small deterministic refit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import Lasso
    for problem, degree, alpha in [("var1", 5, 0.007880462815669913), ("var2", 11, 0.00045)]:
        tr = pd.read_csv(ROOT / "data" / f"IMT2024031_train_{problem}.csv")
        X = tr.drop(columns=["y"]).to_numpy(float); y = tr.y.to_numpy(float)
        model = Pipeline([("poly", PolynomialFeatures(degree=degree, include_bias=False)), ("scale", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=100000, tol=1e-4, selection="cyclic", random_state=42))])
        model.fit(X, y)
        coefficient_sizes(model.named_steps["model"].coef_, f"{problem}: L1 coefficient sizes", PLOTS / f"{problem}_coefficient_sizes.png")
    regularizer_comparison(PLOTS / "regularizer_comparison.png")
    final_metrics(PLOTS / "final_model_metrics.png")

if __name__ == "__main__": main()
