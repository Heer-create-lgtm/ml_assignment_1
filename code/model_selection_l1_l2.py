#!/usr/bin/env python3
"""Per-degree L1 (Lasso) vs L2 (Ridge) comparison for IMT2024031."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 42
KF = KFold(n_splits=5, shuffle=True, random_state=SEED)
L1_ALPHAS = np.unique(np.r_[np.logspace(-5, 1, 30), 0.00045])
L2_ALPHAS = np.logspace(-6, 3, 30)

def evaluate(problem, max_degree):
    train = pd.read_csv(DATA / f"IMT2024031_train_{problem}.csv")
    X = train.drop(columns=["y"]).to_numpy(float)
    y = train["y"].to_numpy(float)
    rows = []
    for degree in range(1, max_degree + 1):
        XP = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)
        e1 = np.zeros(len(L1_ALPHAS)); e2 = np.zeros(len(L2_ALPHAS))
        for tr_idx, va_idx in KF.split(XP):
            sc = StandardScaler()
            Xt = sc.fit_transform(XP[tr_idx]); Xv = sc.transform(XP[va_idx])
            yt = y[tr_idx]; yv = y[va_idx]
            for j, alpha in enumerate(L1_ALPHAS):
                m = Lasso(alpha=float(alpha), max_iter=100000, tol=1e-4,
                          selection="cyclic", random_state=SEED)
                m.fit(Xt, yt)
                e1[j] += np.mean((yv - m.predict(Xv))**2)
            U, s, Vt = np.linalg.svd(Xt, full_matrices=False)
            yc = yt - yt.mean(); Uty = U.T @ yc
            C = Vt.T @ ((Uty[:,None] * s[:,None]) / (s[:,None]**2 + L2_ALPHAS[None,:]))
            preds = Xv @ C + yt.mean()
            e2 += np.mean((yv[:,None] - preds)**2, axis=0)
        i1 = int(np.argmin(e1)); i2 = int(np.argmin(e2))
        l1_mse = float(e1[i1] / 5); l2_mse = float(e2[i2] / 5)
        rows.append({
            "problem": problem, "degree": degree, "terms": XP.shape[1],
            "l1_alpha": float(L1_ALPHAS[i1]), "l1_cv_mse": l1_mse,
            "l2_alpha": float(L2_ALPHAS[i2]), "l2_cv_mse": l2_mse,
            "winner": "L1 (Lasso)" if l1_mse <= l2_mse else "L2 (Ridge)",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"IMT2024031_{problem}_l1_l2_per_degree.csv", index=False)
    return df

if __name__ == "__main__":
    evaluate("var1", 10)
    evaluate("var2", 20)
