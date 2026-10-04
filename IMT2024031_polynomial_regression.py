import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

ROLL = "IMT2024031"
RANDOM_STATE = 42
N_SPLITS = 5


def exhaustive_cv(train_file, max_degree):
    df = pd.read_csv(train_file)
    X = df.drop(columns=["y"]).to_numpy(float)
    y = df["y"].to_numpy(float)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for degree in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        Xp = poly.fit_transform(X)
        mses, r2s = [], []
        for tr, va in kf.split(Xp):
            model = LinearRegression(fit_intercept=False)
            model.fit(Xp[tr], y[tr])
            pred = model.predict(Xp[va])
            mses.append(mean_squared_error(y[va], pred))
            r2s.append(r2_score(y[va], pred))
        rows.append({
            "degree": degree,
            "terms": Xp.shape[1],
            "cv_mse": np.mean(mses),
            "cv_mse_sd": np.std(mses, ddof=1),
            "cv_r2": np.mean(r2s),
            "cv_r2_sd": np.std(r2s, ddof=1),
        })
    return pd.DataFrame(rows)


def train_predict(train_file, test_file, degree):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    X = train.drop(columns=["y"]).to_numpy(float)
    y = train["y"].to_numpy(float)
    Xt = test.to_numpy(float)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = poly.fit_transform(X)
    Xtp = poly.transform(Xt)
    model = LinearRegression(fit_intercept=False)
    model.fit(Xp, y)
    return model.predict(Xtp)


if __name__ == "__main__":
    specs = [("var1", 10), ("var2", 20)]
    for problem, max_degree in specs:
        cv = exhaustive_cv(f"{ROLL}_train_{problem}.csv", max_degree)
        cv.to_csv(f"{ROLL}_{problem}_cv_results.csv", index=False)
        best = cv.loc[cv["cv_mse"].idxmin()]
        pred = train_predict(
            f"{ROLL}_train_{problem}.csv",
            f"{ROLL}_test_{problem}.csv",
            int(best["degree"]),
        )
        pd.DataFrame({"y": pred}).to_csv(
            f"{ROLL}_pred_{problem}.csv", index=False
        )
        print(f"{problem}: degree={int(best['degree'])}, "
              f"terms={int(best['terms'])}, "
              f"CV MSE={best['cv_mse']:.6f}, "
              f"CV R2={best['cv_r2']:.6f}")
