#!/usr/bin/env python3
"""Refit models and compare predictions/OOF results without overwriting them."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from train_final import make_model, CONFIG
ROOT = Path(__file__).resolve().parents[1]
def main():
    rows=[]
    for problem, cfg in CONFIG.items():
        tr=pd.read_csv(ROOT / 'data' / f'IMT2024031_train_{problem}.csv')
        te=pd.read_csv(ROOT / 'data' / f'IMT2024031_test_{problem}.csv')
        submitted=pd.read_csv(ROOT / 'predictions' / f'IMT2024031_pred_{problem}.csv')
        assert list(submitted.columns)==['y'] and len(submitted)==len(te)
        assert np.isfinite(submitted.y).all()
        features=tr.columns.drop('y'); X=tr[features]; y=tr.y
        model=make_model(**cfg)
        oof=cross_val_predict(model,X,y,cv=KFold(5,shuffle=True,random_state=42))
        model.fit(X,y); predicted=model.predict(te[features])
        saved=pd.read_csv(ROOT / 'results' / f'{problem}_oof_predictions.csv')
        np.testing.assert_allclose(submitted.y,predicted,rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(saved.y_true,y)
        np.testing.assert_allclose(saved.y_oof,oof,rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(saved.residual,y-oof,rtol=1e-7,atol=1e-8)
        rows.append({'problem':problem,'prediction_max_abs_difference':float(np.max(np.abs(submitted.y-predicted))), 'oof_mse':mean_squared_error(y,oof),'oof_r2':r2_score(y,oof),'status':'PASS'})
    pd.DataFrame(rows).to_csv(ROOT/'results'/'IMT2024031_verification.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
