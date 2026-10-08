import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,pandas as pd,warnings
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import lasso_path
from sklearn.model_selection import KFold
from scipy.linalg import svd
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
A1=np.sort(np.unique(np.r_[np.logspace(-5,1,30),.00045]))[::-1];A2=np.logspace(-6,3,30)
def run(task):
 v,d=task;df=pd.read_csv(ROOT/f'data/IMT2024031_train_var{v}.csv');X=PolynomialFeatures(d,include_bias=False).fit_transform(df.drop(columns='y'));y=df.y.values;e1=[];e2=[];gaps=[]
 for tr,va in KFold(5,shuffle=True,random_state=42).split(X):
  sc=StandardScaler().fit(X[tr]);B=sc.transform(X[tr]);C=sc.transform(X[va]);mu=y[tr].mean();z=y[tr]-mu
  with warnings.catch_warnings(record=True):_,coef,gap=lasso_path(np.asfortranarray(B),z,alphas=A1,max_iter=(2000 if (v==1 and d>=7) or (v==2 and d>=15) else 10000),tol=1e-4)
  e1.append(np.mean((C@coef+mu-y[va,None])**2,axis=0));gaps.append(gap/(np.mean(z*z)))
  u,s,vt=svd(B,full_matrices=False,check_finite=False);P=(C@vt.T)@((s*(u.T@z))[:,None]/(s[:,None]**2+A2))+mu;e2.append(np.mean((P-y[va,None])**2,axis=0))
 e1=np.mean(e1,axis=0);e2=np.mean(e2,axis=0);gaps=np.max(gaps,axis=0);rows=[]
 for family,aa,ee in [('Lasso',A1,e1),('Ridge',A2,e2)]:
  for j,(a,e) in enumerate(zip(aa,ee)):rows.append(dict(problem=f'var{v}',degree=d,family=family,alpha=a,cv_mse=e,max_relative_dual_gap=gaps[j] if family=='Lasso' else 0,converged=bool(gaps[j]<=1e-4) if family=='Lasso' else True))
 pd.DataFrame(rows).to_csv(ROOT/f'results/search_var{v}_degree{d}.csv',index=False)
 return v,d,float(e1.min()),float(e2.min())
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=6) as p:
  fs=[p.submit(run,(v,d)) for v,md in [(1,10),(2,20)] for d in range(1,md+1) if not (ROOT/f'results/search_var{v}_degree{d}.csv').exists()]
  for f in as_completed(fs):print(f.result(),flush=True)

if __name__=='__main__':
    parts=[pd.read_csv(p) for p in sorted((ROOT/'results').glob('search_var*_degree*.csv'))]
    all_results=pd.concat(parts,ignore_index=True).sort_values(['problem','degree','family','alpha'])
    all_results.to_csv(ROOT/'results'/'search_audit_all_candidates.csv',index=False)
    for problem, group in all_results.groupby('problem'):
        rows=[]
        for (degree,family), candidates in group.groupby(['degree','family']):
            eligible=candidates[candidates.converged]
            best=eligible.loc[eligible.cv_mse.idxmin()] if len(eligible) else None
            rows.append(dict(problem=problem,degree=degree,family=family,
                alpha=best.alpha if best is not None else float('nan'),
                cv_mse=best.cv_mse if best is not None else float('nan'),
                unconverged_candidates=int((~candidates.converged).sum())))
        pd.DataFrame(rows).to_csv(ROOT/'results'/f'{problem}_per_degree_search_audit.csv',index=False)
