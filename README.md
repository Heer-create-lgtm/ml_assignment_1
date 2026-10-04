IMT2024031 - Polynomial Regression Assignment

Files in this package
---------------------
IMT2024031_pred_var1.csv          Final predictions for Problem/Phase 1 (var1)
IMT2024031_pred_var2.csv          Final predictions for Problem/Phase 2 (var2)
IMT2024031_var1_cv_results.csv    Full 5-fold CV results for degrees 1-10
IMT2024031_var2_cv_results.csv    Full 5-fold CV results for degrees 1-20
IMT2024031_var1_cv.png             Degree-selection plot for var1
IMT2024031_var2_cv.png             Degree-selection plot for var2
IMT2024031_polynomial_regression.py Reproducible training/CV/prediction code
IMT2024031_assignment_report.pdf  Report for submission
requirements.txt              Python dependencies
README.md                     This file

Final model selection
---------------------
var1: degree 4, 210 polynomial terms
      5-fold CV MSE = 0.873890
      5-fold CV R^2 = 0.907736

var2: degree 8, 165 polynomial terms
      5-fold CV MSE = 0.251895
      5-fold CV R^2 = 0.994196

Method
------
The final model is ordinary multivariate polynomial regression: polynomial
feature expansion followed by least-squares linear regression. No Ridge/Lasso,
neural network, tree model, or other non-polynomial regressor is used.
The supplied data are already scaled approximately to [-1, 1].

Reproduction
------------
1. Place the four assigned CSV files in the same directory as the script.
2. Install: pip install -r requirements.txt
3. Run: python IMT2024031_polynomial_regression.py

Submission note
---------------
The assignment also asks for a GitHub repository. This folder is repository-ready:
create a GitHub repository and upload the contents of this package (excluding
any private/local files that your course rules prohibit). The prediction CSVs
and report are ready for submission.
