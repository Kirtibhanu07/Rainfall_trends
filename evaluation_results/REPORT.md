# Rainfall forecasting evaluation

The validation-selected recommendation is **historical_mean**. On 20 rolling one-year-ahead
forecasts for 1996-2015 it achieved **MAE 82.18 mm**, **RMSE 108.28 mm**
and MASE 0.693. Its RMSE change versus the original ARIMA(3,1,2) specification
is **13.00% higher**. This is a retrospective
benchmark, not a claim of deployed or independently achieved business impact.

## Evaluation design

- Dataset: 115 national annual observations, 1901-2015.
- Initial training: 1901-1975. Configuration selection: 20 expanding-window origins, 1976-1995.
- Final test: 20 expanding-window origins, 1996-2015. Models and settings were locked before scoring this period.
- Each target is forecast from earlier years only. Previous test years become available at subsequent origins, as in annual retraining.
- Only lagged annual values are predictors. Same-year rainfall components would leak the annual target.
- Six baselines and four model families, 22 configurations in validation. One validation-selected configuration per model family is evaluated on the final test, alongside all fixed baselines and the original ARIMA.
- Selection uses validation RMSE, not the final-test leaderboard. No randomized split or within-year seasonality.
- MASE divides each absolute forecast error by that origin's mean absolute one-step training difference. Positive bias means overprediction.

## Final test results

| Model | MAE (mm) | RMSE (mm) | MASE |
|---|---:|---:|---:|
| arima_3_1_2 | 80.79 | 95.83 | 0.681 |
| rolling_10 | 79.99 | 99.55 | 0.674 |
| rolling_20 | 76.53 | 99.61 | 0.646 |
| rolling_5 | 84.84 | 103.95 | 0.715 |
| forest_depth_2 | 79.51 | 103.98 | 0.671 |
| ridge_100 | 81.53 | 104.57 | 0.688 |
| arima_0_0_0 | 82.18 | 108.28 | 0.693 |
| historical_mean | 82.18 | 108.28 | 0.693 |
| last_year | 105.47 | 126.45 | 0.889 |
| drift | 106.23 | 127.14 | 0.896 |
| prophet_0.001 | 127.72 | 158.11 | 1.076 |


## Statistical checks

The paired circular-block bootstrap estimates the recommendation's RMSE reduction
versus original ARIMA as -12.45 mm, with a 95% interval
of [-25.14, 2.80] mm. An interval containing zero does not establish a clear improvement.
The approximate two-sided HAC forecast loss test has p=0.1806577138172248.
Only 20 paired errors are available, so these are exploratory checks, not strong significance claims.

The training-only OLS trend test estimates 0.836 mm/year,
with a two-sided HAC p-value of 0.3471. This tests a zero linear
slope in 1901-1975. It is not an experimental design or a causal climate analysis.

## Quality and reproducibility

There are no missing numeric observations or duplicate years. Monthly sums agree
with annual totals within 0.30 mm rounding.
Dataset SHA-256: `e9325c98c594fa48721d587b37cb946adb43ddb7e2e38f7f744721a0a5a5b472`.
ARIMA optimizer non-convergence flags: 0 across validation and final-test fits.
All warnings and convergence flags are retained in `diagnostics.json`. No failed fit
is silently replaced with a better-performing fallback. Non-converged fits can make
model results less reliable and must be considered when interpreting the benchmark.

Reproduce with `.venv/bin/python evaluate.py`. See pinned requirements, experiment
plan, environment versions, saved configuration selection, and per-origin predictions.

## Limits

The dataset has only 115 observations and ends in 2015. Its original provenance has
not been independently verified. National averages hide regional variation. The
final test contains only 20 forecast errors and may not represent future climate
conditions. This work adds retrospective model validation and exploratory hypothesis
testing. It does not create randomized experimental design, production ML, cloud
deployment, or professional experience with PySpark, Databricks or AWS S3.
