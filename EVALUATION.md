# Reproducible forecasting evaluation

This extension evaluates the existing rainfall project without changing its original
notebook or exploratory script. It adds chronological model selection, honest
out-of-sample metrics, reference baselines, data integrity checks, statistical
uncertainty and automated leakage tests.

## Run it

Use Python 3.12, from this repository:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-evaluation.lock
.venv/bin/python -m pytest -q test_evaluate.py
.venv/bin/python evaluate.py
.venv/bin/python -m pytest -q test_evaluate.py
```

`requirements-evaluation.txt` lists the direct dependencies. The lock file records
the tested environment, including CmdStanPy 1.2.5 for Prophet 1.1.7 compatibility.
Prophet uses its bundled compiled Stan executable on the tested macOS arm64 system.
Other platforms may need a compatible Prophet wheel or Stan installation.

## What was decided before final testing

See `experiment_plan.json` for the fixed split, model grid and inference settings.
The initial model history is 1901-1975. All 22 configurations are compared over
1976-1995. The recommendation and family configurations are locked in
`selection.json` before scoring 1996-2015.

Each forecast is for the next year, with refitting on earlier observations only.
The model may use an earlier test-year observation once that year has actually
occurred. This is walk-forward evaluation, not one frozen 20-year forecast.

Same-year months and seasonal totals are deliberately excluded as predictors.
Those fields sum to the annual target and would create target leakage. Annual
sampling also cannot support Prophet's within-year seasonality, so it is disabled.

## Results and interpretation

Read [the full report](evaluation_results/REPORT.md), [validation metrics](evaluation_results/validation_metrics.csv),
[final-test metrics](evaluation_results/test_metrics.csv), and [per-year forecasts](evaluation_results/test_predictions.csv).

Run `evaluate.py` to generate the forecast comparison chart at
`evaluation_results/evaluation.png`.

The historical-mean baseline won on validation RMSE. It achieved 82.18 mm MAE and
108.28 mm RMSE on the final test. The original ARIMA(3,1,2) reference achieved
80.79 mm MAE and 95.83 mm RMSE on the same years. We do not relabel the original
ARIMA as a model chosen by the validation process, and do not retune after seeing
the final-test leaderboard.

The original ARIMA's final-test RMSE is 24.22% lower than the previous-year
baseline's 126.45 mm. This is a descriptive result on a small historical sample,
not evidence of universal superiority. Its paired bootstrap interval against the
historical mean includes zero, and the corresponding approximate loss test does
not reject equal expected squared loss at the 5% level.

The training-only trend test also does not reject a zero linear trend. This is a
legitimate hypothesis-test result. Reporting a non-significant result is preferable
to inventing significance. Neither test is a randomized or causal experiment.

## Implementation walkthrough

- `load_data` checks uniqueness, year continuity, nonnegative finite values and
  reconciliation of monthly totals to annual values within rounding tolerance.
- `lag_matrix` creates five strictly prior annual lag features. Ridge preprocessing
  is fitted inside each model's training window.
- `predict` fits the specified baseline, ARIMA, Prophet, Ridge or Random Forest
  using only the history passed in by `backtest`.
- `backtest` saves actual values, predictions, training end dates and MASE scaling
  factors for each origin. It logs convergence warnings without substituting a
  hidden fallback.
- `metrics` computes MAE, RMSE, MASE and bias. Errors are in rainfall millimeters.
- `compare_errors` resamples paired forecast errors in circular four-year blocks
  and reports an approximate HAC loss-comparison test.
- `trend_test` uses only the initial training period and HAC standard errors.

The tests check temporal access, lag construction, exact metric calculations,
invalid data rejection, deterministic inference, and saved-run integrity.

## Scope and ownership

This is an AI-assisted extension completed on 20 October 2026. Review the code,
reproduce the run and understand the validation design before discussing it as
your own project. It does not create professional production ML experience,
Databricks/S3 deployment experience or randomized experimental-design experience.
The source dataset's original provenance remains unverified.
