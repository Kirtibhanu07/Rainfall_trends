"""Reproducible, leakage-aware annual rainfall forecasting evaluation.

Run from the repository: .venv/bin/python evaluate.py
The untouched original analysis remains in rainfall_trends.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    required = ["YEAR", "ANNUAL", "REGION"]
    if any(c not in frame for c in required):
        raise ValueError("Missing YEAR, ANNUAL or REGION column")
    if frame[required].isna().any().any():
        raise ValueError("Missing required values")
    if frame.YEAR.duplicated().any():
        raise ValueError("Duplicate years would leak repeated observations")
    if not np.all(np.isfinite(frame.YEAR)) or not np.all(frame.YEAR == frame.YEAR.astype(int)):
        raise ValueError("Years must be finite integers")
    frame = frame.sort_values("YEAR").reset_index(drop=True)
    if not np.all(np.diff(frame.YEAR) == 1):
        raise ValueError("Missing or irregular years")
    if frame.REGION.nunique() != 1 or frame.REGION.iloc[0] != "INDIA":
        raise ValueError("Expected one national annual series for INDIA")
    if not np.all(np.isfinite(frame.ANNUAL)) or (frame.ANNUAL < 0).any():
        raise ValueError("Rainfall must be finite and nonnegative")
    numeric = frame.select_dtypes(include="number")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Missing or nonfinite numeric data")
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    discrepancy = None
    if all(c in frame for c in months):
        discrepancy = float((frame[months].sum(axis=1) - frame.ANNUAL).abs().max())
        if discrepancy > 0.31:
            raise ValueError("Monthly and annual totals disagree beyond rounding tolerance")
    audit = {"rows": len(frame), "first_year": int(frame.YEAR.min()),
             "last_year": int(frame.YEAR.max()), "missing_numeric_values": 0,
             "duplicate_years": 0, "max_month_sum_discrepancy_mm": discrepancy,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return frame, audit


def configurations(plan: dict) -> list[dict]:
    out = [{"name": n, "family": "baseline"} for n in plan["baselines"]]
    out += [{"name": "arima_" + "_".join(map(str, o)), "family": "arima", "order": o}
            for o in plan["arima_orders"]]
    out += [{"name": f"prophet_{v}", "family": "prophet", "prior": v}
            for v in plan["prophet_changepoint_prior_scales"]]
    out += [{"name": f"ridge_{v:g}", "family": "ridge", "alpha": v}
            for v in plan["ridge_alphas"]]
    out += [{"name": f"forest_depth_{v}", "family": "forest", "depth": v}
            for v in plan["random_forest_depths"]]
    return out


def lag_matrix(history: np.ndarray, lags: int = 5):
    """Every feature precedes its target. Fit preprocessing only on these rows."""
    if len(history) <= lags + 2:
        raise ValueError("Insufficient training history")
    X = np.array([history[i-lags:i][::-1] for i in range(lags, len(history))])
    y = history[lags:].copy()
    return X, y, history[-lags:][::-1].reshape(1, -1)


def predict(history: np.ndarray, years: np.ndarray, cfg: dict) -> tuple[float, dict]:
    """No target/future values accepted by this function."""
    name, family = cfg["name"], cfg["family"]
    if family == "baseline":
        if name == "historical_mean": value = history.mean()
        elif name == "last_year": value = history[-1]
        elif name.startswith("rolling_"): value = history[-int(name.split("_")[1]):].mean()
        elif name == "drift": value = history[-1] + (history[-1] - history[0]) / (len(history)-1)
        else: raise ValueError(name)
        return float(value), {}
    if family == "arima":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = ARIMA(history, order=tuple(cfg["order"])).fit(method_kwargs={"maxiter": 300})
        return float(np.asarray(fitted.forecast(1))[0]), {
            "converged": bool(fitted.mle_retvals.get("converged", True)),
            "warnings": sorted(set(str(w.message) for w in caught))}
    if family == "prophet":
        from prophet import Prophet
        fitted = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                         daily_seasonality=False, changepoint_prior_scale=cfg["prior"],
                         uncertainty_samples=0)
        data = pd.DataFrame({"ds": pd.to_datetime(years.astype(str) + "-01-01"), "y": history})
        fitted.fit(data, seed=42)
        future = pd.DataFrame({"ds": [pd.Timestamp(year=int(years[-1])+1, month=1, day=1)]})
        return float(fitted.predict(future).yhat.iloc[0]), {}
    X, y, next_x = lag_matrix(history)
    if family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=cfg["alpha"]))
    elif family == "forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=cfg["depth"],
                                      min_samples_leaf=5, random_state=42, n_jobs=1)
    else:
        raise ValueError(family)
    model.fit(X, y)
    return float(model.predict(next_x)[0]), {}


def backtest(frame: pd.DataFrame, cfgs: list[dict], first: int, last: int) -> tuple[pd.DataFrame, list]:
    rows, diagnostics = [], []
    values, years = frame.ANNUAL.to_numpy(float), frame.YEAR.to_numpy(int)
    for cfg in cfgs:
        print(f"Evaluating {first}-{last}: {cfg['name']}", flush=True)
        for year in range(first, last+1):
            positions = np.flatnonzero(years == year)
            if len(positions) != 1: raise ValueError(f"No unique target for {year}")
            i = int(positions[0])
            if i < 20: raise ValueError("At least 20 prior annual observations required")
            prediction, diag = predict(values[:i].copy(), years[:i].copy(), cfg)
            if not np.isfinite(prediction): raise ValueError(f"Nonfinite prediction: {cfg['name']}, {year}")
            scale = float(np.abs(np.diff(values[:i])).mean())
            if scale <= 0: raise ValueError("MASE undefined for constant training history")
            rows.append({"model": cfg["name"], "family": cfg["family"], "year": year,
                         "train_start": int(years[0]), "train_end": int(years[i-1]),
                         "n_train": i, "actual_mm": float(values[i]), "predicted_mm": prediction,
                         "mase_scale_mm": scale})
            diagnostics.append({"model": cfg["name"], "year": year, **diag})
    return pd.DataFrame(rows), diagnostics


def metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, family), g in predictions.groupby(["model", "family"], sort=True):
        errors = g.predicted_mm - g.actual_mm
        rows.append({"model": model, "family": family, "n_forecasts": len(g),
                     "MAE_mm": float(np.abs(errors).mean()),
                     "RMSE_mm": float(np.sqrt(np.mean(errors**2))),
                     "MASE": float((np.abs(errors)/g.mase_scale_mm).mean()),
                     "bias_mm": float(errors.mean())})
    return pd.DataFrame(rows).sort_values(["RMSE_mm", "model"]).reset_index(drop=True)


def compare_errors(actual, chosen, reference, seed=42, n_boot=10000, block=4) -> dict:
    """Paired errors, circular block bootstrap, plus approximate HAC loss test."""
    actual, chosen, reference = map(np.asarray, [actual, chosen, reference])
    if not (len(actual) == len(chosen) == len(reference)) or len(actual) < 8:
        raise ValueError("Need aligned paired errors")
    ec, er = chosen-actual, reference-actual
    n = len(actual)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n/block))))
    idx = ((starts[..., None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]
    # Positive difference means the recommendation is better.
    delta = np.sqrt(np.mean(er[idx]**2, axis=1)) - np.sqrt(np.mean(ec[idx]**2, axis=1))
    differential = er**2 - ec**2
    dm_result = sm.OLS(differential, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    se = float(dm_result.bse[0])
    statistic = float(differential.mean()/se * np.sqrt((n-1)/n)) if se > 0 else None
    p = float(2*stats.t.sf(abs(statistic), df=n-1)) if statistic is not None else None
    return {"n_pairs": n, "rmse_reduction_mm": float(np.sqrt(np.mean(er**2))-np.sqrt(np.mean(ec**2))),
            "rmse_reduction_95pct_block_bootstrap_ci_mm": np.quantile(delta, [.025, .975]).tolist(),
            "bootstrap_seed": seed, "bootstrap_resamples": n_boot, "block_length": block,
            "approximate_dm_statistic": statistic, "approximate_dm_two_sided_p": p,
            "warning": "Exploratory small-sample inference, not proof of future performance or a randomized experiment."}


def trend_test(training: pd.DataFrame) -> dict:
    centered = training.YEAR.to_numpy(float) - float(training.YEAR.mean())
    fit = sm.OLS(training.ANNUAL.to_numpy(float), sm.add_constant(centered)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True)
    return {"first_year": int(training.YEAR.min()), "last_year": int(training.YEAR.max()),
            "n": len(training), "slope_mm_per_year": float(fit.params[1]),
            "slope_95pct_ci": fit.conf_int()[1].tolist(), "two_sided_p": float(fit.pvalues[1]),
            "null": "Zero linear rainfall trend", "standard_errors": "HAC, 3 lags",
            "interpretation_limit": "Exploratory association only. Cannot attribute changes to climate or any cause."}


def write_report(out, validation, test, selection, inference, trend, audit, diagnostics):
    chosen = selection["recommended_model"]
    row = test.set_index("model").loc[chosen]
    original = test.set_index("model").loc["arima_3_1_2"]
    difference = 100*(row.RMSE_mm-original.RMSE_mm)/original.RMSE_mm
    direction = "higher" if difference >= 0 else "lower"
    table = "| Model | MAE (mm) | RMSE (mm) | MASE |\n|---|---:|---:|---:|\n"
    for _, r in test.iterrows(): table += f"| {r.model} | {r.MAE_mm:.2f} | {r.RMSE_mm:.2f} | {r.MASE:.3f} |\n"
    nonconverged = [d for d in diagnostics if d.get("converged") is False]
    ci = inference["rmse_reduction_95pct_block_bootstrap_ci_mm"]
    report = f"""# Rainfall forecasting evaluation

The validation-selected recommendation is **{chosen}**. On 20 rolling one-year-ahead
forecasts for 1996-2015 it achieved **MAE {row.MAE_mm:.2f} mm**, **RMSE {row.RMSE_mm:.2f} mm**
and MASE {row.MASE:.3f}. Its RMSE change versus the original ARIMA(3,1,2) specification
is **{abs(difference):.2f}% {direction}**. This is a retrospective
benchmark, not a claim of deployed or independently achieved business impact.

## Evaluation design

- Dataset: {audit['rows']} national annual observations, 1901-2015.
- Initial training: 1901-1975. Configuration selection: 20 expanding-window origins, 1976-1995.
- Final test: 20 expanding-window origins, 1996-2015. Models and settings were locked before scoring this period.
- Each target is forecast from earlier years only. Previous test years become available at subsequent origins, as in annual retraining.
- Only lagged annual values are predictors. Same-year rainfall components would leak the annual target.
- Six baselines and four model families, 22 configurations in validation. One validation-selected configuration per model family is evaluated on the final test, alongside all fixed baselines and the original ARIMA.
- Selection uses validation RMSE, not the final-test leaderboard. No randomized split or within-year seasonality.
- MASE divides each absolute forecast error by that origin's mean absolute one-step training difference. Positive bias means overprediction.

## Final test results

{table}

## Statistical checks

The paired circular-block bootstrap estimates the recommendation's RMSE reduction
versus original ARIMA as {inference['rmse_reduction_mm']:.2f} mm, with a 95% interval
of [{ci[0]:.2f}, {ci[1]:.2f}] mm. An interval containing zero does not establish a clear improvement.
The approximate two-sided HAC forecast loss test has p={inference['approximate_dm_two_sided_p']}.
Only 20 paired errors are available, so these are exploratory checks, not strong significance claims.

The training-only OLS trend test estimates {trend['slope_mm_per_year']:.3f} mm/year,
with a two-sided HAC p-value of {trend['two_sided_p']:.4f}. This tests a zero linear
slope in 1901-1975. It is not an experimental design or a causal climate analysis.

## Quality and reproducibility

There are no missing numeric observations or duplicate years. Monthly sums agree
with annual totals within {audit['max_month_sum_discrepancy_mm']:.2f} mm rounding.
Dataset SHA-256: `{audit['sha256']}`.
ARIMA optimizer non-convergence flags: {len(nonconverged)} across validation and final-test fits.
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
"""
    (out/"REPORT.md").write_text(report)
    naive = test.set_index("model").loc["last_year"]
    reduction = 100*(naive.RMSE_mm-original.RMSE_mm)/naive.RMSE_mm
    bullet = ("Benchmarked 22 forecasting configurations using chronological validation and expanding-window backtesting on 115 annual rainfall observations. "
              f"Measured ARIMA MAE of {original.MAE_mm:.1f} mm and RMSE of {original.RMSE_mm:.1f} mm across 20 final-test years, with {reduction:.1f}% lower RMSE than a previous-year baseline. "
              "Added paired block-bootstrap uncertainty estimates and a training-only trend hypothesis test. "
              f"The validation-selected {chosen} had a final-test RMSE of {row.RMSE_mm:.1f} mm, so model ranking did not fully generalize.")
    (out/"resume_evidence.txt").write_text(bullet+"\n\nThese are results of the current assisted project extension. Review the implementation and reproduce results before presenting it as your own work.\n")


def plot_results(out, predictions, leaderboard, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    pivot = predictions.pivot(index="year", columns="model", values="predicted_mm")
    actual = predictions.drop_duplicates("year").set_index("year").actual_mm.sort_index()
    axes[0].plot(actual.index, actual, color="#222222", marker="o", label="Observed", linewidth=1.7)
    for model, color in [(chosen, "#17658b"), ("arima_3_1_2", "#ba692c")]:
        if model in pivot: axes[0].plot(pivot.index, pivot[model], label=model, color=color)
    axes[0].set(title="One-year-ahead forecasts on final test years", ylabel="Annual rainfall (mm)")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    axes[0].legend(loc="best")
    colors = ["#17658b" if m == chosen else "#c4d0d6" for m in leaderboard.model]
    axes[1].barh(leaderboard.model, leaderboard.RMSE_mm, color=colors)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="RMSE (mm), lower is better", title="Final-test comparison, recommendation chosen on earlier validation")
    fig.savefig(out/"evaluation.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT/"Rainfall Trends.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"evaluation_results")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT/"experiment_plan.json").read_text())
    frame, audit = load_data(args.data)
    if (int(frame.YEAR.min()), int(frame.YEAR.max()), len(frame)) != (1901, 2015, 115):
        raise ValueError("This frozen experiment plan requires the 115-row 1901-2015 dataset")
    cfgs = configurations(plan)
    # Pass only pre-test data into model selection, not the full frame.
    development = frame[frame.YEAR < plan["final_test"][0]].copy()
    val_predictions, val_diag = backtest(development, cfgs, *plan["validation"])
    validation = metrics(val_predictions)
    selected_names = validation.groupby("family", sort=True).head(1).model.tolist()
    chosen = str(validation.iloc[0].model)
    keep = set(plan["baselines"] + selected_names + ["arima_3_1_2"])
    finalists = [c for c in cfgs if c["name"] in keep]
    selection = {"recommended_model": chosen,
                 "family_selections": validation.groupby("family").head(1).to_dict("records"),
                 "final_test_models": [c["name"] for c in finalists],
                 "selection_metric": "validation_RMSE_mm", "configurations_evaluated": len(cfgs)}
    # Persist locked choices before touching final-test targets.
    (out/"selection.json").write_text(json.dumps(selection, indent=2))
    validation.to_csv(out/"validation_metrics.csv", index=False)
    val_predictions.to_csv(out/"validation_predictions.csv", index=False)
    print("Locked recommendation:", chosen, flush=True)
    predictions, test_diag = backtest(frame, finalists, *plan["final_test"])
    test = metrics(predictions)
    predictions.to_csv(out/"test_predictions.csv", index=False)
    test.to_csv(out/"test_metrics.csv", index=False)
    pivot = predictions.pivot(index="year", columns="model", values="predicted_mm")
    actual = predictions.drop_duplicates("year").set_index("year").actual_mm.sort_index()
    inference = compare_errors(actual.to_numpy(), pivot[chosen].to_numpy(), pivot["arima_3_1_2"].to_numpy())
    trend = trend_test(frame[frame.YEAR <= plan["initial_training"][1]])
    diagnostics = val_diag + test_diag
    for name, data in [("audit", audit), ("inference", inference), ("trend_test", trend), ("diagnostics", diagnostics)]:
        (out/f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    versions = {p: importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","statsmodels","prophet","cmdstanpy","matplotlib"]}
    try: sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): sha = "unavailable"
    manifest = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
                "upstream_commit": sha, "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "experiment_plan_sha256": hashlib.sha256((ROOT/"experiment_plan.json").read_bytes()).hexdigest()}
    (out/"environment.json").write_text(json.dumps(manifest, indent=2))
    write_report(out, validation, test, selection, inference, trend, audit, diagnostics)
    plot_results(out, predictions, test, chosen)
    print(test.to_string(index=False), flush=True)
    print("Results written to", out, flush=True)


if __name__ == "__main__":
    main()
