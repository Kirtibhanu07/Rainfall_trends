"""Tests for temporal leakage, metric correctness and data integrity."""
import json
import numpy as np
import pandas as pd
import pytest
import evaluate as ev


def test_each_forecast_sees_only_earlier_years(monkeypatch):
    frame = pd.DataFrame({"YEAR": np.arange(1901, 1941), "ANNUAL": np.arange(40.) + 1000})
    seen = []
    def spy(history, years, cfg):
        seen.append((history.copy(), years.copy()))
        return float(history.mean()), {}
    monkeypatch.setattr(ev, "predict", spy)
    cfg = [{"name": "historical_mean", "family": "baseline"}]
    first, _ = ev.backtest(frame, cfg, 1931, 1935)
    changed = frame.copy()
    changed.loc[changed.YEAR >= 1935, "ANNUAL"] = 999999.
    second, _ = ev.backtest(changed, cfg, 1931, 1935)
    np.testing.assert_array_equal(first.predicted_mm, second.predicted_mm)
    for row, (history, years) in zip(first.itertuples(), seen[:5]):
        assert years.max() == row.year - 1
        assert len(history) == row.n_train
        assert row.train_end < row.year


def test_lag_features_never_include_current_target():
    history = np.arange(20.)
    X, y, next_x = ev.lag_matrix(history)
    np.testing.assert_array_equal(X[0], [4,3,2,1,0])
    assert y[0] == 5
    assert np.all(X[:, 0] == y-1)
    np.testing.assert_array_equal(next_x[0], [19,18,17,16,15])


def test_metrics_have_known_values_and_units():
    p = pd.DataFrame({"model": ["x","x"], "family": ["baseline"]*2,
                      "actual_mm": [10.,20.], "predicted_mm": [13.,16.], "mase_scale_mm": [2.,2.]})
    m = ev.metrics(p).iloc[0]
    assert m.MAE_mm == 3.5
    assert m.RMSE_mm == pytest.approx(np.sqrt(12.5))
    assert m.MASE == 1.75
    assert m.bias_mm == -0.5


@pytest.mark.parametrize("corruption", ["duplicate", "gap", "nan", "negative", "fractional_year"])
def test_invalid_data_is_rejected(tmp_path, corruption):
    frame = pd.DataFrame({"YEAR": np.arange(1901,1931), "ANNUAL": np.arange(30.)+1000., "REGION": "INDIA"})
    if corruption == "duplicate": frame.loc[1,"YEAR"] = 1901
    if corruption == "gap": frame = frame.drop(10)
    if corruption == "nan": frame.loc[1,"ANNUAL"] = np.nan
    if corruption == "negative": frame.loc[1,"ANNUAL"] = -1
    if corruption == "fractional_year": frame = frame.astype({"YEAR": float}); frame.loc[1,"YEAR"] = 1902.5
    path = tmp_path/"bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError): ev.load_data(path)


def test_configuration_grid_matches_frozen_plan():
    plan = json.loads((ev.ROOT/"experiment_plan.json").read_text())
    configs = ev.configurations(plan)
    assert len(configs) == 22
    assert len({c["name"] for c in configs}) == 22
    assert plan["validation"][1] < plan["final_test"][0]


def test_paired_bootstrap_reproducible_and_handles_identical_predictions():
    actual = np.arange(20.)
    predictions = actual+2
    a = ev.compare_errors(actual, predictions, predictions, n_boot=100)
    b = ev.compare_errors(actual, predictions, predictions, n_boot=100)
    assert a == b
    assert a["rmse_reduction_mm"] == 0
    assert a["rmse_reduction_95pct_block_bootstrap_ci_mm"] == [0.,0.]
    assert a["approximate_dm_two_sided_p"] is None


def test_constant_history_mase_is_rejected():
    frame = pd.DataFrame({"YEAR": np.arange(1901,1931), "ANNUAL": np.ones(30)})
    with pytest.raises(ValueError, match="MASE undefined"):
        ev.backtest(frame, [{"name":"historical_mean", "family":"baseline"}], 1930,1930)


def test_saved_run_uses_locked_selection_and_past_only_training():
    out = ev.ROOT/"evaluation_results"
    if not (out/"test_predictions.csv").exists():
        pytest.skip("Run evaluate.py first to check saved outputs")
    selection = json.loads((out/"selection.json").read_text())
    val = pd.read_csv(out/"validation_metrics.csv")
    predictions = pd.read_csv(out/"test_predictions.csv")
    assert len(val) == 22
    assert selection["recommended_model"] == val.sort_values(["RMSE_mm", "model"]).iloc[0].model
    assert predictions.year.min() == 1996 and predictions.year.max() == 2015
    assert (predictions.train_end == predictions.year-1).all()
    assert (predictions.n_train == predictions.year-1901).all()
    assert predictions.groupby("model").size().eq(20).all()
    assert not predictions.duplicated(["model", "year"]).any()
    recalculated = ev.metrics(predictions)
    saved = pd.read_csv(out/"test_metrics.csv")
    pd.testing.assert_frame_equal(recalculated, saved, check_exact=False, rtol=1e-12)
