"""Independent invariants and synthetic trend tests for the audit only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from mighte import panel_ar as pa
from mighte.data import weekly_grid

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = json.loads((ROOT / "config/settings.json").read_text())
RUNTIME = pa.PanelARRuntime.from_dict(SETTINGS["panel_ar_runtime"])
SPEC = pa.PanelARSpec("diagnostic", partial_pooling=True)


def synthetic(rate, future=0):
    dates = pd.date_range("2022-01-01", periods=208 + future, freq="W-SAT")
    rows = []
    for name, base in [("A", 5), ("B", 20), ("C", 80), ("US", 500)]:
        for i, date in enumerate(dates):
            rows.append({"location_name": name, "date": date,
                         "total_hosp": np.expm1(np.log1p(base) + rate * i)})
    return pd.DataFrame(rows)


def test_exact_origin_lags_and_direct_labels():
    frame = synthetic(.03, 4)
    anchor = sorted(frame.date.unique())[207]
    table = pa.prepare_panel_ar_table(frame, RUNTIME)
    rows = table[table.date.eq(anchor)]
    for week in RUNTIME.ar_weeks:
        np.testing.assert_allclose(rows[f"panel_delta_log_week_{week}"], .03, atol=1e-13)
    np.testing.assert_allclose(rows.panel_target_delta_log, .03 * rows.lead_weeks, atol=1e-13)


def test_future_poison_cannot_change_origin_features_or_training_targets():
    frame = synthetic(.03, 4)
    anchor = sorted(frame.date.unique())[207]
    poison = frame.copy()
    poison.loc[poison.date.gt(anchor), "total_hosp"] = 1e20
    a, b = [pa.prepare_panel_ar_table(h, RUNTIME) for h in (frame, poison)]
    cols = [c for c in a if c != "panel_target_delta_log"]
    pd.testing.assert_frame_equal(a.loc[a.date.le(anchor), cols], b.loc[b.date.le(anchor), cols])
    at, _ = pa.panel_ar_training_and_test_rows(a, anchor, RUNTIME)
    bt, _ = pa.panel_ar_training_and_test_rows(b, anchor, RUNTIME)
    pd.testing.assert_frame_equal(at, bt)


def test_missing_calendar_target_is_missing_and_weekly_preparation_keeps_lag_dates():
    frame = synthetic(.03)
    missing = frame.date.max() - pd.Timedelta(weeks=1)
    sparse = frame[frame.date.ne(missing)]
    table = pa.prepare_panel_ar_table(sparse, RUNTIME)
    row = table[table.date.eq(missing - pd.Timedelta(weeks=1)) & table.lead_weeks.eq(1)]
    assert row.panel_target_delta_log.isna().all()
    grid, filled = weekly_grid(sparse, frame.date.max())
    assert filled == 4
    assert grid.groupby("location_name").date.diff().dropna().eq(pd.Timedelta(weeks=1)).all()


def test_known_direct_growth_integrates_once_from_anchor(monkeypatch):
    frame = synthetic(.03)
    anchor = frame.date.max()
    pooled = pa.prepare_panel_ar_table(frame, RUNTIME)
    train, test = pa.panel_ar_training_and_test_rows(pooled, anchor, RUNTIME)
    required = pa.panel_ar_continuous_columns(RUNTIME) + pa.panel_ar_binary_columns() + ["panel_target_delta_log"]
    train = train.dropna(subset=required).reset_index(drop=True)
    _, _, _, names = pa.build_panel_ar_mean_design(train, test, RUNTIME, SPEC)

    def known_fit(x, z, y, p, sp, runtime):
        beta, gamma = np.zeros(x.shape[1]), np.zeros(z.shape[1])
        beta[names.index("intercept")] = .03
        for h in range(2, 5):
            beta[names.index(f"lead_{h}")] = .03 * (h - 1)
        gamma[0] = -1
        return beta, gamma

    monkeypatch.setattr(pa, "fit_distributional_panel_ar", known_fit)
    result = pa.forecast_panel_ar_anchor(pooled, anchor, SPEC, RUNTIME, {x: x for x in frame.location_name.unique()})
    medians = result[result.output_type_id.eq(.5)]
    origins = frame[frame.date.eq(anchor)].set_index("location_name").total_hosp
    expected = np.expm1(np.log1p(medians.location.map(origins)) + .03 * (medians.horizon + 1))
    np.testing.assert_allclose(medians.value, expected, rtol=1e-12)
    assert (pd.to_datetime(medians.target_end_date) - anchor).eq(pd.to_timedelta((medians.horizon + 1) * 7, unit="D")).all()
    assert medians.reference_date.eq((anchor + pd.Timedelta(weeks=1)).date().isoformat()).all()


@pytest.mark.parametrize("rate", [0., .01, .03])
def test_fitted_model_can_continue_flat_and_rising_series(rate):
    frame = synthetic(rate)
    anchor = frame.date.max()
    with threadpool_limits(limits=4):
        result = pa.forecast_panel_ar_anchor(pa.prepare_panel_ar_table(frame, RUNTIME), anchor, SPEC, RUNTIME,
                                             {x: x for x in frame.location_name.unique()})
    assert not result.empty
    medians = result[result.output_type_id.eq(.5)].sort_values(["location", "horizon"])
    origins = frame[frame.date.eq(anchor)].set_index("location_name").total_hosp
    expected = np.expm1(np.log1p(medians.location.map(origins)) + rate * (medians.horizon + 1))
    # A capacity check, not an exact coefficient match: the frozen ridge fit
    # need not recover a deterministic synthetic law with zero error.
    np.testing.assert_allclose(np.log1p(medians.value), np.log1p(expected), atol=.02)
    if rate:
        assert medians.groupby("location").value.diff().dropna().gt(0).all()
    else:
        np.testing.assert_allclose(medians.value, medians.location.map(origins), atol=1e-6)
