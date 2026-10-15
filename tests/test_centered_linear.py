import numpy as np
import pandas as pd
import pytest

from mighte.models import linear
from mighte.panel_ar import (PanelARRuntime, PanelARSpec, build_panel_ar_mean_design,
                             build_panel_ar_scale_design, panel_ar_binary_columns,
                             panel_ar_continuous_columns, panel_ar_training_and_test_rows,
                             prepare_panel_ar_table)


@pytest.fixture
def panel_rows():
    runtime = PanelARRuntime.from_dict({"max_horizons": 2, "ar_weeks": [1, 2],
                                       "rolling_windows": [2], "min_train_rows": 20,
                                       "joint_iterations": 1})
    dates = pd.date_range("2022-01-01", periods=35, freq="7D")
    history = pd.DataFrame([{"location_name": name, "date": date,
                             "total_hosp": (i + 1) * (20 + j + j % 3)}
                            for i, name in enumerate(["Alabama", "Georgia", "US"])
                            for j, date in enumerate(dates)])
    anchor = dates[25]
    pooled = prepare_panel_ar_table(history, runtime)
    train, test = panel_ar_training_and_test_rows(pooled, anchor, runtime)
    required = panel_ar_continuous_columns(runtime) + panel_ar_binary_columns()
    train = train.dropna(subset=required + ["panel_target_delta_log"]).reset_index(drop=True)
    test = test.dropna(subset=required).reset_index(drop=True)
    return history, anchor, runtime, train, test


def test_mean_design_is_invariant_to_location_level_offsets(panel_rows):
    _, _, runtime, train, test = panel_rows
    spec = PanelARSpec("MIGHTE-Linear", partial_pooling=True)
    before_train, before_test = train.copy(), test.copy()
    expected = build_panel_ar_mean_design(train, test, runtime, spec)
    shifted_train, shifted_test = train.copy(), test.copy()
    offsets = {"Alabama": 3., "Georgia": -2., "US": 6.}
    for frame in [shifted_train, shifted_test]:
        frame["panel_log_level_t"] += frame.location_name.map(offsets)
    actual = build_panel_ar_mean_design(shifted_train, shifted_test, runtime, spec)
    for a, b in zip(actual[:3], expected[:3]):
        np.testing.assert_allclose(a, b, atol=1e-12)
    assert actual[3] == expected[3]
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(test, before_test)


def test_test_levels_cannot_fit_training_centers_or_scalers(panel_rows):
    _, _, runtime, train, test = panel_rows
    spec = PanelARSpec("MIGHTE-Linear", partial_pooling=True)
    expected = build_panel_ar_mean_design(train, test, runtime, spec)
    changed = test.copy()
    changed["panel_log_level_t"] += 100
    actual = build_panel_ar_mean_design(train, changed, runtime, spec)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert np.max(np.abs(actual[1] - expected[1])) > 1


def test_centering_preserves_scale_design_and_reconstruction_levels(panel_rows):
    _, _, runtime, train, test = panel_rows
    levels = test.panel_log_level_t.copy()
    expected = build_panel_ar_scale_design(train, test, runtime)
    build_panel_ar_mean_design(train, test, runtime, PanelARSpec("MIGHTE-Linear", True))
    actual = build_panel_ar_scale_design(train, test, runtime)
    for a, b in zip(actual[:3], expected[:3]):
        np.testing.assert_array_equal(a, b)
    pd.testing.assert_series_equal(test.panel_log_level_t, levels)


def test_future_observations_cannot_change_centered_forecasts(panel_rows):
    history, anchor, runtime, _, _ = panel_rows
    settings = {"panel_ar_runtime": {**runtime.__dict__}}
    locations = {"Alabama": "01", "Georgia": "13", "US": "US"}
    reference = (anchor + pd.Timedelta(weeks=1)).date().isoformat()
    expected = linear(history, reference, settings, locations)
    changed = history.copy()
    changed.loc[changed.date > anchor, "total_hosp"] = 1e12
    actual = linear(changed, reference, settings, locations)
    pd.testing.assert_frame_equal(actual, expected)
    assert np.isfinite(actual.value).all() and (actual.value >= 0).all()
    for _, group in actual.groupby(["location", "horizon"]):
        assert (np.diff(group.sort_values("output_type_id").value) >= 0).all()
