"""Unit, boundary, integration and cache checks for the ED response link."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from mighte import models
from mighte.contract import ED, QUANTILES
from mighte.ed_transform import boundary_epsilon, ed_logit

EPSILON = .00005


def test_link_preserves_interior_values_and_has_explicit_units():
    proportions = np.array([.0001, .0031, .025, .1749, .5, .9999])
    original = proportions.copy()
    linked = ed_logit(proportions, EPSILON)
    np.testing.assert_allclose(linked, np.log(proportions / (1 - proportions)), rtol=1e-14)
    np.testing.assert_allclose(expit(linked), proportions, rtol=1e-14)
    np.testing.assert_allclose(ed_logit((100 * proportions) / 100, EPSILON), linked, atol=1e-14)
    np.testing.assert_array_equal(proportions, original)


def test_boundary_handling_is_finite_and_does_not_floor_forecasts():
    linked = ed_logit([0, EPSILON / 2, 1], EPSILON)
    assert np.isfinite(linked).all()
    np.testing.assert_allclose(expit(linked), [EPSILON, EPSILON, 1 - EPSILON])
    # Only input observations are clipped; inverse-link quantiles may fall below epsilon.
    assert 0 < expit(linked[0] - 1) < EPSILON
    assert np.all(np.diff(expit(np.array([-1000., -10., 0., 10., 1000.]))) >= 0)


@pytest.mark.parametrize("values", [[-.01], [1.01], [np.nan], [np.inf]])
def test_response_link_rejects_invalid_proportions(values):
    with pytest.raises(ValueError, match="finite proportions"):
        ed_logit(values, EPSILON)


@pytest.mark.parametrize("epsilon", [0, -.1, .5, np.nan, np.inf])
def test_invalid_boundary_setting_fails(epsilon):
    with pytest.raises(ValueError, match="boundary_epsilon"):
        boundary_epsilon({"name": "logit", "boundary_epsilon": epsilon})


def test_unsupported_link_fails():
    with pytest.raises(ValueError, match="must be logit"):
        boundary_epsilon({"name": "log1p", "boundary_epsilon": EPSILON})


def test_ed_fit_integrates_log_odds_and_exports_proportions(root, tmp_path, monkeypatch):
    settings = json.loads((root / "config/settings.json").read_text())
    settings["runtime"].update(num_bags=1, min_train_rows=1)
    dates = pd.date_range("2025-01-04", periods=8, freq="W-SAT")
    train = pd.DataFrame({"date": dates, "target_date": dates + pd.Timedelta(weeks=1),
                          "total_hosp": [0, .01, .1, .31, 1, 2, 3, 5],
                          "target": [.01, .03, .2, .4, 1.5, 3, 0, 4],
                          "season": "2024/25", "predictor": np.arange(8)})
    test = pd.DataFrame([{"date": pd.Timestamp("2026-09-19"),
                         "target_date": pd.Timestamp("2026-09-26") + pd.Timedelta(weeks=h),
                         "horizon_weeks": h + 1, "location_name": name, "total_hosp": value,
                         "target": np.nan, "season": "2026/27", "predictor": 0}
                        for name, value in [("US", .31), ("Alabama", 0)] for h in range(4)])
    pooled = pd.concat([train, test], ignore_index=True)
    monkeypatch.setattr(models, "pooled_features", lambda *args, **kwargs: (pooled, ["predictor"]))
    fitted = []

    class Center:
        def predict(self, x):
            return np.full(len(x), .2)

    class Scale:
        def predict_sigma(self, x):
            return np.full(len(x), .35)

    def fit(x, y, *args, **kwargs):
        start = np.clip(train.total_hosp.to_numpy() / 100, EPSILON, 1 - EPSILON)
        end = np.clip(train.target.to_numpy() / 100, EPSILON, 1 - EPSILON)
        expected = np.log(end / (1 - end)) - np.log(start / (1 - start))
        np.testing.assert_allclose(y, expected, rtol=1e-14, atol=1e-14)
        fitted.append(True)
        return Center(), Scale()

    monkeypatch.setattr(models.core, "fit_production_lightgbmlss_one_bag", fit)
    checkpoint = tmp_path / "checkpoints"
    args = (pd.DataFrame(), tmp_path, "2026-09-26", settings, ("ww",), ED,
            "base-ed", checkpoint, {"US": "US", "Alabama": "01"})
    actual = models.distributional(*args)
    for location, anchor in [("US", .0031), ("01", EPSILON)]:
        log_odds = np.log(anchor / (1 - anchor)) + .2 + .35 * norm.ppf(QUANTILES)
        expected = 1 / (1 + np.exp(-log_odds))
        for horizon in range(4):
            values = actual.loc[actual.location.eq(location) & actual.horizon.eq(horizon), "value"]
            np.testing.assert_allclose(values, expected, rtol=1e-14)
            assert values.between(0, 1).all() and np.all(np.diff(values) > 0)
    repeat = models.distributional(*args)
    pd.testing.assert_frame_equal(actual, repeat)
    assert len(fitted) == 1
    metadata = json.loads((checkpoint / "fit.json").read_text())
    assert metadata["response_transform"]["boundary_epsilon"] == EPSILON
    assert metadata["response_transform"]["checkpoint_unit"] == "percentage points"
    settings["ed_target_transform"]["boundary_epsilon"] = .0001
    with pytest.raises(ValueError, match="transformation changed"):
        models.distributional(*args)
    settings["ed_target_transform"]["boundary_epsilon"] = EPSILON
    (checkpoint / "response-transform.json").unlink()
    with pytest.raises(ValueError, match="lack response transformation"):
        models.distributional(*args)
