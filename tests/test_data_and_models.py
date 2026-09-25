import json

import numpy as np
import pandas as pd
import pytest

from mighte import core
from mighte.contract import ED, HOSP
from mighte.data import choose_truth, clean_truth, wastewater_weekly, weekly_grid, WW, reconstruct_ed
from mighte.models import equal_quantiles
from mighte.panel_ar import PanelARRuntime, prepare_panel_ar_table


def test_backfill_replaces_prior_values_including_new_suppression():
    old = pd.DataFrame({"date": ["2026-09-05", "2026-09-12"], "location": ["01", "01"], "value": [10, 20]})
    new = old.assign(value=[15, np.nan])
    result, audit = choose_truth([("hub", old, "2026-09-15Z".replace("Z", "T00:00:00Z")),
                                  ("cdc", new, "2026-09-20T00:00:00Z")], HOSP)
    assert result.iloc[0].value == 15
    assert pd.isna(result.iloc[1].value)
    assert audit["priority_newest_first"] == ["cdc", "hub"]


def test_hub_can_be_the_newest_source():
    frame = pd.DataFrame({"date": ["2026-10-03"], "location": ["US"], "value": [.005]})
    result, audit = choose_truth([("CDC", frame, "2026-10-06T00:00:00Z"),
                                  ("hub", frame.assign(value=.006), "2026-10-07T00:00:00Z")], ED)
    assert result.iloc[0].value == .006
    assert audit["priority_newest_first"][0] == "hub"


def test_no_endpoint_extrapolation():
    frame = pd.DataFrame({"location_name": ["A", "A", "A"],
                          "date": pd.to_datetime(["2026-08-29", "2026-09-12", "2026-09-19"]),
                          "total_hosp": [2, 6, np.nan]})
    grid, filled = weekly_grid(frame, pd.Timestamp("2026-09-19"))
    assert grid.total_hosp.iloc[1] == 4
    assert pd.isna(grid.total_hosp.iloc[-1])
    assert filled == 1


def test_wastewater_equal_site_and_calendar_lags():
    rows = [{"site": str(s), "sample_collect_date": day, "pcr_target_mic_lin": 1e-4}
            for day in ["2026-09-01", "2026-09-15"] for s in range(10)]
    # Many samples at one site must not give that site extra national weight.
    rows += [{"site": "0", "sample_collect_date": "2026-09-01", "pcr_target_mic_lin": 1e-2}] * 100
    weekly = wastewater_weekly(pd.DataFrame(rows))
    assert weekly.iloc[0][WW] == pytest.approx(np.log10(.00011))
    assert pd.isna(weekly.iloc[1][WW + "_lag1"])
    assert weekly.iloc[1][WW + "_lag2"] == weekly.iloc[0][WW]
    assert pd.isna(wastewater_weekly(pd.DataFrame(rows).query("site != '9'")).iloc[0][WW])


def test_pooled_targets_match_calendar_dates():
    features = pd.DataFrame({"location_name": ["A", "A"],
                             "date": pd.to_datetime(["2026-09-05", "2026-09-19"]), "total_hosp": [10, 999]})
    examples = core.build_pooled_examples(features, 2)
    first = examples[examples.date.eq("2026-09-05")]
    assert pd.isna(first[first.horizon_weeks.eq(1)].target.iloc[0])
    assert first[first.horizon_weeks.eq(2)].target.iloc[0] == 999


def test_features_do_not_see_future(root, history):
    config = json.loads((root / "config/settings.json").read_text())
    runtime = core.RuntimeConfig.from_dict(config["runtime"])
    cutoff = pd.Timestamp("2025-01-04")
    changed = history.copy()
    changed.loc[changed.date > cutoff, "total_hosp"] = 1e7
    before = core.build_feature_table(history, runtime)
    after = core.build_feature_table(changed, runtime)
    pd.testing.assert_frame_equal(before[before.date.le(cutoff)], after[after.date.le(cutoff)])
    ar = PanelARRuntime.from_dict(config["panel_ar_runtime"])
    before = prepare_panel_ar_table(history, ar)
    after = prepare_panel_ar_table(changed, ar)
    keep = [c for c in before if c != "panel_target_delta_log"]
    pd.testing.assert_frame_equal(before.loc[before.date.le(cutoff), keep], after.loc[after.date.le(cutoff), keep])


def test_covariate_issue_filter(tmp_path):
    file = tmp_path / "cov.csv"
    pd.DataFrame({"date": ["2026-09-12", "2026-09-12"],
                  "available_date": ["2026-09-16", "2026-09-30"], "signal": [2., 999.]}).to_csv(file, index=False)
    features = pd.DataFrame({"date": pd.to_datetime(["2026-09-19"]), "location_name": ["A"]})
    result = core.add_external_covariates(features, str(file), ["signal"], pd.Timestamp("2026-09-26"),
                                          source_lag_weeks=1, max_staleness_weeks=1)
    assert result.signal.iloc[0] == 2


def test_fixed_thirds_and_missing_component(forecast):
    result = equal_quantiles([forecast, forecast.assign(value=forecast.value + 3), forecast.assign(value=forecast.value + 6)])
    assert np.allclose(result.value, forecast.value + 3)
    with pytest.raises(ValueError, match="identical keys"):
        equal_quantiles([forecast, forecast, forecast.iloc[1:]])


def test_ed_proxy_shift_is_explicit(root):
    ili = pd.read_csv(root / "data/historical/ilinet_normalized.csv", parse_dates=["date"])
    obs = ili[(ili.location_name == "US") & ili.date.ge("2022-10-01")].copy()
    obs["total_hosp"] = np.expm1(.8 + .1 * obs.ili)
    result, audit = reconstruct_ed(root, obs[["date", "location_name", "total_hosp"]], {"mode": "post2022_proxy"})
    assert audit["coefficients"] == pytest.approx([.8, .1])
    assert audit["shift_days"] == 728
    expected = ili.loc[(ili.location_name == "US") & ili.date.le("2020-05-30"), "date"].min() + pd.Timedelta(days=728)
    assert result.date.min() == expected
    with pytest.raises(ValueError, match="historical_nssp_file"):
        reconstruct_ed(root, obs, {"mode": "prepandemic_proxy"})
