import json
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from mighte import core
from mighte.contract import ED, HOSP
from mighte.data import (API, Downloader, choose_truth, clean_truth, wastewater_weekly, weekly_grid, WW,
                         reconstruct_ed, hospitalization_history, attach_ed_fraction)
from mighte.models import equal_quantiles
from mighte.panel_ar import PanelARRuntime, prepare_panel_ar_table

ED_TRANSFORM = {"name": "logit", "boundary_epsilon": .00005}


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


def test_hub_values_and_update_timestamp_use_same_commit(tmp_path, monkeypatch):
    downloader = Downloader(tmp_path)
    calls = []

    def get(url, name, params=None):
        calls.append((url, params))
        if url == API + "commits/main":
            return b'{"sha":"fixed-commit"}'
        if url == API + "commits":
            return b'[{"commit":{"committer":{"date":"2026-10-07T15:00:00Z"}}}]'
        return b'date,location,value\n2026-10-03,01,10\n'

    monkeypatch.setattr(downloader, "get", get)
    frame, updated = downloader.hub_truth("target-hospital-admissions.csv")
    assert frame.value.item() == 10
    assert updated == "2026-10-07T15:00:00Z"
    assert "/fixed-commit/target-data/" in calls[1][0]
    assert calls[2][1]["sha"] == "fixed-commit"


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
    obs["total_hosp"] = 100 * expit(-5 + .1 * obs.ili)
    result, audit = reconstruct_ed(root, obs[["date", "location_name", "total_hosp"]],
                                   {"mode": "post2022_proxy"}, ED_TRANSFORM)
    assert audit["coefficients"] == pytest.approx([-5, .1])
    assert audit["response_transform"] == ED_TRANSFORM
    assert audit["shift_days"] == 728
    expected = ili.loc[(ili.location_name == "US") & ili.date.le("2019-06-30"), "date"].min() + pd.Timedelta(days=728)
    assert result.date.min() == expected
    with pytest.raises(ValueError, match="historical_nssp_file"):
        reconstruct_ed(root, obs, {"mode": "prepandemic_proxy"}, ED_TRANSFORM)


def test_supplied_prepandemic_ed_mapping(root, tmp_path):
    shutil.copytree(root / "data/historical", tmp_path / "data/historical")
    ili = pd.read_csv(root / "data/historical/ilinet_normalized.csv", parse_dates=["date"])
    national = ili[ili.location_name.eq("US") & ili.date.lt("2020-03-01")].copy()
    national["value"] = expit(-5 + .1 * national.ili)
    national[["date", "value"]].to_csv(tmp_path / "data/historical/nssp.csv", index=False)
    observed = pd.DataFrame({"location_name": ["US"], "date": pd.to_datetime(["2026-09-19"]), "total_hosp": [.5]})
    settings = {"mode": "prepandemic_proxy", "historical_nssp_file": "data/historical/nssp.csv"}
    result, audit = reconstruct_ed(tmp_path, observed, settings, ED_TRANSFORM)
    assert audit["coefficients"] == pytest.approx([-5, .1])
    assert result.loc[result.date.eq("2026-09-19"), "total_hosp"].item() == .5
    assert not result.date.between("2021-06-27", "2026-09-18").any()
    pd.concat([national[["date", "value"]], national[["date", "value"]].iloc[:1]]).to_csv(
        tmp_path / "data/historical/nssp.csv", index=False)
    with pytest.raises(ValueError, match="one observation"):
        reconstruct_ed(tmp_path, observed, settings, ED_TRANSFORM)


def test_ed_fraction_carries_forward_without_leading_extrapolation():
    dates = pd.date_range("2023-01-07", periods=4, freq="W-SAT")
    frame = pd.DataFrame({"date": dates, "location_name": "A"})
    fractions = pd.DataFrame({"date": dates[1:3], "location_name": "A", "scale_factor": [.5, np.nan]})
    assert attach_ed_fraction(frame, fractions).scale_factor.tolist() == [1, .5, .5, .5]


def test_historical_adjustment_preserves_raw_truth_and_uses_current_revisions(tmp_path):
    directory = tmp_path / "data/historical"
    directory.mkdir(parents=True)
    dates = pd.date_range("2023-01-07", "2025-03-29", freq="W-SAT")
    observed = pd.DataFrame({"date": dates, "location_name": "A",
                              "total_hosp": np.where(dates < "2024-11-01", 99., 199.)})
    original = observed.copy()
    pd.DataFrame({"date": dates, "location_name": "A", "proxy_hosp_continuous": 100.}).to_csv(
        directory / "hospitalization_calibration.csv", index=False)
    pd.DataFrame({"date": ["2021-06-26"], "location_name": ["A"], "total_hosp": [99.]}).to_csv(
        directory / "hospitalization_proxy.csv", index=False)
    pd.DataFrame({"date": dates, "location_name": "A", "scale_factor": .5}).to_csv(
        tmp_path / "ed-fractions.csv", index=False)
    adjusted, _, audit = hospitalization_history(tmp_path, tmp_path, observed, dates[-1])
    get = lambda day: adjusted.loc[adjusted.date.eq(day), "total_hosp"].item()
    assert get("2023-01-07") == 101  # round(99 * .5), then expm1(log1p(50) + log(2))
    assert get("2024-06-01") == 199  # legacy shift alone after May
    assert get("2024-11-02") == 199  # current observations unchanged
    assert get("2021-06-26") == 199  # leading ED fraction defaults to one
    assert audit["locations"][0]["shift_legacy_to_current"] == pytest.approx(np.log(2))
    pd.testing.assert_frame_equal(observed, original)
    revised = observed.copy()
    revised.loc[revised.date.ge("2024-11-01"), "total_hosp"] = 299
    _, _, bounded = hospitalization_history(tmp_path, tmp_path, revised, dates[-1])
    assert bounded["locations"][0]["shift_legacy_to_current"] == pytest.approx(np.log(2))
    revised.loc[revised.date.ge("2024-11-01"), "total_hosp"] = 99
    updated, _, new_audit = hospitalization_history(tmp_path, tmp_path, revised, dates[-1])
    assert new_audit["locations"][0]["shift_legacy_to_current"] == pytest.approx(0)
    assert updated.loc[updated.date.eq("2023-01-07"), "total_hosp"].item() == 50
    # Future observations never supply the minimum eight current-regime weeks.
    early, _, insufficient = hospitalization_history(tmp_path, tmp_path, observed, pd.Timestamp("2024-12-14"))
    assert insufficient["locations"][0]["n_current"] == 7
    assert insufficient["locations"][0]["shift_legacy_to_current"] == 0
    assert early.date.max() == pd.Timestamp("2024-12-14")
    # A real surveillance fraction must survive CSV parsing before half-count rounding.
    fractions = pd.DataFrame({"date": dates, "location_name": "A", "scale_factor": .5})
    fractions.loc[0, "scale_factor"] = .26724137931034486
    fractions.to_csv(tmp_path / "ed-fractions.csv", index=False)
    observed.loc[0, "total_hosp"] = 58
    precise, _, _ = hospitalization_history(tmp_path, tmp_path, observed, dates[-1])
    assert precise.loc[precise.date.eq(dates[0]), "total_hosp"].item() == 33
