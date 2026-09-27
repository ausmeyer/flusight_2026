import json
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from mighte import core
from mighte.contract import ED, HOSP
from mighte.data import (API, Downloader, choose_truth, clean_truth, wastewater_weekly, weekly_grid, WW,
                         ed_ilinet_proxy, hospitalization_history)
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


def test_ed_history_is_built_from_ilinet_like_hospitalizations(root):
    ili = pd.read_csv(root / "data/historical/ilinet_normalized.csv", parse_dates=["date"])
    locations = ["US", "Alabama"]
    observed = ili[ili.location_name.isin(locations) & ili.date.ge("2022-10-01")].copy()
    observed["total_hosp"] = 100 * expit(-5 + .4 * observed.ili)
    proxy, audit = ed_ilinet_proxy(root, observed[["location_name", "date", "total_hosp"]], ED_TRANSFORM)
    # One pooled log-odds regression across locations, as for the hospitalization seed.
    assert audit["coefficients"] == pytest.approx([-5, .4])
    assert audit["mapping_locations"] == 2 and audit["shift_days"] == 728
    assert audit["response_transform"] == ED_TRANSFORM
    source = ili[ili.location_name.isin(locations) & ili.date.le("2019-06-30")]
    assert len(proxy) == len(source)
    assert proxy.date.tolist() == (source.date + pd.Timedelta(days=728)).tolist()
    np.testing.assert_allclose(proxy.total_hosp, np.round(100 * expit(-5 + .4 * source.ili.to_numpy()), 2))
    with pytest.raises(ValueError, match="52 paired weeks"):
        ed_ilinet_proxy(root, observed.head(10)[["location_name", "date", "total_hosp"]], ED_TRANSFORM)


def historical_inputs(root, dates, locations):
    """Seed of 40 admissions a week before July 2021 and an ILINet predictor of 100 every week."""
    directory = root / "data/historical"
    directory.mkdir(parents=True)
    seed_dates = pd.date_range("2021-05-01", "2021-06-26", freq="W-SAT")
    pd.DataFrame([{"date": d, "location_name": n, "total_hosp": 40.} for n in locations for d in seed_dates]).to_csv(
        directory / "hospitalization_proxy.csv", index=False)
    pd.DataFrame([{"date": d, "location_name": n, "proxy_hosp_continuous": 100.} for n in locations for d in dates]).to_csv(
        directory / "hospitalization_calibration.csv", index=False)


def test_hospitalization_history_levels_each_era_to_the_current_regime(tmp_path):
    dates = pd.date_range("2021-07-03", "2025-12-27", freq="W-SAT")
    historical_inputs(tmp_path, dates, ["A", "B", "US"])
    # Admissions per unit of predictor: 1.2 before May 2024, 1.25 while reporting was voluntary,
    # then 1.8 in season and 1.5 off season under the current regime. B reports four times as many.
    y = np.where(dates >= "2024-11-01", np.where(dates.month.isin([11, 12, 1, 2, 3, 4]), 180., 150.), 120.)
    y = np.where((dates >= "2024-05-01") & (dates <= "2024-09-30"), 125., y)
    observed = pd.concat([pd.DataFrame({"date": dates, "location_name": n, "total_hosp": y * (4 if n == "B" else 1)})
                          for n in ["A", "B", "US"]], ignore_index=True)
    observed.loc[observed.date.eq("2025-01-04"), "total_hosp"] += 0.25
    original = observed.copy()
    history, filled, levels = hospitalization_history(tmp_path, observed, dates[-1])
    f = levels.set_index("location_name")
    assert f.loc["A", ["seed", "legacy", "voluntary"]].tolist() == pytest.approx([1.8, 1.5, 1.2], rel=1e-3)
    assert f.loc["B", "seed"] == 2.0 and f.loc["B", "legacy"] == pytest.approx(1.5, rel=1e-3)  # bounded
    get = lambda day: history.loc[history.location_name.eq("A") & history.date.eq(pd.Timestamp(day)), "total_hosp"].item()
    level = lambda value, factor: np.rint(np.expm1(np.log1p(value) + np.log(factor)))
    a = f.loc["A"]
    assert get("2021-06-05") == level(40, a.seed)
    assert get("2021-06-26") == level(40, np.sqrt(a.seed * a.legacy))  # halfway through the splice ramp
    assert get("2023-01-07") == level(120, a.legacy)
    assert get("2024-07-06") == level(125, a.voluntary)
    assert get("2024-10-12") == level(120, a.voluntary ** (1 - 14 / 35))  # hospitals resuming reporting
    # The current regime is exactly as reported, even values that are not whole counts.
    current = history[history.date.ge("2024-11-02")].merge(observed, on=["location_name", "date"])
    assert len(current) == 3 * 61 and current.total_hosp_x.eq(current.total_hosp_y).all()
    assert filled == 0 and history.date.max() == dates[-1]
    pd.testing.assert_frame_equal(observed, original)


def test_history_is_not_leveled_before_eight_current_regime_weeks(tmp_path):
    dates = pd.date_range("2021-07-03", "2024-12-14", freq="W-SAT")
    historical_inputs(tmp_path, dates, ["A"])
    observed = pd.DataFrame({"date": dates, "location_name": "A", "total_hosp": 100.})
    observed.loc[5, "total_hosp"] = np.nan
    history, filled, levels = hospitalization_history(tmp_path, observed, dates[-1])
    assert levels[["seed", "legacy", "voluntary"]].iloc[0].tolist() == [1.0, 1.0, 1.0]
    assert filled == 1 and history.loc[history.date.ge("2021-07-03"), "total_hosp"].eq(100).all()
    assert history.loc[history.date.lt("2021-07-03"), "total_hosp"].eq(40).all()
