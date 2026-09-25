import numpy as np
import pandas as pd
import pytest
import requests

import mighte.evaluate as evaluation
from mighte.contract import QUANTILES, HOSP
from mighte.evaluate import BENCHMARKS, LOCAL_BASELINE, fetch_benchmarks, score_quantiles, summarize, wis


def test_wis_equals_weighted_interval_definition():
    y = 42.
    q = np.linspace(3, 65, len(QUANTILES))
    numerator = .5 * abs(y - q[11])
    for i in range(11):
        alpha = 2 * QUANTILES[i]
        lo, hi = q[i], q[-i-1]
        interval_score = hi - lo + 2 / alpha * max(lo - y, 0) + 2 / alpha * max(y - hi, 0)
        numerator += alpha / 2 * interval_score
    assert wis(np.array([y]), q[None, :])[0] == pytest.approx(numerator / 11.5)


def test_skill_and_coverage_use_matched_denominators(forecast):
    f = pd.concat([forecast.assign(model_id="MIGHTE-Base"),
                   forecast[forecast.horizon.eq(0)].assign(model_id=LOCAL_BASELINE, value=0)])
    truth = pd.DataFrame({"target": HOSP, "location": "01", "date": ["2026-10-10", "2026-10-17"], "value": [21., 22.]})
    scored = score_quantiles(f, truth)
    summary = summarize(scored).set_index("model_id")
    assert summary.loc["MIGHTE-Base", "n"] == 2
    assert summary.loc["MIGHTE-Base", "matched_n"] == 1
    assert summary.loc["MIGHTE-Base", "coverage_50"] == 1
    own = scored[scored.model_id.eq("MIGHTE-Base") & scored.horizon.eq(0)].wis.iloc[0]
    baseline = scored[scored.model_id.eq(LOCAL_BASELINE)].wis.iloc[0]
    assert summary.loc["MIGHTE-Base", "wis_skill"] == pytest.approx(1 - own / baseline)


def test_truth_revision_changes_scores_without_modifying_forecasts(forecast):
    frozen = forecast.copy(deep=True)
    truth = pd.DataFrame({"target": [HOSP], "location": ["01"], "date": ["2026-10-10"], "value": [21.]})
    early = score_quantiles(forecast.assign(model_id="MIGHTE-Base"), truth)
    late = score_quantiles(forecast.assign(model_id="MIGHTE-Base"), truth.assign(value=50.))
    assert early.wis.iloc[0] < late.wis.iloc[0]
    pd.testing.assert_frame_equal(forecast, frozen)


def test_zero_baseline_skill_is_undefined(forecast):
    f = pd.concat([forecast.assign(model_id="MIGHTE-Base", value=1),
                   forecast.assign(model_id=LOCAL_BASELINE, value=0)])
    truth = pd.DataFrame({"target": [HOSP], "location": ["01"], "date": ["2026-10-10"], "value": [0.]})
    summary = summarize(score_quantiles(f, truth)).set_index("model_id")
    assert pd.isna(summary.loc["MIGHTE-Base", "wis_skill"])


def test_comparison_models_refresh_and_work_from_offline_cache(tmp_path, monkeypatch, forecast):
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = forecast.to_csv(index=False).encode()
        return response

    monkeypatch.setattr(evaluation.requests, "get", get)
    online, status = fetch_benchmarks(tmp_path, ["2026-10-10"])
    assert set(online.model_id) == {
        "FluSight-baseline", "FluSight-ensemble", "UMass-flusion", "Google_SAI-FluEns"}
    assert len(requested) == 4
    assert {s["status"] for s in status} == {"refreshed"}

    def unexpected_request(*args, **kwargs):
        raise AssertionError("Offline review must not contact the hub")

    monkeypatch.setattr(evaluation.requests, "get", unexpected_request)
    offline, status = fetch_benchmarks(tmp_path, ["2026-10-10", "2026-10-17"], online=False)
    pd.testing.assert_frame_equal(online, offline)
    assert {s["status"] for s in status if s["reference_date"] == "2026-10-10"} == {"cached (offline)"}
    assert {s["status"] for s in status if s["reference_date"] == "2026-10-17"} == {"not cached"}


@pytest.mark.parametrize("has_prospective_run", [False, True])
def test_displaying_comparators_for_preview_does_not_score_that_week(
        tmp_path, monkeypatch, forecast, has_prospective_run):
    archive = forecast.assign(model_id="MIGHTE-Base") if has_prospective_run else pd.DataFrame()
    monkeypatch.setattr(evaluation, "load_archive", lambda root: archive)
    preview = forecast.assign(
        reference_date="2026-10-17",
        target_end_date=(pd.to_datetime(forecast.target_end_date) + pd.Timedelta(weeks=1)).dt.strftime("%Y-%m-%d"))
    benchmarks = pd.concat([frame.assign(model_id=model) for frame in [forecast, preview]
                            for model in BENCHMARKS], ignore_index=True)

    def fetch(root, references, *, online):
        assert set(references) == ({"2026-10-10", "2026-10-17"} if has_prospective_run else {"2026-10-17"})
        return benchmarks[benchmarks.reference_date.isin(references)], []

    monkeypatch.setattr(evaluation, "fetch_benchmarks", fetch)
    truth = pd.concat([forecast, preview]).rename(columns={"target_end_date": "date"})
    truth[["target", "location", "date", "value"]].drop_duplicates(["target", "location", "date"]).to_csv(
        tmp_path / "truth.csv", index=False)
    scores, _, display = evaluation.evaluate(
        tmp_path, tmp_path, online=False, comparison_references=["2026-10-17"])
    assert set(display[display.reference_date.eq("2026-10-17")].model_id) == set(BENCHMARKS)
    if has_prospective_run:
        assert set(scores.reference_date) == {"2026-10-10"}
        assert set(scores.model_id) == {"MIGHTE-Base", *BENCHMARKS}
    else:
        assert scores.empty
