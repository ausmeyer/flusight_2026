import pandas as pd
import requests

from mighte import evaluate
from mighte.contract import HOSP, PEAK_HEIGHT, PEAK_WEEK


def test_display_accepts_seasonal_null_dates_without_adding_weekly_hindcasts(forecast, peak_forecasts):
    source = pd.concat([forecast, peak_forecasts, forecast.assign(horizon=-1)], ignore_index=True)
    display = evaluate.display_rows(source)
    assert len(display) == len(forecast) + len(peak_forecasts)
    peaks = display[display.target.isin([PEAK_HEIGHT, PEAK_WEEK])]
    assert peaks.horizon.isna().all() and peaks.target_end_date.isna().all()
    assert set(display[display.target.eq(HOSP)].horizon) == {0, 1, 2, 3}
    assert evaluate.display_rows(peak_forecasts.assign(horizon=0, target_end_date="2026-10-10")).empty


def test_peak_only_hub_model_loads_online_and_from_cache(tmp_path, monkeypatch, peak_forecasts):
    def get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = peak_forecasts.to_csv(index=False).encode()
        return response

    monkeypatch.setattr(evaluate.requests, "get", get)
    catalog = {"revision": "commit", "files": [
        {"model": "Peak-only", "reference_date": "2026-10-10", "blob_sha": "blob"}]}
    loaded, _ = evaluate.fetch_benchmarks(tmp_path, ["2026-10-10"], catalog=catalog)
    assert set(loaded.target) == {PEAK_HEIGHT, PEAK_WEEK}
    assert set(loaded.model_id) == {"Peak-only"}
    assert loaded.horizon.isna().all() and loaded.target_end_date.isna().all()
    assert loaded[loaded.target.eq(PEAK_WEEK)].output_type_id.iloc[-1] == "2027-05-29"
    cached, _ = evaluate.fetch_benchmarks(tmp_path, ["2026-10-10"], catalog=catalog, online=False)
    pd.testing.assert_frame_equal(loaded, cached)


def test_peak_distributions_do_not_enter_weekly_accuracy(forecast, peak_forecasts):
    truth = pd.DataFrame({"target": [HOSP], "location": ["01"], "date": ["2026-10-10"], "value": [21]})
    baseline = evaluate.score_quantiles(forecast.assign(model_id="MIGHTE-Base"), truth)
    combined = pd.concat([forecast.assign(model_id="MIGHTE-Base"), peak_forecasts.assign(model_id="Peak-only")])
    scored = evaluate.score_quantiles(combined, truth)
    pd.testing.assert_frame_equal(scored, baseline, check_dtype=False)
    assert evaluate.score_quantiles(peak_forecasts.assign(model_id="Peak-only"), truth).empty
