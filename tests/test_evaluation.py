import numpy as np
import pandas as pd
import pytest

from mighte.contract import QUANTILES, HOSP
from mighte.evaluate import wis, score_quantiles, summarize, LOCAL_BASELINE


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
