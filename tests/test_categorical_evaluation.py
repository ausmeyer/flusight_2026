import json
import shutil

import numpy as np
import pandas as pd
import pytest
import requests

import mighte.evaluate as evaluation
from mighte.contract import CATEGORIES, HOSP, TREND, UNIT
from mighte.evaluate import TREND_BASELINE, score_categories, score_quantiles, summarize


@pytest.fixture
def probabilities():
    return pd.DataFrame([{"model_id": "MIGHTE-Base", "reference_date": "2026-10-10", "target": TREND,
                          "target_end_date": (pd.Timestamp("2026-10-10") + pd.Timedelta(weeks=h)).date().isoformat(),
                          "horizon": h, "location": "01", "output_type": "pmf", "output_type_id": category,
                          "value": value}
                         for h in range(4) for category, value in zip(CATEGORIES, [.1, .2, .4, .2, .1])])


@pytest.fixture
def observed():
    return pd.DataFrame({"target": HOSP, "location": "01",
                         "date": ["2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24"],
                         "value": [100, 120, 90, 200]})


def test_category_scores_use_reference_minus_seven_and_observed_truth(probabilities, observed):
    scored = score_categories(probabilities, observed, {"01": 1_000_000})
    assert list(scored.horizon) == [0, 1, 2]  # No outcome for horizon 3.
    assert list(scored.baseline_truth) == [100, 100, 100]
    assert list(scored.truth_class) == [4, 1, 4]
    np.testing.assert_allclose(scored.rps, [1.4, .6, 1.4])
    np.testing.assert_allclose(scored.brier, [1.06, .86, 1.06])
    np.testing.assert_allclose(scored.log_score, -np.log([.1, .2, .1]))
    assert list(scored.accuracy) == [0, 0, 0]
    assert list(scored.absolute_category_error) == [2, 1, 2]
    assert score_quantiles(probabilities, observed).empty


def test_categorical_skill_matches_common_keys(probabilities, observed):
    baseline = probabilities[probabilities.horizon.eq(0)].assign(model_id=TREND_BASELINE, value=.2)
    scored = score_categories(pd.concat([probabilities, baseline]), observed, {"01": 1_000_000})
    summary = summarize(scored).set_index("model_id")
    assert summary.loc["MIGHTE-Base", "n"] == 3
    assert summary.loc["MIGHTE-Base", "matched_n"] == 1
    assert summary.loc["MIGHTE-Base", "rps_skill"] == pytest.approx(1 - 1.4 / 1.2)
    assert summary.loc[TREND_BASELINE, "rps_skill"] == 0
    no_baseline = summarize(scored[scored.model_id.eq("MIGHTE-Base")]).iloc[0]
    assert no_baseline.matched_n == 0 and pd.isna(no_baseline.rps_skill)


def test_missing_or_revised_truth_and_frozen_population(probabilities, observed):
    frozen = probabilities.assign(population=1_000_000)
    original = score_categories(frozen, observed, {"01": 10_000_000})
    assert original.truth_class.iloc[0] == 4  # Use forecast's frozen population.
    assert score_categories(probabilities, observed[observed.date.ne("2026-10-03")], {"01": 1e6}).empty
    revised = observed.copy()
    revised.loc[revised.date.eq("2026-10-03"), "value"] = 120
    late = score_categories(frozen, revised, {"01": 1e6})
    assert late.truth_class.iloc[0] == 2
    assert late.rps.iloc[0] < original.rps.iloc[0]
    with pytest.raises(ValueError, match="Duplicate"):
        score_categories(pd.concat([probabilities, probabilities.iloc[:1]]), observed, {"01": 1e6})


def test_ties_and_zero_probability_outcomes_are_explicit(probabilities, observed):
    tied = score_categories(probabilities.assign(value=.2), observed, {"01": 1e6})
    assert set(tied.predicted_class) == {2}
    certain = probabilities.assign(value=probabilities.output_type_id.eq("large_increase").astype(float))
    scored = score_categories(certain, observed, {"01": 1e6})
    assert list(scored.zero_observed_probability) == [0, 1, 0]
    assert scored.log_score.iloc[1] == pytest.approx(-np.log(1e-15))
    perfect = summarize(scored[scored.horizon.eq(0)].assign(model_id=TREND_BASELINE))
    assert pd.isna(perfect.rps_skill.iloc[0])


def test_incomplete_pmf_is_not_scored_and_invalid_pmf_fails(probabilities, observed):
    missing = probabilities.drop(index=0)
    assert set(score_categories(missing, observed, {"01": 1e6}).horizon) == {1, 2}
    with pytest.raises(ValueError, match="sum to one"):
        score_categories(probabilities.assign(value=.3), observed, {"01": 1e6})
    with pytest.raises(ValueError, match="Unknown"):
        score_categories(probabilities.replace({"output_type_id": {"stable": "other"}}), observed, {"01": 1e6})


def test_mixed_target_scores_are_summarized_separately(probabilities, observed, forecast):
    continuous = score_quantiles(forecast.assign(model_id="MIGHTE-Base"), observed)
    categorical = score_categories(probabilities, observed, {"01": 1e6})
    summary = summarize(pd.concat([continuous, categorical], ignore_index=True)).set_index("target")
    assert set(summary.index) == {HOSP, TREND}
    assert pd.isna(summary.loc[TREND, "mean_wis"])
    assert pd.isna(summary.loc[HOSP, "rps"])
    assert list(summary.n) == [3, 3]
    assert not categorical.duplicated(["model_id", *UNIT]).any()


def test_categorical_only_hub_models_are_loaded_and_cached(tmp_path, monkeypatch, probabilities):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = probabilities.drop(columns="model_id").to_csv(index=False).encode()
        return response

    monkeypatch.setattr(evaluation.requests, "get", get)
    catalog = {"revision": "commit", "files": [
        {"model": name, "reference_date": "2026-10-10", "blob_sha": name}
        for name in ["Trend-Team", TREND_BASELINE]]}
    loaded, _ = evaluation.fetch_benchmarks(tmp_path, ["2026-10-10"], catalog=catalog)
    assert set(loaded.model_id) == {"Trend-Team", TREND_BASELINE}
    assert set(loaded.target) == {TREND} and len(calls) == 2
    cached, _ = evaluation.fetch_benchmarks(tmp_path, ["2026-10-10"], catalog=catalog, online=False)
    pd.testing.assert_frame_equal(loaded, cached)
    assert len(calls) == 2


@pytest.mark.parametrize("prospective", [False, True])
def test_categorical_preview_exclusion_and_same_population_for_comparisons(
        root, tmp_path, monkeypatch, probabilities, observed, prospective):
    archive = probabilities.assign(population=1e6) if prospective else pd.DataFrame()
    monkeypatch.setattr(evaluation, "load_archive", lambda root: archive)
    peer = probabilities.assign(model_id=TREND_BASELINE)
    preview_peer = peer.assign(reference_date="2026-10-17",
                               target_end_date=(pd.to_datetime(peer.target_end_date) + pd.Timedelta(weeks=1)).dt.strftime("%Y-%m-%d"))
    monkeypatch.setattr(evaluation, "fetch_benchmarks", lambda *args, **kwargs:
                        (pd.concat([peer, preview_peer] if prospective else [preview_peer]), []))
    observed.to_csv(tmp_path / "truth.csv", index=False)
    shutil.copytree(root / "hub-contract", tmp_path / "contract")
    scores, _, display = evaluation.evaluate(tmp_path, tmp_path, online=False,
                                              comparison_references=["2026-10-17"])
    assert "2026-10-17" in set(display.reference_date)
    if prospective:
        assert set(scores.reference_date) == {"2026-10-10"}
        assert set(scores.population) == {1e6}
        a = scores[scores.model_id.eq("MIGHTE-Base")].set_index(UNIT)
        b = scores[scores.model_id.eq(TREND_BASELINE)].set_index(UNIT)
        np.testing.assert_array_equal(a.truth_class, b.truth_class)
        # The report serializer supports a categorical-only accuracy table.
        assert json.loads(scores.to_json(orient="records"))[0]["rps"] is not None
    else:
        assert scores.empty
