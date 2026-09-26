import json
import shutil

import numpy as np
import pandas as pd
import pytest

from mighte import core, ordinal
from mighte.contract import UNIT
from mighte.data import NSSP, WW
from mighte.models import pooled_features


@pytest.fixture
def context(root, tmp_path, history):
    settings = json.loads((root / "tests/fixtures/settings.json").read_text())
    settings["threads"] = 2
    shutil.copytree(root / "hub-contract", tmp_path / "contract")
    locations = pd.read_csv(tmp_path / "contract/locations.csv", dtype={"location": str})
    locations["population"] = 1_000_000  # Synthetic population exercises all five classes.
    locations.to_csv(tmp_path / "contract/locations.csv", index=False)
    dates = pd.date_range("2016-01-02", "2026-10-03", freq="W-SAT")
    weeks = np.arange(len(dates))
    ww = pd.DataFrame({"date": dates, WW: np.round(np.sin(weeks / 5) - 3, 6),
                       "available_date": dates + pd.Timedelta(weeks=2)})
    for lag in [1, 2, 4]:
        ww[f"{WW}_lag{lag}"] = ww[WW].shift(lag)
    ww.to_csv(tmp_path / "wastewater.csv", index=False)
    pd.DataFrame({"date": dates, NSSP: np.round(2 + np.sin(weeks / 9), 6),
                  "available_date": dates + pd.Timedelta(weeks=1)}).to_csv(tmp_path / "nssp.csv", index=False)
    return {"reference_date": "2026-10-10", "settings": settings, "snapshot_path": tmp_path,
            "hospitalization_history": history, "checkpoint_path": tmp_path / "bags"}


@pytest.mark.parametrize("horizon,small,large", [(0, 30, 170), (1, 50, 300), (2, 70, 400), (3, 100, 500)])
def test_cdc_exact_category_boundaries(horizon, small, large):
    delta = np.array([-large, -large + .01, -small, -small + .01, 0, small - .01, small, large - .01, large])
    np.testing.assert_array_equal(ordinal.category_labels(1000 + delta, 1000, 10_000_000, horizon),
                                  [0, 1, 1, 2, 2, 2, 3, 3, 4])
    # A small population still requires at least ten admissions to leave stable.
    np.testing.assert_array_equal(ordinal.category_labels([90, 90.01, 109.99, 110], 100, 1000, horizon),
                                  [0, 2, 2, 4])


@pytest.mark.parametrize("arguments", [(10, 1, 0, 0), (10, 1, np.nan, 0), (10, 1, 1e6, 4),
                                        (-1, 1, 1e6, 0), (1, np.nan, 1e6, 0)])
def test_invalid_label_inputs_fail(arguments):
    with pytest.raises(ValueError):
        ordinal.category_labels(*arguments)


def test_classifier_matches_independent_study_fixture_and_resumes(root, context, monkeypatch):
    result = ordinal.predict(context)
    expected = pd.read_csv(root / "tests/fixtures/ordinal-golden.csv", dtype={"location": str})
    keys = UNIT + ["output_type", "output_type_id"]
    pd.testing.assert_frame_equal(result[keys].reset_index(drop=True), expected[keys], check_dtype=False)
    np.testing.assert_allclose(result.value, expected.value, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(result.groupby(UNIT).value.sum(), 1, rtol=0, atol=1e-8)
    audit = json.loads((context["checkpoint_path"] / "fit.json").read_text())
    assert all(audit["training_class_counts"].values())
    assert audit["training_target_max"] == "2026-10-03"
    runtime = core.RuntimeConfig.from_dict(context["settings"]["runtime"])
    _, base_columns = pooled_features(context["hospitalization_history"], context["snapshot_path"],
                                      context["reference_date"], runtime, ("ww", "nssp"))
    assert audit["features"] == base_columns
    assert {WW, NSSP, *[f"{WW}_lag{lag}" for lag in [1, 2, 4]],
            *[f"{NSSP}_model_lag{lag}" for lag in [1, 2, 4]]} <= set(base_columns)
    assert not {"target", "target_date", "total_hosp"} & set(base_columns)

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Checkpoint resume must not refit")

    monkeypatch.setattr(ordinal.lgb, "train", unexpected_fit)
    pd.testing.assert_frame_equal(result, ordinal.predict(context))
    np.save(context["checkpoint_path"] / "bag-000.npy", np.full((1, 5), .2))
    with pytest.raises(ValueError, match="incompatible forecast support"):
        ordinal.predict(context)


def test_future_outcomes_and_unreleased_covariates_cannot_change_forecast(context):
    result = ordinal.predict(context)
    altered = dict(context)
    altered["checkpoint_path"] = context["snapshot_path"] / "altered-bags"
    future = context["hospitalization_history"].groupby("location_name").tail(1).copy()
    future["date"] += pd.Timedelta(weeks=1)
    future["total_hosp"] = 1e8
    altered["hospitalization_history"] = pd.concat([context["hospitalization_history"], future])
    for file in ["wastewater.csv", "nssp.csv"]:
        path = context["snapshot_path"] / file
        cov = pd.read_csv(path)
        changed = cov.copy()
        changed["available_date"] = "2026-10-11"
        for column in changed.columns.difference(["date", "available_date"]):
            changed[column] = 1e8
        pd.concat([cov, changed]).to_csv(path, index=False)
    pd.testing.assert_frame_equal(result, ordinal.predict(altered))


@pytest.mark.parametrize("p", [[[.2] * 4], [[np.nan, .2, .2, .2, .2]],
                                [[-.1, .3, .3, .3, .2]], [[.1] * 5]])
def test_invalid_probabilities_fail(p):
    with pytest.raises(ValueError):
        ordinal.validate_probabilities(p)
