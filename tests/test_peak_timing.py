import json
import shutil
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mighte import core, peak_height, peak_timing
from mighte.contract import ED, HOSP, MODELS, PEAK_HEIGHT, PEAK_WEEK, TREND, Contract, check_window, read_forecast
from mighte.data import NSSP, WW
from mighte.pipeline import preview_seasonal, run_forecasts, verify_run
from mighte.report import build_report
from mighte.util import digest, write_json
from test_peak_height import examples, peak_context  # noqa: F401


def test_unique_timing_labels_use_prepared_maxima_and_never_enter_features(peak_context):
    history = peak_context["hospitalization_history"].copy()
    history.loc[history.location_name.eq("US") & history.date.eq("2020-12-19"), "total_hosp"] = 50_000
    history.loc[history.location_name.eq("Alabama") & history.date.isin(["2020-12-19", "2021-01-02"]), "total_hosp"] = 500
    frame, columns = examples(dict(peak_context, hospitalization_history=history))
    national = frame[frame.location_name.eq("US") & frame.season_year.eq(2020)]
    first, last = peak_height.season_window(2020)
    assert national.season_peak_date.eq("2020-12-19").all()
    assert national.peak_fraction.eq((pd.Timestamp("2020-12-19") - first) / (last - first)).all()
    tied = frame[frame.location_name.eq("Alabama") & frame.season_year.eq(2020)]
    assert tied.peak_count.eq(500).all() and tied.peak_ties.eq(2).all()
    assert tied.peak_fraction.isna().all() and tied.season_peak_date.isna().all()
    assert not {"season_peak_date", "peak_fraction", "peak_ties", "peak_count", "label_available_date"} & set(columns)
    assert frame.loc[frame.season_year.eq(2026), "peak_fraction"].isna().all()
    labels = frame[frame.peak_fraction.notna()].drop_duplicates(["location_name", "season_year"])
    lengths = set()
    for row in labels.itertuples():
        first, last = peak_height.season_window(int(row.season_year))
        lengths.add(len(pd.date_range(first, last, freq="W-SAT")))
        assert row.peak_fraction == (row.season_peak_date - first) / (last - first)
    assert lengths == {34, 35}


def test_lognormal_week_bins_use_the_observed_anchor_and_normalize_extreme_tails():
    from scipy.stats import lognorm

    reference = pd.Timestamp("2026-12-12")
    weeks = pd.date_range(reference - pd.Timedelta(weeks=3), periods=7, freq="W-SAT")
    mu, sigma = np.log([2, 4]), np.array([.5, 1])
    actual = peak_timing.future_probabilities(mu, sigma, weeks, reference)
    boundaries = np.arange(.5, 4)
    expected = np.diff(lognorm.cdf(boundaries[None, :], s=sigma[:, None], scale=np.exp(mu[:, None])), axis=1)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(actual[:, 4:], expected)
    assert not actual[:, :4].any()
    np.testing.assert_allclose(actual, peak_timing.future_probabilities(mu, sigma, weeks + pd.Timedelta(weeks=1),
                                                                      reference + pd.Timedelta(weeks=1)))
    tails = peak_timing.future_probabilities([-1000, 1000], [.5, .5], weeks, reference)
    np.testing.assert_array_equal(tails, [np.eye(7)[4], np.eye(7)[-1]])
    assert not peak_timing.future_probabilities(mu, sigma, weeks, weeks[-1]).any()
    for mu, sigma in [([np.nan], [1]), ([0], [np.inf]), ([1], [0]), ([1], [-1]), ([1, 2], [1])]:
        with pytest.raises(ValueError, match="Invalid peak-timing"):
            peak_timing.future_probabilities(mu, sigma, weeks, reference)
    for probabilities in [[[.5, .6]], [[-1, 2]], [[np.nan, 1]], [[1]]]:
        with pytest.raises(ValueError, match="Invalid peak-timing"):
            peak_timing.validate_probabilities(probabilities, (1, 2))


def test_repeated_origins_receive_equal_total_location_season_weight():
    frame = pd.DataFrame({"location_name": ["US"] * 5 + ["Alabama"], "season_year": [2020] * 2 + [2021] * 4})
    frame["weight"] = peak_timing.season_weights(frame)
    np.testing.assert_allclose(frame.groupby(["location_name", "season_year"]).weight.sum(), 2)
    assert frame.weight.mean() == 1


def test_weighted_gaussian_scale_uses_weighted_offset_and_gradients(monkeypatch):
    x, y, mu = pd.DataFrame({"x": [1, 2]}), np.array([1., 3.]), np.zeros(2)
    captured = []

    class ConstantBooster:
        def predict(self, x):
            return np.zeros(len(x))

    def train(params, dataset, **kwargs):
        dataset.construct()
        captured.append((params, dataset))
        return ConstantBooster()

    monkeypatch.setattr(core.lgb, "train", train)
    unweighted = core.fit_conditional_gaussian_scale_booster(x, y, mu, {}, 1)
    np.testing.assert_allclose(unweighted.predict_sigma(x), np.sqrt(5))
    weights = np.array([1., 3.])
    weighted = core.fit_conditional_gaussian_scale_booster(x, y, mu, {}, 1, sample_weight=weights)
    np.testing.assert_allclose(weighted.predict_sigma(x), np.sqrt(7))
    params, dataset = captured[-1]
    np.testing.assert_array_equal(dataset.get_weight(), weights)
    grad, hess = params["objective"](np.zeros(2), dataset)
    np.testing.assert_allclose(grad, (1 - y**2 / 7) * weights)
    np.testing.assert_allclose(hess, 2 * y**2 / 7 * weights)
    core.fit_lgbm_regressor(x, y, {}, 1, sample_weight=weights)
    np.testing.assert_array_equal(captured[-1][1].get_weight(), weights)
    for bad in [[1], [0, 1], [-1, 1], [np.nan, 1]]:
        with pytest.raises(ValueError, match="Gaussian scale training"):
            core.fit_conditional_gaussian_scale_booster(x, y, mu, {}, 1, sample_weight=bad)


def test_past_gate_allocates_only_to_observed_maxima_and_converges_at_season_end():
    weeks = pd.date_range("2026-10-10", periods=5, freq="W-SAT")
    history = pd.DataFrame({"location_name": "Alabama", "date": weeks[:4], "total_hosp": [1, 5, np.nan, 5]})
    future = np.array([[0, 0, 0, 0, 1.]])
    result, q = peak_timing.combine([.7], future, history, ["Alabama"], weeks, weeks[3])
    np.testing.assert_allclose(result, [[0, .35, 0, .35, .3]])
    np.testing.assert_allclose(q, [.7])
    np.testing.assert_allclose(future, [[0, 0, 0, 0, 1.]])
    changed = pd.concat([history, pd.DataFrame({"location_name": ["Alabama"], "date": [weeks[-1]], "total_hosp": [999]})])
    np.testing.assert_allclose(result, peak_timing.combine([.7], future, changed, ["Alabama"], weeks, weeks[3])[0])
    history = pd.DataFrame({"location_name": "Alabama", "date": weeks, "total_hosp": [1, 5, 4, 3, 2]})
    result, q = peak_timing.combine([.01], np.zeros((1, 5)), history, ["Alabama"], weeks, weeks[-1])
    np.testing.assert_allclose(result, [[0, 1, 0, 0, 0]])
    np.testing.assert_allclose(q, [1])
    result, q = peak_timing.combine([.9], future, history.iloc[:0], ["Alabama"], weeks, weeks[3])
    np.testing.assert_allclose(result, future)
    np.testing.assert_allclose(q, [0])
    for bad in [[np.nan], [-.1], [1.1], [0, 1]]:
        with pytest.raises(ValueError, match="Invalid peak-timing"):
            peak_timing.combine(bad, future, history, ["Alabama"], weeks, weeks[3])


def test_real_timing_fit_uses_whole_completed_seasons_and_resumes(peak_context, monkeypatch):
    (peak_context["snapshot_path"] / "truth.csv").unlink()
    history = peak_context["hospitalization_history"].copy()
    history.loc[history.location_name.eq("Alabama") & history.date.isin(["2020-12-19", "2021-01-02"]), "total_hosp"] = 500
    peak_context["hospitalization_history"] = history
    result = peak_timing.predict(peak_context)
    contract = Contract(peak_context["snapshot_path"] / "contract")
    assert set(result.target) == {PEAK_WEEK} and len(result) == 3 * 34
    assert result.horizon.isna().all() and result.target_end_date.isna().all()
    contract.validate(result, "MIGHTE-Base", "2026-10-10")
    np.testing.assert_allclose(result.groupby("location").value.sum(), 1, atol=1e-8)
    audit = json.loads((peak_context["checkpoint_path"] / "fit.json").read_text())
    assert audit["training_label_available_max"] == "2026-05-30"
    assert audit["training_seasons"] == list(range(2016, 2026))
    assert {WW, NSSP, f"{WW}_lag4", f"{NSSP}_model_lag4"} <= set(audit["features"])
    assert not {"season_peak_date", "peak_fraction", "peak_ties"} & set(audit["features"])
    assert "candidate_week_boundary" not in audit["features"]
    assert audit["design"] == peak_timing.DESIGN
    assert audit["gate_objective"].startswith("binary log loss")
    assert audit["median_objective"] == "pinball loss at 0.5"
    for bag in audit["bag_audits"]:
        assert set(bag["seasons"]).isdisjoint(bag["omitted_seasons"])
        assert set(bag["seasons"]) | set(bag["omitted_seasons"]) == set(audit["training_seasons"])
        assert len(bag["seasons"]) == 8 and bag["training_label_available_max"] <= "2026-10-03"
        assert bag["folds"] and bag["spread_training_rows"] >= 50
        for fold in bag["folds"]:
            assert max(fold["training_seasons"]) < fold["validation_season"]
            assert fold["training_label_available_max"] < fold["validation_origin_min"]
        assert bag["conditional_rows"] + bag["gate_positive_rows"] == bag["training_origin_rows"]
        assert bag["minimum_future_lead"] >= 1
        assert all(p["log_lead_sigma"] > 0 and 0 <= p["past_peak_probability"] <= 1
                   for p in bag["forecast_parameters"])
    labels_path = peak_context["checkpoint_path"] / "training-labels.csv"
    assert audit["training_labels_sha256"] == digest(labels_path)
    labels = pd.read_csv(labels_path)
    assert len(labels) == 29 and labels.peak_fraction.between(0, 1).all()
    assert audit["tied_location_seasons_excluded"] == 1

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Cached timing forecasts must not refit")

    monkeypatch.setattr(core, "fit_lgbm_regressor", unexpected_fit)
    pd.testing.assert_frame_equal(result, peak_timing.predict(peak_context))
    np.save(peak_context["checkpoint_path"] / "bag-000.npy", np.ones((3, 34)))
    with pytest.raises(ValueError, match="Invalid peak-timing"):
        peak_timing.predict(peak_context)
    path = peak_context["checkpoint_path"] / "bag-000.json"
    old = json.loads(path.read_text())
    old.pop("design")
    write_json(path, old)
    with pytest.raises(ValueError, match="different model design"):
        peak_timing.predict(peak_context)


def test_gate_uses_all_origins_and_median_spread_use_only_positive_log_leads(peak_context, monkeypatch):
    contract = Contract(peak_context["snapshot_path"] / "contract")
    populations = dict(zip(contract.locations.location_name, contract.locations.population))
    runtime = core.RuntimeConfig.from_dict(peak_context["settings"]["runtime"])
    frame, columns = peak_height.seasonal_examples(peak_context["hospitalization_history"],
        peak_context["snapshot_path"], peak_context["reference_date"], runtime, ("ww", "nssp"),
        populations, peak_height.season_window(2026))
    expected = frame[frame.seasonal_origin & frame.season_peak_date.notna()].copy()
    expected["lead"] = (expected.season_peak_date - expected.date).dt.days / 7
    assert expected.lead.min() < 0 < expected.lead.max()
    already, lead = peak_timing.training_targets(expected)
    np.testing.assert_array_equal(already, expected.lead.le(0))
    np.testing.assert_array_equal(lead, expected.lead)
    checked = {"gate": 0, "center": 0, "spread": 0}

    class FixedGate:
        def predict(self, x):
            return np.full(len(x), .35)

    class FixedCenter:
        def predict(self, x):
            return np.full(len(x), np.log(4.))

    class FixedSpread:
        def predict_sigma(self, x):
            return np.full(len(x), .5)

    def check_response(x, y, component, sample_weight):
        matched = x.reset_index(drop=True).merge(expected[columns + ["location_name", "season_year", "lead"]],
                                                 on=columns, how="left", validate="many_to_one")
        if component == "gate":
            np.testing.assert_array_equal(y, matched.lead.le(0))
        else:
            assert matched.lead.gt(0).all()
            np.testing.assert_allclose(y, np.log(matched.lead))
        matched["weight"] = sample_weight
        totals = matched.groupby(["location_name", "season_year"]).weight.sum()
        np.testing.assert_allclose(totals, totals.iloc[0])
        checked[component] += 1

    def center(x, y, params, rounds, *, sample_weight):
        if params["objective"] == "binary":
            assert params["metric"] == "binary_logloss"
            check_response(x, y, "gate", sample_weight)
            return FixedGate()
        assert params["objective"] == "quantile" and params["alpha"] == .5
        check_response(x, y, "center", sample_weight)
        return FixedCenter()

    def spread(x, y, mu, params, rounds, *, sample_weight):
        check_response(x, y, "spread", sample_weight)
        np.testing.assert_allclose(mu, np.log(4.))
        return FixedSpread()

    monkeypatch.setattr(core, "fit_lgbm_regressor", center)
    monkeypatch.setattr(core, "fit_conditional_gaussian_scale_booster", spread)
    result = peak_timing.predict(peak_context)
    weeks = sorted(result.output_type_id.unique())
    anchor = pd.Timestamp(peak_context["reference_date"]) - pd.Timedelta(weeks=1)
    expected_pmf = peak_timing.future_probabilities([np.log(4)], [.5], weeks, anchor)[0]
    for _, group in result.groupby("location"):
        np.testing.assert_allclose(group.sort_values("output_type_id").value, expected_pmf)
    assert checked["center"] > checked["spread"] == checked["gate"] == 2


def test_future_outcomes_and_unreleased_signals_do_not_change_timing(peak_context):
    result = peak_timing.predict(peak_context)
    changed = dict(peak_context, checkpoint_path=peak_context["snapshot_path"] / "changed")
    future = peak_context["hospitalization_history"].groupby("location_name").tail(1).copy()
    future["date"] += pd.Timedelta(weeks=1)
    future["total_hosp"] = 1e8
    changed["hospitalization_history"] = pd.concat([changed["hospitalization_history"], future])
    for name in ["wastewater.csv", "nssp.csv"]:
        path = peak_context["snapshot_path"] / name
        cov = pd.read_csv(path)
        revision = cov.copy()
        revision["available_date"] = "2026-10-11"
        for column in revision.columns.difference(["date", "available_date"]):
            revision[column] = 1e8
        pd.concat([cov, revision]).to_csv(path, index=False)
    pd.testing.assert_frame_equal(result, peak_timing.predict(changed))


def test_timing_contract_requires_exact_week_support_and_base_only(root, peak_forecasts):
    contract = Contract(root / "hub-contract")
    timing = peak_forecasts[peak_forecasts.target.eq(PEAK_WEEK)]
    contract.validate(timing, "MIGHTE-Base", "2026-10-10")
    for bad in [timing.iloc[:-1], timing.assign(value=.5), timing.assign(horizon=0),
                timing.assign(target_end_date="2026-10-10"), timing.assign(output_type_id="2026-10-11"),
                timing.assign(output_type="quantile"), timing.assign(value=-.1)]:
        with pytest.raises(ValueError):
            contract.validate(bad, "MIGHTE-Base", "2026-10-10")
    with pytest.raises(ValueError, match="Incorrect targets"):
        contract.validate(timing, "MIGHTE-Linear", "2026-10-10")


def test_timing_preview_reuses_other_forecasts_and_completed_resume_needs_no_source(root, tmp_path):
    from test_workflow import synthetic_project

    settings, snapshot = synthetic_project(root, tmp_path, with_ordinal=False)
    settings["peak_height"] = True
    write_json(tmp_path / "config/settings.json", settings)
    source = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    source_manifest = verify_run(tmp_path, source)
    original = {name: read_forecast(source / name) for name in source_manifest["output_hashes"]}
    source_history = digest(source / "hospitalization-training.csv")
    preview = preview_seasonal(tmp_path, source, target=PEAK_WEEK)
    manifest = verify_run(tmp_path, preview)
    assert manifest["preview"] and manifest["seasonal_preview_target"] == PEAK_WEEK
    assert manifest["reused_run"] == source_manifest["run_id"] and not source.exists()
    assert digest(preview / "hospitalization-training.csv") == source_history
    for filename, before in original.items():
        after = read_forecast(preview / filename)
        pd.testing.assert_frame_equal(after[after.target.ne(PEAK_WEEK)].reset_index(drop=True), before, check_dtype=False)
    with pytest.raises(ValueError, match="cannot be submitted"):
        verify_run(tmp_path, preview, for_submission=True)
    assert run_forecasts(tmp_path, None, resume=preview) == preview
    write_json(tmp_path / "data/latest.json", {"snapshot_id": snapshot.name})
    report = build_report(tmp_path, preview, online=False)
    payload = json.loads((report.parent / "report-data.json").read_text())
    timing = [r for r in payload["peak_week_forecasts"] if r["model_id"] == "MIGHTE-Base"]
    assert len(timing) == 3 * 34 and payload["scores"] == []
    assert PEAK_HEIGHT in manifest["validation"]["MIGHTE-Base"]["targets"]


def test_official_base_submission_contains_conditional_timing_and_all_other_targets(root, tmp_path, monkeypatch):
    from test_workflow import synthetic_project

    from mighte import submit as submission

    settings, snapshot = synthetic_project(root, tmp_path)
    assert json.loads((root / "config/settings.json").read_text())["peak_timing"] is True
    settings.update(peak_height=True, peak_timing=True)
    settings["runtime"].update(num_bags=100, stage1_rounds=2, stage2_rounds=2)
    write_json(tmp_path / "config/settings.json", settings)
    shutil.copytree(root / "model-metadata", tmp_path / "model-metadata")
    now = datetime.fromisoformat("2026-10-07T16:00:00-04:00")
    snapshot_manifest = json.loads((snapshot / "manifest.json").read_text())
    snapshot_manifest["completed_at"] = now.isoformat()
    write_json(snapshot / "manifest.json", snapshot_manifest)
    monkeypatch.setattr("mighte.pipeline.utc_now", lambda: now.isoformat())
    monkeypatch.setattr("mighte.pipeline.check_window", lambda reference, at=None: check_window(reference, at or now))

    run = run_forecasts(tmp_path, "2026-10-10", snapshot=snapshot)
    manifest = verify_run(tmp_path, run, for_submission=True)
    base_path = "model-output/MIGHTE-Base/2026-10-10-MIGHTE-Base.csv"
    base = read_forecast(run / base_path)
    assert set(base.target) == {HOSP, ED, TREND, PEAK_HEIGHT, PEAK_WEEK}
    timing = base[base.target.eq(PEAK_WEEK)]
    assert len(timing) == 3 * 34 and timing[["horizon", "target_end_date"]].isna().all().all()
    np.testing.assert_allclose(timing.groupby("location").value.sum(), 1, atol=1e-8)
    audit = json.loads((run / "data-audit.json").read_text())["peak_timing"]["base-peak-timing"]
    assert audit["design"] == peak_timing.DESIGN and audit["bags"] == 100 and audit["signals"] == ["ww", "nssp"]
    for model in MODELS[1:]:
        assert manifest["validation"][model]["targets"] == [HOSP]

    uploads = []

    def capture_pr(project, files, **kwargs):
        uploads.append(({name: path.read_bytes() for name, path in files.items()}, kwargs))
        return "https://example.test/forecast-pr"

    monkeypatch.setattr(submission, "fresh_contract", lambda project: Contract(snapshot / "contract"))
    monkeypatch.setattr(submission.requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200, raise_for_status=lambda: None))
    monkeypatch.setattr(submission, "create_pr", capture_pr)
    assert submission.submit(tmp_path, run) == "https://example.test/forecast-pr"
    assert len(uploads) == 1
    assert uploads[0][0][base_path] == (run / base_path).read_bytes()
    assert set(uploads[0][0]) == set(manifest["output_hashes"])
    assert uploads[0][1]["body"] == "2026-10-10 forecasts from MIGHTE-Base, MIGHTE-Linear, MIGHTE-Nsemble."


@pytest.mark.parametrize("missing", ["ww", "anchor"])
def test_timing_follows_base_covariate_and_location_fallbacks(root, tmp_path, missing):
    from test_workflow import synthetic_project

    def mutate(truth, nssp, ww):
        if missing == "ww":
            ww = ww[ww.date.lt("2026-09-19")]
        else:
            truth = truth[~(truth.target.eq(HOSP) & truth.location.eq("US") & truth.date.eq("2026-10-03"))]
        return truth, nssp, ww

    settings, snapshot = synthetic_project(root, tmp_path, with_ordinal=False, mutate=mutate)
    settings["peak_timing"] = True
    write_json(tmp_path / "config/settings.json", settings)
    run = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    manifest = verify_run(tmp_path, run)
    frame = read_forecast(run / "model-output/MIGHTE-Base/2026-10-10-MIGHTE-Base.csv")
    assert set(frame.loc[frame.target.eq(PEAK_WEEK), "location"]) == {"01", "02", "US"}
    audits = json.loads((run / "data-audit.json").read_text())["peak_timing"]
    if missing == "ww":
        assert audits["base-peak-timing"]["signals"] == []
        assert any("MIGHTE-Base peak timing" in notice for notice in manifest["notices"])
    else:
        assert audits["base-peak-timing"]["signals"] == ["ww", "nssp"]
        assert audits["base-peak-timing-fallback"]["signals"] == []
    cached = run / "components/base-peak-timing.csv"
    incomplete = read_forecast(cached)
    incomplete[incomplete.location.ne("01")].to_csv(cached, index=False)
    interrupted = json.loads((run / "manifest.json").read_text())
    interrupted["status"] = "running"
    write_json(run / "manifest.json", interrupted)
    with pytest.raises(ValueError, match="Incomplete peak-timing forecast coverage"):
        run_forecasts(tmp_path, None, resume=run)
