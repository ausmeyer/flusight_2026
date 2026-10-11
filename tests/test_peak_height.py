import json
import shutil
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mighte import core, peak_height
from mighte.contract import COLUMNS, ED, HOSP, MODELS, PEAK_HEIGHT, PEAK_WEEK, QUANTILES, TREND, Contract, check_window, read_forecast
from mighte.data import NSSP, WW
from mighte.pipeline import preview_seasonal, run_forecasts, verify_run
from mighte.report import build_report
from mighte.util import digest, write_json


@pytest.fixture
def peak_context(root, tmp_path, history):
    settings = json.loads((root / "tests/fixtures/settings.json").read_text())
    settings["threads"] = 2
    shutil.copytree(root / "hub-contract", tmp_path / "contract")
    names = {"Alabama": "01", "Alaska": "02", "US": "US"}
    truth = history.rename(columns={"total_hosp": "value"}).assign(target=HOSP)
    truth["location"] = truth.location_name.map(names)
    truth.to_csv(tmp_path / "truth.csv", index=False)
    dates = pd.date_range(history.date.min(), history.date.max(), freq="W-SAT")
    ww = pd.DataFrame({"date": dates, WW: np.sin(np.arange(len(dates)) / 5),
                       "available_date": dates + pd.Timedelta(weeks=2)})
    for lag in [1, 2, 4]:
        ww[f"{WW}_lag{lag}"] = ww[WW].shift(lag)
    ww.to_csv(tmp_path / "wastewater.csv", index=False)
    pd.DataFrame({"date": dates, NSSP: 1 + np.sin(np.arange(len(dates)) / 9),
                  "available_date": dates + pd.Timedelta(weeks=1)}).to_csv(tmp_path / "nssp.csv", index=False)
    return {"reference_date": "2026-10-10", "settings": settings, "snapshot_path": tmp_path,
            "hospitalization_history": history, "checkpoint_path": tmp_path / "bags"}


def examples(context):
    runtime = core.RuntimeConfig.from_dict(context["settings"]["runtime"])
    return peak_height.seasonal_examples(context["hospitalization_history"],
        context["snapshot_path"], context["reference_date"], runtime, ("ww", "nssp"),
        {"Alabama": 1e6, "Alaska": 1e6, "US": 1e7}, peak_height.season_window(2026))


def test_labels_require_complete_prepared_seasons_and_features_use_only_the_prefix(peak_context):
    frame, columns = examples(peak_context)
    train = frame[frame.seasonal_origin & frame.peak_count.notna()]
    assert train.season_year.max() == 2025
    assert frame.loc[frame.season_year.eq(2026), "peak_count"].isna().all()
    assert frame.loc[frame.date.eq("2026-10-03"), "peak_so_far"].eq(0).all()
    assert {WW, NSSP, *[f"{WW}_lag{lag}" for lag in [1, 2, 4]],
            *[f"{NSSP}_model_lag{lag}" for lag in [1, 2, 4]]} <= set(columns)
    assert not {"peak_count", "label_available_date", "season_year", "target", "target_date", "horizon_weeks"} & set(columns)
    history = peak_context["hospitalization_history"].copy()
    history.loc[history.location_name.eq("Alabama") & history.date.eq("2025-12-06"), "total_hosp"] = np.nan
    changed, _ = examples(dict(peak_context, hospitalization_history=history))
    assert changed.loc[changed.location_name.eq("Alabama") & changed.season_year.eq(2025), "peak_count"].isna().all()
    assert changed.loc[changed.location_name.eq("Alaska") & changed.season_year.eq(2025), "peak_count"].notna().all()
    missing_week = changed[changed.location_name.eq("Alabama") & changed.date.eq("2025-12-06")].iloc[0]
    assert missing_week.peak_so_far > 0


def test_labels_and_peak_so_far_use_reconstructed_history_instead_of_raw_pandemic_truth(peak_context):
    history = peak_context["hospitalization_history"].copy()
    history.loc[history.location_name.eq("US") & history.date.eq("2020-12-19"), "total_hosp"] = 43_019
    context = dict(peak_context, hospitalization_history=history)
    truth_path = context["snapshot_path"] / "truth.csv"
    truth = pd.read_csv(truth_path)
    truth.loc[truth.location.eq("US") & truth.date.between("2020-10-01", "2021-06-01"), "value"] = 765
    truth = truth[truth.date.ge("2020-01-01")]
    truth.to_csv(truth_path, index=False)
    frame, _ = examples(context)
    national = frame[frame.location_name.eq("US")]
    assert national[national.season_year.eq(2020)].peak_count.eq(43_019).all()
    assert national[national.date.eq("2020-10-03")].peak_so_far.eq(0).all()
    assert national[national.date.eq("2021-01-02")].peak_so_far.eq(43_019).all()
    assert national[national.season_year.eq(2016)].peak_count.notna().all()
    for (location, year), rows in frame[frame.peak_count.notna()].groupby(["location_name", "season_year"]):
        start, end = peak_height.season_window(int(year))
        values = history.loc[history.location_name.eq(location) & history.date.between(start, end), "total_hosp"]
        assert rows.peak_count.eq(values.max()).all()


def test_real_peak_fit_uses_disjoint_forward_seasons_and_resumes(peak_context, monkeypatch):
    (peak_context["snapshot_path"] / "truth.csv").unlink()
    result = peak_height.predict(peak_context)
    assert set(result.columns) == set(COLUMNS) and set(result.target) == {PEAK_HEIGHT}
    assert len(result) == 3 * 23
    assert result.horizon.isna().all() and result.target_end_date.isna().all()
    Contract(peak_context["snapshot_path"] / "contract").validate(result, "MIGHTE-Base", "2026-10-10")
    audit = json.loads((peak_context["checkpoint_path"] / "fit.json").read_text())
    assert audit["training_label_available_max"] == "2026-05-30"
    assert audit["training_seasons"] == list(range(2016, 2026))
    labels_path = peak_context["checkpoint_path"] / "training-labels.csv"
    assert audit["training_labels_sha256"] == digest(labels_path)
    labels = pd.read_csv(labels_path)
    assert len(labels) == 3 * 10
    for bag in audit["bag_audits"]:
        assert bag["folds"]
        for fold in bag["folds"]:
            assert max(fold["training_seasons"]) < fold["validation_season"]
            assert fold["training_label_available_max"] < fold["validation_origin_min"]

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Cached peak forecasts must not refit")

    monkeypatch.setattr(core, "fit_lgbm_regressor", unexpected_fit)
    pd.testing.assert_frame_equal(result, peak_height.predict(peak_context))
    np.save(peak_context["checkpoint_path"] / "bag-000.npy", np.ones((1, 23)))
    with pytest.raises(ValueError, match="Invalid peak-height"):
        peak_height.predict(peak_context)


def test_future_outcomes_and_unreleased_signals_do_not_change_peak_predictions(peak_context):
    result = peak_height.predict(peak_context)
    changed = dict(peak_context, checkpoint_path=peak_context["snapshot_path"] / "changed")
    future = peak_context["hospitalization_history"].groupby("location_name").tail(1).copy()
    future["date"] += pd.Timedelta(weeks=1)
    future["total_hosp"] = 1e8
    changed["hospitalization_history"] = pd.concat([changed["hospitalization_history"], future])
    path = peak_context["snapshot_path"] / "truth.csv"
    truth = pd.read_csv(path)
    future_truth = truth.groupby("location").tail(1).copy()
    future_truth["date"], future_truth["value"] = "2027-01-02", 1e8
    pd.concat([truth, future_truth]).to_csv(path, index=False)
    for name in ["wastewater.csv", "nssp.csv"]:
        path = peak_context["snapshot_path"] / name
        cov = pd.read_csv(path)
        revision = cov.copy()
        revision["available_date"] = "2026-10-11"
        for column in revision.columns.difference(["date", "available_date"]):
            revision[column] = 1e8
        pd.concat([cov, revision]).to_csv(path, index=False)
    pd.testing.assert_frame_equal(result, peak_height.predict(changed))


def test_peak_center_and_spread_fit_rates_and_quantiles_return_to_count_scale(peak_context, monkeypatch):
    contract = Contract(peak_context["snapshot_path"] / "contract")
    populations = dict(zip(contract.locations.location_name, contract.locations.population))
    runtime = core.RuntimeConfig.from_dict(peak_context["settings"]["runtime"])
    frame, columns = peak_height.seasonal_examples(peak_context["hospitalization_history"],
        peak_context["snapshot_path"], peak_context["reference_date"], runtime, ("ww", "nssp"),
        populations, peak_height.season_window(2026))
    expected = frame[frame.seasonal_origin & frame.peak_count.notna()].copy()
    expected["rate"] = (expected.peak_count + 1) * 100_000 / expected.location_name.map(populations)
    checked = {"center": 0, "spread": 0}

    def check_response(x, y, component):
        matched = x.reset_index(drop=True).merge(expected[columns + ["rate"]], on=columns,
                                                 how="left", validate="many_to_one")
        np.testing.assert_allclose(y, matched.rate)
        checked[component] += 1

    class FixedCenter:
        def predict(self, x):
            return np.full(len(x), 10.0)

    class FixedSpread:
        def predict_sigma(self, x):
            return np.ones(len(x))

    def center(x, y, params, rounds):
        check_response(x, y, "center")
        return FixedCenter()

    def spread(x, y, mu, params, rounds):
        check_response(x, y, "spread")
        np.testing.assert_allclose(mu, 10.0)
        return FixedSpread()

    monkeypatch.setattr(core, "fit_lgbm_regressor", center)
    monkeypatch.setattr(core, "fit_conditional_gaussian_scale_booster", spread)
    result = peak_height.predict(peak_context)
    for row in contract.locations.itertuples(index=False):
        actual = result[result.location.eq(row.location)].sort_values("output_type_id")
        if actual.empty:
            continue
        count_quantiles = np.rint((10 + peak_height.norm.ppf(QUANTILES)) * row.population / 100_000 - 1)
        np.testing.assert_array_equal(actual.value, count_quantiles)
    assert checked["center"] > checked["spread"] > 0


def test_peak_predictions_cannot_fall_below_the_reported_seasonal_maximum(peak_context):
    context = dict(peak_context, reference_date="2026-10-17")
    history = context["hospitalization_history"]
    latest = history.groupby("location_name").tail(1).copy()
    latest["date"] += pd.Timedelta(weeks=1)
    latest["total_hosp"] = 100_000
    context["hospitalization_history"] = pd.concat([history, latest])
    path = context["snapshot_path"] / "truth.csv"
    truth = pd.read_csv(path)
    latest = truth.groupby("location").tail(1).assign(date="2026-10-10", value=100_000)
    pd.concat([truth, latest]).to_csv(path, index=False)
    assert peak_height.predict(context).value.ge(100_000).all()


def test_seasonal_contract_rejects_dates_wrong_models_and_invalid_quantiles(root, forecast, peak_forecasts):
    height = peak_forecasts[peak_forecasts.target.eq(PEAK_HEIGHT)]
    contract = Contract(root / "hub-contract")
    contract.validate(pd.concat([forecast, height]), "MIGHTE-Base", "2026-10-10")
    for bad in [height.assign(horizon=0), height.assign(target_end_date="2027-05-29"),
                height.assign(value=-1), height.assign(value=1.5), height.iloc[:-1], height.assign(target=PEAK_WEEK)]:
        with pytest.raises(ValueError):
            contract.validate(pd.concat([forecast, bad]), "MIGHTE-Base", "2026-10-10")
    with pytest.raises(ValueError, match="Incorrect targets"):
        contract.validate(pd.concat([forecast, height]), "MIGHTE-Linear", "2026-10-10")


def test_peak_pipeline_and_reused_preview_preserve_frozen_submissions(root, tmp_path):
    from test_workflow import synthetic_project

    settings, snapshot = synthetic_project(root, tmp_path, with_ordinal=False)
    first = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    source = tmp_path / "runs/official_submissions/2026-10-10/frozen"
    source.parent.mkdir(parents=True)
    shutil.move(first, source)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.update(run_id=str(source.relative_to(tmp_path / "runs")), preview=False, quick=False)
    write_json(source / "manifest.json", manifest)
    write_json(source / "submission.json", {"url": "already submitted"})
    before = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    settings["peak_height"] = True
    write_json(tmp_path / "config/settings.json", settings)
    preview = preview_seasonal(tmp_path, source)
    result = verify_run(tmp_path, preview)
    assert result["preview"] and result["reused_run"] == manifest["run_id"]
    assert before == {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="cannot be submitted"):
        verify_run(tmp_path, preview, for_submission=True)
    base_path = "model-output/MIGHTE-Base/2026-10-10-MIGHTE-Base.csv"
    base = read_forecast(preview / base_path)
    pd.testing.assert_frame_equal(base[base.target.ne(PEAK_HEIGHT)].reset_index(drop=True),
                                  read_forecast(source / base_path), check_dtype=False)
    assert len(base[base.target.eq(PEAK_HEIGHT)]) == 3 * len(QUANTILES)
    for model in ["MIGHTE-Linear", "MIGHTE-Nsemble"]:
        filename = f"model-output/{model}/2026-10-10-{model}.csv"
        assert digest(preview / filename) == digest(source / filename)
    assert preview_seasonal(tmp_path, source, resume=preview) == preview
    write_json(tmp_path / "data/latest.json", {"snapshot_id": snapshot.name})
    report = build_report(tmp_path, preview, online=False)
    payload = json.loads((report.parent / "report-data.json").read_text())
    assert len([r for r in payload["peak_height_forecasts"] if r["model_id"] == "MIGHTE-Base"]) == 3
    assert payload["scores"] == []
    normal = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    assert PEAK_HEIGHT in verify_run(tmp_path, normal)["validation"]["MIGHTE-Base"]["targets"]


def test_official_pipeline_submits_peak_height_with_other_base_targets(root, tmp_path, monkeypatch):
    """Exercise the production file and submission assembly without contacting GitHub."""
    from test_workflow import synthetic_project

    from mighte import submit as submission

    settings, snapshot = synthetic_project(root, tmp_path)
    assert json.loads((root / "config/settings.json").read_text())["peak_height"] is True
    settings["peak_height"] = True
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
    assert set(base.target) == {HOSP, ED, TREND, PEAK_HEIGHT}
    peak = base[base.target.eq(PEAK_HEIGHT)]
    assert len(peak) == 3 * len(QUANTILES)
    assert peak[["horizon", "target_end_date"]].isna().all().all()
    assert set(peak.location) == {"01", "02", "US"}
    audit = json.loads((run / "data-audit.json").read_text())["peak_height"]["base-peak-height"]
    assert audit["bags"] == 100 and audit["signals"] == ["ww", "nssp"]
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
    files, text = uploads[0]
    assert set(files) == set(manifest["output_hashes"])
    assert files[base_path] == (run / base_path).read_bytes()
    assert text["body"] == "2026-10-10 forecasts from MIGHTE-Base, MIGHTE-Linear, MIGHTE-Nsemble."
    receipt = json.loads((run / "submission.json").read_text())
    assert receipt["output_hashes"] == manifest["output_hashes"]
