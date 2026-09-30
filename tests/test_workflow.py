import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mighte.contract import ED, HOSP, TREND, UNIT, read_forecast
from mighte.data import NSSP, WW
from mighte.evaluate import load_archive, official_runs
from mighte.pipeline import run_forecasts, verify_run
from mighte.report import build_report
from mighte.submit import create_pr
from mighte.util import digest, write_json


ANCHOR = pd.Timestamp("2026-10-03")


def synthetic_project(root, tmp_path, *, with_ordinal=True, mutate=None):
    """A three-location project and sealed snapshot; mutate(truth, nssp, ww) edits inputs first."""
    for directory in ["config", "hub-contract", "data/historical"]:
        shutil.copytree(root / directory, tmp_path / directory)
    settings = json.loads((tmp_path / "config/settings.json").read_text())
    settings["runtime"].update(num_bags=1, stage1_rounds=10, stage2_rounds=10, min_train_rows=100)
    settings["panel_ar_runtime"]["min_train_rows"] = 100
    settings["threads"] = 2
    settings["component_workers"] = 1
    settings["ordinal_plugin"] = "mighte.ordinal:predict" if with_ordinal else None
    write_json(tmp_path / "config/settings.json", settings)
    snapshot = tmp_path / "data/snapshots/test-snapshot"
    shutil.copytree(root / "hub-contract", snapshot / "contract")
    tasks = json.loads((snapshot / "contract/tasks.json").read_text())
    for r in tasks["rounds"]:
        for task in r["model_tasks"]:
            task["task_ids"]["location"]["optional"] = ["01", "02", "US"]
    write_json(snapshot / "contract/tasks.json", tasks)
    dates = pd.date_range("2022-06-04", "2026-10-03", freq="W-SAT")
    w = np.arange(len(dates))
    truth = pd.concat([pd.DataFrame({"date": dates, "location": loc, "target": target,
                                      "value": factor * (2 + np.sin(w / 8))})
                       for loc in ["01", "02", "US"] for target, factor in [(HOSP, 100), (ED, .005)]])
    nssp = pd.DataFrame({"date": dates, NSSP: 1 + .5 * np.sin(w / 8), "available_date": dates + pd.Timedelta(weeks=1)})
    ww = pd.DataFrame({"date": dates, WW: np.sin(w / 8) - 3, "available_date": dates + pd.Timedelta(weeks=2)})
    for lag in [1, 2, 4]:
        ww[f"{WW}_lag{lag}"] = ww[WW].shift(lag)
    if mutate:
        truth, nssp, ww = mutate(truth.reset_index(drop=True), nssp, ww)
    truth.to_csv(snapshot / "truth.csv", index=False)
    nssp.to_csv(snapshot / "nssp.csv", index=False)
    ww.to_csv(snapshot / "wastewater.csv", index=False)
    write_json(snapshot / "manifest.json", {"files": {str(p.relative_to(snapshot)): digest(p)
                for p in snapshot.rglob("*") if p.is_file()}})
    return settings, snapshot


@pytest.mark.parametrize("with_ordinal", [False, True])
def test_whole_offline_preview_and_submission_guard(root, tmp_path, monkeypatch, with_ordinal):
    """Exercise orchestration, serializers and guards in a separate project directory."""
    settings, snapshot = synthetic_project(root, tmp_path, with_ordinal=with_ordinal)
    run = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    serial = {p.relative_to(run): p.read_bytes() for p in (run / "model-output").rglob("*.csv")}
    settings["component_workers"] = 2
    write_json(tmp_path / "config/settings.json", settings)
    parallel_run = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    # The rerun for the same reference date replaced the first run, with identical outputs.
    assert not run.exists() and [p.name for p in run.parent.iterdir()] == [parallel_run.name]
    assert serial == {p.relative_to(parallel_run): p.read_bytes() for p in (parallel_run / "model-output").rglob("*.csv")}
    run = parallel_run
    manifest = verify_run(tmp_path, run)
    assert manifest["validation"]["MIGHTE-Base"]["rows"] == 3 * 2 * 4 * 23 + (3 * 4 * 5 if with_ordinal else 0)
    assert len(manifest["output_hashes"]) == 3 and manifest["notices"] == []
    # ED training history: ILINet proxy through June 2021, observed from 2022, and an unfilled gap between.
    ed = pd.read_csv(run / "ed-training.csv", parse_dates=["date"])
    alabama = ed[ed.location_name.eq("Alabama")].set_index("date").total_hosp
    assert alabama.loc[:"2021-06-26"].notna().all() and alabama.loc["2021-07-03":"2022-05-28"].isna().all()
    assert alabama.loc["2022-06-04":].notna().all()
    for model in ["MIGHTE-Linear", "MIGHTE-Nsemble"]:
        assert manifest["validation"][model]["targets"] == [HOSP]
    if with_ordinal:
        assert TREND in manifest["validation"]["MIGHTE-Base"]["targets"]
        probabilities = read_forecast(run / "components/base-ordinal.csv")
        np.testing.assert_allclose(probabilities.groupby(UNIT).value.sum(), 1, rtol=0, atol=1e-8)
        assert not np.allclose(probabilities.value, .2)  # Real classifier, not a plugin stub.
    with pytest.raises(ValueError, match="cannot be submitted"):
        verify_run(tmp_path, run, for_submission=True)
    assert load_archive(tmp_path).empty
    write_json(tmp_path / "data/latest.json", {"snapshot_id": snapshot.name})
    report = build_report(tmp_path, run, online=False)
    payload = json.loads((report.parent / "report-data.json").read_text())
    assert payload["scores"] == []  # Report generation never scores a preview.
    assert len(payload["categorical_forecasts"]) == (12 if with_ordinal else 0)
    if with_ordinal:
        for unit in payload["categorical_forecasts"]:
            assert sum(unit[c] for c in ["large_decrease", "decrease", "stable", "increase", "large_increase"]) == pytest.approx(1)
        # An interrupted run with an incomplete cached categorical component must fail.
        cached = parallel_run / "components/base-ordinal.csv"
        incomplete = read_forecast(cached)
        incomplete[incomplete.location.ne("02") | incomplete.horizon.ne(0)].to_csv(cached, index=False)
        interrupted = json.loads((parallel_run / "manifest.json").read_text())
        complete = dict(interrupted)
        interrupted["status"] = "running"
        write_json(parallel_run / "manifest.json", interrupted)
        with pytest.raises(ValueError, match="Incomplete forecast coverage.*rate change"):
            run_forecasts(tmp_path, "2026-10-10", resume=parallel_run)
        write_json(parallel_run / "manifest.json", complete)  # restore for the edit check below
    # Verify mutations to the actual exported artifact are caught.
    path = run / next(iter(manifest["output_hashes"]))
    path.write_text(path.read_text().replace(",quantile,", ",changed,", 1))
    with pytest.raises(ValueError, match="edited after validation"):
        verify_run(tmp_path, run)


def locations(run, model, target):
    frame = read_forecast(run / f"model-output/{model}/2026-10-10-{model}.csv")
    return {h: set(g.location) for h, g in frame[frame.target.eq(target)].groupby("horizon")}


SCENARIOS = ["location", "nssp_stale", "nssp_missing", "nssp_blank", "ww_missing", "plausibility"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_missing_inputs_use_the_fallback_without_omitting_forecasts(root, tmp_path, scenario):
    week = pd.Timedelta(weeks=1)

    def mutate(truth, nssp, ww):
        hosp_anchor = truth.target.eq(HOSP) & truth.date.eq(ANCHOR)
        if scenario == "location":
            truth = truth[~(hosp_anchor & truth.location.eq("US"))]
        if scenario == "nssp_stale":
            nssp = nssp[nssp.date.lt(ANCHOR)]
        if scenario == "nssp_missing":
            nssp = nssp[nssp.date.lt(ANCHOR - week)]
        if scenario == "nssp_blank":  # the newest row exists but is empty, so the model would receive NaN
            nssp.loc[nssp.date.eq(ANCHOR), NSSP] = np.nan
        if scenario == "ww_missing":
            ww = ww[ww.date.lt(ANCHOR - 2 * week)]
        if scenario == "plausibility":
            alaska = truth.target.eq(ED) & truth.location.eq("02")
            truth.loc[alaska, "value"] = .28 + .01 * np.sin(np.arange(alaska.sum()) / 8)
        return truth, nssp, ww

    _, snapshot = synthetic_project(root, tmp_path, mutate=mutate)
    run = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    manifest = verify_run(tmp_path, run)
    notices = " | ".join(manifest["notices"])
    everywhere = {h: {"01", "02", "US"} for h in range(4)}
    # Nothing is omitted: every model covers every location, target and horizon.
    assert manifest["models"] == ["MIGHTE-Base", "MIGHTE-Linear", "MIGHTE-Nsemble"]
    for model in manifest["models"]:
        assert locations(run, model, HOSP) == everywhere
    for target in [ED, TREND]:
        assert locations(run, "MIGHTE-Base", target) == everywhere
    component = lambda name: read_forecast(run / f"components/{name}.csv")
    base = read_forecast(run / "model-output/MIGHTE-Base/2026-10-10-MIGHTE-Base.csv")
    ordered = lambda frame: frame.sort_values(UNIT + ["output_type_id"]).value.to_numpy()
    if scenario == "location":
        fallback = component("fallback-hospitalizations")
        for model in manifest["models"]:
            frame = read_forecast(run / f"model-output/{model}/2026-10-10-{model}.csv")
            us = frame[frame.target.eq(HOSP) & frame.location.eq("US")]
            np.testing.assert_allclose(ordered(us), np.rint(ordered(fallback[fallback.location.eq("US")])))
        assert "no 2026-10-03 value for US (last reported 2026-09-26)" in notices
        assert set(component("base-ordinal-fallback").location) == {"US"}
        assert set(component("base-ordinal").location) == {"01", "02"}
    if scenario == "nssp_stale":
        assert "used 2026-09-26" in notices and "without wastewater" not in notices
    if scenario == "nssp_blank":
        assert "National NSSP unavailable" in notices and not (run / "components/base-hospitalizations.csv").exists()
    if scenario == "nssp_missing":
        assert ("National NSSP unavailable: MIGHTE-Base admissions, the Nsemble NSSP part, MIGHTE-Base trends "
                "used MIGHTE-Base without wastewater and NSSP") in notices
        np.testing.assert_allclose(ordered(base[base.target.eq(HOSP)]), np.rint(ordered(component("fallback-hospitalizations"))))
        assert not (run / "components/nssp-hospitalizations.csv").exists()
    if scenario == "ww_missing":
        assert "WastewaterSCAN unavailable" in notices and "MIGHTE-Base ED visits" in notices
        assert (run / "components/fallback-ed.csv").exists() and not (run / "components/base-ed.csv").exists()
    if scenario == "plausibility":
        assert base[base.target.eq(ED)].value.max() > .25  # flagged for review, never altered
        assert "MIGHTE-Base ED visits: Alaska h0,1,2,3 above the hub's plausibility limit" in notices


@pytest.mark.parametrize("target", [HOSP, ED])
def test_an_unreleased_week_stops_the_run_instead_of_carrying_forward(root, tmp_path, target):
    def mutate(truth, nssp, ww):  # no location has the anchor week yet
        return truth[~(truth.target.eq(target) & truth.date.eq(ANCHOR))], nssp, ww

    _, snapshot = synthetic_project(root, tmp_path, with_ordinal=False, mutate=mutate)
    with pytest.raises(ValueError, match="data for the week ending 2026-10-03 are not released yet"):
        run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)


def test_preview_without_a_date_anchors_to_the_most_recent_released_week(root, tmp_path):
    def mutate(truth, nssp, ww):  # ED lags hospitalizations by a week
        return truth[~(truth.target.eq(ED) & truth.date.eq(ANCHOR))], nssp, ww

    _, snapshot = synthetic_project(root, tmp_path, with_ordinal=False, mutate=mutate)
    run = run_forecasts(tmp_path, None, preview=True, snapshot=snapshot)
    manifest = verify_run(tmp_path, run)
    assert manifest["reference_date"] == "2026-10-03" and run.parent.name == "2026-10-03"
    assert not any("no 2026-09-26 value" in notice for notice in manifest["notices"])  # nothing carried forward


def test_run_without_any_producible_forecast_fails_loudly(root, tmp_path):
    def mutate(truth, nssp, ww):
        return truth[truth.target.ne(HOSP)], nssp, ww

    _, snapshot = synthetic_project(root, tmp_path, with_ordinal=False, mutate=mutate)
    with pytest.raises(ValueError, match="No forecasts could be produced"):
        run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)


def test_select_submitted_run_without_double_counting(tmp_path):
    base = tmp_path / "runs/official_submissions/2026-10-10"
    for name, completed in [("first", "2026-10-07T15:00:00-04:00"),
                             ("later", "2026-10-07T16:00:00-04:00"),
                             ("late", "2026-10-08T10:00:00-04:00")]:
        write_json(base / name / "manifest.json", {"preview": False, "status": "complete",
                     "reference_date": "2026-10-10", "created_at": completed, "completed_at": completed})
    assert official_runs(tmp_path) == [base / "later"]
    write_json(base / "first/submission.json", {"url": "test", "submitted_at": "2026-10-07T15:30:00-04:00"})
    assert official_runs(tmp_path) == [base / "first"]


def test_submission_upload_allowlist_rejects_other_files(tmp_path):
    path = tmp_path / "private.txt"
    path.write_text("do not upload")
    with pytest.raises(ValueError, match="allowlist"):
        create_pr(tmp_path, {"private.txt": path}, branch="codex/test", title="test", body="test")


def test_snapshot_freshness_uses_eastern_calendar(root, tmp_path, monkeypatch):
    for directory in ["config", "hub-contract"]:
        shutil.copytree(root / directory, tmp_path / directory)
    snapshot = tmp_path / "data/snapshots/tuesday-evening"
    write_json(snapshot / "manifest.json", {"files": {}, "completed_at": "2026-10-07T01:00:00+00:00"})
    monkeypatch.setattr("mighte.pipeline.check_window", lambda *args: None)
    with pytest.raises(ValueError, match="fresh Wednesday"):
        run_forecasts(tmp_path, "2026-10-10", snapshot=snapshot)


def test_deadline_rechecked_after_github_preparation(tmp_path, monkeypatch):
    from mighte import submit as submission

    checks, requests = [], []

    def window(reference):
        checks.append(reference)
        if len(checks) == 2:
            raise ValueError("Submission window closed during upload")

    def github(arguments, payload=None):
        endpoint = arguments[1]
        requests.append((endpoint, payload))
        if endpoint == "user":
            return {"login": "test-user"}
        if endpoint == "repos/test-user/FluSight-forecast-hub":
            return {"fork": True, "parent": {"full_name": submission.UPSTREAM}}
        if "pulls?" in endpoint:
            return []
        if endpoint.endswith("git/ref/heads/main"):
            return {"object": {"sha": "upstream-head"}}
        if endpoint.endswith("git/commits/upstream-head"):
            return {"tree": {"sha": "upstream-tree"}}
        return {"sha": "new-object"}

    monkeypatch.setattr(submission, "check_window", window)
    monkeypatch.setattr(submission, "gh_json", github)
    path = tmp_path / "forecast.csv"
    path.write_text("validated forecast bytes")
    with pytest.raises(ValueError, match="closed during upload"):
        create_pr(tmp_path, {"model-output/MIGHTE-Base/2026-10-10-MIGHTE-Base.csv": path},
                  branch="codex/test", title="test", body="test", reference="2026-10-10")
    assert checks == ["2026-10-10", "2026-10-10"]
    assert not any(endpoint.endswith("/pulls") for endpoint, _ in requests)


def test_open_retry_pr_is_updated_in_place(tmp_path, monkeypatch):
    import base64

    from mighte import submit as submission

    requests = []
    unchanged, changed = tmp_path / "same.yml", tmp_path / "new.yml"
    unchanged.write_text("same bytes")
    changed.write_text("new bytes")
    retry = "codex/mighte-2026-27-metadata-20260926T005603392477"
    open_pr = {"number": 3710, "html_url": "https://github.com/cdcepi/FluSight-forecast-hub/pull/3710",
               "title": "old title", "body": "old body", "user": {"login": "test-user"},
               "head": {"ref": retry, "sha": "pr-head"}}

    def github(arguments, payload=None):
        endpoint = arguments[1]
        requests.append((endpoint, arguments[3] if len(arguments) > 3 else "GET", payload))
        if endpoint == "user":
            return {"login": "test-user"}
        if endpoint == "repos/test-user/FluSight-forecast-hub":
            return {"fork": True, "parent": {"full_name": submission.UPSTREAM}}
        if endpoint.startswith(f"repos/{submission.UPSTREAM}/pulls?state=open"):
            others = [{**open_pr, "number": n, "user": {"login": "someone"}} for n in range(99)]
            return others + [open_pr] if "page=1" in endpoint else []
        if "/contents/" in endpoint:
            text = "same bytes" if "Base" in endpoint else "old bytes"
            return {"content": base64.b64encode(text.encode()).decode()}
        if endpoint.endswith("git/commits/pr-head"):
            return {"tree": {"sha": "pr-tree"}}
        return {"sha": "new-object"}

    monkeypatch.setattr(submission, "gh_json", github)
    url = create_pr(tmp_path, {"model-metadata/MIGHTE-Base.yml": unchanged,
                               "model-metadata/MIGHTE-Nsemble.yml": changed},
                    branch="codex/mighte-2026-27-metadata", title="new title", body="new body")
    assert url == open_pr["html_url"]
    blobs = [payload for endpoint, _, payload in requests if endpoint.endswith("git/blobs")]
    assert [base64.b64decode(b["content"]) for b in blobs] == [b"new bytes"]
    assert (f"repos/test-user/FluSight-forecast-hub/git/refs/heads/{retry}", "PATCH",
            {"sha": "new-object", "force": False}) in requests
    assert (f"repos/{submission.UPSTREAM}/pulls/3710", "PATCH", {"title": "new title", "body": "new body"}) in requests
    # No second PR, no new branch and no pull-request history lookup.
    assert not any(endpoint.endswith("/pulls") or endpoint.endswith("git/refs") or "state=all" in endpoint
                   for endpoint, _, _ in requests)


def test_registration_cancellation_makes_no_github_request(root, monkeypatch, capsys):
    from mighte import cli

    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["mighte", "register"])
    monkeypatch.setattr("builtins.input", lambda prompt: "cancel")
    monkeypatch.setattr(cli, "register", lambda root: pytest.fail("Registration was not approved"))
    cli.main()
    assert "Registration cancelled" in capsys.readouterr().out


@pytest.mark.parametrize("joint_designated", [False, True])
def test_registration_includes_joint_retirement_and_enforces_two_designated(root, tmp_path, monkeypatch, joint_designated):
    from mighte import submit as submission
    from mighte.contract import Contract

    shutil.copytree(root / "model-metadata", tmp_path / "model-metadata")
    (tmp_path / "docs").mkdir()
    shutil.copyfile(root / "docs/REGISTRATION_PR.md", tmp_path / "docs/REGISTRATION_PR.md")
    joint = tmp_path / "model-metadata/MIGHTE-Joint.yml"
    if joint_designated:
        joint.write_text(joint.read_text().replace("designated_model: false", "designated_model: true"))
    monkeypatch.setattr(submission, "fresh_contract", lambda _: Contract(root / "hub-contract"))
    calls = []

    def capture_pr(project, files, **kwargs):
        calls.append(files)
        return "reviewed-metadata-pr"

    monkeypatch.setattr(submission, "create_pr", capture_pr)
    if joint_designated:
        with pytest.raises(ValueError, match="More than two designated"):
            submission.register(tmp_path)
        assert calls == []
    else:
        assert submission.register(tmp_path) == "reviewed-metadata-pr"
        assert len(calls) == 1
        assert set(calls[0]) == {"model-metadata/MIGHTE-Base.yml", "model-metadata/MIGHTE-Linear.yml",
                                 "model-metadata/MIGHTE-Nsemble.yml", "model-metadata/MIGHTE-Joint.yml"}


def test_a_completed_run_replaces_earlier_runs_for_the_same_date(tmp_path):
    from mighte.pipeline import replace_earlier_runs
    day = tmp_path / "runs/previews/2026-09-26"
    for name in ["A-submitted", "B-earlier", "C-new"]:
        (day / name).mkdir(parents=True)
        (tmp_path / "reports/previews/2026-09-26" / name).mkdir(parents=True)
    (day / "A-submitted/submission.json").write_text("{}")
    (tmp_path / "runs/previews/2026-10-03/other").mkdir(parents=True)  # other dates are untouched
    assert replace_earlier_runs(tmp_path, day / "C-new") == 1
    assert sorted(p.name for p in day.iterdir()) == ["A-submitted", "C-new"]
    assert sorted(p.name for p in (tmp_path / "reports/previews/2026-09-26").iterdir()) == ["A-submitted", "C-new"]
    assert (tmp_path / "runs/previews/2026-10-03/other").exists()
