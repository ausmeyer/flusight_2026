import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mighte.contract import ED, HOSP, TREND, UNIT, read_forecast
from mighte.data import NSSP, WW
from mighte.evaluate import load_archive, prospective_runs
from mighte.pipeline import run_forecasts, verify_run
from mighte.report import build_report
from mighte.submit import create_pr
from mighte.util import digest, write_json


@pytest.mark.parametrize("with_ordinal", [False, True])
def test_whole_offline_preview_and_submission_guard(root, tmp_path, monkeypatch, with_ordinal):
    """Exercise orchestration, serializers and guards in a separate project directory."""
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
    truth.to_csv(snapshot / "truth.csv", index=False)
    pd.DataFrame({"date": dates.repeat(3), "location_name": ["Alabama", "Alaska", "US"] * len(dates),
                  "scale_factor": 1.}).to_csv(snapshot / "ed-fractions.csv", index=False)
    pd.DataFrame({"date": dates, NSSP: 1 + .5 * np.sin(w / 8),
                   "available_date": dates + pd.Timedelta(weeks=1)}).to_csv(snapshot / "nssp.csv", index=False)
    ww = pd.DataFrame({"date": dates, WW: np.sin(w / 8) - 3,
                       "available_date": dates + pd.Timedelta(weeks=2)})
    for lag in [1, 2, 4]:
        ww[f"{WW}_lag{lag}"] = ww[WW].shift(lag)
    ww.to_csv(snapshot / "wastewater.csv", index=False)
    write_json(snapshot / "manifest.json", {"files": {str(p.relative_to(snapshot)): digest(p)
                for p in snapshot.rglob("*") if p.is_file()}})
    run = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    settings["component_workers"] = 2
    write_json(tmp_path / "config/settings.json", settings)
    parallel_run = run_forecasts(tmp_path, "2026-10-10", preview=True, snapshot=snapshot)
    for path in (run / "model-output").rglob("*.csv"):
        assert path.read_bytes() == (parallel_run / path.relative_to(run)).read_bytes()
    manifest = verify_run(tmp_path, run)
    assert manifest["validation"]["MIGHTE-Base"]["rows"] == 3 * 2 * 4 * 23 + (3 * 4 * 5 if with_ordinal else 0)
    assert len(manifest["output_hashes"]) == 3
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
        interrupted["status"] = "running"
        write_json(parallel_run / "manifest.json", interrupted)
        with pytest.raises(ValueError, match="Incomplete forecast coverage.*rate change"):
            run_forecasts(tmp_path, "2026-10-10", resume=parallel_run)
    # Verify mutations to the actual exported artifact are caught.
    path = run / next(iter(manifest["output_hashes"]))
    path.write_text(path.read_text().replace(",quantile,", ",changed,", 1))
    with pytest.raises(ValueError, match="edited after validation"):
        verify_run(tmp_path, run)


def test_select_submitted_run_without_double_counting(tmp_path):
    base = tmp_path / "runs/prospective/2026-10-10"
    for name, completed in [("first", "2026-10-07T15:00:00-04:00"),
                             ("later", "2026-10-07T16:00:00-04:00"),
                             ("late", "2026-10-08T10:00:00-04:00")]:
        write_json(base / name / "manifest.json", {"preview": False, "status": "complete",
                     "reference_date": "2026-10-10", "created_at": completed, "completed_at": completed})
    assert prospective_runs(tmp_path) == [base / "later"]
    write_json(base / "first/submission.json", {"url": "test", "submitted_at": "2026-10-07T15:30:00-04:00"})
    assert prospective_runs(tmp_path) == [base / "first"]


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
