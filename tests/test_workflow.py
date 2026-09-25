import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mighte.contract import ED, HOSP, read_forecast
from mighte.data import NSSP, WW
from mighte.evaluate import load_archive, prospective_runs
from mighte.pipeline import run_forecasts, verify_run
from mighte.submit import create_pr
from mighte.util import digest, write_json


def test_whole_offline_preview_and_submission_guard(root, tmp_path):
    """Exercise orchestration, serializers and guards in a separate project directory."""
    for directory in ["config", "hub-contract", "data/historical"]:
        shutil.copytree(root / directory, tmp_path / directory)
    settings = json.loads((tmp_path / "config/settings.json").read_text())
    settings["runtime"].update(num_bags=1, stage1_rounds=10, stage2_rounds=10, min_train_rows=100)
    settings["panel_ar_runtime"]["min_train_rows"] = 100
    settings["threads"] = 2
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
    manifest = verify_run(tmp_path, run)
    assert manifest["validation"]["MIGHTE-Base"]["rows"] == 3 * 2 * 4 * 23
    assert len(manifest["output_hashes"]) == 3
    with pytest.raises(ValueError, match="cannot be submitted"):
        verify_run(tmp_path, run, for_submission=True)
    assert load_archive(tmp_path).empty
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
