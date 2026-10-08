import json
import shutil
import sys

import pandas as pd
import pytest
import requests

from mighte import cli, evaluate, report
from mighte.contract import HOSP
from mighte.util import digest, write_json


@pytest.fixture
def pending_pr(monkeypatch, forecast):
    source = {
        "state": "open", "changed_files": 2,
        "head": {"sha": "a" * 40, "repo": {"full_name": "reichlab/FluSight-forecast-hub"}},
    }
    filename = "model-output/UMass-flusion/2026-10-10-UMass-flusion.csv"
    files = [{"filename": "model-metadata/UMass-flusion.yml", "status": "modified"},
             {"filename": filename, "status": "added"}]
    mock = {"pr": source, "files": files, "csv": forecast.to_csv(index=False).encode(), "calls": []}

    def get(url, **kwargs):
        mock["calls"].append(url)
        response = requests.Response()
        response.status_code = 200
        if url.endswith("/pulls/3766"):
            response._content = json.dumps(mock["pr"]).encode()
        elif url.endswith("/pulls/3766/files"):
            page = kwargs["params"]["page"]
            response._content = json.dumps(mock["files"][(page - 1) * 100:page * 100]).encode()
        else:
            assert url == f"https://raw.githubusercontent.com/reichlab/FluSight-forecast-hub/{'a' * 40}/{filename}"
            response._content = mock["csv"]
            if mock.get("change_head"):
                mock["pr"]["head"]["sha"] = "b" * 40
        return response

    monkeypatch.setattr(evaluate.requests, "get", get)
    return mock


def test_pr_reads_only_forecast_csv_at_pinned_commit_and_keeps_cache_separate(tmp_path, pending_pr):
    frame, sources = evaluate.fetch_pr_forecasts(tmp_path, [3766, 3766], {"2026-10-10"})
    assert set(frame.model_id) == {"UMass-flusion (pending PR #3766)"}
    assert set(frame.location) == {"01"}
    assert len(sources) == 1 and sources[0]["head_sha"] == "a" * 40
    cache = tmp_path / "data/benchmarks/pull-requests/3766" / ("a" * 40)
    path = cache / "model-output/UMass-flusion/2026-10-10-UMass-flusion.csv"
    assert path.read_bytes() == pending_pr["csv"]
    assert sources[0]["files"][0]["sha256"] == digest(path)
    assert not (tmp_path / "data/benchmarks/UMass-flusion").exists()
    assert not (tmp_path / "data/benchmarks/catalog.json").exists()
    assert len(pending_pr["calls"]) == 4


def test_pr_files_are_paginated(tmp_path, pending_pr):
    pending_pr["files"] = [{"filename": f"notes/{i}.md", "status": "added"} for i in range(100)] + pending_pr["files"]
    pending_pr["pr"]["changed_files"] = 102
    frame, _ = evaluate.fetch_pr_forecasts(tmp_path, [3766], {"2026-10-10"})
    assert not frame.empty
    assert sum(url.endswith("/files") for url in pending_pr["calls"]) == 2


@pytest.mark.parametrize("problem,message", [
    ("closed", "not open"), ("different_week", "no supported forecasts"),
    ("removed", "no supported forecasts"), ("truncated", "incomplete"),
    ("head_changed", "changed during download"), ("wrong_date", "reference date"),
    ("missing_column", "columns"), ("duplicate", "duplicate forecast rows"),
    ("negative", "invalid or duplicate"),
])
def test_bad_or_changed_pr_does_not_produce_a_comparison(tmp_path, pending_pr, forecast, problem, message):
    references = {"2026-10-10"}
    if problem == "closed":
        pending_pr["pr"]["state"] = "closed"
    elif problem == "different_week":
        references = {"2026-10-17"}
    elif problem == "removed":
        pending_pr["files"][1]["status"] = "removed"
    elif problem == "truncated":
        pending_pr["pr"]["changed_files"] = 3
    elif problem == "head_changed":
        pending_pr["change_head"] = True
    else:
        frame = {"wrong_date": forecast.assign(reference_date="2026-10-17"),
                 "missing_column": forecast.drop(columns="value"),
                 "duplicate": pd.concat([forecast, forecast.iloc[:1]]),
                 "negative": forecast.assign(value=-1)}[problem]
        pending_pr["csv"] = frame.to_csv(index=False).encode()
    with pytest.raises(ValueError, match=message):
        evaluate.fetch_pr_forecasts(tmp_path, [3766], references)


@pytest.mark.parametrize("preview", [False, True])
def test_pending_report_preserves_official_report_scores_and_model_colors(
        root, tmp_path, monkeypatch, forecast, pending_pr, preview):
    kind = "previews" if preview else "official_submissions"
    run = tmp_path / "runs" / kind / "2026-10-10/fixture"
    filename = "model-output/MIGHTE-Base/2026-10-10-MIGHTE-Base.csv"
    (run / filename).parent.mkdir(parents=True)
    forecast.to_csv(run / filename, index=False)
    manifest = {"run_id": f"{kind}/2026-10-10/fixture", "reference_date": "2026-10-10",
                "preview": preview, "quick": False, "settings": {"season": "2026-2027", "runtime": {"num_bags": 100}},
                "output_hashes": {filename: digest(run / filename)}}
    write_json(run / "manifest.json", manifest)
    write_json(run / "data-audit.json", {})
    snapshot = tmp_path / "data/snapshots/fixture"
    shutil.copytree(root / "hub-contract", snapshot / "contract")
    pd.DataFrame({"target": [HOSP], "location": ["01"], "date": ["2026-10-10"], "value": [20]}).to_csv(
        snapshot / "truth.csv", index=False)
    monkeypatch.setattr(report, "latest_snapshot", lambda root: snapshot)
    monkeypatch.setattr(report, "verify_snapshot", lambda path: None)
    monkeypatch.setattr(report, "verify_run", lambda *a: manifest)
    monkeypatch.setattr(report, "preview_runs", lambda root: {})
    monkeypatch.setattr(report, "discover_benchmarks", lambda *a, **k: (None, {}))
    monkeypatch.setattr(evaluate, "load_archive", lambda root:
                        pd.DataFrame() if preview else forecast.assign(model_id="MIGHTE-Base"))
    # The same model can have a merged forecast and a pending revision without replacing either.
    monkeypatch.setattr(evaluate, "fetch_benchmarks", lambda *a, **k:
                        (forecast.assign(model_id="UMass-flusion", value=1), []))
    normal = report.build_report(tmp_path, run, online=False)
    before = {p: p.read_bytes() for directory in [normal.parent, run] for p in directory.rglob("*") if p.is_file()}
    comparison = report.build_report(tmp_path, run, include_prs=[3766])
    assert comparison.is_relative_to(tmp_path / "reports/local-comparisons")
    assert all(p.read_bytes() == content for p, content in before.items())
    original = json.loads((normal.parent / "report-data.json").read_text())
    payload = json.loads((comparison.parent / "report-data.json").read_text())
    assert payload["scores"] == original["scores"]
    assert payload["baseline_models"] == original["baseline_models"]
    assert all("pending" not in row["model_id"] for row in payload["scores"])
    assert {row["model_id"] for row in payload["forecasts"]} == {
        "MIGHTE-Base", "UMass-flusion", "UMass-flusion (pending PR #3766)"}
    assert payload["model_colors"]["UMass-flusion (pending PR #3766)"] == report.model_color("UMass-flusion")
    assert payload["default_models"] == ["MIGHTE-Base"]
    assert payload["included_prs"][0]["number"] == 3766


def test_review_cli_passes_optional_prs_without_submission_or_publication(root, monkeypatch, capsys):
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mighte", "review", "--preview", "--include-pr", "3766",
                                     "--include-pr", "3767", "--no-open"])
    run = root / "runs/previews/fixture"
    monkeypatch.setattr(cli, "latest_run", lambda _, preview: run if preview else pytest.fail("Expected preview"))
    received = {}

    def build(root, selected, **kwargs):
        assert selected == run
        received.update(kwargs)
        return root / "reports/local-comparisons/index.html"

    monkeypatch.setattr(cli, "build_report", build)
    monkeypatch.setattr(cli, "submit", lambda *a: pytest.fail("Review must never submit"))
    monkeypatch.setattr(cli, "publish_report", lambda *a: pytest.fail("Review must never publish"))
    cli.main()
    assert received == {"online": True, "include_prs": [3766, 3767]}
    assert "local-comparisons" in capsys.readouterr().out


@pytest.mark.parametrize("flags", [["--offline", "--include-pr", "3766"], ["--include-pr", "0"]])
def test_invalid_pr_arguments_fail_before_any_download(root, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["mighte", "review", *flags])
    monkeypatch.setattr(cli, "build_report", lambda *a, **k: pytest.fail("Must fail before review"))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
