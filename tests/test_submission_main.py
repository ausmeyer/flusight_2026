import base64
import json

import pandas as pd
import pytest
import requests

from mighte import submit as submission
from mighte.contract import MODELS, Contract


REFERENCE = "2026-10-24"
FILE = f"model-output/MIGHTE-Base/{REFERENCE}-MIGHTE-Base.csv"
FORK = "repos/test-user/FluSight-forecast-hub"
UPSTREAM = f"repos/{submission.UPSTREAM}"


@pytest.fixture
def main_submission(tmp_path, monkeypatch):
    path = tmp_path / "forecast.csv"
    path.write_text("validated forecast bytes")
    state = {"calls": [], "open": [], "files": [], "tree": "new-tree", "sync_error": False}

    def github(arguments, payload=None):
        endpoint = arguments[1]
        state["calls"].append((endpoint, payload))
        if endpoint == "user":
            return {"login": "test-user"}
        if endpoint == FORK:
            return {"fork": True, "parent": {"full_name": submission.UPSTREAM}}
        if endpoint.startswith(f"{UPSTREAM}/pulls?state=open"):
            return state["open"]
        if endpoint.endswith("/files?per_page=100"):
            return state["files"]
        if endpoint == f"{FORK}/merge-upstream":
            if state["sync_error"]:
                raise RuntimeError("HTTP 409: merge conflict")
            return {"merge_type": "fast-forward"}
        if endpoint.endswith("git/ref/heads/main"):
            return {"object": {"sha": "synced-head" if endpoint.startswith(FORK) else "upstream-head"}}
        if "/compare/" in endpoint:
            return {"files": state["files"]}
        if endpoint == f"{FORK}/git/commits/synced-head":
            return {"sha": "synced-head", "tree": {"sha": "synced-tree"}}
        if "/contents/" in endpoint:
            return {"content": base64.b64encode(b"older forecast").decode()}
        if endpoint == f"{FORK}/git/trees":
            return {"sha": state["tree"]}
        if endpoint == f"{UPSTREAM}/pulls":
            return {"html_url": "https://example.test/weekly"}
        if endpoint in {f"{FORK}/git/blobs", f"{FORK}/git/commits", f"{FORK}/git/refs/heads/main"}:
            return {"sha": "new-object"}
        pytest.fail(f"Unexpected GitHub request: {arguments}")

    monkeypatch.setattr(submission, "gh_json", github)
    monkeypatch.setattr(submission, "check_window", lambda reference: None)
    return state, lambda **kwargs: submission.create_pr(tmp_path, {FILE: path}, branch="main", title="weekly",
                                                        body="forecast", reference=REFERENCE, **kwargs)


def test_weekly_pr_uses_synced_account_main_without_creating_a_branch(main_submission, tmp_path):
    state, submit = main_submission
    assert submit() == "https://example.test/weekly"
    assert (f"{FORK}/merge-upstream", {"branch": "main"}) in state["calls"]
    assert (f"{FORK}/git/refs/heads/main", {"sha": "new-object", "force": False}) in state["calls"]
    assert (f"{UPSTREAM}/pulls", {"title": "weekly", "head": "test-user:main",
                                 "base": "main", "body": "forecast"}) in state["calls"]
    assert not any(endpoint.endswith("/git/refs") or "state=all" in endpoint
                   for endpoint, _ in state["calls"])
    assert json.loads((tmp_path / ".runtime/main.json").read_text())["branch"] == "main"


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("changed", [{"filename": "unrelated.txt"},
                                     {"filename": FILE, "previous_filename": "unrelated.txt"}])
def test_main_rejects_other_unmerged_files_before_uploading(main_submission, existing, changed):
    state, submit = main_submission
    state["files"] = [changed]
    if existing:
        state["open"] = [{"number": 10, "user": {"login": "test-user"}, "head": {"ref": "main"}}]
    with pytest.raises(ValueError, match="other"):
        submit()
    assert not any(endpoint.endswith(("/git/blobs", "/git/commits", "/git/refs/heads/main", "/pulls"))
                   for endpoint, _ in state["calls"])


def test_legacy_open_forecast_pr_prevents_a_duplicate_main_pr(main_submission):
    state, submit = main_submission
    state["open"] = [{"number": 10, "html_url": "https://example.test/legacy", "user": {"login": "test-user"},
                       "head": {"ref": f"codex/mighte-{REFERENCE}"}}]
    with pytest.raises(ValueError, match="already has an open PR"):
        submit()
    assert not any(endpoint.endswith(("/merge-upstream", "/git/blobs", "/git/commits", "/git/refs/heads/main", "/pulls"))
                   for endpoint, _ in state["calls"])


def test_open_main_pr_is_updated_without_creating_another(main_submission):
    state, submit = main_submission
    state["files"] = [{"filename": FILE}]
    state["open"] = [{"number": 10, "html_url": "https://example.test/existing", "title": "weekly",
                       "body": "forecast", "user": {"login": "test-user"},
                       "head": {"ref": "main", "sha": "synced-head"}}]
    assert submit() == "https://example.test/existing"
    assert (f"{FORK}/git/refs/heads/main", {"sha": "new-object", "force": False}) in state["calls"]
    assert not any(endpoint.endswith(("/merge-upstream", "/pulls", "/git/refs"))
                   for endpoint, _ in state["calls"])


def test_main_recovers_an_interrupted_pr_creation_without_another_commit(main_submission):
    state, submit = main_submission
    state["files"], state["tree"] = [{"filename": FILE}], "synced-tree"
    assert submit() == "https://example.test/weekly"
    assert not any(endpoint.endswith(("/git/commits", "/git/refs/heads/main"))
                   for endpoint, _ in state["calls"])


def test_already_merged_forecasts_do_not_create_an_empty_pr(main_submission):
    state, submit = main_submission
    state["tree"] = "synced-tree"
    with pytest.raises(ValueError, match="already match upstream"):
        submit()
    assert not any(endpoint.endswith(("/git/commits", "/git/refs/heads/main", "/pulls"))
                   for endpoint, _ in state["calls"])


def test_fork_sync_conflicts_stop_before_uploading(main_submission):
    state, submit = main_submission
    state["sync_error"] = True
    with pytest.raises(RuntimeError, match="merge conflict"):
        submit()
    assert not any(endpoint.endswith(("/git/blobs", "/git/commits", "/git/refs/heads/main", "/pulls"))
                   for endpoint, _ in state["calls"])


def test_deadline_is_checked_before_advancing_main(main_submission, monkeypatch):
    state, submit = main_submission
    checks = []

    def window(reference):
        checks.append(reference)
        if len(checks) == 2:
            raise ValueError("Submission window closed")

    monkeypatch.setattr(submission, "check_window", window)
    with pytest.raises(ValueError, match="window closed"):
        submit()
    assert checks == [REFERENCE, REFERENCE]
    assert not any(endpoint.endswith(("/git/refs/heads/main", "/pulls")) for endpoint, _ in state["calls"])


@pytest.mark.parametrize("existing", [False, True])
def test_explicit_window_override_reaches_new_and_updated_prs(main_submission, monkeypatch, existing):
    state, submit = main_submission
    if existing:
        state["files"] = [{"filename": FILE}]
        state["open"] = [{"number": 10, "html_url": "https://example.test/existing", "title": "weekly",
                           "body": "forecast", "user": {"login": "test-user"},
                           "head": {"ref": "main", "sha": "synced-head"}}]
    monkeypatch.setattr(submission, "check_window", lambda reference: pytest.fail("Window override was ignored"))
    assert submit(ignore_window=True) == f"https://example.test/{'existing' if existing else 'weekly'}"
    assert (f"{FORK}/git/refs/heads/main", {"sha": "new-object", "force": False}) in state["calls"]


@pytest.mark.parametrize("ignore_window", [False, True])
def test_weekly_entry_point_uses_main_and_preserves_pr_text(root, tmp_path, monkeypatch, forecast, ignore_window):
    frame = forecast.copy()
    frame["reference_date"] = REFERENCE
    frame["target_end_date"] = (pd.to_datetime(frame.target_end_date) + pd.Timedelta(weeks=2)).dt.strftime("%Y-%m-%d")
    paths = {}
    for model in MODELS:
        name = f"model-output/{model}/{REFERENCE}-{model}.csv"
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        frame.to_csv(path, index=False)
        paths[name] = "validated-hash"
    manifest = {"reference_date": REFERENCE, "output_hashes": paths, "models": list(MODELS)}
    validation = []
    monkeypatch.setattr(submission, "verify_run", lambda *args, **kwargs: validation.append(kwargs) or manifest)
    monkeypatch.setattr(submission, "fresh_contract", lambda _: Contract(root / "hub-contract"))
    response = requests.Response()
    response.status_code = 200
    monkeypatch.setattr(submission.requests, "get", lambda *args, **kwargs: response)
    captured = []
    monkeypatch.setattr(submission, "create_pr", lambda *args, **kwargs:
                        captured.append(kwargs) or "https://example.test/weekly")
    assert submission.submit(root, tmp_path, ignore_window=ignore_window) == "https://example.test/weekly"
    override = {"ignore_window": True} if ignore_window else {}
    assert validation == [{"for_submission": True, **override}]
    assert captured == [{"branch": "main", "title": f"MIGHTE forecasts for {REFERENCE}",
                         "body": f"{REFERENCE} forecasts from MIGHTE-Base, MIGHTE-Linear, MIGHTE-Nsemble.",
                         "reference": REFERENCE, **override}]
    assert json.loads((tmp_path / "submission.json").read_text())["window_override"] is ignore_window


@pytest.mark.parametrize("command", ["forecast", "resume"])
@pytest.mark.parametrize("ignore_window", [False, True])
def test_forecast_cli_passes_explicit_window_override(tmp_path, monkeypatch, command, ignore_window):
    from mighte import cli

    arguments = ["mighte", command, *([str(tmp_path)] if command == "resume" else []), "--no-open"]
    if ignore_window:
        arguments.append("--ignore-window")
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "reference_saturday", lambda: REFERENCE)
    captured = []
    monkeypatch.setattr(cli, "run_forecasts", lambda *args, **kwargs: captured.append((args, kwargs)) or tmp_path)
    monkeypatch.setattr(cli, "build_report", lambda *args: tmp_path / "report.html")
    cli.main()
    assert captured == [((tmp_path, REFERENCE), {"preview": False, "quick": False,
                        "resume": tmp_path if command == "resume" else None,
                        "snapshot": None, "ignore_window": ignore_window})]


def test_submit_cli_passes_explicit_window_override(tmp_path, monkeypatch):
    from mighte import cli

    monkeypatch.setattr("sys.argv", ["mighte", "submit", "--run", str(tmp_path), "--ignore-window"])
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(cli, "verify_run", lambda *args, **kwargs:
                        calls.append(("verify", args, kwargs)) or {"reference_date": REFERENCE})
    monkeypatch.setattr(cli, "submit", lambda *args, **kwargs:
                        calls.append(("submit", args, kwargs)) or "https://example.test/weekly")
    cli.main()
    assert calls == [("verify", (tmp_path, tmp_path), {"for_submission": True, "ignore_window": True}),
                     ("submit", (tmp_path, tmp_path), {"ignore_window": True})]


def test_preview_cli_rejects_window_override(monkeypatch):
    from mighte import cli

    monkeypatch.setattr("sys.argv", ["mighte", "preview", "--ignore-window"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
