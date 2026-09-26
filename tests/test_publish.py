import base64
import hashlib
import json
import subprocess
import sys

import pytest
import yaml

from mighte import cli, publish
from mighte.util import digest, write_json


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    run = tmp_path / "runs/prospective/2026-10-10/fixture"
    manifest = {"run_id": "prospective/2026-10-10/fixture", "reference_date": "2026-10-10",
                "preview": False, "quick": False, "settings": {"runtime": {"num_bags": 100}},
                "output_hashes": {"model-output/example.csv": "forecast-hash"}}
    write_json(run / "manifest.json", manifest)
    directory = tmp_path / "reports" / manifest["run_id"]
    write_json(directory / "report-data.json", {"manifest": manifest})
    (directory / "index.html").write_text("<!doctype html><p>Reviewed forecast</p>")
    write_json(directory / "review.json", {"run_id": manifest["run_id"],
        "manifest_sha256": digest(run / "manifest.json"),
        "data_sha256": digest(directory / "report-data.json"),
        "report_sha256": digest(directory / "index.html")})
    monkeypatch.setattr(publish, "verify_run", lambda *args: manifest)
    return tmp_path, run, directory, manifest


def test_preparation_preserves_exact_reviewed_bytes(reviewed):
    root, run, directory, manifest = reviewed
    html, metadata = publish.prepare_publication(root, run)
    assert html == (directory / "index.html").read_bytes()
    assert metadata["forecast_output_hashes"] == manifest["output_hashes"]


@pytest.mark.parametrize("changed", ["index.html", "report-data.json", "manifest.json"])
def test_changed_artifacts_require_another_review(reviewed, changed):
    root, run, directory, _ = reviewed
    path = (run if changed == "manifest.json" else directory) / changed
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after review"):
        publish.prepare_publication(root, run)


def test_preview_and_reduced_fit_guards(reviewed):
    root, run, directory, manifest = reviewed
    manifest["preview"] = True
    with pytest.raises(ValueError, match="--preview explicitly"):
        publish.prepare_publication(root, run)
    manifest["quick"] = True
    with pytest.raises(ValueError, match="100-fit"):
        publish.prepare_publication(root, run, preview=True)
    manifest["preview"] = manifest["quick"] = False
    (directory / "review.json").unlink()
    with pytest.raises(ValueError, match="Build and review"):
        publish.prepare_publication(root, run)


@pytest.mark.parametrize("remote,expected", [
    ("https://github.com/owner/flusight_2026.git", "owner/flusight_2026"),
    ("git@github.com:owner/flusight_2026.git", "owner/flusight_2026"),
    ("https://github.com/owner/FluSight-forecast-hub.git", None)])
def test_destination_comes_from_pipeline_origin(tmp_path, monkeypatch, remote, expected):
    monkeypatch.setattr(publish.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess([], 0, stdout=remote))
    if expected:
        assert publish.repository(tmp_path) == expected
    else:
        with pytest.raises(ValueError, match="never the forecast hub"):
            publish.repository(tmp_path)


@pytest.mark.parametrize("existing", [False, True])
def test_publish_uploads_only_report_and_dispatches_exact_commit(reviewed, monkeypatch, existing):
    root, run, _, _ = reviewed
    html, metadata = publish.prepare_publication(root, run)
    calls = []
    api = "repos/owner/flusight_2026"

    def github(arguments, payload=None):
        endpoint = arguments[1]
        calls.append((arguments, payload))
        assert endpoint.startswith(api) and "pulls" not in endpoint
        if endpoint == api:
            return {"default_branch": "main"}
        if endpoint == api + "/pages":
            return {"build_type": "workflow", "html_url": "https://owner.github.io/flusight_2026/"}
        if endpoint.endswith("/actions/workflows/pages.yml"):
            return {"state": "active"}
        if "/git/ref/heads/" in endpoint:
            if not existing:
                raise RuntimeError("gh: Not Found (HTTP 404)")
            return {"object": {"sha": "previous-commit"}}
        if endpoint.endswith("/git/trees/previous-commit"):
            return {"tree": [{"path": name} for name in publish.SITE_FILES]}
        if "/contents/publication.json?" in endpoint:
            return {"content": base64.b64encode(json.dumps({"generator": publish.GENERATOR}).encode()).decode()}
        if endpoint.endswith("/git/blobs"):
            assert base64.b64decode(payload["content"]) == html
            return {"sha": "report-blob"}
        if endpoint.endswith("/git/trees"):
            assert {item["path"] for item in payload["tree"]} == publish.SITE_FILES
            return {"sha": "site-tree"}
        if endpoint.endswith("/git/commits"):
            assert payload["parents"] == (["previous-commit"] if existing else [])
            return {"sha": "reviewed-commit"}
        if endpoint.endswith("/dispatches"):
            assert payload == {"ref": "main", "inputs": {"dashboard_commit": "reviewed-commit"}}
            return None
        if "/git/refs" in endpoint:
            assert payload["sha"] == "reviewed-commit"
            if existing:
                assert payload["force"] is False
            else:
                assert payload["ref"] == "refs/heads/codex/dashboard"
            return None
        pytest.fail(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(publish, "gh_json", github)
    receipt = publish.publish_report(root, "owner/flusight_2026", html, metadata)
    assert receipt["status"] == "requested" and receipt["commit"] == "reviewed-commit"
    assert json.loads((root / ".runtime/pages-publication.json").read_text()) == receipt
    assert calls[-1][0][1].endswith("/dispatches")


def test_rejects_hub_destination_before_any_network_request(reviewed, monkeypatch):
    root, run, _, _ = reviewed
    html, metadata = publish.prepare_publication(root, run)
    monkeypatch.setattr(publish, "gh_json", lambda *a, **k: pytest.fail("No network request allowed"))
    with pytest.raises(ValueError, match="Invalid pipeline repository"):
        publish.publish_report(root, "cdcepi/FluSight-forecast-hub", html, metadata)
    with pytest.raises(ValueError, match="bytes differ"):
        publish.publish_report(root, "owner/flusight_2026", html + b"changed", metadata)


@pytest.mark.parametrize("problem,message", [("files", "unrelated files"),
                                             ("owner", "not created by MIGHTE"),
                                             ("network", "HTTP 502")])
def test_existing_branch_and_network_errors_never_trigger_writes(reviewed, monkeypatch, problem, message):
    root, run, _, _ = reviewed
    html, metadata = publish.prepare_publication(root, run)

    def github(arguments, payload=None):
        assert payload is None, "Must reject this publication before creating GitHub objects"
        endpoint = arguments[1]
        if endpoint.endswith("/pages"):
            return {"build_type": "workflow"}
        if "/git/ref/heads/" in endpoint:
            if problem == "network":
                raise RuntimeError("HTTP 502")
            return {"object": {"sha": "previous"}}
        if "/git/trees/" in endpoint:
            names = publish.SITE_FILES | ({"unrelated.txt"} if problem == "files" else set())
            return {"tree": [{"path": name} for name in names]}
        if "/contents/" in endpoint:
            return {"content": base64.b64encode(b'{}').decode()}
        return {"default_branch": "main"}

    monkeypatch.setattr(publish, "gh_json", github)
    with pytest.raises((ValueError, RuntimeError), match=message):
        publish.publish_report(root, "owner/flusight_2026", html, metadata)


def test_publish_cancellation_has_no_external_effect(root, monkeypatch, capsys):
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["mighte", "publish", "--preview"])
    monkeypatch.setattr(cli, "latest_run", lambda *a, **k: root / "runs/example")
    monkeypatch.setattr(cli, "prepare_publication", lambda *a, **k:
                        (b"report", {"reference_date": "2026-09-26", "preview": True}))
    monkeypatch.setattr(cli, "repository", lambda *a: "owner/flusight_2026")
    monkeypatch.setattr("builtins.input", lambda prompt: "cancel")
    monkeypatch.setattr(cli, "publish_report", lambda *a: pytest.fail("Publication was cancelled"))
    monkeypatch.setattr(cli, "submit", lambda *a: pytest.fail("Never submit when publishing"))
    cli.main()
    assert "Publication cancelled" in capsys.readouterr().out


def test_deployment_is_explicit_and_checks_artifact_hash(root, tmp_path):
    workflow = yaml.load((root / ".github/workflows/pages.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    steps = workflow["jobs"]["deploy"]["steps"]
    assert steps[0]["with"]["ref"] == "${{ inputs.dashboard_commit }}"
    script = next(step["run"] for step in steps if "run" in step)
    script = script.removeprefix("python - <<'PY'\n").removesuffix("PY\n")
    source = tmp_path / "reviewed"
    source.mkdir()
    (source / ".git").mkdir()
    (source / ".nojekyll").touch()
    html = b"<!doctype html><p>Reviewed preview</p>"
    (source / "index.html").write_bytes(html)
    write_json(source / "publication.json", {"generator": publish.GENERATOR,
               "report_sha256": hashlib.sha256(html).hexdigest()})
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in (tmp_path / "_site").iterdir()} == publish.SITE_FILES
    assert (tmp_path / "_site/index.html").read_bytes() == html
    (source / "index.html").write_bytes(b"unreviewed")
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True)
    assert result.returncode != 0
