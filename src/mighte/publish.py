"""Publish an existing reviewed dashboard to this pipeline repository's Pages site."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path

from .pipeline import verify_run
from .submit import gh_json
from .util import digest, utc_now, write_json

BRANCH = "codex/dashboard"
SITE_FILES = {"index.html", "publication.json", ".nojekyll"}
GENERATOR = "mighte-pages-v1"


def repository(root: Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "remote", "get-url", "origin"],
                            capture_output=True, text=True, check=False)
    remote = result.stdout.strip().rstrip("/").removesuffix(".git")
    match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+)", remote)
    if result.returncode or not match:
        raise ValueError("Dashboard publishing requires a GitHub origin remote")
    repo = match.group(1)
    if repo.lower().endswith("/flusight-forecast-hub"):
        raise ValueError("Dashboard publishing must target the pipeline repository, never the forecast hub")
    return repo


def prepare_publication(root: Path, run: Path, *, preview=False) -> tuple[bytes, dict]:
    manifest = verify_run(root, run)
    if manifest["run_id"] != run.resolve().relative_to((root / "runs").resolve()).as_posix():
        raise ValueError("Run identity does not match its archive directory")
    if manifest["preview"] and not preview:
        raise ValueError("Use --preview explicitly to publish a labeled preseason preview")
    if manifest["quick"] or manifest["settings"]["runtime"]["num_bags"] != 100:
        raise ValueError("Only complete 100-fit reports can be published; quick runs remain local")
    directory = root / "reports" / manifest["run_id"]
    if not (directory / "review.json").exists():
        raise ValueError("Build and review this report first with ./mighte review (or review --preview)")
    html = (directory / "index.html").read_bytes()
    html_hash = hashlib.sha256(html).hexdigest()
    expected = {"run_id": manifest["run_id"], "manifest_sha256": digest(run / "manifest.json"),
                "report_sha256": html_hash, "data_sha256": digest(directory / "report-data.json")}
    payload = json.loads((directory / "report-data.json").read_text())
    if json.loads((directory / "review.json").read_text()) != expected or payload["manifest"] != manifest:
        raise ValueError("Report or forecast changed after review generation; rebuild and review it again")
    return html, {"generator": GENERATOR, "run_id": manifest["run_id"],
                  "reference_date": manifest["reference_date"], "preview": manifest["preview"],
                  "report_sha256": html_hash, "forecast_output_hashes": manifest["output_hashes"]}


def publish_report(root: Path, repo: str, html: bytes, metadata: dict) -> dict:
    # Keep this boundary explicit even when called outside the command-line entry point.
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or repo.lower().endswith("/flusight-forecast-hub"):
        raise ValueError("Invalid pipeline repository for dashboard publishing")
    if hashlib.sha256(html).hexdigest() != metadata["report_sha256"]:
        raise ValueError("Publication bytes differ from the reviewed report")
    api = f"repos/{repo}"
    info = gh_json(["api", api])
    pages = gh_json(["api", f"{api}/pages"])
    if pages.get("build_type") != "workflow":
        raise ValueError("Set this repository's Settings → Pages → Source to GitHub Actions first")
    gh_json(["api", f"{api}/actions/workflows/pages.yml"])
    try:
        head = gh_json(["api", f"{api}/git/ref/heads/{BRANCH}"])["object"]["sha"]
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        head = None
    if head:
        tree = gh_json(["api", f"{api}/git/trees/{head}"])
        if {entry["path"] for entry in tree["tree"]} != SITE_FILES:
            raise ValueError("Existing dashboard branch contains unrelated files; refusing to replace it")
        previous = gh_json(["api", f"{api}/contents/publication.json?ref={BRANCH}"])
        if json.loads(base64.b64decode(previous["content"])).get("generator") != GENERATOR:
            raise ValueError("Existing dashboard branch was not created by MIGHTE publishing")
    publication = {**metadata, "published_at": utc_now()}
    blob = gh_json(["api", f"{api}/git/blobs", "--method", "POST"],
                   {"content": base64.b64encode(html).decode(), "encoding": "base64"})
    tree = gh_json(["api", f"{api}/git/trees", "--method", "POST"], {"tree": [
        {"path": "index.html", "mode": "100644", "type": "blob", "sha": blob["sha"]},
        {"path": "publication.json", "mode": "100644", "type": "blob", "content": json.dumps(publication, indent=2) + "\n"},
        {"path": ".nojekyll", "mode": "100644", "type": "blob", "content": ""}]})
    commit = gh_json(["api", f"{api}/git/commits", "--method", "POST"],
                     {"message": f"Publish reviewed dashboard for {metadata['reference_date']}",
                      "tree": tree["sha"], "parents": [head] if head else []})["sha"]
    if head:
        gh_json(["api", f"{api}/git/refs/heads/{BRANCH}", "--method", "PATCH"],
                 {"sha": commit, "force": False})
    else:
        gh_json(["api", f"{api}/git/refs", "--method", "POST"],
                 {"ref": f"refs/heads/{BRANCH}", "sha": commit})
    gh_json(["api", f"{api}/actions/workflows/pages.yml/dispatches", "--method", "POST"],
             {"ref": info["default_branch"], "inputs": {"dashboard_commit": commit}})
    receipt = {**publication, "commit": commit, "repository": repo, "status": "requested",
               "url": pages["html_url"], "workflow_url": f"https://github.com/{repo}/actions/workflows/pages.yml"}
    write_json(root / ".runtime/pages-publication.json", receipt)
    return receipt
