"""Create small, allowlisted hub PRs through the authenticated GitHub CLI."""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import requests

from .contract import Contract, MODELS, check_window, read_forecast
from .data import HUB
from .pipeline import verify_run
from .util import digest, utc_now, write_json

UPSTREAM = "cdcepi/FluSight-forecast-hub"


def gh_json(arguments: list[str], payload: dict | None = None):
    args = ["gh", *arguments]
    if payload is not None:
        args += ["--input", "-"]
    result = subprocess.run(args, input=json.dumps(payload) if payload is not None else None,
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"GitHub command failed: {result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip() else None


def fresh_contract(root: Path) -> Contract:
    directory = root / ".runtime/submission-contract"
    directory.mkdir(parents=True, exist_ok=True)
    for remote, local in [("hub-config/tasks.json", "tasks.json"),
                          ("hub-config/model-metadata-schema.json", "model-metadata-schema.json"),
                          ("auxiliary-data/locations.csv", "locations.csv")]:
        response = requests.get(HUB + remote, timeout=(20, 60))
        response.raise_for_status()
        (directory / local).write_bytes(response.content)
    return Contract(directory)


def open_pull_request(login: str, branch: str) -> dict | None:
    """Find the user's open PR, allowing retry suffixes only for legacy named branches."""
    page = 1
    while True:
        pulls = gh_json(["api", f"repos/{UPSTREAM}/pulls?state=open&per_page=100&page={page}"])
        for pr in pulls:
            ref = pr["head"]["ref"]
            if pr["user"]["login"] == login and (ref == branch or (
                    branch != "main" and ref.startswith(branch + "-"))):
                return pr
        if len(pulls) < 100:
            return None
        page += 1


def update_pr(root: Path, files: dict[str, Path], pr: dict, fork: str, *, title: str, body: str,
              reference: str | None) -> str:
    """Add one commit with the changed files to an open PR and refresh its text."""
    ref, head = pr["head"]["ref"], pr["head"]["sha"]
    changed = {}
    for name, path in sorted(files.items()):
        try:
            remote = gh_json(["api", f"repos/{fork}/contents/{name}?ref={head}"])
            same = base64.b64decode(remote.get("content") or "") == path.read_bytes()
        except RuntimeError:
            same = False
        if not same:
            changed[name] = path
    commit = head
    if changed:
        base = gh_json(["api", f"repos/{fork}/git/commits/{head}"])
        entries = []
        for name, path in changed.items():
            blob = gh_json(["api", f"repos/{fork}/git/blobs", "--method", "POST"],
                           {"content": base64.b64encode(path.read_bytes()).decode(), "encoding": "base64"})
            entries.append({"path": name, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        tree = gh_json(["api", f"repos/{fork}/git/trees", "--method", "POST"],
                       {"base_tree": base["tree"]["sha"], "tree": entries})
        commit = gh_json(["api", f"repos/{fork}/git/commits", "--method", "POST"],
                         {"message": title, "tree": tree["sha"], "parents": [head]})["sha"]
        if reference:
            check_window(reference)
        gh_json(["api", f"repos/{fork}/git/refs/heads/{ref}", "--method", "PATCH"], {"sha": commit, "force": False})
    if pr["title"] != title or (pr.get("body") or "").strip() != body.strip():
        gh_json(["api", f"repos/{UPSTREAM}/pulls/{pr['number']}", "--method", "PATCH"], {"title": title, "body": body})
    write_json(root / ".runtime" / (ref.replace("/", "-") + ".json"),
               {"url": pr["html_url"], "branch": ref, "commit": commit, "updated_at": utc_now(),
                "updated_files": sorted(changed), "files": {name: digest(path) for name, path in files.items()}})
    print(f"Updated open PR {pr['html_url']}" + (f": {', '.join(sorted(changed))}" if changed else " (files unchanged)"))
    return pr["html_url"]


def create_pr(root: Path, files: dict[str, Path], *, branch: str, title: str, body: str,
              reference: str | None = None) -> str:
    """Update an open PR, or prepare allowlisted changes on the requested fork branch."""
    if not files or any(not (name in {f"model-metadata/{m}.yml" for m in MODELS}
                              or any(name.startswith(f"model-output/{m}/") and name.endswith(f"-{m}.csv")
                                     for m in MODELS)) for name in files):
        raise ValueError("Refusing to upload files outside the MIGHTE submission allowlist")
    if reference:
        check_window(reference)
    login = gh_json(["api", "user"])["login"]
    fork = f"{login}/FluSight-forecast-hub"
    try:
        repo = gh_json(["api", f"repos/{fork}"])
    except RuntimeError:
        repo = gh_json(["api", f"repos/{UPSTREAM}/forks", "--method", "POST"], {})
    if not repo.get("fork") or repo.get("parent", {}).get("full_name", UPSTREAM) != UPSTREAM:
        raise ValueError(f"{fork} is not a fork of {UPSTREAM}")
    existing = open_pull_request(login, branch)
    if existing:
        if branch == "main":
            changed = gh_json(["api", f"repos/{UPSTREAM}/pulls/{existing['number']}/files?per_page=100"])
            if any(f["filename"] not in files or f.get("previous_filename", f["filename"]) not in files
                   for f in changed):
                raise ValueError("Your fork's main branch has an open PR with other files; finish that PR first")
        return update_pr(root, files, existing, fork, title=title, body=body, reference=reference)
    if branch == "main":
        if reference:
            legacy = open_pull_request(login, f"codex/mighte-{reference}")
            if legacy:
                raise ValueError(f"This forecast already has an open PR: {legacy['html_url']}; no duplicate was opened")
        # Sync without discarding fork commits, then exclude unrelated unmerged work.
        gh_json(["api", f"repos/{fork}/merge-upstream", "--method", "POST"], {"branch": "main"})
        head = gh_json(["api", f"repos/{fork}/git/ref/heads/main"])["object"]["sha"]
        upstream = gh_json(["api", f"repos/{UPSTREAM}/git/ref/heads/main"])["object"]["sha"]
        comparison = gh_json(["api", f"repos/{fork}/compare/{upstream}...{head}?per_page=1"])
        if any(f["filename"] not in files or f.get("previous_filename", f["filename"]) not in files
               for f in comparison["files"]):
            raise ValueError("Your fork's main branch has other unmerged changes; review them before submitting")
    else:
        history = gh_json(["api", f"repos/{UPSTREAM}/pulls?state=all&head={login}:{branch}"])
        if history:
            # Legacy named branches retain their closed-PR retry behavior.
            branch += "-" + utc_now().replace("-", "").replace(":", "").replace(".", "").split("+")[0]
        head = gh_json(["api", f"repos/{UPSTREAM}/git/ref/heads/main"])["object"]["sha"]
    commit = gh_json(["api", f"repos/{fork if branch == 'main' else UPSTREAM}/git/commits/{head}"])
    # Build on synced fork main, or the fresh upstream base for a legacy named branch.
    tree_entries = []
    for name, path in sorted(files.items()):
        blob = gh_json(["api", f"repos/{fork}/git/blobs", "--method", "POST"],
                       {"content": base64.b64encode(path.read_bytes()).decode(), "encoding": "base64"})
        tree_entries.append({"path": name, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    tree = gh_json(["api", f"repos/{fork}/git/trees", "--method", "POST"],
                   {"base_tree": commit["tree"]["sha"], "tree": tree_entries})
    if branch == "main" and tree["sha"] == commit["tree"]["sha"]:
        if not comparison["files"]:
            raise ValueError("These files already match upstream; no new PR is needed")
        new_commit = {"sha": head}  # Recover a push that succeeded before PR creation was interrupted.
    else:
        new_commit = gh_json(["api", f"repos/{fork}/git/commits", "--method", "POST"],
                             {"message": title, "tree": tree["sha"], "parents": [head]})
    if branch == "main":
        if reference:
            check_window(reference)
        if new_commit["sha"] != head:
            gh_json(["api", f"repos/{fork}/git/refs/heads/main", "--method", "PATCH"],
                     {"sha": new_commit["sha"], "force": False})
    else:
        # Unique legacy branch names preserve earlier work, including closed PRs.
        try:
            gh_json(["api", f"repos/{fork}/git/refs", "--method", "POST"],
                    {"ref": "refs/heads/" + branch, "sha": new_commit["sha"]})
        except RuntimeError as exc:
            # Recover an interrupted PR creation only for the intended tree.
            # Never reset or overwrite an unrelated branch.
            try:
                ref = gh_json(["api", f"repos/{fork}/git/ref/heads/{branch}"])
                prior = gh_json(["api", f"repos/{fork}/git/commits/{ref['object']['sha']}"])
            except RuntimeError:
                raise exc
            if prior["tree"]["sha"] != tree["sha"]:
                raise ValueError(f"Branch {branch} already exists with different contents; inspect it on GitHub") from exc
            new_commit = prior
    if reference:
        check_window(reference)  # Network preparation must not carry a submission past the deadline.
    result = gh_json(["api", f"repos/{UPSTREAM}/pulls", "--method", "POST"],
                     {"title": title, "head": f"{login}:{branch}", "base": "main", "body": body})
    url = result["html_url"]
    receipt = {"url": url, "branch": branch, "commit": new_commit["sha"],
               "submitted_at": utc_now(), "files": {name: digest(path) for name, path in files.items()}}
    write_json(root / ".runtime" / (branch.replace("/", "-") + ".json"), receipt)
    return url


def register(root: Path) -> str:
    fresh_contract(root).validate_metadata(root / "model-metadata")
    files = {f"model-metadata/{model}.yml": root / "model-metadata" / f"{model}.yml" for model in MODELS}
    return create_pr(root, files, branch="main",
                     title="Update MIGHTE model metadata for 2026–27",
                     body=(root / "docs/REGISTRATION_PR.md").read_text())


def submit(root: Path, run: Path) -> str:
    manifest = verify_run(root, run, for_submission=True)
    contract = fresh_contract(root)
    contract.validate_metadata(root / "model-metadata")
    files = {}
    for filename in manifest["output_hashes"]:
        path = run / filename
        contract.validate(read_forecast(path), path.parent.name, manifest["reference_date"])
        # Metadata must already exist upstream before submitting a weekly forecast.
        meta = requests.get(HUB + f"model-metadata/{path.parent.name}.yml", timeout=(15, 40))
        if meta.status_code == 404:
            raise ValueError("Register the new models first with ./mighte register; wait for CDC to merge the metadata PR")
        meta.raise_for_status()
        files[filename] = path
    reference = manifest["reference_date"]
    models = manifest.get("models", list(MODELS))
    title = f"MIGHTE forecasts for {reference}"
    body = f"{reference} forecasts from {', '.join(models)}."
    url = create_pr(root, files, branch="main", title=title, body=body, reference=reference)
    write_json(run / "submission.json", {"url": url, "submitted_at": utc_now(), "output_hashes": manifest["output_hashes"]})
    return url
