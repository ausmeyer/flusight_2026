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
REGISTRATION_MODELS = (*MODELS, "MIGHTE-Joint")


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
    """The user's open hub PR on this branch or on a timestamped retry of it."""
    page = 1
    while True:
        pulls = gh_json(["api", f"repos/{UPSTREAM}/pulls?state=open&per_page=100&page={page}"])
        for pr in pulls:
            ref = pr["head"]["ref"]
            if pr["user"]["login"] == login and (ref == branch or ref.startswith(branch + "-")):
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
    """Update the open PR for this branch, or build a new commit from upstream main."""
    if not files or any(not (name in {f"model-metadata/{m}.yml" for m in REGISTRATION_MODELS}
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
        return update_pr(root, files, existing, fork, title=title, body=body, reference=reference)
    history = gh_json(["api", f"repos/{UPSTREAM}/pulls?state=all&head={login}:{branch}"])
    if history:
        # A deliberately closed PR stays closed. A later, explicitly requested
        # submission uses a new branch instead of reopening the old discussion.
        branch += "-" + utc_now().replace("-", "").replace(":", "").replace(".", "").split("+")[0]
    head = gh_json(["api", f"repos/{UPSTREAM}/git/ref/heads/main"])["object"]["sha"]
    commit = gh_json(["api", f"repos/{UPSTREAM}/git/commits/{head}"])
    # Obtain the upstream commit in the fork without resetting any branch.
    # Fork networks share Git objects; the tree is rooted at the fresh upstream SHA.
    tree_entries = []
    for name, path in sorted(files.items()):
        blob = gh_json(["api", f"repos/{fork}/git/blobs", "--method", "POST"],
                       {"content": base64.b64encode(path.read_bytes()).decode(), "encoding": "base64"})
        tree_entries.append({"path": name, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    tree = gh_json(["api", f"repos/{fork}/git/trees", "--method", "POST"],
                   {"base_tree": commit["tree"]["sha"], "tree": tree_entries})
    new_commit = gh_json(["api", f"repos/{fork}/git/commits", "--method", "POST"],
                         {"message": title, "tree": tree["sha"], "parents": [head]})
    # Unique branch names prevent overwriting earlier work, including failed/rejected PRs.
    try:
        gh_json(["api", f"repos/{fork}/git/refs", "--method", "POST"],
                {"ref": "refs/heads/" + branch, "sha": new_commit["sha"]})
    except RuntimeError as exc:
        # Recover an interrupted PR creation only if the existing branch is exactly
        # the intended tree. Never reset or overwrite an unrelated branch.
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
    fresh_contract(root).validate_metadata(root / "model-metadata", models=REGISTRATION_MODELS)
    files = {f"model-metadata/{model}.yml": root / "model-metadata" / f"{model}.yml" for model in REGISTRATION_MODELS}
    return create_pr(root, files, branch="codex/mighte-2026-27-metadata",
                     title="Register MIGHTE-Base and MIGHTE-Linear; update MIGHTE metadata for 2026–27",
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
    notices = manifest.get("notices", [])
    title = f"MIGHTE forecasts for {reference}"
    body = (f"Prospective {reference} forecasts from {', '.join(models)}.\n\n"
            "MIGHTE-Base combines its targets in one file; the other models provide hospitalization forecasts. "
            "No peak timing or peak height targets are included.\n\n"
            + ("This week:\n" + "".join(f"- {notice}\n" for notice in notices) + "\n" if notices else "")
            + "All files passed local validation against the current hub configuration. "
            f"Inputs were frozen in snapshot `{manifest['snapshot_id']}` and reviewed locally before submission.")
    url = create_pr(root, files, branch=f"codex/mighte-{reference}", title=title, body=body, reference=reference)
    write_json(run / "submission.json", {"url": url, "submitted_at": utc_now(), "output_hashes": manifest["output_hashes"]})
    return url
