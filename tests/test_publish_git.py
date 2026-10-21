import hashlib
import json
import subprocess

import pytest

from mighte import publish


@pytest.fixture
def local_transport(tmp_path, monkeypatch):
    original = subprocess.run
    root, remote = tmp_path / "checkout", tmp_path / "remote.git"
    root.mkdir()
    original(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
    original(["git", "-C", str(root), "config", "user.email", "austin.g.meyer@gmail.com"], check=True)
    original(["git", "init", "--bare", "--quiet", str(remote)], check=True, capture_output=True)
    (root / "uncommitted.txt").write_text("Keep local work unchanged\n")
    commands = []

    def run(arguments, **kwargs):
        commands.append(arguments)
        assert not any(arg.startswith("--force") for arg in arguments)
        arguments = [str(remote) if arg == "https://github.com/owner/flusight_2026.git" else arg
                     for arg in arguments]
        return original(arguments, **kwargs)

    def github(arguments, payload=None):
        assert arguments == ["api", "user"] and payload is None
        return {"login": "ausmeyer", "name": "Austin G. Meyer", "id": 123}

    monkeypatch.setattr(publish.subprocess, "run", run)
    monkeypatch.setattr(publish, "gh_json", github)

    def remote_git(*arguments):
        return original(["git", "--git-dir", str(remote), *arguments],
                        capture_output=True, check=True).stdout

    return root, remote_git, commands


def publication(html):
    return {"generator": publish.GENERATOR, "reference_date": "2026-10-24",
            "report_sha256": hashlib.sha256(html).hexdigest()}


@pytest.mark.parametrize("existing", [False, True])
def test_large_dashboard_transfers_exact_bytes_and_only_site_files(local_transport, existing):
    root, remote_git, commands = local_transport
    head = publish._push_dashboard(root, "owner/flusight_2026", b"old", publication(b"old"), None) if existing else None
    # Larger than the report that failed through the base64 JSON API upload.
    html = b"<!doctype html>\n" + b"x" * (36 * 1024 * 1024)
    metadata = publication(html)
    commit = publish._push_dashboard(root, "owner/flusight_2026", html, metadata, head)
    assert remote_git("rev-parse", "refs/heads/dashboard").decode().strip() == commit
    assert remote_git("show", f"{commit}:index.html") == html
    assert json.loads(remote_git("show", f"{commit}:publication.json")) == metadata
    assert set(remote_git("ls-tree", "--name-only", commit).decode().splitlines()) == publish.SITE_FILES
    assert remote_git("log", "-1", "--format=%an|%ae", commit).decode().strip() == (
        "Austin G. Meyer|austin.g.meyer@gmail.com")
    parents = remote_git("rev-list", "--parents", "-n", "1", commit).decode().split()
    assert parents == ([commit, head] if head else [commit])
    assert (root / "uncommitted.txt").read_text() == "Keep local work unchanged\n"
    assert any("credential.helper=!gh auth git-credential" in call for call in commands)


def test_remote_change_before_fetch_stops_publication(local_transport):
    root, remote_git, commands = local_transport
    old = publish._push_dashboard(root, "owner/flusight_2026", b"old", publication(b"old"), None)
    current = publish._push_dashboard(root, "owner/flusight_2026", b"current", publication(b"current"), old)
    pushes = sum("push" in call for call in commands)
    with pytest.raises(ValueError, match="branch changed"):
        publish._push_dashboard(root, "owner/flusight_2026", b"stale", publication(b"stale"), old)
    assert sum("push" in call for call in commands) == pushes
    assert remote_git("rev-parse", "refs/heads/dashboard").decode().strip() == current


def test_remote_change_after_fetch_is_not_overwritten(local_transport, monkeypatch):
    root, remote_git, _ = local_transport
    old = publish._push_dashboard(root, "owner/flusight_2026", b"old", publication(b"old"), None)
    original = publish.subprocess.run
    competing = []

    def run(arguments, **kwargs):
        if "push" in arguments and not competing:
            competing.append(None)
            competing[0] = publish._push_dashboard(
                root, "owner/flusight_2026", b"current", publication(b"current"), old)
        return original(arguments, **kwargs)

    monkeypatch.setattr(publish.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="GitHub dashboard transfer failed"):
        publish._push_dashboard(root, "owner/flusight_2026", b"stale", publication(b"stale"), old)
    assert remote_git("rev-parse", "refs/heads/dashboard").decode().strip() == competing[0]
    assert remote_git("show", f"{competing[0]}:index.html") == b"current"
