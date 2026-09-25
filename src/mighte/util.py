from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def project_root() -> Path:
    # The launcher sets an explicit root; plain CLI use searches from the user's cwd.
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "config/settings.json").is_file() and (path / "hub-contract").is_dir():
            return path
    raise ValueError("Run mighte from the downloaded flusight_2026 repository")


def code_hash(root: Path) -> str:
    files = sorted((root / "src/mighte").glob("*.py"))
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()
