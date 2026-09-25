from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import requests

from .contract import Contract, reference_saturday
from .data import refresh, latest_snapshot
from .pipeline import latest_run, run_forecasts, verify_run
from .report import build_report
from .submit import register, submit
from .util import project_root


def main():
    parser = argparse.ArgumentParser(description="MIGHTE weekly forecasts: forecast → review → submit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("refresh", help="Download complete current hub/CDC histories, including revisions")
    for command in ["forecast", "preview"]:
        p = sub.add_parser(command, help="Generate all three submission models" if command == "forecast"
                           else "Non-submittable rehearsal, including pre-season dates")
        p.add_argument("--reference-date", default=None)
        p.add_argument("--no-open", action="store_true")
        if command == "preview":
            p.add_argument("--quick", action="store_true", help="Two bags and fewer rounds; never submittable")
            p.add_argument("--snapshot", type=Path, help="Reuse a local data snapshot for this preview only")
    p = sub.add_parser("resume", help="Continue an interrupted run using its original immutable inputs")
    p.add_argument("run", type=Path)
    p.add_argument("--no-open", action="store_true")
    for command in ["review", "validate", "submit"]:
        p = sub.add_parser(command)
        p.add_argument("--run", type=Path, default=None)
        if command != "submit":
            p.add_argument("--preview", action="store_true")
        if command == "review":
            p.add_argument("--refresh", action="store_true", help="Refresh revised truth before recalculating accuracy")
            p.add_argument("--no-open", action="store_true")
            p.add_argument("--offline", action="store_true", help="Skip fetching public benchmark forecasts")
        if command == "submit":
            p.add_argument("--yes", action="store_true", help="Confirm that these exact files were reviewed")
    sub.add_parser("register", help="Open the metadata-only registration/update PR")
    sub.add_parser("check", help="Check local metadata, environment and fixed model settings")
    args = parser.parse_args()
    root = project_root()
    try:
        if args.command == "refresh":
            refresh(root)
            return
        if args.command == "check":
            Contract(root / "hub-contract").validate_metadata(root / "model-metadata")
            from .pipeline import environment
            print(json.dumps(environment(), indent=2))
            print("Metadata and standalone imports OK")
            return
        if args.command == "register":
            print(register(root))
            return
        if args.command in {"forecast", "preview", "resume"}:
            run = run_forecasts(root, getattr(args, "reference_date", None) or reference_saturday(),
                                preview=args.command == "preview", quick=getattr(args, "quick", False),
                                resume=args.run if args.command == "resume" else None,
                                snapshot=getattr(args, "snapshot", None))
            path = build_report(root, run)
        elif args.command == "review":
            if args.refresh:
                refresh(root)
            run = args.run or latest_run(root, preview=args.preview)
            path = build_report(root, run, online=not args.offline)
        elif args.command == "validate":
            run = args.run or latest_run(root, preview=args.preview)
            manifest = verify_run(root, run)
            print(json.dumps(manifest["validation"], indent=2))
            return
        else:
            run = args.run or latest_run(root)
            manifest = verify_run(root, run, for_submission=True)
            print(f"Submit the three reviewed model files for {manifest['reference_date']}?\nRun: {run}")
            if not args.yes and input("Type submit to open the CDC pull request: ").strip() != "submit":
                print("Submission cancelled.")
                return
            print(submit(root, run))
            return
        print(f"Review: {path}")
        if not args.no_open:
            webbrowser.open(path.resolve().as_uri())
    except (ValueError, RuntimeError, FileNotFoundError, requests.RequestException) as exc:
        print(f"MIGHTE: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
