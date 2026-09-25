"""Score only on-time prospective runs against the latest observed truth."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .contract import CATEGORIES, ED, HOSP, MODELS, QUANTILES, TREND, UNIT, check_window, read_forecast
from .data import API, HUB
from .pipeline import verify_run
from .util import digest, utc_now, write_json

LOCAL_BASELINE = "Local-persistence"
BENCHMARKS = ("FluSight-baseline", "FluSight-ensemble", "UMass-flusion", "Google_SAI-FluEns")


def discover_benchmarks(root: Path, reference_dates, *, season: str, online=True):
    """Discover actual season submissions, not merely registered model names."""
    path = root / "data/benchmarks/catalog.json"
    catalog = json.loads(path.read_text()) if path.exists() else None
    status = {"model": "Season model catalog", "status": "cached (offline)" if catalog else "not cached"}
    if online:
        try:
            response = requests.get(API + "commits/main", timeout=(15, 40))
            response.raise_for_status()
            revision = response.json()["sha"]
            if catalog and catalog["revision"] == revision:
                status["status"] = "unchanged"
            else:
                response = requests.get(API + f"git/trees/{revision}", params={"recursive": "1"}, timeout=(15, 40))
                response.raise_for_status()
                tree = response.json()
                if tree.get("truncated"):
                    raise ValueError("Hub model catalog is truncated; refusing a partial model list")
                files = []
                for entry in tree["tree"]:
                    match = re.fullmatch(r"model-output/([A-Za-z0-9][A-Za-z0-9_.-]*)/(\d{4}-\d{2}-\d{2})-\1\.csv", entry["path"])
                    if match and entry["type"] == "blob":
                        files.append({"model": match[1], "reference_date": match[2], "blob_sha": entry["sha"]})
                catalog = {"revision": revision, "retrieved_at": utc_now(), "files": files}
                write_json(path, catalog)
                status["status"] = "refreshed"
        except (requests.RequestException, ValueError) as exc:
            status.update(status="cached (refresh failed)" if catalog else "unavailable", reason=str(exc))
    if catalog is None:
        return None, status
    start_year, end_year = season.split("-")
    # The hub's allowed dates also contain previous seasons.
    dates = {day for day in reference_dates if f"{start_year}-07-01" <= day < f"{end_year}-07-01"}
    return {**catalog, "files": [f for f in catalog["files"] if f["reference_date"] in dates]}, status


def prospective_runs(root: Path) -> list[Path]:
    selected = {}
    for path in sorted((root / "runs/prospective").glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("preview", True) or manifest.get("status") != "complete":
            continue
        reference = manifest["reference_date"]
        try:
            check_window(reference, datetime.fromisoformat(manifest["created_at"]))
            check_window(reference, datetime.fromisoformat(manifest["completed_at"]))
        except ValueError:
            continue
        receipt = path.parent / "submission.json"
        # Submitted forecasts take precedence. Otherwise use the last on-time run.
        submitted = json.loads(receipt.read_text()) if receipt.exists() else {}
        rank = (bool(submitted), submitted.get("submitted_at", manifest["completed_at"]))
        if reference not in selected or rank > selected[reference][0]:
            selected[reference] = (rank, path.parent)
    return [value[1] for _, value in sorted(selected.items())]


def load_archive(root: Path) -> pd.DataFrame:
    frames = []
    for run in prospective_runs(root):
        manifest = verify_run(root, run)
        for filename in manifest["output_hashes"]:
            path = run / filename
            frames.append(read_forecast(path).assign(model_id=path.parent.name))
        frames.append(read_forecast(run / "local-baseline.csv").assign(model_id=LOCAL_BASELINE))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_benchmarks(root: Path, references: list[str], *, online=True, catalog=None) -> tuple[pd.DataFrame, list[dict]]:
    frames, status = [], []
    models = sorted(set(BENCHMARKS) | {f["model"] for f in catalog["files"]}) if catalog else BENCHMARKS
    published = {(f["model"], f["reference_date"]): f["blob_sha"] for f in catalog["files"]} if catalog else {}
    for reference in sorted(set(references)):
        for model in models:
            if model in MODELS:  # The immutable local runs are authoritative for our models.
                continue
            blob = published.get((model, reference))
            if catalog is not None and blob is None:
                status.append({"model": model, "reference_date": reference, "status": "not published"})
                continue
            filename = f"{reference}-{model}.csv"
            path = root / "data/benchmarks" / model / filename
            source_path = path.with_suffix(".source.json")
            source = json.loads(source_path.read_text()) if source_path.exists() else {}
            unchanged = bool(blob and blob == source.get("blob_sha") and path.exists()
                             and digest(path) == source.get("sha256"))
            base = HUB.removesuffix("main/") + catalog["revision"] + "/" if catalog else HUB
            url = base + f"model-output/{model}/{filename}"
            try:
                if online and not unchanged:
                    response = requests.get(url, timeout=(15, 40))
                    if response.status_code == 404:
                        status.append({"model": model, "reference_date": reference, "status": "not published"})
                        continue
                    response.raise_for_status()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(response.content)
                    write_json(source_path, {"url": url, "retrieved_at": utc_now(),
                                            "sha256": digest(path), "blob_sha": blob})
                    state = "refreshed"
                elif path.exists():
                    state = "unchanged" if online else "cached (offline)"
                else:
                    status.append({"model": model, "reference_date": reference, "status": "not cached"})
                    continue
            except requests.RequestException as exc:
                state = "cached (refresh failed)" if path.exists() else "unavailable"
                status.append({"model": model, "reference_date": reference, "status": state, "reason": str(exc)})
                if not path.exists():
                    continue
            frame = read_forecast(path)
            frame = frame[frame.target.isin([HOSP, ED]) & frame.horizon.isin([0, 1, 2, 3])
                          & frame.output_type.eq("quantile")]
            if not frame.empty:
                frames.append(frame.assign(model_id=model))
            status.append({"model": model, "reference_date": reference, "status": state})
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), status


def wis(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    error = y[:, None] - q
    return (2 * np.maximum(QUANTILES[None, :] * error, (QUANTILES[None, :] - 1) * error)).mean(axis=1)


def score_quantiles(forecasts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    frame = forecasts[forecasts.output_type.eq("quantile")].copy()
    frame["output_type_id"] = pd.to_numeric(frame.output_type_id)
    keys = ["model_id", *UNIT]
    if frame.duplicated(keys + ["output_type_id"]).any():
        raise ValueError("Duplicate forecasts would inflate accuracy denominators")
    matrix = frame.pivot(index=keys, columns="output_type_id", values="value")
    matrix = matrix.reindex(columns=QUANTILES).dropna().reset_index()
    observed = truth[["target", "location", "date", "value"]].rename(
        columns={"date": "target_end_date", "value": "truth"})
    scored = matrix.merge(observed, on=["target", "location", "target_end_date"],
                           how="inner", validate="many_to_one").dropna(subset=["truth"])
    if scored.empty:
        return scored
    q = scored[list(QUANTILES)].to_numpy(float)
    y = scored.truth.to_numpy(float)
    if not np.isfinite(q).all() or not np.isfinite(y).all() or (q < 0).any() or (y < 0).any():
        raise ValueError("Invalid values in scoring inputs")
    scored["wis"] = wis(y, q)
    scored["wis_log1p"] = wis(np.log1p(y), np.log1p(q))
    scored["ae"] = abs(y - scored[.5])
    scored["se"] = (y - scored[.5]) ** 2
    scored["bias"] = scored[.5] - y
    for coverage, lo, hi in [(50, .25, .75), (80, .1, .9), (95, .025, .975)]:
        scored[f"coverage_{coverage}"] = ((y >= scored[lo]) & (y <= scored[hi])).astype(float)
    return scored


def geometric(values) -> float:
    values = np.asarray(values, dtype=float)
    return 0. if (values == 0).any() else float(np.exp(np.log(values).mean()))


def relative_scores(group: pd.DataFrame, metric: str, baseline: str) -> dict:
    """scoringutils-style geometric mean of pairwise ratios; keys include target."""
    models = group.model_id.unique()
    theta = {}
    for model in models:
        ratios = []
        for other in models:
            a = group[group.model_id.eq(model)][UNIT + [metric]]
            b = group[group.model_id.eq(other)][UNIT + [metric]]
            pair = a.merge(b, on=UNIT, suffixes=("_a", "_b"), validate="one_to_one")
            den = pair[f"{metric}_b"].sum()
            if pair.empty or den <= 0:
                ratios.append(np.nan)
            else:
                ratios.append(pair[f"{metric}_a"].sum() / den)
        theta[model] = geometric(ratios) if np.isfinite(ratios).all() else np.nan
    denominator = theta.get(baseline, np.nan)
    return {k: v / denominator if denominator > 0 else np.nan for k, v in theta.items()}


def summarize(scored: pd.DataFrame, *, baseline=LOCAL_BASELINE, location="states", horizon="all") -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    work = scored.copy()
    if location == "states":
        work = work[work.location.ne("US")]
    elif location != "all":
        work = work[work.location.eq(location)]
    if horizon != "all":
        work = work[work.horizon.eq(int(horizon))]
    records = []
    for target, target_group in work.groupby("target"):
        raw_relative = relative_scores(target_group, "wis", baseline)
        log_relative = relative_scores(target_group, "wis_log1p", baseline)
        base = target_group[target_group.model_id.eq(baseline)][UNIT + ["wis", "wis_log1p", "ae"]]
        for model, g in target_group.groupby("model_id"):
            pair = g.merge(base, on=UNIT, suffixes=("", "_baseline"), validate="one_to_one")
            den = pair.wis_baseline.sum()
            relative = float(pair.wis.sum() / den) if len(pair) and den > 0 else np.nan
            record = {"target": target, "model_id": model, "n": len(g), "matched_n": len(pair),
                      "mean_wis": g.wis.mean(), "geo_mean_wis": geometric(g.wis),
                      "geo_mean_wis_log1p": geometric(g.wis_log1p), "mean_mae": g.ae.mean(),
                      "rmse": np.sqrt(g.se.mean()), "bias": g.bias.mean(),
                      "relative_wis": relative, "wis_skill": 1 - relative,
                      "reichlab_relative_wis": raw_relative.get(model, np.nan),
                      "cdc_relative_wis": log_relative.get(model, np.nan),
                      **{f"coverage_{p}": g[f"coverage_{p}"].mean() for p in [50, 80, 95]}}
            records.append(record)
    return pd.DataFrame(records)


def evaluate(root: Path, snapshot: Path, *, online=True, comparison_references=(), catalog=None):
    archive = load_archive(root)
    prospective = [] if archive.empty else archive.reference_date.unique().tolist()
    references = sorted(set(prospective) | set(comparison_references))
    benchmarks, status = fetch_benchmarks(root, references, online=online, catalog=catalog)
    scoring_archive = archive
    if not benchmarks.empty:
        # A peer forecast can be displayed for a rehearsal, but only genuine
        # prospective MIGHTE weeks contribute to the accuracy comparison.
        eligible = benchmarks[benchmarks.reference_date.isin(prospective)]
        scoring_archive = pd.concat([archive, eligible], ignore_index=True)
        archive = pd.concat([archive, benchmarks], ignore_index=True)
    truth = pd.read_csv(snapshot / "truth.csv", dtype={"location": str})
    scores = score_quantiles(scoring_archive, truth)
    return scores, status, archive
