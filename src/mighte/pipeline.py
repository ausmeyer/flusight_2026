from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contract import COLUMNS, ED, HOSP, TREND, MODELS, EASTERN, Contract, check_window, read_forecast
from .data import prepare_inputs, refresh, verify_snapshot
from .models import distributional, equal_quantiles, linear, persistence_baseline
from .util import code_hash, digest, utc_now, write_json


TARGET_LABELS = {HOSP: "hospital admissions", ED: "ED visits", TREND: "hospitalization trends"}
COMPONENT_INPUTS = {"linear": ("hosp",), "base-hospitalizations": ("hosp", "ww", "nssp"),
                    "ww-hospitalizations": ("hosp", "ww"), "nssp-hospitalizations": ("hosp", "nssp"),
                    "base-ed": ("ed", "ww"), "base-ordinal": ("hosp", "ww", "nssp")}


def environment() -> dict:
    return {"python": platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "lightgbm"]}}


def input_hashes(root: Path, settings: dict) -> dict:
    paths = sorted((root / "data/historical").glob("*.csv"))
    return {str(p.relative_to(root)): digest(p) for p in paths}


def write_forecast(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame[COLUMNS].sort_values(["target", "location", "horizon", "output_type", "output_type_id"]).to_csv(
        temp, index=False, float_format="%.12g")
    temp.replace(path)


def fit_component(name, target, signals, run, snapshot, reference, settings, location_map):
    """Each process owns one component's checkpoints and output file."""
    path = run / "components" / f"{name}.csv"
    if path.exists():
        return read_forecast(path)
    history = pd.read_csv(run / ("hospitalization-training.csv" if target == HOSP else "ed-training.csv"),
                          parse_dates=["date"])
    with threadpool_limits(limits=int(settings["threads"])):
        result = distributional(history, snapshot, reference, settings, signals, target,
                                name, run / "checkpoints" / name, location_map)
    write_forecast(path, result)
    return read_forecast(path)


def run_forecasts(root: Path, reference: str, *, preview=False, quick=False,
                  resume: Path | None = None, snapshot: Path | None = None) -> Path:
    settings = json.loads((root / "config/settings.json").read_text())
    if quick and not preview:
        raise ValueError("Reduced fits are permitted only in a non-submittable preview")
    if resume:
        run = resume
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["code_sha256"] != code_hash(root) or manifest["environment"] != environment():
            raise ValueError("Code or numerical environment changed; start a new run")
        if manifest["settings_sha256"] != digest(root / "config/settings.json"):
            raise ValueError("Settings changed; start a new run")
        if manifest["historical_input_hashes"] != input_hashes(root, settings):
            raise ValueError("Historical inputs changed; start a new run")
        reference, preview = manifest["reference_date"], manifest["preview"]
        settings = manifest["settings"]
        snapshot = root / "data/snapshots" / manifest["snapshot_id"]
        if manifest["status"] == "complete":
            verify_run(root, run)
            return run
    else:
        if not preview:
            check_window(reference)
            if reference not in Contract(root / "hub-contract").by_target[HOSP]["task_ids"]["reference_date"]["optional"]:
                raise ValueError("Reference date is outside FluSight rounds. Use ./mighte preview before the season.")
        snapshot = snapshot or refresh(root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = root / "runs" / ("previews" if preview else "prospective") / reference / stamp
        run.mkdir(parents=True)
        effective = json.loads(json.dumps(settings))
        effective["runtime"]["num_threads"] = settings["threads"]
        if quick:
            effective["runtime"].update(num_bags=2, stage1_rounds=25, stage2_rounds=15)
        manifest = {"run_id": str(run.relative_to(root / "runs")), "reference_date": reference,
                    "created_at": utc_now(), "status": "running", "preview": preview,
                    "quick": quick, "settings": effective,
                    "settings_sha256": digest(root / "config/settings.json"),
                    "code_sha256": code_hash(root), "environment": environment(),
                    "historical_input_hashes": input_hashes(root, settings), "snapshot_id": snapshot.name,
                    "snapshot_manifest_sha256": digest(snapshot / "manifest.json")}
        settings = effective
        write_json(run / "manifest.json", manifest)
    verify_snapshot(snapshot)
    if digest(snapshot / "manifest.json") != manifest["snapshot_manifest_sha256"]:
        raise ValueError("Input snapshot manifest changed")
    if not preview:
        check_window(reference)
        # Resumes never mix fresh data into an existing fit; refetch by starting a new run.
        retrieved_day = pd.Timestamp(verify_snapshot(snapshot)["completed_at"]).tz_convert(EASTERN).date()
        if retrieved_day < (pd.Timestamp(reference) - pd.Timedelta(days=3)).date():
            raise ValueError("Production forecasts require a fresh Wednesday data snapshot")
    contract = Contract(snapshot / "contract")
    location_map = dict(zip(contract.locations.location_name, contract.locations.location))
    names = dict(zip(contract.locations.location, contract.locations.location_name))
    histories, audit = prepare_inputs(root, snapshot, reference, settings)
    notices = audit["notices"]
    write_json(run / "data-audit.json", audit)
    components = run / "components"
    components.mkdir(exist_ok=True)
    for target, history in histories.items():
        history.to_csv(run / ("hospitalization-training.csv" if target == HOSP else "ed-training.csv"), index=False)

    # A component runs only when its target and covariates are available this week.
    available = {"hosp": HOSP in histories, "ed": ED in histories,
                 **{key: audit["covariates"][key]["available"] for key in ["ww", "nssp"]}}
    runnable = {name for name, needs in COMPONENT_INPUTS.items() if all(available[x] for x in needs)}
    results = {}
    if "linear" in runnable:
        with threadpool_limits(limits=int(settings["threads"])):
            linear_file = components / "linear.csv"
            if not linear_file.exists():
                print("Fitting MIGHTE-Linear...", flush=True)
                write_forecast(linear_file, linear(histories[HOSP], reference, settings, location_map))
            results["linear"] = read_forecast(linear_file)
    jobs = [job for job in [("base-hospitalizations", HOSP, ("ww", "nssp")),
                            ("ww-hospitalizations", HOSP, ("ww",)),
                            ("nssp-hospitalizations", HOSP, ("nssp",)), ("base-ed", ED, ("ww",))]
            if job[0] in runnable]
    args = (run, snapshot, reference, settings, location_map)
    workers = int(settings.get("component_workers", 2))
    if workers not in {1, 2}:
        raise ValueError("component_workers must be 1 or 2")
    if workers == 1:
        fitted = [fit_component(*job, *args) for job in jobs]
    else:
        # Spawn avoids inheriting native OpenMP state from the linear fit.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            futures = [executor.submit(fit_component, *job, *args) for job in jobs]
            fitted = [future.result() for future in futures]
    results.update({job[0]: frame for job, frame in zip(jobs, fitted)})
    if settings.get("ordinal_plugin") and "base-ordinal" in runnable:
        module, name = settings["ordinal_plugin"].split(":")
        if not module.startswith("mighte."):
            raise ValueError("Keep ordinal plugin code inside this standalone mighte package")
        fn = getattr(importlib.import_module(module), name)
        ordinal_path = components / "base-ordinal.csv"
        ordinal_checkpoint = run / "checkpoints/base-ordinal"
        if not ordinal_path.exists():
            # Use the identical serialized history consumed by the boosted Base fit.
            ordinal_history = pd.read_csv(run / "hospitalization-training.csv", parse_dates=["date"])
            with threadpool_limits(limits=int(settings["threads"])):
                ordinal = fn({"reference_date": reference, "snapshot_path": snapshot,
                              "hospitalization_history": ordinal_history,
                              "base_hospitalization_quantiles": results["base-hospitalizations"].copy(),
                              "settings": settings, "checkpoint_path": ordinal_checkpoint})
            if ordinal.empty or set(ordinal.columns) != set(COLUMNS):
                raise ValueError("Enabled ordinal plugin must return nonempty hub-format rate-change forecasts")
            write_forecast(ordinal_path, ordinal)
        ordinal = read_forecast(ordinal_path)
        if ordinal.empty or set(ordinal.target) != {TREND} or set(ordinal.columns) != set(COLUMNS):
            raise ValueError("Enabled ordinal plugin must return nonempty hub-format rate-change forecasts")
        results["base-ordinal"] = ordinal
        if (ordinal_checkpoint / "fit.json").exists():
            audit["ordinal"] = json.loads((ordinal_checkpoint / "fit.json").read_text())
            write_json(run / "data-audit.json", audit)

    forecasts, expected_targets = {}, {}
    base_parts = {HOSP: "base-hospitalizations", ED: "base-ed"}
    if settings.get("ordinal_plugin"):
        base_parts[TREND] = "base-ordinal"
    parts = {target: component for target, component in base_parts.items() if component in results}
    if parts:
        forecasts["MIGHTE-Base"] = pd.concat([results[c] for c in parts.values()], ignore_index=True)
        expected_targets["MIGHTE-Base"] = set(parts)
        if set(parts) != set(base_parts):
            notices.append("MIGHTE-Base produced without " + ", ".join(
                TARGET_LABELS[t] for t in base_parts if t not in parts))
    if "linear" in results:
        forecasts["MIGHTE-Linear"] = results["linear"]
        expected_targets["MIGHTE-Linear"] = {HOSP}
    if {"ww-hospitalizations", "nssp-hospitalizations", "linear"} <= set(results):
        forecasts["MIGHTE-Nsemble"] = equal_quantiles(
            [results["ww-hospitalizations"], results["nssp-hospitalizations"], results["linear"]])
        expected_targets["MIGHTE-Nsemble"] = {HOSP}
    missing_models = [model for model in MODELS if model not in forecasts]
    if missing_models:
        notices.append("Not produced this week: " + ", ".join(missing_models))
    if not forecasts:
        raise ValueError("No forecasts could be produced:\n- " + "\n- ".join(notices))

    validation = {}
    for model, frame in forecasts.items():
        hosp = frame.target.eq(HOSP)
        frame.loc[hosp, "value"] = np.rint(frame.loc[hosp, "value"])
        # The hub flags these quantiles for review; remove only the affected location-horizons.
        flagged = contract.implausible_units(frame)
        if not flagged.empty:
            frame = frame.merge(flagged.assign(_flagged=True), on=["target", "location", "horizon"], how="left")
            frame = frame[frame.pop("_flagged").isna()].reset_index(drop=True)
            for target, group in flagged.groupby("target"):
                units = "; ".join(f"{names[location]} h{','.join(str(h) for h in sorted(g.horizon))}"
                                  for location, g in group.groupby("location"))
                notices.append(f"{model} {TARGET_LABELS[target]}: removed {units} "
                               "(quantiles above the hub's plausibility limit)")
        for target in expected_targets[model]:
            expected = set(audit[HOSP if target == TREND else target]["locations"])
            for horizon in range(4):
                removed = set(flagged.loc[flagged.target.eq(target) & flagged.horizon.eq(horizon), "location"])
                actual = set(frame.loc[frame.target.eq(target) & frame.horizon.eq(horizon), "location"])
                if actual != expected - removed:
                    raise ValueError(f"Incomplete forecast coverage in {model}, {target}, horizon {horizon}")
        path = run / "model-output" / model / f"{reference}-{model}.csv"
        write_forecast(path, frame)
        # Validate the serialized bytes, which are exactly what will be submitted.
        validation[model] = contract.validate(read_forecast(path), model, reference, preview=preview)
    baseline = pd.concat([persistence_baseline(history, reference, target, location_map, audit[target]["locations"])
                          for target, history in histories.items()])
    write_forecast(run / "local-baseline.csv", baseline)
    if not preview:
        check_window(reference)
    manifest.update(status="complete", completed_at=utc_now(), validation=validation,
                    models=sorted(forecasts), notices=notices,
                    output_hashes={str(p.relative_to(run)): digest(p) for p in (run / "model-output").rglob("*.csv")},
                    baseline_sha256=digest(run / "local-baseline.csv"))
    write_json(run / "manifest.json", manifest)
    write_json(root / "runs" / ("latest-preview.json" if preview else "latest.json"),
               {"run_id": manifest["run_id"]})
    print(f"Validated {len(forecasts)} model file(s): {run}", flush=True)
    print_notices(notices)
    return run


def print_notices(notices: list[str]) -> None:
    if notices:
        print("Notices (these affect this week's output):\n- " + "\n- ".join(notices), flush=True)


def latest_run(root: Path, *, preview=False) -> Path:
    path = root / "runs" / ("latest-preview.json" if preview else "latest.json")
    if not path.exists():
        raise ValueError("No completed run. Generate forecasts first.")
    return root / "runs" / json.loads(path.read_text())["run_id"]


def verify_run(root: Path, run: Path, *, for_submission=False) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Run is incomplete")
    snapshot = root / "data/snapshots" / manifest["snapshot_id"]
    verify_snapshot(snapshot)
    if digest(snapshot / "manifest.json") != manifest["snapshot_manifest_sha256"]:
        raise ValueError("Snapshot manifest changed")
    contract = Contract(snapshot / "contract")
    for filename, expected in manifest["output_hashes"].items():
        path = run / filename
        if digest(path) != expected:
            raise ValueError(f"Forecast was edited after validation: {filename}")
        contract.validate(read_forecast(path), path.parent.name, manifest["reference_date"],
                          preview=manifest["preview"])
    models = manifest.get("models", list(MODELS))
    if not models or not set(models) <= set(MODELS):
        raise ValueError("Run lists no valid submission model")
    expected_paths = {f"model-output/{model}/{manifest['reference_date']}-{model}.csv" for model in models}
    if set(manifest["output_hashes"]) != expected_paths:
        raise ValueError("Run files do not match its recorded models")
    if digest(run / "local-baseline.csv") != manifest["baseline_sha256"]:
        raise ValueError("Frozen local baseline changed")
    if for_submission:
        if manifest["preview"] or manifest["quick"] or manifest["settings"]["runtime"]["num_bags"] != 100:
            raise ValueError("Preview/reduced runs cannot be submitted")
        check_window(manifest["reference_date"])
        check_window(manifest["reference_date"], datetime.fromisoformat(manifest["created_at"]))
        check_window(manifest["reference_date"], datetime.fromisoformat(manifest["completed_at"]))
    return manifest
