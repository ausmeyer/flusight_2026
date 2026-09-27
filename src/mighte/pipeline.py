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
from .models import covariate_weeks, distributional, equal_quantiles, linear, persistence_baseline
from .util import code_hash, digest, utc_now, write_json


TARGET_LABELS = {HOSP: "hospital admissions", ED: "ED visits", TREND: "hospitalization trends"}
# Components with surveillance covariates; see the fallback in run_forecasts.
COMPONENTS = {"base-hospitalizations": (HOSP, ("ww", "nssp")), "ww-hospitalizations": (HOSP, ("ww",)),
              "nssp-hospitalizations": (HOSP, ("nssp",)), "base-ed": (ED, ("ww",))}
FALLBACK = {HOSP: "fallback-hospitalizations", ED: "fallback-ed"}
COMPONENT_LABELS = {"base-hospitalizations": "MIGHTE-Base admissions", "ww-hospitalizations": "the Nsemble wastewater part",
                    "nssp-hospitalizations": "the Nsemble NSSP part", "base-ed": "MIGHTE-Base ED visits",
                    "base-ordinal": "MIGHTE-Base trends"}


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


def fit_component(name, target, signals, run, snapshot, reference, settings, location_map, carry_forward=False):
    """Each process owns one component's checkpoints and output file."""
    path = run / "components" / f"{name}.csv"
    if path.exists():
        return read_forecast(path)
    history = pd.read_csv(run / ("hospitalization-training.csv" if target == HOSP else "ed-training.csv"),
                          parse_dates=["date"])
    with threadpool_limits(limits=int(settings["threads"])):
        result = distributional(history, snapshot, reference, settings, signals, target,
                                name, run / "checkpoints" / name, location_map, carry_forward)
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
    audit["covariates"] = covariate_weeks(snapshot, reference)
    for key, label in [("nssp", "National NSSP"), ("ww", "WastewaterSCAN")]:
        week = audit["covariates"][key]
        if week["available"] and week["week_used"] != week["expected_week"]:
            notices.append(f"{label} for {week['expected_week']} not yet available; used {week['week_used']} "
                           "(the model's one-week staleness allowance)")
    write_json(run / "data-audit.json", audit)
    components = run / "components"
    components.mkdir(exist_ok=True)
    for target, history in histories.items():
        history.to_csv(run / ("hospitalization-training.csv" if target == HOSP else "ed-training.csv"), index=False)

    # Fallback: MIGHTE-Base without wastewater and NSSP. It replaces any part whose covariate is
    # unavailable and forecasts locations without an anchor value from their last reported value.
    available = {key: audit["covariates"][key]["available"] for key in ["ww", "nssp"]}
    missing = {target: set(audit[target]["anchor_missing"]) for target in histories}
    jobs = {name: (target, signals, False) for name, (target, signals) in COMPONENTS.items()
            if target in histories and all(available[x] for x in signals)
            and len(missing[target]) < len(audit[target]["locations"])}
    for target in histories:
        if missing[target] or any(t == target and name not in jobs for name, (t, _) in COMPONENTS.items()):
            jobs[FALLBACK[target]] = (target, (), True)
    use_ordinal = bool(settings.get("ordinal_plugin")) and HOSP in histories
    replaced = [name for name, (t, _) in COMPONENTS.items() if t in histories and name not in jobs
                and not all(available.values())] + (["base-ordinal"] if use_ordinal and not all(available.values()) else [])
    if replaced:
        lost = " and ".join(label for key, label in [("nssp", "National NSSP"), ("ww", "WastewaterSCAN")]
                            if not available[key])
        notices.append(f"{lost} unavailable: " + ", ".join(COMPONENT_LABELS[n] for n in replaced)
                       + " used MIGHTE-Base without wastewater and NSSP")
    linear_frame = None
    if HOSP in histories and len(missing[HOSP]) < len(audit[HOSP]["locations"]):
        with threadpool_limits(limits=int(settings["threads"])):
            linear_file = components / "linear.csv"
            if not linear_file.exists():
                print("Fitting MIGHTE-Linear...", flush=True)
                write_forecast(linear_file, linear(histories[HOSP], reference, settings, location_map))
            linear_frame = read_forecast(linear_file)
    args = (run, snapshot, reference, settings, location_map)
    workers = int(settings.get("component_workers", 2))
    if workers not in {1, 2}:
        raise ValueError("component_workers must be 1 or 2")
    if workers == 1:
        results = {name: fit_component(name, target, signals, *args, carry)
                   for name, (target, signals, carry) in jobs.items()}
    else:
        # Spawn avoids inheriting native OpenMP state from the linear fit.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            futures = {name: executor.submit(fit_component, name, target, signals, *args, carry)
                       for name, (target, signals, carry) in jobs.items()}
            results = {name: future.result() for name, future in futures.items()}

    def part(frame, target):
        """A part's forecasts for anchored locations, plus the fallback wherever it is missing."""
        wanted = set(audit[target]["locations"])
        fallback = results.get(FALLBACK[target])
        if frame is None:
            return fallback[fallback.location.isin(wanted)]
        frame = frame[~frame.location.isin(missing[target])]
        if missing[target]:
            frame = pd.concat([frame, fallback[fallback.location.isin(missing[target])]], ignore_index=True)
        return frame

    def classify(name, signals, locations):
        path, checkpoint = components / f"{name}.csv", run / "checkpoints" / name
        if not path.exists():
            # Use the identical serialized history consumed by the boosted Base fit.
            history = pd.read_csv(run / "hospitalization-training.csv", parse_dates=["date"])
            with threadpool_limits(limits=int(settings["threads"])):
                frame = fn({"reference_date": reference, "snapshot_path": snapshot, "hospitalization_history": history,
                            "base_hospitalization_quantiles": part(results.get("base-hospitalizations"), HOSP),
                            "settings": settings, "checkpoint_path": checkpoint, "signals": signals,
                            "locations": locations})
            if frame.empty or set(frame.columns) != set(COLUMNS):
                raise ValueError("Enabled ordinal plugin must return nonempty hub-format rate-change forecasts")
            write_forecast(path, frame)
        frame = read_forecast(path)
        if frame.empty or set(frame.target) != {TREND} or set(frame.columns) != set(COLUMNS):
            raise ValueError("Enabled ordinal plugin must return nonempty hub-format rate-change forecasts")
        return frame

    ordinal = None
    if use_ordinal:
        module, name = settings["ordinal_plugin"].split(":")
        if not module.startswith("mighte."):
            raise ValueError("Keep ordinal plugin code inside this standalone mighte package")
        fn = getattr(importlib.import_module(module), name)
        anchored = [names[x] for x in audit[HOSP]["locations"] if x not in missing[HOSP]]
        fallback_locations = [names[x] for x in missing[HOSP]]
        # The fallback (no wastewater or NSSP) classifies locations without an anchor value, and all
        # locations when a covariate is unavailable.
        if all(available.values()):
            calls = [("base-ordinal", ("ww", "nssp"), anchored), ("base-ordinal-fallback", (), fallback_locations)]
        else:
            calls = [("base-ordinal", (), anchored + fallback_locations)]
        ordinal = pd.concat([classify(*call) for call in calls if call[2]], ignore_index=True)
        if (run / "checkpoints/base-ordinal/fit.json").exists():
            audit["ordinal"] = json.loads((run / "checkpoints/base-ordinal/fit.json").read_text())
            write_json(run / "data-audit.json", audit)

    forecasts = {}
    if HOSP in histories:
        base = [part(results.get("base-hospitalizations"), HOSP)]
        if ED in histories:
            base.append(part(results.get("base-ed"), ED))
        if ordinal is not None:
            base.append(ordinal)
        forecasts["MIGHTE-Base"] = pd.concat(base, ignore_index=True)
        forecasts["MIGHTE-Linear"] = part(linear_frame, HOSP)
        ensemble = None if linear_frame is None else equal_quantiles(
            [part(results.get(name), HOSP)[lambda f: ~f.location.isin(missing[HOSP])]
             for name in ["ww-hospitalizations", "nssp-hospitalizations"]] + [linear_frame])
        forecasts["MIGHTE-Nsemble"] = part(ensemble, HOSP)
    if not forecasts:
        raise ValueError("No forecasts could be produced:\n- " + "\n- ".join(notices))

    validation = {}
    for model, frame in forecasts.items():
        hosp = frame.target.eq(HOSP)
        frame.loc[hosp, "value"] = np.rint(frame.loc[hosp, "value"])
        # The hub flags quantiles above ED 0.25 or admissions of 30% of population; review, not changed.
        flagged = contract.implausible_units(frame)
        for target, group in flagged.groupby("target"):
            units = "; ".join(f"{names[location]} h{','.join(str(h) for h in sorted(g.horizon))}"
                              for location, g in group.groupby("location"))
            notices.append(f"{model} {TARGET_LABELS[target]}: {units} above the hub's plausibility limit; "
                           "review before submitting")
        targets = {HOSP} | ({ED} if model == "MIGHTE-Base" and ED in histories else set()) | (
            {TREND} if model == "MIGHTE-Base" and ordinal is not None else set())
        for target in targets:
            expected = set(audit[HOSP if target == TREND else target]["locations"])
            for horizon in range(4):
                actual = set(frame.loc[frame.target.eq(target) & frame.horizon.eq(horizon), "location"])
                if actual != expected:
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
