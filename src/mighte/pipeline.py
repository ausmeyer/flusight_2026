from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contract import COLUMNS, ED, HOSP, TREND, MODELS, EASTERN, Contract, check_window, read_forecast
from .data import prepare_inputs, refresh, verify_snapshot
from .models import distributional, equal_quantiles, linear, persistence_baseline
from .util import code_hash, digest, utc_now, write_json


def environment() -> dict:
    return {"python": platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "lightgbm"]}}


def input_hashes(root: Path, settings: dict) -> dict:
    paths = sorted((root / "data/historical").glob("*.csv"))
    if settings["ed_history"].get("historical_nssp_file"):
        paths.append(root / settings["ed_history"]["historical_nssp_file"])
    return {str(p.relative_to(root)): digest(p) for p in paths}


def write_forecast(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame[COLUMNS].sort_values(["target", "location", "horizon", "output_type", "output_type_id"]).to_csv(
        temp, index=False, float_format="%.12g")
    temp.replace(path)


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
    histories, audit = prepare_inputs(root, snapshot, reference, settings)
    write_json(run / "data-audit.json", audit)
    components = run / "components"
    components.mkdir(exist_ok=True)
    for target, history in histories.items():
        history.to_csv(run / ("hospitalization-training.csv" if target == HOSP else "ed-training.csv"), index=False)

    def fit_component(name, target, signals):
        path = components / f"{name}.csv"
        if path.exists():
            return read_forecast(path)
        result = distributional(histories[target], snapshot, reference, settings, signals, target,
                                name, run / "checkpoints" / name, location_map)
        write_forecast(path, result)
        return read_forecast(path)

    with threadpool_limits(limits=int(settings["threads"])):
        linear_file = components / "linear.csv"
        if not linear_file.exists():
            print("Fitting MIGHTE-Linear...", flush=True)
            write_forecast(linear_file, linear(histories[HOSP], reference, settings, location_map))
        linear_frame = read_forecast(linear_file)
        base_hosp = fit_component("base-hospitalizations", HOSP, ("ww", "nssp"))
        ww = fit_component("ww-hospitalizations", HOSP, ("ww",))
        nssp = fit_component("nssp-hospitalizations", HOSP, ("nssp",))
        ed = fit_component("base-ed", ED, ("ww",))
    forecasts = {"MIGHTE-Base": pd.concat([base_hosp, ed], ignore_index=True),
                 "MIGHTE-Linear": linear_frame,
                 "MIGHTE-Nsemble": equal_quantiles([ww, nssp, linear_frame])}
    if settings.get("ordinal_plugin"):
        module, name = settings["ordinal_plugin"].split(":")
        if not module.startswith("mighte."):
            raise ValueError("Keep ordinal plugin code inside this standalone mighte package")
        fn = getattr(importlib.import_module(module), name)
        ordinal = fn({"reference_date": reference, "snapshot_path": snapshot,
                      "hospitalization_history": histories[HOSP].copy(),
                      "base_hospitalization_quantiles": base_hosp.copy(), "settings": settings})
        if ordinal.empty or set(ordinal.target) != {TREND} or set(ordinal.columns) != set(COLUMNS):
            raise ValueError("Enabled ordinal plugin must return nonempty hub-format rate-change forecasts")
        forecasts["MIGHTE-Base"] = pd.concat([forecasts["MIGHTE-Base"], ordinal], ignore_index=True)
    validation = {}
    for model, frame in forecasts.items():
        hosp = frame.target.eq(HOSP)
        frame.loc[hosp, "value"] = np.rint(frame.loc[hosp, "value"])
        for target in [HOSP, ED] if model == "MIGHTE-Base" else [HOSP]:
            expected = set(audit[target]["locations"])
            for horizon in range(4):
                actual = set(frame.loc[frame.target.eq(target) & frame.horizon.eq(horizon), "location"])
                if actual != expected:
                    raise ValueError(f"Incomplete forecast coverage in {model}, {target}, horizon {horizon}")
        path = run / "model-output" / model / f"{reference}-{model}.csv"
        write_forecast(path, frame)
        # Validate the serialized bytes, which are exactly what will be submitted.
        validation[model] = contract.validate(read_forecast(path), model, reference, preview=preview)
    baseline = pd.concat([persistence_baseline(histories[t], reference, t, location_map) for t in [HOSP, ED]])
    write_forecast(run / "local-baseline.csv", baseline)
    if not preview:
        check_window(reference)
    manifest.update(status="complete", completed_at=utc_now(), validation=validation,
                    output_hashes={str(p.relative_to(run)): digest(p) for p in (run / "model-output").rglob("*.csv")},
                    baseline_sha256=digest(run / "local-baseline.csv"))
    write_json(run / "manifest.json", manifest)
    write_json(root / "runs" / ("latest-preview.json" if preview else "latest.json"),
               {"run_id": manifest["run_id"]})
    print(f"Validated all three model files: {run}", flush=True)
    return run


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
    expected_paths = {f"model-output/{model}/{manifest['reference_date']}-{model}.csv" for model in MODELS}
    if set(manifest["output_hashes"]) != expected_paths:
        raise ValueError("Run must contain exactly the three expected model files")
    if digest(run / "local-baseline.csv") != manifest["baseline_sha256"]:
        raise ValueError("Frozen local baseline changed")
    if for_submission:
        if manifest["preview"] or manifest["quick"] or manifest["settings"]["runtime"]["num_bags"] != 100:
            raise ValueError("Preview/reduced runs cannot be submitted")
        check_window(manifest["reference_date"])
        check_window(manifest["reference_date"], datetime.fromisoformat(manifest["created_at"]))
        check_window(manifest["reference_date"], datetime.fromisoformat(manifest["completed_at"]))
    return manifest
