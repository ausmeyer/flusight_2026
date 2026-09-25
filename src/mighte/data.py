"""Fetch complete revised surveillance histories and preserve each retrieval."""
from __future__ import annotations

import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .contract import Contract, HOSP, ED
from .util import digest, utc_now, write_json

HUB = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/"
API = "https://api.github.com/repos/cdcepi/FluSight-forecast-hub/"
CDC = "https://data.cdc.gov/"
WW = "wastewaterscan_flu_a_log10_pmmov"
NSSP = "nssp_influenza_pct_ed_visits"


class Downloader:
    def __init__(self, directory: Path):
        self.directory = directory
        self.sources = []
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MIGHTE-FluSight/2026.1"
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

    def get(self, url: str, name: str, params=None) -> bytes:
        response = self.session.get(url, params=params, timeout=(20, 120))
        response.raise_for_status()
        data = response.content
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.sources.append({"url": response.url, "file": name, "retrieved_at": utc_now(),
                             "sha256": digest(path), "bytes": len(data)})
        return data

    def socrata(self, dataset: str, select: str, where: str = "") -> tuple[pd.DataFrame, str]:
        metadata = json.loads(self.get(CDC + f"api/views/{dataset}.json", f"raw/{dataset}-metadata.json"))
        updated = datetime.fromtimestamp(metadata["rowsUpdatedAt"], timezone.utc).isoformat()
        rows = []
        for offset in range(0, 5_000_000, 50_000):
            params = {"$select": select, "$order": ":id", "$limit": 50_000, "$offset": offset}
            if where:
                params["$where"] = where
            page = json.loads(self.get(CDC + f"resource/{dataset}.json",
                                      f"raw/{dataset}-{offset}.json", params))
            if not isinstance(page, list):
                raise ValueError(f"Unexpected CDC response for {dataset}")
            rows.extend(page)
            if len(page) < 50_000:
                break
        else:
            raise ValueError(f"CDC pagination limit reached for {dataset}; refusing truncated data")
        # A dataset changing during pagination can duplicate or omit rows.
        after = json.loads(self.get(CDC + f"api/views/{dataset}.json", f"raw/{dataset}-metadata-after.json"))
        if after["rowsUpdatedAt"] != metadata["rowsUpdatedAt"]:
            raise ValueError(f"CDC updated {dataset} during download. Run refresh again.")
        if not rows:
            raise ValueError(f"No rows returned by CDC dataset {dataset}")
        return pd.DataFrame(rows), updated

    def hub_truth(self, filename: str) -> tuple[pd.DataFrame, str]:
        data = self.get(HUB + "target-data/" + filename, "raw/hub-" + filename)
        commits = json.loads(self.get(API + "commits", "raw/" + filename + "-commits.json",
                                      {"path": "target-data/" + filename, "per_page": 1}))
        updated = commits[0]["commit"]["committer"]["date"]
        return pd.read_csv(io.BytesIO(data), dtype={"location": str}), updated


def clean_truth(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame[["date", "location", "value"]].copy()
    out["location"] = out.location.astype(str).str.zfill(2)
    out["date"] = pd.to_datetime(out.date, errors="raise").dt.normalize()
    out["value"] = pd.to_numeric(out.value, errors="coerce")
    if out.duplicated(["date", "location"]).any():
        raise ValueError(f"Duplicate source observations for {target}")
    if not out.date.dt.weekday.eq(5).all():
        raise ValueError("Surveillance observations must be Saturday week endings")
    if (out.value.dropna() < 0).any() or np.isinf(out.value).any():
        raise ValueError(f"Invalid values in {target}")
    if target == ED and (out.value.dropna() > 1).any():
        raise ValueError("ED truth must be a proportion, not a percentage")
    out["target"] = target
    return out


def choose_truth(sources: list[tuple[str, pd.DataFrame, str]], target: str):
    """Most recently updated source wins on overlapping keys, including suppression."""
    ordered = sorted(sources, key=lambda s: pd.Timestamp(s[2]))
    parts = []
    for name, data, updated in ordered:
        part = clean_truth(data, target)
        part["source"] = name
        part["source_updated_at"] = updated
        parts.append(part)
    result = pd.concat(parts).drop_duplicates(["date", "location"], keep="last")
    result = result.sort_values(["location", "date"]).reset_index(drop=True)
    audit = {"priority_newest_first": [s[0] for s in reversed(ordered)],
             "source_row_counts": result.groupby("source").size().to_dict(),
             "latest_week": result.loc[result.value.notna(), "date"].max().date().isoformat()}
    return result, audit


def wastewater_weekly(raw: pd.DataFrame) -> pd.DataFrame:
    """Same equal-site median, log transform and lag block as the selected study."""
    raw = raw.copy()
    dates = pd.to_datetime(raw.sample_collect_date).dt.normalize()
    raw["date"] = dates + pd.to_timedelta((5 - dates.dt.weekday) % 7, unit="D")
    concentration = pd.to_numeric(raw.pcr_target_mic_lin, errors="coerce")
    raw[WW] = np.log10(concentration.where(concentration >= 0) + 1e-5)
    site = raw.groupby(["date", "site"])[WW].median().dropna().reset_index()
    week = site.groupby("date")[WW].agg(["median", "count"]).reset_index()
    week = week.rename(columns={"median": WW, "count": "site_count"})
    week.loc[week.site_count < 10, WW] = np.nan
    week["available_date"] = week.date + pd.Timedelta(weeks=2)
    lookup = week.set_index("date")[WW]
    for lag in [1, 2, 4]:
        week[f"{WW}_lag{lag}"] = (week.date - pd.Timedelta(weeks=lag)).map(lookup)
    return week


def refresh(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "data/snapshots" / stamp
    staging = destination.with_name(stamp + ".partial")
    staging.mkdir(parents=True)
    download = Downloader(staging)
    print("Refreshing hub rules and complete CDC histories (including backfill)...", flush=True)
    try:
        for remote, local in [("hub-config/tasks.json", "tasks.json"),
                              ("hub-config/model-metadata-schema.json", "model-metadata-schema.json"),
                              ("hub-config/validations.yml", "validations.yml"),
                              ("auxiliary-data/locations.csv", "locations.csv")]:
            download.get(HUB + remote, "contract/" + local)
        contract = Contract(staging / "contract")
        location_names = dict(zip(contract.locations.location_name, contract.locations.location))
        abbreviations = dict(zip(contract.locations.abbreviation, contract.locations.location))
        abbreviations["USA"] = "US"
        hosp_sources = []
        for dataset in ["mpgq-jmmr", "ua7e-t2fy"]:
            raw, updated = download.socrata(dataset, "jurisdiction,weekendingdate,totalconfflunewadm")
            frame = pd.DataFrame({"date": raw.weekendingdate, "location": raw.jurisdiction.map(abbreviations),
                                  "value": raw.totalconfflunewadm})
            hosp_sources.append(("CDC " + dataset, frame.dropna(subset=["location"]), updated))
        hub_hosp, hub_hosp_time = download.hub_truth("target-hospital-admissions.csv")
        hosp_sources.append(("FluSight hub", hub_hosp, hub_hosp_time))
        hosp, hosp_audit = choose_truth(hosp_sources, HOSP)

        raw_ed, ed_time = download.socrata("rdmq-nq56", "week_end,geography,percent_visits_influenza",
                                           "county='All'")
        frame = pd.DataFrame({"date": raw_ed.week_end,
                              "location": raw_ed.geography.replace({"United States": "US"}).map(location_names),
                              "value": pd.to_numeric(raw_ed.percent_visits_influenza, errors="coerce") / 100})
        hub_ed, hub_ed_time = download.hub_truth("target-ed-visits-prop.csv")
        ed, ed_audit = choose_truth([("CDC rdmq-nq56", frame.dropna(subset=["location"]), ed_time),
                                     ("FluSight hub", hub_ed, hub_ed_time)], ED)
        truth = pd.concat([hosp, ed], ignore_index=True)
        truth.to_csv(staging / "truth.csv", index=False, date_format="%Y-%m-%d")
        national = ed[ed.location.eq("US")][["date", "value"]].copy()
        national[NSSP] = national.pop("value") * 100
        # This is the frozen recipe's nominal reference-week availability, not a claim
        # about the actual publication date. Retrieval timestamps establish what was known.
        national["available_date"] = national.date + pd.Timedelta(weeks=1)
        national.to_csv(staging / "nssp.csv", index=False, date_format="%Y-%m-%d")
        raw_ww, _ = download.socrata("ymmh-divb", "site,sample_collect_date,pcr_target_mic_lin",
                                     "source='WastewaterSCAN' AND pcr_target='fluav'")
        wastewater_weekly(raw_ww).to_csv(staging / "wastewater.csv", index=False, date_format="%Y-%m-%d")
        manifest = {"snapshot_id": stamp, "completed_at": utc_now(), "sources": download.sources,
                    "hospitalizations": hosp_audit, "ed": ed_audit,
                    "files": {str(p.relative_to(staging)): digest(p) for p in sorted(staging.rglob("*"))
                              if p.is_file()}}
        write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
        write_json(root / "data/latest.json", {"snapshot_id": stamp})
        print(f"Data snapshot {stamp}: hospitals through {hosp_audit['latest_week']}; "
              f"ED through {ed_audit['latest_week']}", flush=True)
        return destination
    except Exception:
        # Keep failed acquisitions for diagnosis, but never promote them to latest.
        print(f"Refresh failed. Partial inputs retained at {staging}; no stale fallback was used.", flush=True)
        raise


def latest_snapshot(root: Path) -> Path:
    pointer = root / "data/latest.json"
    if not pointer.exists():
        raise ValueError("No snapshot yet. Run ./mighte refresh or ./mighte forecast")
    return root / "data/snapshots" / json.loads(pointer.read_text())["snapshot_id"]


def verify_snapshot(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Snapshot has changed: {name}")
    return manifest


def weekly_grid(frame: pd.DataFrame, anchor: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    parts, interpolated = [], 0
    for location, group in frame.groupby("location_name", sort=True):
        values = group.set_index("date").total_hosp.sort_index()
        values = values.reindex(pd.date_range(values.first_valid_index(), anchor, freq="W-SAT"))
        before = values.isna().sum()
        values = values.interpolate(method="linear", limit_area="inside")
        interpolated += int(before - values.isna().sum())
        parts.append(pd.DataFrame({"location_name": location, "date": values.index,
                                    "total_hosp": values.to_numpy()}))
    return pd.concat(parts, ignore_index=True), interpolated


def reconstruct_ed(root: Path, observed: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, dict]:
    mode = settings["mode"]
    if mode == "observed":
        return observed, {"mode": mode, "proxy_rows": 0}
    if mode not in {"post2022_proxy", "prepandemic_proxy"}:
        raise ValueError("ed_history.mode must be observed, post2022_proxy or prepandemic_proxy")
    ili = pd.read_csv(root / "data/historical/ilinet_normalized.csv", parse_dates=["date"])
    if mode == "prepandemic_proxy":
        filename = settings.get("historical_nssp_file")
        if not filename:
            raise ValueError("Set ed_history.historical_nssp_file to a repository-local CSV of date,value (proportion)")
        path = (root / filename).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Historical NSSP must be stored inside this standalone repository")
        national = pd.read_csv(path, parse_dates=["date"])
        if not national.value.between(0, 1).all() or (national.date >= "2020-03-01").any():
            raise ValueError("Historical national NSSP must contain pre-pandemic proportions in [0,1]")
        national = national.rename(columns={"value": "total_hosp"})
        national["total_hosp"] *= 100
    else:
        national = observed[observed.location_name.eq("US")]
    paired = ili[ili.location_name.eq("US")].merge(national[["date", "total_hosp"]], on="date").dropna()
    if len(paired) < 52 or paired.ili.std() < 1e-8:
        raise ValueError("At least 52 paired weeks with varying ILINet are required for ED reconstruction")
    design = np.column_stack([np.ones(len(paired)), paired.ili])
    coefficients = np.linalg.lstsq(design, np.log1p(paired.total_hosp), rcond=None)[0]
    # The same 728-day shift and biological cutoff as the hospitalization proxy.
    proxy = ili[ili.date.le("2020-05-30")].copy()
    proxy["total_hosp"] = np.clip(np.expm1(coefficients[0] + coefficients[1] * proxy.ili), 0, 100)
    proxy["date"] += pd.Timedelta(days=728)
    proxy = proxy[proxy.location_name.isin(observed.location_name.unique())]
    proxy = proxy[["location_name", "date", "total_hosp"]]
    merged = pd.concat([proxy, observed]).drop_duplicates(["location_name", "date"], keep="last")
    return merged, {"mode": mode, "proxy_rows": len(proxy), "mapping_weeks": len(paired),
                     "coefficients": coefficients.tolist(), "shift_days": 728,
                     "note": "Training proxy only. The post2022 option assumes the national ILI-to-ED relationship transfers across eras and locations."}


def prepare_inputs(root: Path, snapshot: Path, reference: str, settings: dict) -> tuple[dict, dict]:
    anchor = pd.Timestamp(reference) - pd.Timedelta(weeks=1)
    contract = Contract(snapshot / "contract")
    names = contract.locations.set_index("location").location_name.to_dict()
    truth = pd.read_csv(snapshot / "truth.csv", dtype={"location": str}, parse_dates=["date"])
    truth = truth[truth.date.le(anchor)].copy()
    truth["location_name"] = truth.location.map(names)
    inputs, audit = {}, {}
    for target in [HOSP, ED]:
        history = truth[truth.target.eq(target)].copy()
        current = history[history.date.eq(anchor) & history.value.notna()]
        eligible = set(current.location)
        allowed = set(contract.by_target[target]["task_ids"]["location"]["optional"])
        if target == HOSP and eligible != allowed:
            raise ValueError(f"Hospitalizations missing at {anchor.date()}: {sorted(allowed - eligible)}. "
                             "Wait for this week's data release; forecasts will not be relabeled.")
        if "US" not in eligible:
            raise ValueError(f"National {target} missing at {anchor.date()}")
        history = history[history.location.isin(eligible)]
        model = history[["location_name", "date", "value"]].rename(columns={"value": "total_hosp"})
        if target == HOSP:
            seed = pd.read_csv(root / "data/historical/hospitalization_proxy.csv", parse_dates=["date"])
            model = pd.concat([seed, model[model.date.ge("2022-06-04")]], ignore_index=True)
        else:
            model["total_hosp"] *= 100  # model in percentage points; submit in proportions
            model, reconstruction = reconstruct_ed(root, model, settings["ed_history"])
            audit["ed_reconstruction"] = reconstruction
        model = model[model.location_name.isin(current.location_name)]
        model, n_filled = weekly_grid(model, anchor)
        inputs[target] = model
        audit[target] = {"anchor": anchor.date().isoformat(), "locations": sorted(eligible),
                          "excluded_locations": sorted(allowed - eligible), "training_rows": len(model),
                          "interpolated_training_values": n_filled}
    nssp = pd.read_csv(snapshot / "nssp.csv", parse_dates=["date"])
    if nssp.loc[nssp.date.eq(anchor), NSSP].dropna().empty:
        raise ValueError("Missing current national NSSP covariate")
    ww = pd.read_csv(snapshot / "wastewater.csv", parse_dates=["date"])
    usable = ww[ww.date.le(anchor - pd.Timedelta(weeks=1)) & ww[WW].notna()]
    if usable.empty or usable.date.max() < anchor - pd.Timedelta(weeks=2):
        raise ValueError("WastewaterSCAN covariate is missing or beyond the frozen staleness limit")
    audit["wastewater_week_used"] = usable.date.max().date().isoformat()
    return inputs, audit
