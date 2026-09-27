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
from scipy.special import expit
from urllib3.util.retry import Retry

from .contract import Contract, HOSP, ED
from .ed_transform import boundary_epsilon, ed_logit
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
        self.hub_revision = None
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

    def pin_hub(self) -> str:
        head = json.loads(self.get(API + "commits/main", "raw/hub-head.json"))
        self.hub_revision = head["sha"]
        return self.hub_revision

    def hub_file(self, remote: str, local: str) -> bytes:
        if self.hub_revision is None:
            self.pin_hub()
        return self.get(HUB.removesuffix("main/") + self.hub_revision + "/" + remote, local)

    def hub_truth(self, filename: str) -> tuple[pd.DataFrame, str]:
        data = self.hub_file("target-data/" + filename, "raw/hub-" + filename)
        commits = json.loads(self.get(API + "commits", "raw/" + filename + "-commits.json",
                                      {"path": "target-data/" + filename, "per_page": 1,
                                       "sha": self.hub_revision}))
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
            download.hub_file(remote, "contract/" + local)
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

        raw_ed, ed_time = download.socrata("rdmq-nq56", "week_end,geography,percent_visits_influenza", "county='All'")
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
                    "hub_revision": download.hub_revision,
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


def weekly_grid(frame: pd.DataFrame, anchor: pd.Timestamp, *, interpolate=True) -> tuple[pd.DataFrame, int]:
    """Complete weekly grid per location; inside-only linear interpolation, never extrapolation."""
    parts, interpolated = [], 0
    for location, group in frame.groupby("location_name", sort=True):
        values = group.set_index("date").total_hosp.sort_index()
        values = values.reindex(pd.date_range(values.first_valid_index(), anchor, freq="W-SAT"))
        if interpolate:
            before = values.isna().sum()
            values = values.interpolate(method="linear", limit_area="inside")
            interpolated += int(before - values.isna().sum())
        parts.append(pd.DataFrame({"location_name": location, "date": values.index,
                                    "total_hosp": values.to_numpy()}))
    return pd.concat(parts, ignore_index=True), interpolated


# Reporting eras of the hospitalization history. Each earlier era is leveled to the current NHSN
# regime (mandatory reporting from November 2024).
CURRENT_REGIME = pd.Timestamp("2024-11-01")
MANDATORY_REPORTING = (pd.Timestamp("2022-02-01"), pd.Timestamp("2024-04-30"))
VOLUNTARY_REPORTING = (pd.Timestamp("2024-05-01"), pd.Timestamp("2024-09-30"))
IN_SEASON, OFF_SEASON = [11, 12, 1, 2, 3, 4], [5, 6, 7, 8, 9]
LEVEL_BOUNDS = (0.5, 2.0)
# The log level changes linearly over the four weeks around the seed/NHSN splice, the four weeks
# after mandatory reporting ended, and the five weeks in which hospitals resumed reporting before
# the November 2024 mandate.
LEVEL_RAMPS = [("2021-06-12", "2021-07-10"), ("2024-04-27", "2024-05-25"), ("2024-09-28", "2024-11-02")]


def era_levels(observed: pd.DataFrame, predictor: pd.DataFrame) -> pd.DataFrame:
    """Multipliers that put each era on the current NHSN level.

    Levels are volume-weighted ratios of reported admissions to the ILINet/FluSurv predictor over
    matching seasons. The seed takes the location's current in-season (Nov-Apr) ratio; NHSN from
    July 2021 to April 2024 takes that ratio over the mandatory-reporting in-season ratio; the
    voluntary period takes one national factor, the US current off-season (May-Sep) ratio over the
    voluntary ratio, because several states nearly stopped reporting. Levels are 1 without eight
    current and 26 mandatory in-season weeks (eight off-season weeks each for the national factor).
    """
    paired = observed[observed.date.ge("2021-07-01")].merge(predictor, on=["location_name", "date"], validate="one_to_one")
    ratio = lambda g: g.total_hosp.sum() / g.proxy_hosp_continuous.sum()
    month = paired.date.dt.month
    current = paired[paired.date.ge(CURRENT_REGIME) & month.isin(IN_SEASON)]
    mandatory = paired[paired.date.between(*MANDATORY_REPORTING) & month.isin(IN_SEASON)]
    us = paired[paired.location_name.eq("US")]
    us_off = us[us.date.ge(CURRENT_REGIME) & us.date.dt.month.isin(OFF_SEASON)]
    us_voluntary = us[us.date.between(*VOLUNTARY_REPORTING)]
    voluntary = ratio(us_off) / ratio(us_voluntary) if len(us_off) >= 8 and len(us_voluntary) >= 8 else 1.0
    rows = []
    for name in sorted(observed.location_name.unique()):
        c, m = current[current.location_name.eq(name)], mandatory[mandatory.location_name.eq(name)]
        enough = len(c) >= 8 and len(m) >= 26
        rows.append({"location_name": name, "seed": ratio(c) if enough else 1.0,
                     "legacy": ratio(c) / ratio(m) if enough else 1.0, "voluntary": voluntary,
                     "current_in_season_weeks": len(c), "mandatory_in_season_weeks": len(m)})
    levels = pd.DataFrame(rows)
    levels[["seed", "legacy", "voluntary"]] = levels[["seed", "legacy", "voluntary"]].clip(*LEVEL_BOUNDS)
    return levels


def level_history(history: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Apply the levels in log1p space, like the study's shift; the current regime is never changed."""
    x = [pd.Timestamp(day).toordinal() for ramp in LEVEL_RAMPS for day in ramp]
    out = history.reset_index(drop=True).copy()
    shift = np.zeros(len(out))
    factors = levels.set_index("location_name")
    for name, rows in out.groupby("location_name").indices.items():
        f = factors.loc[name]
        y = np.log([f.seed, f.legacy, f.legacy, f.voluntary, f.voluntary, 1.0])
        shift[rows] = np.interp(out.date.iloc[rows].map(pd.Timestamp.toordinal), x, y)
    moved = shift != 0
    out.loc[moved, "total_hosp"] = np.rint(np.maximum(np.expm1(np.log1p(out.total_hosp[moved].clip(lower=0)) + shift[moved]), 0))
    return out


def hospitalization_history(root: Path, observed: pd.DataFrame, anchor: pd.Timestamp) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """ILINet/FluSurv seed through June 2021 (dates shifted 728 days), then NHSN counts, with each
    era leveled to the current NHSN regime from data available at the anchor."""
    observed = observed[observed.date.le(anchor)]
    seed = pd.read_csv(root / "data/historical/hospitalization_proxy.csv", parse_dates=["date"])
    seed = seed[seed.location_name.isin(observed.location_name.unique()) & seed.date.le(anchor)]
    model = pd.concat([seed, observed[observed.date.ge("2021-07-01")]], ignore_index=True)
    history, filled = weekly_grid(model[["location_name", "date", "total_hosp"]], anchor)
    predictor = pd.read_csv(root / "data/historical/hospitalization_calibration.csv", parse_dates=["date"],
                            float_precision="round_trip")
    levels = era_levels(observed, predictor[predictor.date.le(anchor)])
    return level_history(history, levels), filled, levels


def ed_ilinet_proxy(root: Path, observed: pd.DataFrame, transform: dict) -> tuple[pd.DataFrame, dict]:
    """ILINet-based ED history built like the hospitalization seed.

    A pooled regression of observed ED log-odds on normalized ILINet, fitted where both exist,
    is applied to each location's ILINet through June 2019. Dates shift forward 728 days and
    percentages are rounded to NSSP's 0.01-point reporting precision.
    """
    epsilon = boundary_epsilon(transform)
    ili = pd.read_csv(root / "data/historical/ilinet_normalized.csv", parse_dates=["date"])
    paired = ili.merge(observed.dropna(subset=["total_hosp"]), on=["location_name", "date"], validate="one_to_one")
    if len(paired) < 52 or paired.ili.std() < 1e-8:
        raise ValueError("At least 52 paired weeks with varying ILINet are required for ED reconstruction")
    design = np.column_stack([np.ones(len(paired)), paired.ili])
    coefficients = np.linalg.lstsq(design, ed_logit(paired.total_hosp / 100, epsilon), rcond=None)[0]
    proxy = ili[ili.date.le("2019-06-30") & ili.location_name.isin(observed.location_name.unique())].copy()
    proxy["total_hosp"] = np.round(100 * expit(coefficients[0] + coefficients[1] * proxy.ili), 2)
    proxy["date"] += pd.Timedelta(days=728)
    return proxy[["location_name", "date", "total_hosp"]], {
        "mode": "ilinet_proxy", "proxy_rows": len(proxy), "mapping_rows": len(paired),
        "mapping_locations": int(paired.location_name.nunique()), "coefficients": coefficients.tolist(),
        "source_end": "2019-06-30", "shift_days": 728, "response_transform": transform}


LABELS = {HOSP: "Hospital admissions", ED: "ED visits"}


def prepare_inputs(root: Path, snapshot: Path, reference: str, settings: dict) -> tuple[dict, dict]:
    """Training histories and input availability.

    Every location reported in the past year is forecast. Locations without an anchor value are
    recorded so the pipeline can use its fallback.
    """
    epsilon = boundary_epsilon(settings["ed_target_transform"])
    anchor = pd.Timestamp(reference) - pd.Timedelta(weeks=1)
    contract = Contract(snapshot / "contract")
    names = contract.locations.set_index("location").location_name.to_dict()
    truth = pd.read_csv(snapshot / "truth.csv", dtype={"location": str}, parse_dates=["date"])
    truth = truth[truth.date.le(anchor)].copy()
    truth["location_name"] = truth.location.map(names)
    inputs, audit, notices = {}, {}, []
    for target in [HOSP, ED]:
        allowed = set(contract.by_target[target]["task_ids"]["location"]["optional"])
        observed = truth[truth.target.eq(target) & truth.location.isin(allowed) & truth.value.notna()]
        expected = set(observed.loc[observed.date.gt(anchor - pd.Timedelta(weeks=52)), "location"])
        last = observed.groupby("location").date.max()
        missing = {x: last[x].date().isoformat() for x in sorted(expected) if last[x] < anchor}
        audit[target] = {"anchor": anchor.date().isoformat(), "locations": sorted(expected),
                         "anchor_missing": missing, "excluded_locations": sorted(allowed - expected)}
        if not expected:
            notices.append(f"{LABELS[target]}: nothing reported in the past year; target omitted")
            continue
        if missing:
            dates = sorted(set(missing.values()))
            which = (f"all locations (last reported {dates[0]}{'' if len(dates) == 1 else ' to ' + dates[-1]})"
                     if len(missing) == len(expected) else ", ".join(f"{names[x]} (last reported {day})"
                                                                    for x, day in missing.items()))
            notices.append(f"{LABELS[target]}: no {anchor.date()} value for {which}; forecast by MIGHTE-Base "
                           "without wastewater and NSSP from the last reported value")
        model = observed[["location_name", "date", "value"]].rename(columns={"value": "total_hosp"})
        if target == HOSP:
            model, n_filled, levels = hospitalization_history(root, model, anchor)
            audit["hospitalization_levels"] = levels.round(4).to_dict("records")
        else:
            model["total_hosp"] *= 100  # predictors in percentage points; response link uses proportions
            mode = settings["ed_history"]["mode"]
            if mode == "ilinet_proxy":
                proxy, audit["ed_reconstruction"] = ed_ilinet_proxy(root, model, settings["ed_target_transform"])
                model, n_filled = weekly_grid(model, anchor)
                proxy, _ = weekly_grid(proxy, proxy.date.max())
                # Keep the gap between the proxy and observed eras missing rather than interpolated.
                model, _ = weekly_grid(pd.concat([proxy, model]), anchor, interpolate=False)
            elif mode == "observed":
                audit["ed_reconstruction"] = {"mode": mode, "proxy_rows": 0}
                model, n_filled = weekly_grid(model, anchor)
            else:
                raise ValueError("ed_history.mode must be observed or ilinet_proxy")
            proportions = model.total_hosp / 100
            ed_logit(proportions.dropna(), epsilon)
            boundary = (proportions < epsilon) | (proportions > 1 - epsilon)
            audit["ed_target_transform"] = {**settings["ed_target_transform"],
                "boundary_training_observations": int(boundary.sum()),
                "boundary_anchor_observations": int((boundary & model.date.eq(anchor)).sum()),
                "boundary_policy": "Clip response-link inputs only; do not alter truth or predictor history"}
        inputs[target] = model
        audit[target].update(training_rows=len(model), interpolated_training_values=n_filled)
    audit["notices"] = notices
    return inputs, audit
