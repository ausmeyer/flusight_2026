"""The FluSight contract, with stricter checks from the hub's written rules."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import jsonschema
import numpy as np
import pandas as pd
import yaml

HOSP = "wk inc flu hosp"
ED = "wk inc flu prop ed visits"
TREND = "wk flu hosp rate change"
MODELS = ("MIGHTE-Base", "MIGHTE-Linear", "MIGHTE-Nsemble")
COLUMNS = ["reference_date", "target", "horizon", "target_end_date", "location",
           "output_type", "output_type_id", "value"]
UNIT = ["reference_date", "target", "horizon", "target_end_date", "location"]
QUANTILES = np.array([.01, .025, .05, .1, .15, .2, .25, .3, .35, .4, .45,
                      .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .975, .99])
CATEGORIES = ["large_decrease", "decrease", "stable", "increase", "large_increase"]
EASTERN = ZoneInfo("America/New_York")


def read_forecast(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"location": str, "output_type_id": str})


def reference_saturday(now: datetime | None = None) -> str:
    day = (now or datetime.now(EASTERN)).astimezone(EASTERN).date()
    return (day + timedelta(days=(5 - day.weekday()) % 7)).isoformat()


def check_window(reference: str, now: datetime | None = None) -> None:
    now = (now or datetime.now(EASTERN)).astimezone(EASTERN)
    ref = datetime.fromisoformat(reference).replace(tzinfo=EASTERN)
    start = ref - timedelta(days=6)
    deadline = (ref - timedelta(days=3)).replace(hour=23)
    if not start <= now <= deadline:
        raise ValueError(f"Submission window: {start.isoformat()} through {deadline.isoformat()}. "
                         "Use preview for an out-of-window rehearsal; it cannot be submitted.")


class Contract:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.tasks = json.loads((self.directory / "tasks.json").read_text())
        self.locations = pd.read_csv(self.directory / "locations.csv", dtype={"location": str})
        self.population = self.locations.set_index("location")["population"].to_dict()
        self.by_target = {}
        for round_spec in self.tasks["rounds"]:
            for task in round_spec["model_tasks"]:
                for name in task["task_ids"]["target"].get("optional", []) or []:
                    self.by_target[name] = task

    def validate(self, frame: pd.DataFrame, model: str, reference: str, *, preview=False) -> dict:
        if model not in MODELS:
            raise ValueError(f"Unknown submission model: {model}")
        if set(frame.columns) != set(COLUMNS) or len(frame.columns) != len(COLUMNS):
            raise ValueError(f"Exactly these columns are required: {COLUMNS}")
        if frame.empty or frame.isna().any().any():
            raise ValueError("Empty forecasts and missing values are forbidden")
        df = frame.copy()
        for col in ["reference_date", "target_end_date"]:
            if not df[col].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
                raise ValueError(f"{col} must use YYYY-MM-DD")
            dt = pd.to_datetime(df[col], errors="raise")
            if not dt.dt.weekday.eq(5).all():
                raise ValueError(f"{col} must be Saturday")
        if set(df.reference_date) != {reference}:
            raise ValueError("Reference dates must match the filename/run")
        h = pd.to_numeric(df.horizon, errors="raise")
        # These models intentionally submit only horizons 0..3, including the nowcast.
        if not h.isin([0, 1, 2, 3]).all():
            raise ValueError("This pipeline supports only horizons 0, 1, 2, 3")
        expected = pd.to_datetime(df.reference_date) + pd.to_timedelta(h * 7, unit="D")
        if not expected.eq(pd.to_datetime(df.target_end_date)).all():
            raise ValueError("target_end_date must equal reference_date + 7*horizon")
        allowed = {HOSP, ED, TREND} if model == "MIGHTE-Base" else {HOSP}
        if not set(df.target) <= allowed or HOSP not in set(df.target):
            raise ValueError(f"Incorrect targets for {model}")
        df["value"] = pd.to_numeric(df.value, errors="raise")
        if not np.isfinite(df.value).all() or (df.value < 0).any():
            raise ValueError("Forecast values must be finite and nonnegative")
        if df.duplicated(UNIT + ["output_type", "output_type_id"]).any():
            raise ValueError("Duplicate forecast keys")
        for target, g in df.groupby("target"):
            task = self.by_target[target]
            ids = task["task_ids"]
            if not set(g.location) <= set(ids["location"]["optional"]):
                raise ValueError(f"Invalid location for {target}; preserve leading FIPS zeros")
            if not preview:
                for name in ["reference_date", "target_end_date"]:
                    if not set(g[name]) <= set(ids[name]["optional"]):
                        raise ValueError(f"{name} is outside the hub's accepted rounds")
            output_type = "pmf" if target == TREND else "quantile"
            if set(g.output_type) != {output_type}:
                raise ValueError(f"{target} requires {output_type}")
            required = task["output_type"][output_type]["output_type_id"]["required"]
            for _, unit in g.groupby(UNIT, dropna=False):
                q = (pd.to_numeric(unit.output_type_id, errors="raise").to_numpy()
                     if output_type == "quantile" else unit.output_type_id.to_numpy())
                if len(q) != len(required) or set(q) != set(required):
                    raise ValueError(f"Incomplete or duplicate output_type_id set for {target}")
                if output_type == "quantile" and (np.diff(unit.value.to_numpy()[np.argsort(q)]) < 0).any():
                    raise ValueError("Crossing quantiles")
                if output_type == "pmf" and (not np.isclose(unit.value.sum(), 1, atol=1e-8)
                                              or (unit.value > 1).any()):
                    raise ValueError("Category probabilities must be in [0,1] and sum to one")
            if target == HOSP:
                if not np.equal(g.value, np.floor(g.value)).all():
                    raise ValueError("Hospitalization forecasts must be integer-valued")
                if (g.value > g.location.map(self.population) * .30).any():
                    raise ValueError("Hospitalization quantile exceeds the hub's 30%-of-population bound")
            if target == ED and (g.value > .25).any():
                raise ValueError("ED quantile exceeds the hub's 0.25 plausibility bound; inspect the fit")
        return {"rows": len(df), "targets": sorted(df.target.unique()),
                "locations": df.groupby("target").location.nunique().to_dict(), "valid": True}

    def validate_metadata(self, directory: Path, *, models=MODELS) -> list[dict]:
        schema = json.loads((self.directory / "model-metadata-schema.json").read_text())
        records = []
        for model in models:
            data = yaml.safe_load((directory / f"{model}.yml").read_text())
            jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(data)
            if f"{data['team_abbr']}-{data['model_abbr']}" != model:
                raise ValueError(f"Metadata filename does not match model ID: {model}")
            records.append(data)
        if sum(d["designated_model"] for d in records) > 2:
            raise ValueError("More than two designated models requires CDC agreement")
        return records
