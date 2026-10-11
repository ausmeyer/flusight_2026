"""Prospective, direct seasonal peak-height regression for MIGHTE-Base."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import core
from .contract import COLUMNS, PEAK_HEIGHT, PEAK_WEEK, QUANTILES, Contract
from .models import pooled_features
from .util import digest, write_json


def season_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Historical analogues of the hub's October–May seasonal target window."""
    first, last = pd.Timestamp(year, 10, 4), pd.Timestamp(year + 1, 5, 31)
    return first + pd.Timedelta(days=(5 - first.weekday()) % 7), last - pd.Timedelta(days=(last.weekday() - 5) % 7)


def seasonal_examples(history, snapshot, reference, runtime, signals, populations, current_window):
    """Origin-only predictors and complete-season labels from the same prepared history."""
    anchor = pd.Timestamp(reference) - pd.Timedelta(weeks=1)
    history = history[pd.to_datetime(history.date).le(anchor)].copy()
    pooled, columns = pooled_features(history, snapshot, reference, replace(runtime, max_horizons=1), signals)
    columns = [c for c in columns if c not in {"horizon_weeks", "horizon_sin", "horizon_cos"}]
    issue = pooled.date + pd.Timedelta(weeks=1)
    pooled["season_year"] = issue.dt.year - issue.dt.month.lt(7).astype(int)
    current_year = pd.Timestamp(current_window[0]).year
    frames, labels = [], []
    for year, frame in pooled.groupby("season_year", sort=True):
        first, last = current_window if year == current_year else season_window(int(year))
        seasonal = history[history.date.between(first, last)].sort_values("date").copy()
        for location, group in seasonal.groupby("location_name"):
            weeks = pd.date_range(first, last, freq="W-SAT")
            if last <= anchor and group.set_index("date").total_hosp.reindex(weeks).notna().all():
                maximum = float(group.total_hosp.max())
                peak_dates = group.loc[group.total_hosp.eq(maximum), "date"]
                peak_date = peak_dates.iloc[0] if len(peak_dates) == 1 else pd.NaT
                labels.append({"location_name": location, "season_year": year,
                               "peak_count": maximum, "label_available_date": last,
                               "season_peak_date": peak_date, "peak_ties": len(peak_dates),
                               "peak_fraction": (peak_date - first) / (last - first) if len(peak_dates) == 1 else np.nan})
        seasonal = seasonal.dropna(subset=["total_hosp"])
        frame = frame.sort_values("date").copy()
        if seasonal.empty:
            frame["peak_so_far"] = 0.0
            frame["weeks_since_maximum"] = -1.0
        else:
            seasonal["peak_so_far"] = seasonal.groupby("location_name").total_hosp.cummax()
            seasonal["peak_date"] = seasonal.date.where(seasonal.total_hosp.eq(seasonal.peak_so_far))
            seasonal["peak_date"] = seasonal.groupby("location_name").peak_date.ffill()
            frame = pd.merge_asof(frame, seasonal[["date", "location_name", "peak_so_far", "peak_date"]],
                                  on="date", by="location_name", direction="backward")
            frame["peak_so_far"] = frame.peak_so_far.fillna(0)
            frame["weeks_since_maximum"] = ((frame.date - frame.peak_date).dt.days / 7).fillna(-1)
            frame = frame.drop(columns="peak_date")
        frame["season_week"] = (frame.date + pd.Timedelta(weeks=1) - first).dt.days / 7
        frame["log_anchor"] = np.log1p(frame.total_hosp)
        frame["log_peak_so_far"] = np.log1p(frame.peak_so_far)
        frame["log_population"] = np.log(frame.location_name.map(populations))
        frame["seasonal_origin"] = (frame.date + pd.Timedelta(weeks=1)).between(first, last)
        frames.append(frame)
    examples = pd.concat(frames, ignore_index=True)
    label_table = pd.DataFrame(labels, columns=["location_name", "season_year", "peak_count", "label_available_date",
                                               "season_peak_date", "peak_ties", "peak_fraction"])
    examples = examples.merge(label_table, on=["location_name", "season_year"], how="left", validate="many_to_one")
    columns += ["season_week", "log_anchor", "log_peak_so_far", "weeks_since_maximum", "log_population"]
    return examples, columns


def tree_params(seed, runtime, *, spread=False):
    params = core.lightgbmlss_stage2_params(seed, runtime, "joint_base") if spread else core.central_model_params(seed, runtime)
    params.update(num_leaves=7, max_depth=3, min_child_samples=50, lambda_l2=5.0, force_col_wise=True)
    return params


def forward_centers(train, columns, y, runtime, seed):
    """Every validation season is absent from every earlier-season center fit."""
    predictions = np.full(len(train), np.nan)
    audit = []
    for year in sorted(train.season_year.unique())[1:]:
        fit, held = train.season_year.lt(year).to_numpy(), train.season_year.eq(year).to_numpy()
        if fit.sum() < runtime.min_train_rows:
            continue
        model = core.fit_lgbm_regressor(train.loc[fit, columns], y[fit], tree_params(seed + int(year), runtime),
                                       runtime.stage1_rounds)
        predictions[held] = model.predict(train.loc[held, columns].astype(float))
        audit.append({"training_seasons": sorted(int(x) for x in train.loc[fit, "season_year"].unique()),
                      "validation_season": int(year), "fit_rows": int(fit.sum()), "validation_rows": int(held.sum()),
                      "training_label_available_max": train.loc[fit, "label_available_date"].max().date().isoformat(),
                      "validation_origin_min": train.loc[held, "date"].min().date().isoformat()})
    valid = np.isfinite(predictions)
    if valid.sum() < 50:
        raise ValueError("Peak height has too few whole-season forward predictions to learn uncertainty")
    return valid, predictions, audit


def predict(context):
    reference, settings = context["reference_date"], context["settings"]
    snapshot, checkpoint = Path(context["snapshot_path"]), Path(context["checkpoint_path"])
    contract = Contract(snapshot / "contract")
    location_map = dict(zip(contract.locations.location_name, contract.locations.location))
    populations = dict(zip(contract.locations.location_name, contract.locations.population))
    weeks = contract.by_target[PEAK_WEEK]["output_type"]["pmf"]["output_type_id"]["required"]
    window = (pd.Timestamp(min(weeks)), pd.Timestamp(max(weeks)))
    if not pd.Timestamp(reference) <= window[1]:
        raise ValueError("Peak-height reference date is after the target season")
    runtime = core.RuntimeConfig.from_dict({**settings["runtime"], "num_threads": settings["threads"]})
    signals = context.get("signals", ("ww", "nssp"))
    examples, columns = seasonal_examples(context["hospitalization_history"], snapshot, reference,
                                          runtime, signals, populations, window)
    anchor = pd.Timestamp(reference) - pd.Timedelta(weeks=1)
    train = examples[examples.seasonal_origin & examples.peak_count.notna()
                     & examples.label_available_date.le(anchor) & examples.total_hosp.notna()].reset_index(drop=True)
    test = examples[examples.date.eq(anchor)].copy()
    if context.get("locations") is not None:
        test = test[test.location_name.isin(context["locations"])]
    test = test.sort_values("location_name").reset_index(drop=True)
    floors = test.peak_so_far.to_numpy()[:, None]
    if len(train) < runtime.min_train_rows or test.empty:
        raise ValueError("Insufficient completed-season peak-height labels or no forecast locations")
    seasons = sorted(train.season_year.unique())
    if len(seasons) < 3:
        raise ValueError("Peak height requires at least three completed training seasons")
    checkpoint.mkdir(parents=True, exist_ok=True)
    labels_path = checkpoint / "training-labels.csv"
    train[["location_name", "season_year", "peak_count", "label_available_date"]].drop_duplicates().sort_values(
        ["season_year", "location_name"]).to_csv(labels_path, index=False)
    seed = int(settings["seed"])
    rng = np.random.default_rng(seed + int(anchor.value // 10**9))
    bag_size = max(3, int(round(len(seasons) * runtime.bag_frac)))
    y = (train.peak_count.to_numpy() + 1) * 100_000 / train.location_name.map(populations).to_numpy()
    count_scale = test.location_name.map(populations).to_numpy()[:, None] / 100_000
    predictions, bag_audits = [], []
    for bag in range(runtime.num_bags):
        sampled = sorted(rng.choice(seasons, size=bag_size, replace=False))
        path = checkpoint / f"bag-{bag:03d}.npy"
        if path.exists():
            quantiles = np.load(path, allow_pickle=False)
            bag_audit = json.loads(path.with_suffix(".json").read_text())
        else:
            selected = train.season_year.isin(sampled).to_numpy()
            subset, outcome = train.loc[selected].reset_index(drop=True), y[selected]
            valid, centers, folds = forward_centers(subset, columns, outcome, runtime, seed + bag)
            center = core.fit_lgbm_regressor(subset[columns], outcome, tree_params(seed + bag, runtime), runtime.stage1_rounds)
            scale = core.fit_conditional_gaussian_scale_booster(subset.loc[valid, columns], outcome[valid], centers[valid],
                                                                tree_params(seed + bag, runtime, spread=True), runtime.stage2_rounds)
            x = test[columns].astype(float)
            mu, sigma = center.predict(x), scale.predict_sigma(x)
            quantiles = norm.ppf(QUANTILES[None, :], loc=mu[:, None], scale=sigma[:, None]) * count_scale - 1
            quantiles = np.maximum(quantiles, floors)
            bag_audit = {"bag": bag, "seasons": [int(s) for s in sampled], "folds": folds,
                         "sigma_min": float(sigma.min()), "sigma_max": float(sigma.max())}
            with path.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, quantiles, allow_pickle=False)
            path.with_suffix(".tmp").replace(path)
            write_json(path.with_suffix(".json"), bag_audit)
            print(f"base-peak-height: bag {bag + 1}/{runtime.num_bags}", flush=True)
        if quantiles.shape != (len(test), len(QUANTILES)) or not np.isfinite(quantiles).all():
            raise ValueError("Invalid peak-height prediction/checkpoint")
        if np.any(np.diff(quantiles, axis=1) < 0) or np.any(quantiles < floors):
            raise ValueError("Peak-height quantiles cross or fall below the observed seasonal maximum")
        predictions.append(quantiles)
        bag_audits.append(bag_audit)
    values = np.rint(np.median(np.stack(predictions), axis=0))
    rows = [{"reference_date": reference, "target": PEAK_HEIGHT, "horizon": np.nan, "target_end_date": None,
             "location": location_map[row.location_name], "output_type": "quantile", "output_type_id": q, "value": float(value)}
            for row, vector in zip(test.itertuples(index=False), values) for q, value in zip(QUANTILES, vector)]
    audit = {"component": "MIGHTE-Base peak height", "response_transform": "(seasonal peak + 1) / population * 100000",
             "label_source": "complete Base hospitalization history: reconstructed seed and era-adjusted observations",
             "training_labels_sha256": digest(labels_path),
             "training_rows": len(train), "training_seasons": [int(s) for s in seasons],
             "training_label_available_max": train.label_available_date.max().date().isoformat(),
             "features": columns, "signals": list(signals), "bags": len(predictions),
             "aggregation": "pointwise median of bag quantiles", "window": [w.date().isoformat() for w in window],
             "historical_vintage_policy": "frozen current snapshot, not exact historical issue vintages",
             "observed_peak_policy": "current-season maximum in Base history is fixed; revisions update next run",
             "bag_audits": bag_audits}
    write_json(checkpoint / "fit.json", audit)
    return pd.DataFrame(rows, columns=COLUMNS)
