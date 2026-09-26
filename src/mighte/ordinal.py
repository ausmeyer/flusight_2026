"""Prospective MIGHTE-Base-Ordinal: direct five-class LightGBM probabilities."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import core
from .contract import CATEGORIES, COLUMNS, TREND, Contract
from .models import pooled_features
from .util import write_json

STABLE_RATES = np.array([.3, .5, .7, 1.0])
LARGE_RATES = np.array([1.7, 3.0, 4.0, 5.0])


def category_labels(future, baseline, population, horizon):
    """CDC categories, ordered from large decrease to large increase."""
    future, baseline, population, horizon = np.broadcast_arrays(
        np.asarray(future, float), np.asarray(baseline, float),
        np.asarray(population, float), np.asarray(horizon))
    if np.any(~np.isfinite(population) | (population <= 0)):
        raise ValueError("Population must be finite and positive")
    if np.any(~np.isin(horizon, [0, 1, 2, 3])):
        raise ValueError("Hub horizons must be 0-3")
    if np.any(~np.isfinite(future) | ~np.isfinite(baseline) | (future < 0) | (baseline < 0)):
        raise ValueError("Counts must be finite and nonnegative")
    h = horizon.astype(int)
    small = np.maximum(10, population * STABLE_RATES[h] / 1e5)
    large = np.maximum(10, population * LARGE_RATES[h] / 1e5)
    delta = future - baseline
    return np.select([delta <= -large, delta <= -small, delta < small, delta < large],
                     [0, 1, 2, 3], default=4).astype(int)


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, float)
    if p.ndim != 2 or p.shape[1] != len(CATEGORIES) or not np.isfinite(p).all():
        raise ValueError("Expected five finite category probabilities per instance")
    if np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Category probabilities must lie in [0,1] and sum to one")


def fit_ordinal_bags(pooled, feature_cols, anchor, runtime, seed, population_map, bag_dir: Path):
    """Port of the study classifier, with the same season draws and tree settings."""
    train = pooled[(pooled.target_date <= anchor) & (pooled.date <= anchor)].dropna(
        subset=["total_hosp", "target"]).copy()
    test = pooled[(pooled.date == anchor) & (pooled.horizon_weeks <= runtime.max_horizons)].copy()
    if len(train) < runtime.min_train_rows or test.empty or test.total_hosp.isna().any():
        raise ValueError("Insufficient ordinal training data or missing forecast anchor")
    populations = train.location_name.map(population_map).to_numpy()
    labels = category_labels(train.target, train.total_hosp, populations, train.horizon_weeks - 1)
    seasons = sorted(train.season.dropna().unique())
    rng = np.random.default_rng(int(seed) + int(anchor.value // 10**9))
    bag_size = max(1, int(round(len(seasons) * runtime.bag_frac)))
    x, x_test = train.loc[:, feature_cols].astype(float), test.loc[:, feature_cols].astype(float)
    bag_dir.mkdir(parents=True, exist_ok=True)
    probabilities = []
    for bag in range(runtime.num_bags):
        sampled = rng.choice(seasons, size=bag_size, replace=False)
        path = bag_dir / f"bag-{bag:03d}.npy"
        if path.exists():
            prediction = np.load(path, allow_pickle=False)
        else:
            mask = train.season.isin(sampled).to_numpy()
            if mask.sum() < runtime.min_train_rows:
                raise ValueError(f"Ordinal bag {bag}: insufficient training rows")
            params = core.central_model_params(seed + bag, runtime)
            params.update(objective="multiclass", num_class=5, metric="multi_logloss")
            dataset = lgb.Dataset(x.loc[mask], label=labels[mask], params={"verbose": -1})
            model = lgb.train(params, dataset, num_boost_round=runtime.stage1_rounds, callbacks=[])
            prediction = model.predict(x_test, num_threads=runtime.num_threads)
            validate_probabilities(prediction)
            with path.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, prediction, allow_pickle=False)
            path.with_suffix(".tmp").replace(path)
            print(f"base-ordinal: bag {bag + 1}/{runtime.num_bags}", flush=True)
        validate_probabilities(prediction)
        if prediction.shape != (len(test), len(CATEGORIES)):
            raise ValueError("Ordinal checkpoint has incompatible forecast support")
        probabilities.append(prediction)
    result = np.mean(probabilities, axis=0)
    validate_probabilities(result)
    audit = {"component": "MIGHTE-Base-Ordinal", "objective": "multiclass",
             "anchor_date": anchor.date().isoformat(), "training_rows": len(train), "test_rows": len(test),
             "features": list(feature_cols), "training_target_max": train.target_date.max().date().isoformat(),
             "training_class_counts": dict(zip(CATEGORIES, np.bincount(labels, minlength=5).tolist())),
             "bags": len(probabilities), "seed": int(seed), "boosting_rounds": runtime.stage1_rounds,
             "aggregation": "arithmetic mean of bag probabilities", "population_by_location": population_map,
             "stable_rates_per_100k": STABLE_RATES.tolist(), "large_rates_per_100k": LARGE_RATES.tolist(),
             "stable_count_threshold": 10, "baseline_reference_offset_days": -7}
    write_json(bag_dir / "fit.json", audit)
    return test, result, audit


def predict(context):
    """Fit one current origin using the frozen snapshot; no quantile conversion."""
    reference, settings = context["reference_date"], context["settings"]
    snapshot = Path(context["snapshot_path"])
    contract = Contract(snapshot / "contract")
    location_map = dict(zip(contract.locations.location_name, contract.locations.location))
    populations = dict(zip(contract.locations.location_name, contract.locations.population))
    runtime = core.RuntimeConfig.from_dict({**settings["runtime"], "num_threads": settings["threads"]})
    anchor = pd.Timestamp(reference) - pd.Timedelta(weeks=1)
    history = context["hospitalization_history"].copy()
    history = history[pd.to_datetime(history.date).le(anchor)]
    pooled, columns = pooled_features(history, snapshot, reference, runtime, ("ww", "nssp"))
    test, probabilities, _ = fit_ordinal_bags(pooled, columns, anchor, runtime, int(settings["seed"]),
                                             populations, Path(context["checkpoint_path"]))
    rows = []
    for row, vector in zip(test.itertuples(index=False), probabilities):
        for category, value in zip(CATEGORIES, vector):
            rows.append({"reference_date": reference, "target": TREND, "horizon": int(row.horizon_weeks) - 1,
                         "target_end_date": row.target_date.date().isoformat(),
                         "location": location_map[row.location_name], "output_type": "pmf",
                         "output_type_id": category, "value": float(value)})
    return pd.DataFrame(rows, columns=COLUMNS)
