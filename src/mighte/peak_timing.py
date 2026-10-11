"""Past-peak probability and conditional future peak timing for MIGHTE-Base."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp

from . import core
from .contract import COLUMNS, PEAK_WEEK, Contract
from .peak_height import seasonal_examples, tree_params
from .util import digest, write_json


DESIGN = "past-gate-conditional-log-lead-v1"


def future_probabilities(medians, sigmas, weeks, anchor):
    """Integrate lognormal lead time over the remaining weekly peak bins."""
    mu, sigma = np.asarray(medians, float), np.asarray(sigmas, float)
    if (mu.ndim != 1 or sigma.shape != mu.shape or not np.isfinite(mu).all()
            or not np.isfinite(sigma).all() or np.any(sigma <= 0)):
        raise ValueError("Invalid peak-timing median or spread")
    leads = (pd.DatetimeIndex(weeks) - pd.Timestamp(anchor)).days.to_numpy() / 7
    future = leads > 0
    probabilities = np.zeros((len(mu), len(weeks)))
    if not future.any():
        return probabilities
    k = leads[future]
    lower = (np.log(k[None, :] - .5) - mu[:, None]) / sigma[:, None]
    upper = (np.log(k[None, :] + .5) - mu[:, None]) / sigma[:, None]
    # Use survival probabilities in the right tail to avoid subtracting two ones.
    right = lower > 0
    larger = log_ndtr(np.where(right, -lower, upper))
    smaller = log_ndtr(np.where(right, -upper, lower))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mass = larger + np.log(-np.expm1(smaller - larger))
    probabilities[:, future] = np.exp(log_mass - logsumexp(log_mass, axis=1, keepdims=True))
    validate_probabilities(probabilities, (len(mu), len(weeks)))
    return probabilities


def validate_probabilities(probabilities, shape):
    p = np.asarray(probabilities)
    if (p.shape != shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Invalid peak-timing prediction/checkpoint")


def combine(q, future, history, locations, weeks, anchor):
    """Assign past probability to reported maxima and future probability to later weeks."""
    q, future = np.asarray(q, float).copy(), np.asarray(future, float)
    if (q.shape != (len(locations),) or future.shape != (len(locations), len(weeks))
            or not np.isfinite(q).all() or np.any((q < 0) | (q > 1))):
        raise ValueError("Invalid peak-timing gate or future probabilities")
    dates = pd.DatetimeIndex(weeks)
    future_mask = dates > pd.Timestamp(anchor)
    if not np.isfinite(future).all() or np.any(future[:, ~future_mask] != 0):
        raise ValueError("Invalid peak-timing future probabilities")
    if future_mask.any():
        validate_probabilities(future, future.shape)
    result = np.zeros_like(future)
    for index, location in enumerate(locations):
        observed = history[history.location_name.eq(location) & history.date.le(anchor)].set_index("date").total_hosp
        values = observed.reindex(dates)
        known = values.notna().to_numpy() & ~future_mask
        if known.any():
            maximum = known & values.eq(values[known].max()).to_numpy()
            if not future_mask.any():
                q[index] = 1.
            result[index, maximum] = q[index] / maximum.sum()
        else:
            q[index] = 0.
        result[index] += (1 - q[index]) * future[index]
    validate_probabilities(result, result.shape)
    return result, q


def season_weights(frame):
    counts = frame.groupby(["location_name", "season_year"]).location_name.transform("size").to_numpy()
    weights = 1 / counts
    return weights / weights.mean()


def training_targets(frame):
    leads = (frame.season_peak_date - frame.date).dt.days.to_numpy() / 7
    if not np.isfinite(leads).all() or not np.allclose(leads, np.round(leads)):
        raise ValueError("Peak-timing labels must be whole weeks from the observed anchor")
    return leads <= 0, leads


def fit_gate(train, columns, already, runtime, seed):
    params = tree_params(seed, runtime)
    params.update(objective="binary", metric="binary_logloss")
    return core.fit_lgbm_regressor(train[columns], already, params, runtime.stage1_rounds,
                                   sample_weight=season_weights(train))


def fit_median(train, columns, y, runtime, seed):
    params = tree_params(seed, runtime)
    params.update(objective="quantile", metric="quantile", alpha=.5)
    return core.fit_lgbm_regressor(train[columns], y, params, runtime.stage1_rounds,
                                   sample_weight=season_weights(train))


def forward_medians(train, columns, y, runtime, seed):
    """Learn spread from median errors on wholly unseen, later seasons."""
    predictions = np.full(len(train), np.nan)
    audit = []
    for year in sorted(train.season_year.unique())[1:]:
        fit, held = train.season_year.lt(year).to_numpy(), train.season_year.eq(year).to_numpy()
        if fit.sum() < runtime.min_train_rows:
            continue
        model = fit_median(train.loc[fit], columns, y[fit], runtime, seed + int(year))
        predictions[held] = model.predict(train.loc[held, columns].astype(float))
        audit.append({"training_seasons": sorted(int(x) for x in train.loc[fit, "season_year"].unique()),
                      "validation_season": int(year), "fit_rows": int(fit.sum()), "validation_rows": int(held.sum()),
                      "training_label_available_max": train.loc[fit, "label_available_date"].max().date().isoformat(),
                      "validation_origin_min": train.loc[held, "date"].min().date().isoformat()})
    valid = np.isfinite(predictions)
    if valid.sum() < 50:
        raise ValueError("Peak timing has too few whole-season forward predictions to learn uncertainty")
    return valid, predictions, audit


def predict(context):
    reference, settings = context["reference_date"], context["settings"]
    snapshot, checkpoint = Path(context["snapshot_path"]), Path(context["checkpoint_path"])
    contract = Contract(snapshot / "contract")
    location_map = dict(zip(contract.locations.location_name, contract.locations.location))
    populations = dict(zip(contract.locations.location_name, contract.locations.population))
    weeks = sorted(contract.by_target[PEAK_WEEK]["output_type"]["pmf"]["output_type_id"]["required"])
    window = (pd.Timestamp(weeks[0]), pd.Timestamp(weeks[-1]))
    if pd.Timestamp(reference) > window[1]:
        raise ValueError("Peak-timing reference date is after the target season")
    runtime = core.RuntimeConfig.from_dict({**settings["runtime"], "num_threads": settings["threads"]})
    signals = context.get("signals", ("ww", "nssp"))
    history = context["hospitalization_history"]
    examples, columns = seasonal_examples(history, snapshot, reference, runtime, signals, populations, window)
    anchor = pd.Timestamp(reference) - pd.Timedelta(weeks=1)
    eligible = examples.seasonal_origin & examples.peak_count.notna() & examples.label_available_date.le(anchor)
    train = examples[eligible & examples.peak_fraction.notna() & examples.total_hosp.notna()].reset_index(drop=True)
    test = examples[examples.date.eq(anchor)].copy()
    if context.get("locations") is not None:
        test = test[test.location_name.isin(context["locations"])]
    test = test.sort_values("location_name").reset_index(drop=True)
    seasons = sorted(train.season_year.unique())
    if len(train) < runtime.min_train_rows or len(seasons) < 3 or test.empty:
        raise ValueError("Insufficient complete, unique-peak seasons or no peak-timing forecast locations")
    checkpoint.mkdir(parents=True, exist_ok=True)
    label_columns = ["location_name", "season_year", "season_peak_date", "peak_fraction", "label_available_date"]
    labels_path = checkpoint / "training-labels.csv"
    train[label_columns].drop_duplicates().sort_values(["season_year", "location_name"]).to_csv(labels_path, index=False)
    already, leads = training_targets(train)
    x_test = test[columns].astype(float)
    seed = int(settings["seed"])
    rng = np.random.default_rng(seed + int(anchor.value // 10**9))
    bag_size = max(3, int(round(len(seasons) * runtime.bag_frac)))
    predictions, bag_audits = [], []
    for bag in range(runtime.num_bags):
        sampled = sorted(rng.choice(seasons, size=bag_size, replace=False))
        path = checkpoint / f"bag-{bag:03d}.npy"
        if path.exists():
            bag_audit = json.loads(path.with_suffix(".json").read_text())
            if bag_audit.get("design") != DESIGN:
                raise ValueError("Peak-timing checkpoint uses a different model design; start a new run")
            prediction = np.load(path, allow_pickle=False)
        else:
            selected = train.season_year.isin(sampled).to_numpy()
            subset = train.loc[selected].reset_index(drop=True)
            gate = fit_gate(subset, columns, already[selected], runtime, seed + bag)
            future = selected & ~already
            remaining, outcome = train.loc[future].reset_index(drop=True), np.log(leads[future])
            valid, medians, folds = forward_medians(remaining, columns, outcome, runtime, seed + bag)
            center = fit_median(remaining, columns, outcome, runtime, seed + bag)
            scale = core.fit_conditional_gaussian_scale_booster(remaining.loc[valid, columns], outcome[valid], medians[valid],
                tree_params(seed + bag, runtime, spread=True), runtime.stage2_rounds,
                sample_weight=season_weights(remaining.loc[valid]))
            q = gate.predict(x_test)
            mu, sigma = center.predict(x_test), scale.predict_sigma(x_test)
            prediction, past_mass = combine(q, future_probabilities(mu, sigma, weeks, anchor),
                                             history, test.location_name, weeks, anchor)
            bag_audit = {"bag": bag, "design": DESIGN, "seasons": [int(s) for s in sampled], "folds": folds,
                         "omitted_seasons": [int(s) for s in seasons if s not in sampled],
                         "training_origin_rows": int(selected.sum()),
                         "gate_positive_rows": int(already[selected].sum()),
                         "conditional_rows": len(remaining), "minimum_future_lead": float(leads[future].min()),
                         "training_label_available_max": subset.label_available_date.max().date().isoformat(),
                         "spread_training_rows": int(valid.sum()),
                         "forecast_parameters": [{"location": row.location_name, "past_peak_probability": float(past),
                                                  "applied_past_probability": float(applied),
                                                  "log_lead_median": float(median), "log_lead_sigma": float(sd)}
                                                 for row, past, applied, median, sd in
                                                 zip(test.itertuples(index=False), q, past_mass, mu, sigma)]}
            with path.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, prediction, allow_pickle=False)
            path.with_suffix(".tmp").replace(path)
            write_json(path.with_suffix(".json"), bag_audit)
            print(f"base-peak-timing: bag {bag + 1}/{runtime.num_bags}", flush=True)
        validate_probabilities(prediction, (len(test), len(weeks)))
        predictions.append(prediction)
        bag_audits.append(bag_audit)
    values = np.mean(predictions, axis=0)
    validate_probabilities(values, (len(test), len(weeks)))
    audit = {"component": "MIGHTE-Base peak timing", "design": DESIGN,
             "gate_objective": "binary log loss: final peak on or before observed anchor",
             "median_objective": "pinball loss at 0.5", "spread_objective": "Gaussian NLL around held-season medians",
             "label_source": "unique maxima of complete Base reconstructed and era-adjusted hospitalization history",
             "training_rows": len(train), "training_labels_sha256": digest(labels_path),
             "training_seasons": [int(s) for s in seasons],
             "training_label_available_max": train.label_available_date.max().date().isoformat(),
             "tied_location_seasons_excluded": len(examples.loc[eligible & examples.peak_ties.gt(1),
                                                               ["location_name", "season_year"]].drop_duplicates()),
             "response_transform": "log((final peak date - observed anchor date) / 7 days), conditional on positive lead",
             "features": columns, "signals": list(signals),
             "median_rounds": runtime.stage1_rounds, "spread_rounds": runtime.stage2_rounds,
             "bags": len(predictions), "aggregation": "arithmetic mean of bag probabilities",
             "weighting": "equal total weight per location-season in each median and spread fit",
             "distribution": "past-peak classifier plus conditional lognormal future weekly-bin mass",
             "window": weeks, "historical_vintage_policy": "frozen current snapshot, not exact historical issue vintages",
             "observed_peak_policy": "past mass split over reported maxima; no observed seasonal values implies future-only; terminal support implies past-only",
             "bag_audits": bag_audits}
    write_json(checkpoint / "fit.json", audit)
    return pd.DataFrame([{"reference_date": reference, "target": PEAK_WEEK, "horizon": np.nan,
                          "target_end_date": None, "location": location_map[row.location_name], "output_type": "pmf",
                          "output_type_id": week, "value": float(value)}
                         for row, vector in zip(test.itertuples(index=False), values)
                         for week, value in zip(weeks, vector)], columns=COLUMNS)
