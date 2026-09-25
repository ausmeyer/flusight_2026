"""Single-origin fits for the three fixed submission models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from . import core
from .contract import COLUMNS, ED, HOSP, QUANTILES, UNIT
from .data import NSSP, WW
from .ed_transform import boundary_epsilon, ed_logit
from .panel_ar import PanelARRuntime, PanelARSpec, forecast_panel_ar_anchor, prepare_panel_ar_table
from .util import write_json


def pooled_features(history: pd.DataFrame, snapshot: Path, reference: str,
                    runtime: core.RuntimeConfig, signals: tuple[str, ...], *, shifted=True):
    features = core.build_feature_table(history, runtime)
    sources = []
    if "ww" in signals:
        sources.append({"file": str(snapshot / "wastewater.csv"),
                        "columns": [WW, *[f"{WW}_lag{lag}" for lag in [1, 2, 4]]],
                        "source_lag_weeks": 1, "max_staleness_weeks": 1,
                        "add_availability_features": True})
    if "nssp" in signals:
        sources.append({"file": str(snapshot / "nssp.csv"), "columns": [NSSP],
                        "source_lag_weeks": 0, "max_staleness_weeks": 1,
                        "add_availability_features": True})
    features = core.add_external_covariate_sources(features, sources, pd.Timestamp(reference),
                                                    shift_stitched_proxy_dates=shifted)
    if "nssp" in signals:
        features = core.add_external_feature_lags(features, {NSSP: [1, 2, 4]})
    pooled = core.build_pooled_examples(features, runtime.max_horizons)
    return pooled, core.feature_columns(pooled)


def distributional(history: pd.DataFrame, snapshot: Path, reference: str, settings: dict,
                   signals: tuple[str, ...], target: str, component: str, checkpoint: Path,
                   location_map: dict) -> pd.DataFrame:
    runtime = core.RuntimeConfig.from_dict(settings["runtime"])
    anchor = pd.Timestamp(reference) - pd.Timedelta(weeks=1)
    pooled, columns = pooled_features(history, snapshot, reference, runtime, signals,
                                      shifted=target == HOSP or settings["ed_history"]["mode"] != "observed")
    train = pooled[(pooled.target_date <= anchor) & (pooled.date <= anchor)].copy()
    train = train.dropna(subset=["total_hosp", "target"]).reset_index(drop=True)
    test = pooled[pooled.date.eq(anchor)].reset_index(drop=True)
    if len(train) < runtime.min_train_rows or test.empty or test.total_hosp.isna().any():
        raise ValueError(f"Insufficient training data or missing forecast anchor: {component}")
    transform = None
    if target == ED:
        epsilon = boundary_epsilon(settings["ed_target_transform"])
        transform = {"name": "logit", "boundary_epsilon": epsilon,
                     "input_unit": "percentage points", "checkpoint_unit": "percentage points"}
        y = ed_logit(train.target.to_numpy() / 100, epsilon) - ed_logit(train.total_hosp.to_numpy() / 100, epsilon)
        anchor_link = ed_logit(test.total_hosp.to_numpy() / 100, epsilon)
    else:
        y = np.log1p(train.target.to_numpy()) - np.log1p(train.total_hosp.to_numpy())
        anchor_link = np.log1p(test.total_hosp.to_numpy())
    seasons = sorted(train.season.unique())
    bag_size = max(1, int(round(len(seasons) * runtime.bag_frac)))
    seed = int(settings["seed"])
    rng = np.random.default_rng(seed + int(anchor.value // 10**9))
    predictions = []
    audit = []
    checkpoint.mkdir(parents=True, exist_ok=True)
    if transform is not None:
        transform_path = checkpoint / "response-transform.json"
        if transform_path.exists():
            if json.loads(transform_path.read_text()) != transform:
                raise ValueError("ED checkpoint transformation changed; use a new checkpoint directory")
        elif any(checkpoint.glob("bag-*.npy")):
            raise ValueError("ED checkpoints lack response transformation metadata; use a new checkpoint directory")
        else:
            write_json(transform_path, transform)
    for bag in range(runtime.num_bags):
        sampled = rng.choice(seasons, size=bag_size, replace=False)
        path = checkpoint / f"bag-{bag:03d}.npy"
        if path.exists():
            quantiles = np.load(path, allow_pickle=False)
        else:
            mask = train.season.isin(sampled).to_numpy()
            if mask.sum() < runtime.min_train_rows:
                raise ValueError(f"{component} bag {bag}: insufficient rows; no bags may be silently skipped")
            print(f"{component}: bag {bag + 1}/{runtime.num_bags}", flush=True)
            center, scale = core.fit_production_lightgbmlss_one_bag(
                train.loc[mask, columns], y[mask], train.loc[mask, "target_date"].to_numpy(),
                runtime, seed + bag, fit_config_profile="joint_base", oof_audit=audit,
            )
            if any(row["fold"] in {"short_history_fallback", "primary_only_in_sample_fallback"}
                   for row in audit):
                raise ValueError("Out-of-fold uncertainty fit fell back to in-sample residuals")
            x = test[columns].astype(float)
            mu, sigma = center.predict(x), scale.predict_sigma(x)
            q_log = norm.ppf(QUANTILES[None, :], loc=mu[:, None], scale=sigma[:, None])
            q_log += anchor_link[:, None]
            quantiles = 100 * expit(q_log) if target == ED else np.maximum(np.expm1(q_log), 0)
            with path.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, quantiles, allow_pickle=False)
            path.with_suffix(".tmp").replace(path)
        if quantiles.shape != (len(test), len(QUANTILES)) or not np.isfinite(quantiles).all():
            raise ValueError(f"Invalid prediction/checkpoint in {component} bag {bag}")
        if target == ED and (np.any((quantiles < 0) | (quantiles > 100)) or np.any(np.diff(quantiles, axis=1) < 0)):
            raise ValueError(f"Unordered or out-of-bounds ED quantiles in bag {bag}")
        predictions.append(quantiles)
    # Preserve the study's pointwise median across season-subsampled fits.
    values = np.median(np.stack(predictions), axis=0)
    if target == ED:
        values = values / 100
    records = []
    for row, vector in zip(test.itertuples(index=False), values):
        for q, value in zip(QUANTILES, vector):
            records.append({"reference_date": reference, "target": target,
                            "horizon": int(row.horizon_weeks) - 1,
                            "target_end_date": row.target_date.date().isoformat(),
                            "location": location_map[row.location_name], "output_type": "quantile",
                            "output_type_id": q, "value": float(value)})
    write_json(checkpoint / "fit.json", {"component": component, "bags": len(predictions),
                                        "training_rows": len(train), "features": columns,
                                        "seed": seed, "signals": list(signals),
                                        "aggregation": "pointwise median of bag quantiles",
                                        "response_transform": transform or {"name": "log1p"}})
    return pd.DataFrame(records, columns=COLUMNS)


def linear(history: pd.DataFrame, reference: str, settings: dict, location_map: dict) -> pd.DataFrame:
    runtime = PanelARRuntime.from_dict(settings["panel_ar_runtime"])
    pooled = prepare_panel_ar_table(history, runtime)
    result = forecast_panel_ar_anchor(pooled, pd.Timestamp(reference) - pd.Timedelta(weeks=1),
                                      PanelARSpec("MIGHTE-Linear", partial_pooling=True), runtime, location_map)
    if result.empty:
        raise ValueError("MIGHTE-Linear produced no forecasts")
    return result[COLUMNS]


def equal_quantiles(components: list[pd.DataFrame]) -> pd.DataFrame:
    key = UNIT + ["output_type", "output_type_id"]
    series = []
    for component in components:
        if component.duplicated(key).any():
            raise ValueError("Duplicate ensemble component keys")
        series.append(component.set_index(key).value.sort_index())
    if len(series) != 3 or any(not series[0].index.equals(s.index) for s in series[1:]):
        raise ValueError("The equal-third ensemble requires all three components on identical keys")
    return ((series[0] + series[1] + series[2]) / 3).rename("value").reset_index()[COLUMNS]


def persistence_baseline(history: pd.DataFrame, reference: str, target: str, location_map: dict) -> pd.DataFrame:
    """Local comparison baseline: zero-centered, symmetric historical weekly changes.

    This is explicitly a local baseline, not the official FluSight-baseline.
    It is fitted and frozen alongside each prospective forecast.
    """
    rows = []
    anchor = pd.Timestamp(reference) - pd.Timedelta(weeks=1)
    for name, group in history[history.date.le(anchor)].groupby("location_name"):
        values = group.sort_values("date").total_hosp
        increments = values.diff().dropna().tail(104).to_numpy()
        errors = np.concatenate([increments, -increments, [0.]])
        for h in range(4):
            q = np.maximum(values.iloc[-1] + np.quantile(errors, QUANTILES) * np.sqrt(h + 1), 0)
            if target == ED:
                q = np.clip(q / 100, 0, 1)
            else:
                q = np.rint(q)
            for probability, value in zip(QUANTILES, q):
                rows.append({"reference_date": reference, "target": target, "horizon": h,
                              "target_end_date": (pd.Timestamp(reference) + pd.Timedelta(weeks=h)).date().isoformat(),
                              "location": location_map[name], "output_type": "quantile",
                              "output_type_id": probability, "value": float(value)})
    return pd.DataFrame(rows, columns=COLUMNS)
