#!/usr/bin/env python3
"""Leakage-safe pooled distributional panel autoregression.

The response for an origin ``t`` and lead ``h`` is
``log1p(y[t+h]) - log1p(y[t])``.  The newest autoregressive input is explicitly
``log1p(y[t]) - log1p(y[t-1])``.  Thus integration returns forecasts to the
level scale without exposing ``y[t+1]`` (or any later observation) to the
feature row at ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .core import infer_flu_season
from .contract import QUANTILES as CDC_QUANTILES, HOSP as TARGET_NAME


@dataclass(frozen=True)
class PanelARRuntime:
    max_horizons: int
    anchor_step_weeks: int
    min_train_rows: int
    ar_weeks: Tuple[int, ...]
    rolling_windows: Tuple[int, ...]
    mean_ridge: float
    location_intercept_ridge: float
    horizon_interaction_ridge: float
    partial_slope_ridge: float
    scale_ridge: float
    joint_iterations: int
    scale_max_iterations: int

    @classmethod
    def from_dict(cls, raw: dict) -> "PanelARRuntime":
        runtime = cls(
            max_horizons=int(raw.get("max_horizons", 4)),
            anchor_step_weeks=int(raw.get("anchor_step_weeks", 1)),
            min_train_rows=int(raw.get("min_train_rows", 1000)),
            ar_weeks=tuple(int(value) for value in raw.get("ar_weeks", [1, 2, 3, 4, 8, 12, 52])),
            rolling_windows=tuple(int(value) for value in raw.get("rolling_windows", [2, 4, 8, 12])),
            mean_ridge=float(raw.get("mean_ridge", 0.01)),
            location_intercept_ridge=float(raw.get("location_intercept_ridge", 0.02)),
            horizon_interaction_ridge=float(raw.get("horizon_interaction_ridge", 0.02)),
            partial_slope_ridge=float(raw.get("partial_slope_ridge", 0.20)),
            scale_ridge=float(raw.get("scale_ridge", 0.02)),
            joint_iterations=int(raw.get("joint_iterations", 5)),
            scale_max_iterations=int(raw.get("scale_max_iterations", 100)),
        )
        if not runtime.ar_weeks or any(week < 1 for week in runtime.ar_weeks):
            raise ValueError("panel_ar_runtime.ar_weeks must contain positive week numbers")
        if not runtime.rolling_windows or any(window < 2 for window in runtime.rolling_windows):
            raise ValueError("panel_ar_runtime.rolling_windows must be at least two weeks")
        if runtime.max_horizons < 1 or runtime.joint_iterations < 1:
            raise ValueError("Panel AR horizons and joint iterations must be positive")
        for value in (
            runtime.mean_ridge,
            runtime.location_intercept_ridge,
            runtime.horizon_interaction_ridge,
            runtime.partial_slope_ridge,
            runtime.scale_ridge,
        ):
            if value < 0.0:
                raise ValueError("Panel AR ridge penalties must be nonnegative")
        return runtime


@dataclass(frozen=True)
class PanelARSpec:
    model_id: str
    partial_pooling: bool
    distribution: str = "normal"

    @classmethod
    def from_dict(cls, raw: dict) -> "PanelARSpec":
        spec = cls(
            model_id=str(raw["model_id"]),
            partial_pooling=bool(raw.get("partial_pooling", False)),
            distribution=str(raw.get("distribution", "normal")),
        )
        if spec.distribution != "normal":
            raise ValueError("The initial panel AR implementation supports normal distributions only")
        return spec


def build_panel_ar_feature_table(
    df_long: pd.DataFrame,
    runtime: PanelARRuntime,
) -> pd.DataFrame:
    """Build origin-indexed features using observations no later than each row date."""

    df = df_long.sort_values(["location_name", "date"]).reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["panel_log_level_t"] = np.log1p(np.clip(df["total_hosp"].astype(float), 0.0, None))
    by_location = df.groupby("location_name", group_keys=False)

    # This is the crucial forecast-time invariant: at row t, delta_log_t uses
    # y[t] and y[t-1].  Every AR week below uses a nonnegative shift of it.
    df["panel_delta_log_t"] = by_location["panel_log_level_t"].diff(1)
    delta_by_location = df.groupby("location_name", group_keys=False)["panel_delta_log_t"]
    for week in runtime.ar_weeks:
        df[f"panel_delta_log_week_{week}"] = delta_by_location.shift(week - 1)
    for window in runtime.rolling_windows:
        df[f"panel_delta_roll_mean_{window}"] = delta_by_location.transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        df[f"panel_delta_roll_std_{window}"] = delta_by_location.transform(
            lambda values: values.rolling(window, min_periods=window).std()
        )

    delta_pivot = df.pivot(
        index="date", columns="location_name", values="panel_delta_log_t"
    ).sort_index()
    state_delta = delta_pivot.drop(columns=["US"], errors="ignore")
    if state_delta.shape[1] == 0:
        state_delta = delta_pivot
    national_mean = state_delta.mean(axis=1)
    national_std = state_delta.std(axis=1)
    national = pd.DataFrame(index=delta_pivot.index)
    for week in runtime.ar_weeks:
        national[f"panel_nat_delta_log_week_{week}"] = national_mean.shift(week - 1)
    national["panel_nat_delta_std_week_1"] = national_std
    df = df.merge(national.reset_index(), on="date", how="left")

    phase = df["date"].dt.dayofyear.astype(float) / 365.25
    df["panel_origin_week_sin"] = np.sin(2.0 * np.pi * phase)
    df["panel_origin_week_cos"] = np.cos(2.0 * np.pi * phase)
    df["panel_origin_quarter_sin"] = np.sin(8.0 * np.pi * phase)
    df["panel_origin_quarter_cos"] = np.cos(8.0 * np.pi * phase)
    month = df["date"].dt.month
    df["panel_origin_is_flu"] = ((month >= 10) | (month <= 3)).astype(float)
    df["panel_origin_is_peak"] = month.isin([12, 1, 2]).astype(float)
    df["panel_regime_pre_2022"] = (df["date"] < pd.Timestamp("2022-07-01")).astype(float)
    df["panel_regime_2022_2024"] = (
        (df["date"] >= pd.Timestamp("2022-07-01"))
        & (df["date"] < pd.Timestamp("2024-11-01"))
    ).astype(float)
    df["panel_regime_post_2024"] = (df["date"] >= pd.Timestamp("2024-11-01")).astype(float)
    df["season"] = df["date"].map(infer_flu_season)
    return df


def build_panel_ar_examples(
    feature_df: pd.DataFrame,
    max_horizons: int,
) -> pd.DataFrame:
    """Stack direct leads; future values appear only in ``panel_target_delta_log``."""

    future_levels = feature_df.loc[
        :, ["location_name", "date", "panel_log_level_t"]
    ].rename(
        columns={
            "date": "target_date",
            "panel_log_level_t": "panel_target_log_level",
        }
    )
    parts = []
    for lead in range(1, max_horizons + 1):
        frame = feature_df.copy()
        frame["lead_weeks"] = lead
        frame["target_date"] = frame["date"] + pd.Timedelta(weeks=lead)
        # Match the realized target by its exact calendar date.  A positional
        # group shift can silently select a later week when a vintage has a
        # missing date, mislabeling that value as the intended horizon.
        frame = frame.merge(
            future_levels,
            on=["location_name", "target_date"],
            how="left",
            validate="many_to_one",
        )
        frame["panel_target_delta_log"] = (
            frame["panel_target_log_level"] - frame["panel_log_level_t"]
        )
        frame = frame.drop(columns="panel_target_log_level")
        target_phase = frame["target_date"].dt.dayofyear.astype(float) / 365.25
        frame["panel_target_week_sin"] = np.sin(2.0 * np.pi * target_phase)
        frame["panel_target_week_cos"] = np.cos(2.0 * np.pi * target_phase)
        frame["panel_target_quarter_sin"] = np.sin(8.0 * np.pi * target_phase)
        frame["panel_target_quarter_cos"] = np.cos(8.0 * np.pi * target_phase)
        target_month = frame["target_date"].dt.month
        frame["panel_target_is_flu"] = ((target_month >= 10) | (target_month <= 3)).astype(float)
        frame["panel_target_is_peak"] = target_month.isin([12, 1, 2]).astype(float)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def panel_ar_continuous_columns(runtime: PanelARRuntime) -> List[str]:
    columns = ["panel_log_level_t"]
    columns.extend(f"panel_delta_log_week_{week}" for week in runtime.ar_weeks)
    for window in runtime.rolling_windows:
        columns.extend(
            [f"panel_delta_roll_mean_{window}", f"panel_delta_roll_std_{window}"]
        )
    columns.extend(f"panel_nat_delta_log_week_{week}" for week in runtime.ar_weeks)
    columns.extend(
        [
            "panel_nat_delta_std_week_1",
            "panel_origin_week_sin",
            "panel_origin_week_cos",
            "panel_origin_quarter_sin",
            "panel_origin_quarter_cos",
            "panel_target_week_sin",
            "panel_target_week_cos",
            "panel_target_quarter_sin",
            "panel_target_quarter_cos",
        ]
    )
    return columns


def panel_ar_binary_columns() -> List[str]:
    return [
        "panel_origin_is_flu",
        "panel_origin_is_peak",
        "panel_target_is_flu",
        "panel_target_is_peak",
        "panel_regime_pre_2022",
        "panel_regime_2022_2024",
        "panel_regime_post_2024",
    ]


def panel_ar_horizon_interaction_columns(runtime: PanelARRuntime) -> List[str]:
    columns = ["panel_log_level_t"]
    columns.extend(f"panel_delta_log_week_{week}" for week in runtime.ar_weeks)
    columns.extend(f"panel_delta_roll_mean_{window}" for window in runtime.rolling_windows)
    columns.extend(f"panel_nat_delta_log_week_{week}" for week in runtime.ar_weeks)
    return columns


def _standardize_from_training(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = train.loc[:, columns].mean(axis=0)
    scale = train.loc[:, columns].std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return (
        (train.loc[:, columns] - mean) / scale,
        (test.loc[:, columns] - mean) / scale,
    )


def build_panel_ar_mean_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    runtime: PanelARRuntime,
    spec: PanelARSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    continuous = panel_ar_continuous_columns(runtime)
    binary = panel_ar_binary_columns()
    train_scaled, test_scaled = _standardize_from_training(train, test, continuous)
    locations = sorted(set(train["location_name"].astype(str)) | set(test["location_name"].astype(str)))
    reference_location = locations[0]

    train_parts = [np.ones((len(train), 1)), train_scaled.to_numpy(dtype=float)]
    test_parts = [np.ones((len(test), 1)), test_scaled.to_numpy(dtype=float)]
    names = ["intercept", *continuous]
    penalties = [0.0, *([runtime.mean_ridge] * len(continuous))]

    train_parts.append(train.loc[:, binary].to_numpy(dtype=float))
    test_parts.append(test.loc[:, binary].to_numpy(dtype=float))
    names.extend(binary)
    penalties.extend([runtime.mean_ridge] * len(binary))

    for lead in range(2, runtime.max_horizons + 1):
        train_indicator = train["lead_weeks"].eq(lead).to_numpy(dtype=float)[:, None]
        test_indicator = test["lead_weeks"].eq(lead).to_numpy(dtype=float)[:, None]
        train_parts.append(train_indicator)
        test_parts.append(test_indicator)
        names.append(f"lead_{lead}")
        penalties.append(runtime.mean_ridge)

    for location in locations:
        if location == reference_location:
            continue
        train_indicator = train["location_name"].astype(str).eq(location).to_numpy(dtype=float)[:, None]
        test_indicator = test["location_name"].astype(str).eq(location).to_numpy(dtype=float)[:, None]
        train_parts.append(train_indicator)
        test_parts.append(test_indicator)
        names.append(f"location_{location}")
        penalties.append(runtime.location_intercept_ridge)

    continuous_index = {name: index for index, name in enumerate(continuous)}
    for lead in range(2, runtime.max_horizons + 1):
        train_indicator = train["lead_weeks"].eq(lead).to_numpy(dtype=float)[:, None]
        test_indicator = test["lead_weeks"].eq(lead).to_numpy(dtype=float)[:, None]
        for column in panel_ar_horizon_interaction_columns(runtime):
            index = continuous_index[column]
            train_parts.append(train_scaled.iloc[:, [index]].to_numpy(dtype=float) * train_indicator)
            test_parts.append(test_scaled.iloc[:, [index]].to_numpy(dtype=float) * test_indicator)
            names.append(f"lead_{lead}_x_{column}")
            penalties.append(runtime.horizon_interaction_ridge)

    if spec.partial_pooling:
        partial_columns = [
            "panel_log_level_t",
            "panel_delta_log_week_1",
            "panel_delta_log_week_2",
        ]
        for location in locations:
            if location == reference_location:
                continue
            train_indicator = train["location_name"].astype(str).eq(location).to_numpy(dtype=float)[:, None]
            test_indicator = test["location_name"].astype(str).eq(location).to_numpy(dtype=float)[:, None]
            for column in partial_columns:
                index = continuous_index[column]
                train_parts.append(train_scaled.iloc[:, [index]].to_numpy(dtype=float) * train_indicator)
                test_parts.append(test_scaled.iloc[:, [index]].to_numpy(dtype=float) * test_indicator)
                names.append(f"location_{location}_x_{column}")
                penalties.append(runtime.partial_slope_ridge)

    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(test_parts, axis=1),
        np.asarray(penalties, dtype=float),
        names,
    )


def build_panel_ar_scale_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    runtime: PanelARRuntime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    scale_window = 4 if 4 in runtime.rolling_windows else runtime.rolling_windows[0]
    scale_columns = [
        "panel_log_level_t",
        "panel_delta_log_week_1",
        f"panel_delta_roll_std_{scale_window}",
        "panel_nat_delta_std_week_1",
    ]
    train_scale = train.loc[:, scale_columns].copy()
    test_scale = test.loc[:, scale_columns].copy()
    train_scale["panel_delta_log_week_1"] = train_scale["panel_delta_log_week_1"].abs()
    test_scale["panel_delta_log_week_1"] = test_scale["panel_delta_log_week_1"].abs()
    train_scaled, test_scaled = _standardize_from_training(train_scale, test_scale, scale_columns)

    train_parts = [np.ones((len(train), 1)), train_scaled.to_numpy(dtype=float)]
    test_parts = [np.ones((len(test), 1)), test_scaled.to_numpy(dtype=float)]
    names = ["scale_intercept", *scale_columns]
    penalties = [0.0, *([runtime.scale_ridge] * len(scale_columns))]
    for lead in range(2, runtime.max_horizons + 1):
        train_parts.append(train["lead_weeks"].eq(lead).to_numpy(dtype=float)[:, None])
        test_parts.append(test["lead_weeks"].eq(lead).to_numpy(dtype=float)[:, None])
        names.append(f"scale_lead_{lead}")
        penalties.append(runtime.scale_ridge)
    for column in ["panel_target_is_flu", "panel_target_is_peak"]:
        train_parts.append(train[[column]].to_numpy(dtype=float))
        test_parts.append(test[[column]].to_numpy(dtype=float))
        names.append(f"scale_{column}")
        penalties.append(runtime.scale_ridge)
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(test_parts, axis=1),
        np.asarray(penalties, dtype=float),
        names,
    )


def solve_penalized_wls(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    penalties: np.ndarray,
) -> np.ndarray:
    n_rows = max(1, len(target))
    weights = np.asarray(weights, dtype=float)
    weighted_design = design * weights[:, None]
    system = design.T @ weighted_design / n_rows
    system.flat[:: system.shape[0] + 1] += penalties + 1e-10
    rhs = design.T @ (weights * target) / n_rows
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def fit_log_scale_coefficients(
    scale_design: np.ndarray,
    residual: np.ndarray,
    penalties: np.ndarray,
    initial: np.ndarray,
    max_iterations: int,
) -> np.ndarray:
    n_rows = max(1, len(residual))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        raw_eta = scale_design @ coefficients
        eta = np.clip(raw_eta, -10.0, 10.0)
        squared_standardized = np.square(residual) * np.exp(-2.0 * eta)
        loss = float(np.mean(eta + 0.5 * squared_standardized))
        loss += float(0.5 * np.sum(penalties * np.square(coefficients)))
        eta_gradient = 1.0 - squared_standardized
        active = ((raw_eta > -10.0) & (raw_eta < 10.0)).astype(float)
        gradient = scale_design.T @ (eta_gradient * active) / n_rows
        gradient += penalties * coefficients
        return loss, gradient

    result = optimize.minimize(
        fun=lambda coefficients: objective(coefficients)[0],
        x0=initial,
        jac=lambda coefficients: objective(coefficients)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations), "ftol": 1e-10},
    )
    if not np.all(np.isfinite(result.x)):
        raise ValueError("Panel AR scale optimization returned non-finite coefficients")
    return np.asarray(result.x, dtype=float)


def fit_distributional_panel_ar(
    mean_design: np.ndarray,
    scale_design: np.ndarray,
    target: np.ndarray,
    mean_penalties: np.ndarray,
    scale_penalties: np.ndarray,
    runtime: PanelARRuntime,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate-minimize penalized Gaussian NLL over linear mu and log-sigma."""

    weights = np.ones(len(target), dtype=float)
    mean_coefficients = solve_penalized_wls(
        mean_design, target, weights, mean_penalties
    )
    residual = target - mean_design @ mean_coefficients
    initial_scale = max(float(np.sqrt(np.mean(np.square(residual)))), 1e-4)
    scale_coefficients = np.zeros(scale_design.shape[1], dtype=float)
    scale_coefficients[0] = np.log(initial_scale)

    for _ in range(runtime.joint_iterations):
        residual = target - mean_design @ mean_coefficients
        scale_coefficients = fit_log_scale_coefficients(
            scale_design,
            residual,
            scale_penalties,
            scale_coefficients,
            runtime.scale_max_iterations,
        )
        log_sigma = np.clip(scale_design @ scale_coefficients, -8.0, 8.0)
        weights = np.clip(np.exp(-2.0 * log_sigma), 1e-4, 1e4)
        weights /= max(float(np.mean(weights)), 1e-12)
        mean_coefficients = solve_penalized_wls(
            mean_design, target, weights, mean_penalties
        )

    residual = target - mean_design @ mean_coefficients
    scale_coefficients = fit_log_scale_coefficients(
        scale_design,
        residual,
        scale_penalties,
        scale_coefficients,
        runtime.scale_max_iterations,
    )
    return mean_coefficients, scale_coefficients


def panel_ar_training_and_test_rows(
    pooled: pd.DataFrame,
    anchor_date: pd.Timestamp,
    runtime: PanelARRuntime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict realized targets and predictors to their forecast-time support."""

    anchor = pd.Timestamp(anchor_date)
    train = pooled[
        (pooled["target_date"] <= anchor) & (pooled["date"] <= anchor)
    ].copy()
    test = pooled[
        (pooled["date"] == anchor)
        & (pooled["lead_weeks"] <= runtime.max_horizons)
    ].copy()
    return train, test


def forecast_panel_ar_anchor(
    pooled: pd.DataFrame,
    anchor_date: pd.Timestamp,
    spec: PanelARSpec,
    runtime: PanelARRuntime,
    location_map: Dict[str, str],
) -> pd.DataFrame:
    train, test = panel_ar_training_and_test_rows(pooled, anchor_date, runtime)
    required = list(dict.fromkeys(
        panel_ar_continuous_columns(runtime)
        + panel_ar_binary_columns()
        + ["panel_target_delta_log"]
    ))
    train = train.dropna(subset=required).reset_index(drop=True)
    test_required = [column for column in required if column != "panel_target_delta_log"]
    test = test.dropna(subset=test_required).reset_index(drop=True)
    if len(train) < runtime.min_train_rows or test.empty:
        return pd.DataFrame()
    if pd.to_datetime(test["date"]).max() > pd.Timestamp(anchor_date):
        raise ValueError("Panel AR test features extend beyond the forecast anchor")

    mean_train, mean_test, mean_penalties, _ = build_panel_ar_mean_design(
        train, test, runtime, spec
    )
    scale_train, scale_test, scale_penalties, _ = build_panel_ar_scale_design(
        train, test, runtime
    )
    target = train["panel_target_delta_log"].to_numpy(dtype=float)
    mean_coefficients, scale_coefficients = fit_distributional_panel_ar(
        mean_train,
        scale_train,
        target,
        mean_penalties,
        scale_penalties,
        runtime,
    )

    mu = mean_test @ mean_coefficients
    sigma = np.exp(np.clip(scale_test @ scale_coefficients, -8.0, 8.0))
    quantile_delta = stats.norm.ppf(
        CDC_QUANTILES[None, :], loc=mu[:, None], scale=sigma[:, None]
    )
    # Integration uses the origin level in the same test row; no target or
    # future level participates in this reconstruction.
    origin_log = test["panel_log_level_t"].to_numpy(dtype=float)
    quantiles = np.maximum(np.expm1(origin_log[:, None] + quantile_delta), 0.0)

    reference_date = pd.Timestamp(anchor_date) + pd.Timedelta(weeks=1)
    records = []
    for row_index, row in test.iterrows():
        location = location_map.get(str(row["location_name"]), str(row["location_name"]))
        for quantile_index, quantile in enumerate(CDC_QUANTILES):
            records.append(
                {
                    "model_id": spec.model_id,
                    "reference_date": reference_date.date().isoformat(),
                    "target": TARGET_NAME,
                    "horizon": int(row["lead_weeks"]) - 1,
                    "target_end_date": pd.Timestamp(row["target_date"]).date().isoformat(),
                    "location": location,
                    "output_type": "quantile",
                    "output_type_id": float(quantile),
                    "value": float(quantiles[row_index, quantile_index]),
                }
            )
    return pd.DataFrame.from_records(records)


def prepare_panel_ar_table(
    df: pd.DataFrame,
    runtime: PanelARRuntime,
) -> pd.DataFrame:
    features = build_panel_ar_feature_table(df, runtime)
    return build_panel_ar_examples(features, runtime.max_horizons)
