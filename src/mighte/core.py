"""Numerical kernels ported from the frozen MIGHTE study. See docs/MODELS.md."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

STAGE1_OBJECTIVE_L2 = "l2"
STAGE1_OBJECTIVE_PINBALL_MEDIAN = "pinball_median"
VALID_STAGE1_OBJECTIVES = frozenset({STAGE1_OBJECTIVE_L2, STAGE1_OBJECTIVE_PINBALL_MEDIAN})
LOG_SIGMA_NUMERIC_MIN = -12.0
LOG_SIGMA_NUMERIC_MAX = 12.0


@dataclass(frozen=True)
class RuntimeConfig:
    max_horizons: int
    anchor_step_weeks: int
    num_bags: int
    bag_frac: float
    stage1_rounds: int
    stage2_rounds: int
    min_train_rows: int
    num_threads: int
    own_lags: List[int]
    donor_top_k: int
    donor_lags: List[int]
    donor_min_overlap: int

    @classmethod
    def from_dict(cls, raw: dict) -> "RuntimeConfig":
        return cls(
            max_horizons=int(raw["max_horizons"]),
            anchor_step_weeks=int(raw.get("anchor_step_weeks", 1)),
            num_bags=int(raw.get("num_bags", 1)),
            bag_frac=float(raw.get("bag_frac", 1.0)),
            stage1_rounds=int(raw.get("stage1_rounds", 250)),
            stage2_rounds=int(raw.get("stage2_rounds", 120)),
            min_train_rows=int(raw.get("min_train_rows", 1000)),
            num_threads=int(raw.get("num_threads", 1)),
            own_lags=[int(x) for x in raw["own_lags"]],
            donor_top_k=int(raw.get("donor_top_k", 0)),
            donor_lags=[int(x) for x in raw.get("donor_lags", [])],
            donor_min_overlap=int(raw.get("donor_min_overlap", 40)),
        )


def infer_flu_season(ts: pd.Timestamp) -> str:
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def add_global_features(df: pd.DataFrame, lag_set: Sequence[int]) -> pd.DataFrame:
    pivot = df.pivot(index="date", columns="location_name", values="total_hosp").sort_index()
    us = pivot["US"] if "US" in pivot.columns else pivot.mean(axis=1)
    states = pivot.drop(columns=["US"], errors="ignore")
    nat_mean = states.mean(axis=1) if states.shape[1] else us
    nat_std = states.std(axis=1) if states.shape[1] else pd.Series(0.0, index=pivot.index)

    g = pd.DataFrame(index=pivot.index)
    for lag in sorted(set(lag_set)):
        g[f"us_lag_{lag}"] = us.shift(lag)
        g[f"nat_mean_lag_{lag}"] = nat_mean.shift(lag)
    g["nat_std_lag_1"] = nat_std.shift(1)
    g["nat_mean_diff_1"] = nat_mean.diff(1).shift(1)
    g["nat_mean_diff_4"] = nat_mean.diff(4).shift(1)
    return df.merge(g.reset_index(), on="date", how="left")


def build_feature_table(
    df_long: pd.DataFrame,
    runtime: RuntimeConfig,
) -> pd.DataFrame:
    df = df_long.sort_values(["location_name", "date"]).reset_index(drop=True).copy()
    by_loc = df.groupby("location_name", group_keys=False)

    for lag in runtime.own_lags:
        df[f"y_lag_{lag}"] = by_loc["total_hosp"].shift(lag)

    df["y_diff_1"] = by_loc["total_hosp"].diff(1)
    df["y_diff_4"] = by_loc["total_hosp"].diff(4)
    df["y_pct_change_1"] = by_loc["total_hosp"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    df["y_pct_change_4"] = by_loc["total_hosp"].pct_change(4).replace([np.inf, -np.inf], np.nan)
    for window in (2, 4, 8, 12):
        df[f"y_roll_mean_{window}"] = by_loc["total_hosp"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"y_roll_std_{window}"] = by_loc["total_hosp"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).std()
        )

    phase = df["date"].dt.dayofyear.astype(float) / 365.25
    df["week_sin"] = np.sin(2.0 * np.pi * phase)
    df["week_cos"] = np.cos(2.0 * np.pi * phase)
    df["quarter_sin"] = np.sin(8.0 * np.pi * phase)
    df["quarter_cos"] = np.cos(8.0 * np.pi * phase)
    month = df["date"].dt.month
    df["is_flu_season"] = ((month >= 10) | (month <= 3)).astype(float)
    df["is_peak_flu"] = month.isin([12, 1, 2]).astype(float)

    regime2 = pd.Timestamp("2022-07-01")
    regime3 = pd.Timestamp("2024-11-01")
    df["regime_pre_2022"] = (df["date"] < regime2).astype(float)
    df["regime_2022_2024"] = ((df["date"] >= regime2) & (df["date"] < regime3)).astype(float)
    df["regime_post_2024"] = (df["date"] >= regime3).astype(float)
    df["regime_post_2024_x_flu"] = df["regime_post_2024"] * df["is_flu_season"]

    lag_set = sorted(set(runtime.own_lags + runtime.donor_lags + [1, 2, 4, 8, 12, 52]))
    df = add_global_features(df, lag_set)
    dummies = pd.get_dummies(df["location_name"], prefix="loc", dtype=float)
    df = pd.concat([df, dummies], axis=1)
    df["season"] = df["date"].map(infer_flu_season)
    return df


def build_pooled_examples(feature_df: pd.DataFrame, max_horizons: int) -> pd.DataFrame:
    parts = []
    for h in range(1, max_horizons + 1):
        tmp = feature_df.copy()
        tmp["horizon_weeks"] = h
        tmp["horizon_sin"] = np.sin(2.0 * np.pi * h / 8.0)
        tmp["horizon_cos"] = np.cos(2.0 * np.pi * h / 8.0)
        tmp["target_date"] = tmp["date"] + pd.Timedelta(weeks=h)
        lookup = feature_df[["location_name", "date", "total_hosp"]].rename(
            columns={"date": "target_date", "total_hosp": "target"}
        )
        tmp = tmp.merge(lookup, on=["location_name", "target_date"], how="left", validate="many_to_one")
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)


def feature_columns(pooled_df: pd.DataFrame) -> List[str]:
    excluded = {
        "location_name", "date", "total_hosp", "target", "target_date", "season",
        "target_name", "target_available_date", "target_output_multiplier",
        "target_output_upper_bound",
    }
    return [c for c in pooled_df.columns if c not in excluded]


@lru_cache(maxsize=16)
def _read_external_covariate_file_cached(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path, low_memory=False)


def read_external_covariate_file(path: Path) -> pd.DataFrame:
    resolved = path.resolve()
    return _read_external_covariate_file_cached(
        str(resolved),
        resolved.stat().st_mtime_ns,
    ).copy()


def add_external_covariates(
    feature_df: pd.DataFrame,
    covariate_file: Optional[str],
    covariate_names: Optional[Sequence[str]],
    reference_date: Optional[pd.Timestamp] = None,
    source_lag_weeks: int = 0,
    max_staleness_weeks: Optional[int] = None,
    add_availability_features: bool = False,
    shift_stitched_proxy_dates: bool = True,
) -> pd.DataFrame:
    """Attach national or location-matched covariates without zero filling.

    ``source_lag_weeks`` is measured from the model anchor Saturday.  The hub
    reference date is one week after that anchor, so 0 represents a one-week
    publication lag and 1 represents a two-week publication lag.

    If the input contains multiple issue rows per source date, ``available_date``
    is used to select the latest issue known by ``reference_date``.  Values may
    carry forward only through ``max_staleness_weeks``.  Earlier structural
    missingness remains NaN for LightGBM's native missing-value handling.  An
    optional constant ``model_date_min`` column in the covariate file forces the
    signal to remain missing on earlier displayed model dates; this prevents
    NHSN-predecessor ILI proxy rows from receiving ILI as a predictor.  A file
    containing ``location_name`` is matched within location; otherwise its
    values are treated as national and shared across locations.
    """

    names = [str(name) for name in (covariate_names or []) if str(name)]
    if not covariate_file or not names:
        return feature_df
    if int(source_lag_weeks) < 0:
        raise ValueError("source_lag_weeks must be non-negative")
    if max_staleness_weeks is not None and int(max_staleness_weeks) < 0:
        raise ValueError("max_staleness_weeks must be non-negative or None")

    cov_path = Path(covariate_file)
    if not cov_path.exists():
        raise FileNotFoundError(f"External covariate file not found: {cov_path}")

    cov = read_external_covariate_file(cov_path)
    required = {"date", *names}
    missing = required - set(cov.columns)
    if missing:
        raise ValueError(f"{cov_path} missing external covariate columns: {sorted(missing)}")

    cov = cov.copy()
    model_date_min = None
    if "model_date_min" in cov.columns:
        model_date_min_values = pd.to_datetime(
            cov["model_date_min"], errors="coerce"
        ).dropna().dt.normalize().unique()
        if len(model_date_min_values) > 1:
            raise ValueError(f"{cov_path} contains multiple model_date_min values")
        if len(model_date_min_values) == 1:
            model_date_min = pd.Timestamp(model_date_min_values[0])
    cov["date"] = pd.to_datetime(cov["date"], errors="coerce").dt.normalize()
    location_specific = "location_name" in cov.columns
    covariate_keys = ["location_name", "date"] if location_specific else ["date"]
    if location_specific:
        cov["location_name"] = cov["location_name"].astype(str)
    cov = cov.dropna(subset=covariate_keys)
    sort_columns = list(covariate_keys)
    if "available_date" in cov.columns:
        cov["available_date"] = pd.to_datetime(cov["available_date"], errors="coerce").dt.normalize()
        cov = cov.dropna(subset=["available_date"])
        if reference_date is not None:
            cov = cov[cov["available_date"] <= pd.Timestamp(reference_date).normalize()].copy()
        sort_columns.append("available_date")
    for name in names:
        cov[name] = pd.to_numeric(cov[name], errors="coerce")
    cov = cov.sort_values(sort_columns).drop_duplicates(covariate_keys, keep="last")

    if location_specific:
        if "location_name" not in feature_df.columns:
            raise ValueError("Location-specific external covariates require feature_df.location_name")
        dates = feature_df.loc[:, ["location_name", "date"]].drop_duplicates().copy()
        dates["location_name"] = dates["location_name"].astype(str)
        merge_keys = ["location_name", "date"]
    else:
        dates = pd.DataFrame({"date": sorted(pd.to_datetime(feature_df["date"].dropna().unique()))})
        merge_keys = ["date"]
    dates["date"] = pd.to_datetime(dates["date"], errors="coerce").dt.normalize()
    # The stitched training series excises the 2020-21 and 2021-22 seasons:
    # displayed dates before 2022-06-01 are ILINet-derived proxy rows shifted
    # forward by 728 days, so their biological covariate date is 728 days prior.
    shifted_proxy_cutoff = pd.Timestamp("2022-05-28")
    shifted_proxy_offset = pd.Timedelta(days=728)
    dates["biological_source_date"] = np.where(
        (dates["date"] <= shifted_proxy_cutoff) & bool(shift_stitched_proxy_dates),
        dates["date"] - shifted_proxy_offset,
        dates["date"],
    )
    dates["biological_source_date"] = pd.to_datetime(dates["biological_source_date"])
    dates["covariate_lookup_date"] = dates["biological_source_date"] - pd.Timedelta(
        weeks=int(source_lag_weeks)
    )

    if cov.empty:
        for name in names:
            dates[name] = np.nan
            if add_availability_features:
                dates[f"{name}_available"] = 0.0
                dates[f"{name}_age_weeks"] = np.nan
    else:
        cov_columns = ["location_name"] if location_specific else []
        cov = cov.loc[:, [*cov_columns, "date", *names]].rename(
            columns={"date": "matched_covariate_date"}
        )
        merge_kwargs = {"by": "location_name"} if location_specific else {}
        left_sort = ["covariate_lookup_date", "location_name"] if location_specific else [
            "covariate_lookup_date"
        ]
        right_sort = ["matched_covariate_date", "location_name"] if location_specific else [
            "matched_covariate_date"
        ]
        dates = pd.merge_asof(
            dates.sort_values(left_sort),
            cov.sort_values(right_sort),
            left_on="covariate_lookup_date",
            right_on="matched_covariate_date",
            direction="backward",
            **merge_kwargs,
        )
        age_weeks = (
            (dates["covariate_lookup_date"] - dates["matched_covariate_date"]).dt.days / 7.0
        )
        age_valid = age_weeks.ge(0)
        if max_staleness_weeks is not None:
            age_valid &= age_weeks.le(int(max_staleness_weeks))
        if model_date_min is not None:
            age_valid &= dates["date"].ge(model_date_min)
        for name in names:
            values = pd.to_numeric(dates[name], errors="coerce")
            available = values.notna() & age_valid
            dates[name] = values.where(available, np.nan)
            if add_availability_features:
                dates[f"{name}_available"] = available.astype(float)
                dates[f"{name}_age_weeks"] = age_weeks.where(available, np.nan)

    audit_columns = [
        "biological_source_date",
        "covariate_lookup_date",
        "matched_covariate_date",
    ]
    return feature_df.merge(
        dates.drop(columns=audit_columns, errors="ignore"), on=merge_keys, how="left"
    )


def add_external_covariate_sources(
    feature_df: pd.DataFrame,
    sources: Optional[Sequence[dict]],
    reference_date: Optional[pd.Timestamp] = None,
    shift_stitched_proxy_dates: bool = True,
) -> pd.DataFrame:
    """Attach multiple sources while preserving each source's own latency rule."""

    prepared: list[tuple[dict, list[str]]] = []
    seen_names: set[str] = set()
    for source in sources or []:
        cfg = dict(source)
        names = cfg.get("columns", cfg.get("names", []))
        if isinstance(names, str):
            names = [names]
        names = [str(name) for name in names if str(name)]
        duplicates = seen_names.intersection(names)
        if duplicates:
            raise ValueError(f"External covariate names repeated across sources: {sorted(duplicates)}")
        seen_names.update(names)
        prepared.append((cfg, names))

    out = feature_df
    for cfg, names in prepared:
        out = add_external_covariates(
            out,
            covariate_file=cfg.get("file"),
            covariate_names=names,
            reference_date=reference_date,
            source_lag_weeks=int(cfg.get("source_lag_weeks", 0)),
            max_staleness_weeks=(
                None
                if cfg.get("max_staleness_weeks") is None
                else int(cfg["max_staleness_weeks"])
            ),
            add_availability_features=bool(cfg.get("add_availability_features", False)),
            shift_stitched_proxy_dates=shift_stitched_proxy_dates,
        )
    return out


def add_external_feature_lags(
    feature_df: pd.DataFrame,
    lag_spec: Optional[Dict[str, Sequence[int]]],
) -> pd.DataFrame:
    """Create forecast-time lag blocks after availability-constrained attachment."""

    if not lag_spec:
        return feature_df
    out = feature_df.sort_values(["location_name", "date"]).copy()
    grouped = out.groupby("location_name", group_keys=False)
    for source_name, raw_lags in lag_spec.items():
        if source_name not in out.columns:
            raise ValueError(
                f"Cannot lag unavailable external feature column: {source_name}"
            )
        lags = sorted(set(int(lag) for lag in raw_lags))
        if not lags or any(lag < 1 for lag in lags):
            raise ValueError(
                f"External feature lags for {source_name} must be positive"
            )
        for lag in lags:
            out[f"{source_name}_model_lag{lag}"] = grouped[source_name].shift(lag)
    return out


def central_model_params(seed: int, runtime: RuntimeConfig) -> dict:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": int(seed),
        "num_threads": runtime.num_threads,
    }


def quantile_model_params(seed: int, runtime: RuntimeConfig, alpha: float) -> dict:
    params = central_model_params(seed, runtime)
    params.update(
        {
            "objective": "quantile",
            "metric": "quantile",
            "alpha": float(alpha),
        }
    )
    return params


def stage1_model_params(
    seed: int,
    runtime: RuntimeConfig,
    stage1_objective: str = STAGE1_OBJECTIVE_L2,
) -> dict:
    """Return an otherwise identical Stage 1 configuration for one center objective."""

    if stage1_objective == STAGE1_OBJECTIVE_L2:
        return central_model_params(seed, runtime)
    if stage1_objective == STAGE1_OBJECTIVE_PINBALL_MEDIAN:
        return quantile_model_params(seed, runtime, alpha=0.5)
    raise ValueError(
        f"Unknown Stage 1 objective={stage1_objective}; "
        f"expected one of {sorted(VALID_STAGE1_OBJECTIVES)}"
    )


def spread_model_params(seed: int, runtime: RuntimeConfig) -> dict:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.04,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "random_state": int(seed),
        "num_threads": runtime.num_threads,
    }


def fit_lgbm_regressor(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    rounds: int,
) -> lgb.Booster:
    dtrain = lgb.Dataset(X.astype(float), label=np.asarray(y, dtype=float), params={"verbose": -1})
    return lgb.train(params, dtrain, num_boost_round=rounds, callbacks=[])


def make_oof_predictions(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    runtime: RuntimeConfig,
    seed: int,
    stage1_objective: str = STAGE1_OBJECTIVE_L2,
    target_available_dates: Optional[Sequence[pd.Timestamp]] = None,
    auxiliary_rows: Optional[Sequence[bool]] = None,
    oof_audit: Optional[list[dict]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rolling-origin row indices and corresponding Stage 1 predictions."""

    available = (
        pd.to_datetime(np.asarray(target_available_dates))
        if target_available_dates is not None else None
    )
    auxiliary = np.asarray(auxiliary_rows, dtype=bool) if auxiliary_rows is not None else None
    if (available is None) != (auxiliary is None):
        raise ValueError("Auxiliary OOF availability requires dates and stream flags together")
    if available is not None and (len(available) != len(train_df) or len(auxiliary) != len(train_df)):
        raise ValueError("Auxiliary OOF metadata has mismatched row counts")
    primary = ~auxiliary if auxiliary is not None else np.ones(len(train_df), dtype=bool)
    # Auxiliary coverage must not move the paired primary model's fold cutoffs.
    dates = np.array(sorted(train_df.loc[primary, "target_date"].dropna().unique()))
    if len(dates) < 10:
        indices = np.where(primary)[0]
        if oof_audit is not None:
            oof_audit.append({"fold": "short_history_fallback", "fit_primary": int(primary.sum()), "fit_auxiliary": 0, "validation_auxiliary": 0})
        return indices, np.repeat(float(np.nanmedian(y[primary])), len(indices))

    cut_points = np.linspace(0.55, 0.9, 4)
    pred = np.full(len(train_df), np.nan)
    for i, frac in enumerate(cut_points):
        cut = dates[min(len(dates) - 2, max(1, int(len(dates) * frac)))]
        next_cut = dates[min(len(dates) - 1, max(2, int(len(dates) * (frac + 0.1))))]
        fit_mask = train_df["target_date"].to_numpy() <= cut
        if available is not None:
            # The modeled target week is Saturday and forecasts are issued the
            # following Saturday. Revised auxiliary labels released after that
            # issue must not enter an earlier rolling fit. The existing primary
            # target's historical-vintage convention remains unchanged.
            issue_cut = pd.Timestamp(cut) + pd.Timedelta(weeks=1)
            fit_mask &= ~auxiliary | (available <= issue_cut)
        val_mask = (train_df["target_date"].to_numpy() > cut) & (train_df["target_date"].to_numpy() <= next_cut)
        if oof_audit is not None:
            oof_audit.append({
                "fold": i, "cutoff": pd.Timestamp(cut).date().isoformat(),
                "fit_primary": int((fit_mask & primary).sum()),
                "fit_auxiliary": int((fit_mask & ~primary).sum()),
                "validation_auxiliary": int((val_mask & ~primary).sum()),
                "fit_skipped": bool(fit_mask.sum() < runtime.min_train_rows or val_mask.sum() == 0),
            })
        if fit_mask.sum() < runtime.min_train_rows or val_mask.sum() == 0:
            continue
        model = fit_lgbm_regressor(
            train_df.loc[fit_mask, feature_cols],
            y[fit_mask],
            stage1_model_params(seed + i, runtime, stage1_objective),
            runtime.stage1_rounds,
        )
        pred[val_mask] = model.predict(train_df.loc[val_mask, feature_cols].astype(float))

    valid = np.isfinite(pred)
    if valid.sum() < max(100, int(0.1 * len(train_df))):
        full_model = fit_lgbm_regressor(
            train_df.loc[primary, feature_cols],
            y[primary],
            stage1_model_params(seed, runtime, stage1_objective),
            max(25, runtime.stage1_rounds // 3),
        )
        pred[:] = np.nan
        pred[primary] = full_model.predict(train_df.loc[primary, feature_cols].astype(float))
        valid = np.isfinite(pred)
        if oof_audit is not None:
            oof_audit.append({"fold": "primary_only_in_sample_fallback", "fit_primary": int(primary.sum()), "fit_auxiliary": 0, "validation_auxiliary": 0})

    return np.where(valid)[0], pred[valid]


class ConditionalGaussianLogSigmaObjective:
    """Gaussian NLL for log-sigma conditional on fixed, row-specific means.

    The booster prediction is an additive correction to ``base_log_sigma``.
    Unlike the former two-parameter LightGBMLSS workaround, the objective sees
    the actual outcome and its matching frozen mean for every training row.
    """

    def __init__(self, frozen_mu: np.ndarray, base_log_sigma: float):
        self.frozen_mu = np.asarray(frozen_mu, dtype=float)
        self.base_log_sigma = float(base_log_sigma)

    def __call__(self, predt: np.ndarray, data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(data.get_label(), dtype=float)
        predt = np.asarray(predt, dtype=float)
        if len(y) != len(self.frozen_mu) or len(predt) != len(y):
            raise ValueError("Gaussian scale objective received mismatched row counts.")

        eta = np.clip(
            self.base_log_sigma + predt,
            LOG_SIGMA_NUMERIC_MIN,
            LOG_SIGMA_NUMERIC_MAX,
        )
        squared_standardized_error = np.square(y - self.frozen_mu) * np.exp(-2.0 * eta)
        squared_standardized_error = np.nan_to_num(
            squared_standardized_error,
            nan=1.0,
            posinf=1e12,
            neginf=1.0,
        )
        squared_standardized_error = np.clip(squared_standardized_error, 0.0, 1e12)

        # For l(eta) = eta + 0.5 * residual^2 * exp(-2 eta):
        #   dl/deta = 1 - residual^2 / sigma^2
        #   d2l/deta2 = 2 * residual^2 / sigma^2
        # The analytic Hessian is non-negative, avoiding the unstable negative
        # Hessians induced by the former bounded-sigmoid scale link.
        grad = 1.0 - squared_standardized_error
        hess = np.maximum(2.0 * squared_standardized_error, 1e-8)

        weights = data.get_weight()
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            grad *= weights
            hess *= weights
        return grad, hess


@dataclass
class ConditionalGaussianScaleBooster:
    """A fitted log-sigma booster with its likelihood-optimal constant offset."""

    booster: lgb.Booster
    base_log_sigma: float

    def predict_sigma(self, X: pd.DataFrame) -> np.ndarray:
        eta = self.base_log_sigma + np.asarray(self.booster.predict(X.astype(float)), dtype=float)
        return np.exp(np.clip(eta, LOG_SIGMA_NUMERIC_MIN, LOG_SIGMA_NUMERIC_MAX))


def fit_conditional_gaussian_scale_booster(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    frozen_mu: np.ndarray,
    params: dict,
    rounds: int,
) -> ConditionalGaussianScaleBooster:
    """Fit conditional Gaussian scale by NLL, without a learned-width target or model cap."""

    X = X_train.astype(float)
    y = np.asarray(y_train, dtype=float)
    mu = np.asarray(frozen_mu, dtype=float)
    if len(X) != len(y) or len(y) != len(mu):
        raise ValueError("Gaussian scale training inputs must have identical row counts.")

    residual = y - mu
    finite = np.isfinite(residual)
    if not finite.all():
        X = X.loc[finite].reset_index(drop=True)
        y = y[finite]
        mu = mu[finite]
        residual = residual[finite]
    if len(y) == 0:
        raise ValueError("Gaussian scale training has no finite residuals.")

    # This is the exact constant-sigma Gaussian MLE. It anchors boosting at the
    # likelihood optimum instead of at the midpoint of a user-selected range.
    base_sigma = float(np.sqrt(np.mean(np.square(residual))))
    base_sigma = max(base_sigma, float(np.exp(LOG_SIGMA_NUMERIC_MIN)))
    base_log_sigma = float(np.log(base_sigma))

    objective = ConditionalGaussianLogSigmaObjective(mu, base_log_sigma)
    train_params = dict(params)
    train_params.update({"objective": objective, "metric": "None"})
    dtrain = lgb.Dataset(X, label=y, params={"verbose": -1}, free_raw_data=False)
    booster = lgb.train(train_params, dtrain, num_boost_round=rounds, callbacks=[])
    return ConditionalGaussianScaleBooster(booster=booster, base_log_sigma=base_log_sigma)


def fit_production_lightgbmlss_one_bag(
    X_train: pd.DataFrame,
    y_train_model: np.ndarray,
    target_dates: Sequence[pd.Timestamp],
    runtime: RuntimeConfig,
    seed: int,
    fit_config_profile: str = "production",
    stage1_objective: str = STAGE1_OBJECTIVE_L2,
    target_available_dates: Optional[Sequence[pd.Timestamp]] = None,
    auxiliary_rows: Optional[Sequence[bool]] = None,
    oof_audit: Optional[list[dict]] = None,
):
    X = X_train.astype(float)
    y = np.asarray(y_train_model, dtype=float)

    # Tree settings remain matched to joint base. A declared center-objective
    # ablation may replace L2 with median pinball loss without changing any
    # other Stage 1 setting.
    p1 = stage1_model_params(seed, runtime, stage1_objective)
    d1 = lgb.Dataset(X, label=y, params={"verbose": -1})
    stage1 = lgb.train(p1, d1, num_boost_round=runtime.stage1_rounds, callbacks=[])

    oof_table = X.copy()
    oof_table["target_date"] = pd.to_datetime(np.asarray(target_dates))
    oof_idx, oof_mu = make_oof_predictions(
        oof_table,
        list(X.columns),
        y,
        runtime,
        seed + 10000,
        stage1_objective=stage1_objective,
        target_available_dates=target_available_dates,
        auxiliary_rows=auxiliary_rows,
        oof_audit=oof_audit,
    )
    if len(oof_idx) < max(50, int(0.05 * len(X))):
        raise ValueError("Too few rolling-origin predictions to fit Gaussian scale.")

    p2 = lightgbmlss_stage2_params(seed, runtime, fit_config_profile)
    stage2 = fit_conditional_gaussian_scale_booster(
        X_train=X.iloc[oof_idx].reset_index(drop=True),
        y_train=y[oof_idx],
        frozen_mu=oof_mu,
        params=p2,
        rounds=runtime.stage2_rounds,
    )

    return stage1, stage2


def lightgbmlss_stage2_params(
    seed: int,
    runtime: RuntimeConfig,
    fit_config_profile: str = "production",
) -> dict:
    """Return tree settings for the conditional Gaussian-NLL sigma fit.

    The joint-base ablation uses the same Stage 2 tree capacity and seed offset
    as the native-L2 spread model. ``objective`` and ``metric`` are attached by
    ``fit_conditional_gaussian_scale_booster`` after the fixed means are known.
    """

    if fit_config_profile == "joint_base":
        params = spread_model_params(seed + 20000, runtime)
        params.pop("objective", None)
        params.pop("metric", None)
        params.update(
            {
                "feature_pre_filter": False,
                "force_col_wise": True,
            }
        )
        return params
    if fit_config_profile != "production":
        raise ValueError(f"Unknown LightGBMLSS fit_config_profile={fit_config_profile}")
    return {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "num_threads": runtime.num_threads,
        "verbosity": -1,
        "random_state": int(seed),
    }
