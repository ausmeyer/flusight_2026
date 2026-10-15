"""Controlled diagnostics, never candidate production outputs or tuned alternatives."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from mighte import panel_ar as pa
from mighte.contract import HOSP

RUN = ROOT / "runs/official_submissions/2026-10-17/20261015T043845714774Z"
manifest = json.loads((RUN / "manifest.json").read_text())
runtime = pa.PanelARRuntime.from_dict(manifest["settings"]["panel_ar_runtime"])
spec = pa.PanelARSpec("MIGHTE-Linear", partial_pooling=True)
history = pd.read_csv(RUN / "hospitalization-training.csv", parse_dates=["date"])
pooled = pa.prepare_panel_ar_table(history, runtime)
train, test = pa.panel_ar_training_and_test_rows(pooled, pd.Timestamp("2026-10-10"), runtime)
required = pa.panel_ar_continuous_columns(runtime) + pa.panel_ar_binary_columns() + ["panel_target_delta_log"]
train = train.dropna(subset=required).reset_index(drop=True)
test = test.dropna(subset=[c for c in required if c != "panel_target_delta_log"]).reset_index(drop=True)
del pooled
origin_log = test.panel_log_level_t.to_numpy()
coefficients = np.load(OUT / "fit_coefficients.npz")
x, xt, penalty, names = pa.build_panel_ar_mean_design(train, test, runtime, spec)
z, zt, scale_penalty, _ = pa.build_panel_ar_scale_design(train, test, runtime)
beta = coefficients["beta"]
y = train.panel_target_delta_log.to_numpy()
rows = []


def record(label, xtest, b, test_rows=test, origin=origin_log, **metadata):
    mu = xtest @ b
    medians = np.maximum(np.expm1(origin + mu), 0)
    frame = test_rows[["location_name", "lead_weeks"]].assign(scenario=label, mu=mu, median=medians)
    for key, value in metadata.items():
        frame[key] = value
    rows.extend(frame.to_dict("records"))
    print(label, np.round(medians[test_rows.location_name.eq("US")], 2), flush=True)


with threadpool_limits(limits=4):
    record("saved_model", xt, beta)
    hlevel = [i for i, name in enumerate(names) if name.startswith("lead_") and name.endswith("_x_panel_log_level_t")]
    ablated = xt.copy()
    ablated[:, hlevel] = 0
    record("frozen_coefficients_zero_horizon_level_contribution", ablated, beta)

    # The integration anchor stays original; only the predictor level changes here.
    state_level = train.loc[train.location_name.ne("US"), "panel_log_level_t"].mean()
    changed = test.copy()
    changed.loc[changed.location_name.eq("US"), "panel_log_level_t"] = state_level
    _, xt_changed, _, _ = pa.build_panel_ar_mean_design(train, changed, runtime, spec)
    record("frozen_coefficients_US_level_at_state_training_mean", xt_changed, beta)

    # Remove three design columns, with all other priors and iterations unchanged.
    keep = [i for i in range(len(names)) if i not in hlevel]
    b, g = pa.fit_distributional_panel_ar(x[:, keep], z, y, penalty[keep], scale_penalty, runtime)
    record("refit_without_three_horizon_level_interactions", xt[:, keep], b)

    # Center the predictor within location using training rows only. Targets and
    # integration stay log1p(count); scale inputs remain original to isolate mean design.
    level_means = train.groupby("location_name").panel_log_level_t.mean()
    centered_train, centered_test = train.copy(), test.copy()
    centered_train["panel_log_level_t"] -= centered_train.location_name.map(level_means)
    centered_test["panel_log_level_t"] -= centered_test.location_name.map(level_means)
    cx, cxt, cp, _ = pa.build_panel_ar_mean_design(centered_train, centered_test, runtime, spec)
    b, g = pa.fit_distributional_panel_ar(cx, z, y, cp, scale_penalty, runtime)
    record("refit_location_centered_mean_level", cxt, b)

    # Reconstructed seed versus NHSN influence; keep current origin features fixed.
    nhsn_train = train[train.date.ge("2021-07-01")].copy()
    nx, nxt, np_, _ = pa.build_panel_ar_mean_design(nhsn_train, test, runtime, spec)
    nz, nzt, nsp, _ = pa.build_panel_ar_scale_design(nhsn_train, test, runtime)
    b, g = pa.fit_distributional_panel_ar(nx, nz, nhsn_train.panel_target_delta_log.to_numpy(), np_, nsp, runtime)
    record("refit_NHSN_origins_only_same_prepared_history", nxt, b)

    # Test the stated coordinate-NLL interpretation. The production mean update
    # normalizes inverse-variance weights, changing the effective ridge strength
    # relative to a literal fixed-penalty Gaussian NLL. This is a secondary audit,
    # not a proposed change to the scientific settings.
    gamma = coefficients["gamma"]
    raw_weights = np.clip(np.exp(-2 * np.clip(z @ gamma, -8, 8)), 1e-4, 1e4)
    conditional_beta = pa.solve_penalized_wls(x, y, raw_weights, penalty)
    literal_gradient = -x.T @ (raw_weights * (y - x @ beta)) / len(y) + penalty * beta
    normalized_gradient = -x.T @ ((raw_weights / raw_weights.mean()) * (y - x @ beta)) / len(y) + penalty * beta
    (OUT / "objective_check.json").write_text(json.dumps({
        "raw_inverse_variance_weight_mean": float(raw_weights.mean()),
        "fixed_penalty_NLL_mean_gradient_max_abs": float(np.max(np.abs(literal_gradient))),
        "normalized_weight_mean_gradient_max_abs": float(np.max(np.abs(normalized_gradient))),
        "interpretation": "The normalized weighted-ridge update is not the mean-coordinate minimizer of the same unnormalized Gaussian NLL plus the recorded fixed penalty."
    }, indent=2) + "\n")
    record("conditional_mean_update_unnormalized_weights_fixed_final_scale", xt, conditional_beta)

    # Counterfactual ascertainment: replace ONLY the newest US level by constant
    # continuation of the preceding log change; use frozen beta and scalers, no refit.
    h = history.copy()
    u = h[h.location_name.eq("US")].sort_values("date")
    prev, prev2 = u.iloc[-2].total_hosp, u.iloc[-3].total_hosp
    hypothetical_anchor = np.expm1(2 * np.log1p(prev) - np.log1p(prev2))
    h.loc[h.location_name.eq("US") & h.date.eq("2026-10-10"), "total_hosp"] = hypothetical_anchor
    cp = pa.prepare_panel_ar_table(h, runtime)
    _, ct = pa.panel_ar_training_and_test_rows(cp, pd.Timestamp("2026-10-10"), runtime)
    ct = ct.dropna(subset=[c for c in required if c != "panel_target_delta_log"]).reset_index(drop=True)
    _, cxt, _, _ = pa.build_panel_ar_mean_design(train, ct, runtime, spec)
    record("frozen_coefficients_US_anchor_continues_prior_log_growth", cxt, beta, ct,
           ct.panel_log_level_t.to_numpy(), hypothetical_anchor=float(hypothetical_anchor))

pd.DataFrame(rows).to_csv(OUT / "counterfactual_forecasts.csv", index=False)

# Exact additive decomposition, using shared fitted coefficients. Collinear
# individual coefficients are descriptive; these grouped changes support ablation.
c = pd.read_csv(OUT / "mean_contributions.csv")
def group(name):
    if "panel_log_level_t" in name:
        return "level and level interactions"
    if "nat_delta" in name:
        return "cross-jurisdiction growth/spread"
    if "delta" in name:
        return "own growth and rolling moments"
    if any(k in name for k in ["week_sin", "week_cos", "quarter", "is_flu", "is_peak", "regime"]):
        return "calendar and regime"
    return "intercepts and lead offsets"
c["group"] = c.term.map(group)
c.groupby(["location_name", "lead_weeks", "group"]).mu_contribution.sum().unstack().to_csv(
    OUT / "grouped_contributions.csv")
print("Controlled counterfactuals complete. No production file changed.", flush=True)
