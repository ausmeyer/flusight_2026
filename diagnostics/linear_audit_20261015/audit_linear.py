"""Read-only MIGHTE-Linear audit; all products stay beside this script.

Run from repository root: .venv/bin/python diagnostics/linear_audit_20261015/audit_linear.py
This fits only the single linear model, never the boosted production components.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from mighte import panel_ar as pa
from mighte.contract import HOSP, QUANTILES, read_forecast
from mighte.data import prepare_inputs, verify_snapshot
from mighte.util import code_hash, digest

RUN = ROOT / "runs/official_submissions/2026-10-17/20261015T043845714774Z"
MANIFEST = json.loads((RUN / "manifest.json").read_text())
SNAPSHOT = ROOT / "data/snapshots" / MANIFEST["snapshot_id"]
ANCHOR = pd.Timestamp("2026-10-10")
RUNTIME = pa.PanelARRuntime.from_dict(MANIFEST["settings"]["panel_ar_runtime"])
SPEC = pa.PanelARSpec("MIGHTE-Linear", partial_pooling=True)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def main():
    start = time.monotonic()
    baseline = {str(p.relative_to(ROOT)): digest(p) for p in [
        *sorted((ROOT / "src/mighte").glob("*.py")), ROOT / "config/settings.json",
        *sorted(RUN.rglob("*.csv")), RUN / "manifest.json", RUN / "submission.json",
        ROOT / "reports" / MANIFEST["run_id"] / "index.html",
        ROOT / "reports" / MANIFEST["run_id"] / "report-data.json",
    ]}
    write_json("before_hashes.json", baseline)
    (OUT / "git_status_before.txt").write_text(subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True))
    snapshot_manifest = verify_snapshot(SNAPSHOT)
    assert digest(SNAPSHOT / "manifest.json") == MANIFEST["snapshot_manifest_sha256"]
    for name, expected in MANIFEST["output_hashes"].items():
        assert digest(RUN / name) == expected
    history = pd.read_csv(RUN / "hospitalization-training.csv", parse_dates=["date"])
    histories, audit = prepare_inputs(ROOT, SNAPSHOT, "2026-10-17", MANIFEST["settings"])
    pd.testing.assert_frame_equal(history, histories[HOSP], check_exact=True)
    print("Snapshot hashes, exported hashes, and exact raw-to-training reproduction: PASS", flush=True)
    locations = pd.read_csv(SNAPSHOT / "contract/locations.csv", dtype={"location": str})
    location_map = locations.set_index("location_name").location.to_dict()
    pooled = pa.prepare_panel_ar_table(history, RUNTIME)
    train, test = pa.panel_ar_training_and_test_rows(pooled, ANCHOR, RUNTIME)
    required = list(dict.fromkeys(pa.panel_ar_continuous_columns(RUNTIME)
                                 + pa.panel_ar_binary_columns() + ["panel_target_delta_log"]))
    train = train.dropna(subset=required).reset_index(drop=True)
    test = test.dropna(subset=[c for c in required if c != "panel_target_delta_log"]).reset_index(drop=True)
    del pooled
    test.to_csv(OUT / "anchor_features.csv", index=False)
    x, xt, penalty, names = pa.build_panel_ar_mean_design(train, test, RUNTIME, SPEC)
    z, zt, scale_penalty, scale_names = pa.build_panel_ar_scale_design(train, test, RUNTIME)
    y = train.panel_target_delta_log.to_numpy()
    us = test.location_name.eq("US").to_numpy()
    iterations, optimizations = [], []
    original_solve = pa.solve_penalized_wls
    original_minimize = pa.optimize.minimize

    def solve(*args, **kwargs):
        beta = original_solve(*args, **kwargs)
        iterations.append({"stage": len(iterations), "us_mu": (xt[us] @ beta).tolist(),
                           "us_median": np.expm1(test.loc[us, "panel_log_level_t"].to_numpy()
                                                 + xt[us] @ beta).tolist(),
                           "mean_weight": float(np.mean(args[2]))})
        return beta

    def minimize(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        optimizations.append({"success": bool(result.success), "message": str(result.message),
                              "nit": int(result.nit), "objective": float(result.fun),
                              "gradient_max_abs": float(np.max(np.abs(result.jac)))})
        return result

    pa.solve_penalized_wls = solve
    pa.optimize.minimize = minimize
    with threadpool_limits(limits=4):
        beta, gamma = pa.fit_distributional_panel_ar(x, z, y, penalty, scale_penalty, RUNTIME)
    pa.solve_penalized_wls = original_solve
    pa.optimize.minimize = original_minimize
    mu, sigma = xt @ beta, np.exp(np.clip(zt @ gamma, -8, 8))
    np.savez(OUT / "fit_coefficients.npz", beta=beta, gamma=gamma, names=names, scale_names=scale_names)
    predicted = np.maximum(np.expm1(test.panel_log_level_t.to_numpy()[:, None]
                                   + stats.norm.ppf(QUANTILES[None, :], loc=mu[:, None], scale=sigma[:, None])), 0)
    component = read_forecast(RUN / "components/linear.csv")
    component["output_type_id"] = pd.to_numeric(component.output_type_id)
    rebuilt = pd.DataFrame([{ "location": location_map[row.location_name], "horizon": row.lead_weeks - 1,
                             "output_type_id": q, "value": predicted[i, j]}
                           for i, row in test.iterrows() for j, q in enumerate(QUANTILES)])
    comparison = rebuilt.merge(component, on=["location", "horizon", "output_type_id"], suffixes=("_rebuilt", "_saved"))
    delta = float(np.max(np.abs(comparison.value_rebuilt - comparison.value_saved)))
    assert len(comparison) == len(component)
    assert np.allclose(comparison.value_rebuilt, comparison.value_saved, rtol=1e-10, atol=1e-7)
    print(f"Linear refit {time.monotonic()-start:.1f}s; {len(train)} rows x {x.shape[1]} mean terms; max saved-quantile difference {delta:.3g}", flush=True)
    np.savez(OUT / "fit_coefficients.npz", beta=beta, gamma=gamma, names=names, scale_names=scale_names)
    pd.DataFrame({"term": names, "coefficient": beta, "ridge": penalty}).to_csv(OUT / "mean_coefficients.csv", index=False)
    pd.DataFrame({"term": scale_names, "coefficient": gamma, "ridge": scale_penalty}).to_csv(OUT / "scale_coefficients.csv", index=False)
    test[["location_name", "lead_weeks"]].assign(mu=mu, sigma=sigma).to_csv(OUT / "forecast_mu_sigma.csv", index=False)
    contributions = []
    for i, row in test.iterrows():
        for term, coefficient, feature in zip(names, beta, xt[i]):
            contributions.append({"location_name": row.location_name, "lead_weeks": row.lead_weeks,
                                  "term": term, "standardized_design_value": feature,
                                  "coefficient": coefficient, "mu_contribution": feature * coefficient})
    pd.DataFrame(contributions).to_csv(OUT / "mean_contributions.csv", index=False)
    standardization = pd.DataFrame({"mean": train[pa.panel_ar_continuous_columns(RUNTIME)].mean(),
                                    "std": train[pa.panel_ar_continuous_columns(RUNTIME)].std()})
    standardization.to_csv(OUT / "standardization.csv")
    train.assign(era=np.where(train.date.lt("2021-07-01"), "reconstructed", "NHSN")).groupby(
        ["era", "lead_weeks"]).agg(rows=("date", "size"), target_mean=("panel_target_delta_log", "mean"),
                                  target_std=("panel_target_delta_log", "std")).to_csv(OUT / "training_eras.csv")
    write_json("fit_summary.json", {"elapsed_seconds": time.monotonic()-start,
        "run_id": MANIFEST["run_id"], "snapshot_id": SNAPSHOT.name,
        "run_code_hash": MANIFEST["code_sha256"], "current_code_hash": code_hash(ROOT),
        "snapshot_completed": snapshot_manifest["completed_at"], "rows": len(train),
        "mean_terms": x.shape[1], "scale_terms": z.shape[1], "train_target_latest": train.target_date.max(),
        "train_origin_latest": train.date.max(), "test_rows": len(test),
        "optimizer_results": optimizations, "mean_iterations": iterations,
        "max_saved_quantile_absolute_difference": delta,
        "anchor_missing": audit[HOSP]["anchor_missing"], "interpolated_training_values": audit[HOSP]["interpolated_training_values"]})
    after = {name: digest(ROOT / name) for name in baseline}
    assert after == baseline
    write_json("after_hashes.json", after)
    print("Audit complete; protected source/settings/run/review hashes unchanged.", flush=True)


if __name__ == "__main__":
    main()
