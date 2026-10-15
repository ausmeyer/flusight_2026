"""Build source-backed audit tables and a standalone scientific diagnostic report."""
from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/official_submissions/2026-10-17/20261015T043845714774Z"
MANIFEST = json.loads((RUN / "manifest.json").read_text())
SNAPSHOT = ROOT / "data/snapshots" / MANIFEST["snapshot_id"]
REPORT = ROOT / "reports" / MANIFEST["run_id"]
HOSP = "wk inc flu hosp"
KEYS = ["reference_date", "target", "horizon", "target_end_date", "location", "output_type", "output_type_id"]


def read(path):
    return pd.read_csv(path, dtype={"location": str})


def save_json(filename, obj):
    (OUT / filename).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")


def table(frame, digits=2):
    return frame.to_html(index=False, border=0, classes="evidence", float_format=lambda v: f"{v:,.{digits}f}")


def source(name, lines):
    return f'<a href="../../{name}"><code>{name}:{lines}</code></a>'


def main():
    locations = read(SNAPSHOT / "contract/locations.csv")
    names = locations.set_index("location").location_name.to_dict()
    truth = read(SNAPSHOT / "truth.csv")
    truth = truth[truth.target.eq(HOSP)]
    observed = truth.pivot(index="location", columns="date", values="value")
    linear = read(RUN / "model-output/MIGHTE-Linear/2026-10-17-MIGHTE-Linear.csv")
    component = read(RUN / "components/linear.csv")
    medians = linear[linear.output_type_id.eq(.5)].pivot(index="location", columns="horizon", values="value")
    slopes = pd.DataFrame({"location": medians.index, "location_name": medians.index.map(names)})
    for day in ["2026-09-12", "2026-09-19", "2026-09-26", "2026-10-03", "2026-10-10"]:
        slopes[day] = slopes.location.map(observed[day])
    for lead in range(4):
        slopes[f"forecast_h{lead}"] = slopes.location.map(medians[lead])
    cols = [f"forecast_h{h}" for h in range(4)]
    slopes["forecast_OLS_admissions_per_week"] = slopes[cols].to_numpy() @ np.array([-1.5, -.5, .5, 1.5]) / 5
    slopes["forecast_endpoint_admissions_per_week"] = (slopes.forecast_h3 - slopes.forecast_h0) / 3
    slopes["forecast_h0_to_h3_percent"] = 100 * (slopes.forecast_h3 / slopes.forecast_h0 - 1)
    slopes["observed_last_week_percent"] = 100 * (slopes["2026-10-10"] / slopes["2026-10-03"] - 1)
    slopes["observed_Sep19_to_Oct10_endpoint_admissions_per_week"] = (slopes["2026-10-10"] - slopes["2026-09-19"]) / 3
    slopes["forecast_range_over_h0_percent"] = 100 * (slopes[cols].max(axis=1) - slopes[cols].min(axis=1)) / slopes.forecast_h0
    slopes["flat_under_stated_10pct_range_rule"] = slopes.forecast_range_over_h0_percent.le(10)
    slopes.sort_values("forecast_h0_to_h3_percent").to_csv(OUT / "jurisdiction_slopes.csv", index=False)

    # Validate rounding, ensemble construction, coverage, and plotted five quantiles.
    final_check = linear.merge(component, on=KEYS, suffixes=("_submitted", "_component"), validate="one_to_one")
    assert len(final_check) == 4876
    assert (final_check.value_submitted == np.rint(final_check.value_component)).all()
    ww = read(RUN / "components/ww-hospitalizations.csv").set_index(KEYS).value.sort_index()
    nssp = read(RUN / "components/nssp-hospitalizations.csv").set_index(KEYS).value.sort_index()
    lseries = component.set_index(KEYS).value.sort_index()
    ensemble = read(RUN / "model-output/MIGHTE-Nsemble/2026-10-17-MIGHTE-Nsemble.csv").set_index(KEYS).value.sort_index()
    assert ww.index.equals(nssp.index) and ww.index.equals(lseries.index) and ww.index.equals(ensemble.index)
    assert (ensemble == np.rint((ww + nssp + lseries) / 3)).all()
    payload = json.loads((REPORT / "report-data.json").read_text())
    chart = pd.DataFrame(payload["forecasts"])
    chart = chart[chart.model_id.eq("MIGHTE-Linear") & chart.reference_date.eq("2026-10-17")]
    plotted = []
    for q, col in [(.025, "lo95"), (.25, "lo50"), (.5, "median"), (.75, "hi50"), (.975, "hi95")]:
        p = linear[linear.output_type_id.eq(q)].merge(chart, on=KEYS[:5], validate="one_to_one")
        assert len(p) == 212 and np.max(np.abs(p.value - p[col])) == 0
        plotted.append({"quantile": q, "rows": len(p), "max_difference": 0})
    plotted_truth = pd.DataFrame(payload["truth"])
    plotted_truth = plotted_truth[plotted_truth.target.eq(HOSP)]
    joined = truth.merge(plotted_truth, on=["date", "target", "location"], suffixes=("_run", "_plot"), validate="one_to_one")
    assert len(joined) == len(truth) == len(plotted_truth)
    assert joined.value_run.fillna(-1).equals(joined.value_plot.fillna(-1))
    save_json("export_and_plot_checks.json", {"submitted_quantiles": len(final_check), "rounding_exact": True,
        "ensemble_unrounded_equal_thirds_then_rounding_exact": True,
        "plot_comparisons": plotted, "truth_snapshot": payload["truth_snapshot"],
        "truth_rows_compared": len(joined), "truth_differences": 0,
        "live_visual_inspection": "US / 2026-10-17 / hospital admissions / Linear selected; rising observations, flat median and expanding intervals visible."})

    # Across-vintage evidence, including genuinely preliminary versus published files.
    vintages = []
    for path in sorted((ROOT / "data/snapshots").glob("*/truth.csv")):
        if path.parent.name < "20261007":
            continue
        frame = read(path)
        frame = frame[frame.target.eq(HOSP) & frame.location.eq("US") & frame.date.ge("2026-09-12")]
        vintages.extend(frame.assign(snapshot=path.parent.name).to_dict("records"))
    pd.DataFrame(vintages).to_csv(OUT / "national_truth_vintages.csv", index=False)
    source_rows = []
    for dataset, label in [("mpgq-jmmr", "CDC preliminary"), ("ua7e-t2fy", "CDC published")]:
        metadata = json.loads((SNAPSHOT / "raw" / f"{dataset}-metadata.json").read_text())
        frame = pd.read_json(SNAPSHOT / "raw" / f"{dataset}-0.json")
        frame = frame[frame.jurisdiction.eq("USA") & frame.weekendingdate.ge("2026-09-12")].copy()
        frame["source"] = label
        frame["source_updated_at"] = str(pd.Timestamp(metadata["rowsUpdatedAt"], unit="s", tz="UTC"))
        source_rows.extend(frame.to_dict("records"))
    pd.DataFrame(source_rows).to_csv(OUT / "raw_preliminary_vs_published_US.csv", index=False)

    us_sigma = pd.read_csv(OUT / "forecast_mu_sigma.csv").query("location_name == 'US'")
    us_sigma["implied_count_median"] = np.expm1(np.log1p(3835) + us_sigma.mu)
    # At US levels the zero-floor probability is negligible; this is the
    # untruncated shifted-lognormal mean, distinct from the submitted median.
    us_sigma["implied_count_mean_before_zero_floor"] = np.expm1(np.log1p(3835) + us_sigma.mu + .5 * us_sigma.sigma ** 2)
    us_sigma.to_csv(OUT / "national_mean_vs_median.csv", index=False)
    flat = slopes[slopes.flat_under_stated_10pct_range_rule].sort_values("forecast_h0_to_h3_percent")
    flat_display = flat[["location_name", *cols, "forecast_endpoint_admissions_per_week", "forecast_h0_to_h3_percent"]].rename(
        columns={"location_name": "Location", "forecast_h0": "Oct 17", "forecast_h1": "Oct 24", "forecast_h2": "Oct 31", "forecast_h3": "Nov 7",
                 "forecast_endpoint_admissions_per_week": "Slope / week", "forecast_h0_to_h3_percent": "% change"})
    cf = pd.read_csv(OUT / "counterfactual_forecasts.csv")
    hypothetical_anchor = cf.loc[cf.scenario.eq("frozen_coefficients_US_anchor_continues_prior_log_growth"), "hypothetical_anchor"].dropna().iloc[0]
    cf_us = cf[cf.location_name.eq("US")].pivot(index="scenario", columns="lead_weeks", values="median")
    cf_us["% change"] = 100 * (cf_us[4] / cf_us[1] - 1)
    cf_us.to_csv(OUT / "national_counterfactual_summary.csv")
    summary = json.loads((OUT / "fit_summary.json").read_text())
    remote = json.loads((OUT / "remote_verification.json").read_text())
    objective = json.loads((OUT / "objective_check.json").read_text())
    us = truth[truth.location.eq("US") & truth.date.ge("2026-08-29")].sort_values("date")
    submitted_us = chart[chart.location.eq("US")].sort_values("horizon")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=submitted_us.target_end_date, y=submitted_us.lo95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=submitted_us.target_end_date, y=submitted_us.hi95, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(219,135,68,.10)", name="Submitted 95% interval"))
    fig.add_trace(go.Scatter(x=submitted_us.target_end_date, y=submitted_us.lo50, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=submitted_us.target_end_date, y=submitted_us.hi50, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(219,135,68,.23)", name="Submitted 50% interval"))
    fig.add_trace(go.Scatter(x=us.date, y=us.value, name="Observed at run time (latest preliminary)", mode="lines+markers", line=dict(color="#253e4c", width=2)))
    fig.add_trace(go.Scatter(x=submitted_us.target_end_date, y=submitted_us["median"], name="Submitted Linear median", mode="lines+markers", line=dict(color="#db8744", width=3)))
    fig.update_layout(template="plotly_white", title="US admissions: run-time observations and submitted forecasts", height=460,
                      margin=dict(l=65, r=20, t=60, b=60), yaxis=dict(title="Weekly admissions", rangemode="tozero"),
                      xaxis=dict(title="Week ending Saturday"), legend=dict(orientation="h", y=-.23), font=dict(size=13))
    fig_html = fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})
    fig2 = go.Figure()
    labels = {"saved_model": ("Submitted model (unrounded)", "#db8744", "solid"),
              "frozen_coefficients_zero_horizon_level_contribution": ("Frozen fit: level-by-lead term removed", "#6a7780", "dot"),
              "refit_location_centered_mean_level": ("Diagnostic refit: location-centered level", "#007f89", "dash")}
    dates = ["2026-10-17", "2026-10-24", "2026-10-31", "2026-11-07"]
    for key, (label, color, dash) in labels.items():
        fig2.add_trace(go.Scatter(x=dates, y=cf_us.loc[key, [1, 2, 3, 4]], mode="lines+markers", name=label,
                                  line=dict(color=color, dash=dash, width=2.5)))
    fig2.update_layout(template="plotly_white", title="US median sensitivity: controlled diagnostics", height=390,
                      margin=dict(l=65, r=20, t=60, b=60), yaxis=dict(title="Weekly admissions", rangemode="tozero"),
                      xaxis=dict(title="Forecast week ending Saturday"), legend=dict(orientation="h", y=-.25), font=dict(size=13))
    cf_html = fig2.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
    # Concrete, unapplied proposal. Outcome reconstruction still uses the original
    # panel_log_level_t; only copies passed to mean-design standardization change.
    original = (ROOT / "src/mighte/panel_ar.py").read_text()
    old = "    train_scaled, test_scaled = _standardize_from_training(train, test, continuous)\n"
    new = """    # Predictor level is relative to each location's training-only history.
    # Keep original origin levels intact for log1p/count reconstruction.
    level_centers = train.groupby("location_name")["panel_log_level_t"].mean()
    train_features, test_features = train.copy(), test.copy()
    train_features["panel_log_level_t"] -= train_features["location_name"].map(level_centers)
    test_features["panel_log_level_t"] -= test_features["location_name"].map(level_centers)
    train_scaled, test_scaled = _standardize_from_training(train_features, test_features, continuous)
"""
    assert original.count(old) == 1
    import difflib
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), original.replace(old, new, 1).splitlines(keepends=True),
                                         fromfile="a/src/mighte/panel_ar.py", tofile="b/src/mighte/panel_ar.py"))
    (OUT / "candidate_centering_UNAPPLIED.patch").write_text(patch)
    notes = [
        ("Implementation", "Raw-to-training", "PASS", "Prepared hospitalization training exactly reproduced from the pinned snapshot and bundled historical inputs."),
        ("Implementation", "Numerical reproduction", "PASS", f"143,006 training examples; 316 mean terms; six successful scale optimizations; maximum quantile discrepancy {summary['max_saved_quantile_absolute_difference']:.3g}."),
        ("Implementation", "Horizon / leakage / integration", "PASS in targeted tests", "Exact-calendar labels; t to t+h direct response; h=1..4 maps to hub 0..3; one anchor integration; future-poison invariant passed."),
        ("Implementation", "Floor / rounding / fallback / cache", "PASS for this run", "All 53 anchors observed, all 212 test rows retained, no hospitalization fallback; submitted values equal rounded Linear component at every key. US floor inactive."),
        ("Implementation", "Ensemble / public export / plot", "PASS", "4,876 ensemble values equal rounded means of three unrounded components; merged CSVs and live HTML byte-identical; all 1,060 displayed Linear quantiles exact."),
        ("Structural", "Absolute level pooled across locations", "CONCERN", "A shared negative level-by-lead coefficient treats population-scale count differences as forecasting signals; no location-specific lead offsets can absorb them."),
        ("Structural", "Historical seed and preliminary slowdown", "CONTRIBUTORS / NOT SUFFICIENT", "NHSN-only refit remains flat; counterfactual continued newest-week US growth remains flat under frozen coefficients."),
        ("Numerical definition", "Normalized inverse-variance weights", "SECONDARY CONCERN", "Mean step optimizes normalized weighted ridge, not literal fixed-penalty coordinate Gaussian NLL. Recorded weight mean 2.945; not the flatness explanation by itself."),
        ("Predictive validity", "Replacement accuracy / calibration", "NOT ASSESSED", "No rolling-origin comparison or new calibration performed; current round outcomes not yet known. Ablations do not establish a better forecasting model."),
        ("Provenance", "Full run-time source snapshot", "LIMITED", "Whole-package code hash differs from the run; package has pre-existing report/evaluation/CLI edits. Relevant Linear reproduction is numerically exact, but no source bundle was stored."),
    ]
    pd.DataFrame(notes, columns=["Type", "Check", "Status", "Evidence / limit"]).to_csv(OUT / "claim_ledger.csv", index=False)
    before = json.loads((OUT / "before_hashes.json").read_text())
    import hashlib
    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in before}
    assert after == before
    save_json("final_protected_file_verification.json", {"protected_files": len(before), "unchanged": True, "sha256": after})
    tests = {"repository_data_models_golden_contract": {"passed": 41, "seconds": 3.36},
             "isolated_synthetic_and_invariants": {"passed": 7, "seconds": 1.93},
             "repository_workflow_and_PR_review": {"passed": 46, "seconds": 33.51},
             "total_passed": 94,
             "workflow_log": (OUT / "workflow-tests.log").read_text()}
    save_json("test_results.json", tests)
    test_text = html.escape(tests["workflow_log"].strip())
    cf_display = cf_us.reset_index().rename(columns={"scenario": "Diagnostic (unvalidated)", 1: "Oct 17", 2: "Oct 24", 3: "Oct 31", 4: "Nov 7"})
    intervals = linear[linear.location.eq("US") & linear.output_type_id.isin([.025, .25, .5, .75, .975])].pivot(
        index="target_end_date", columns="output_type_id", values="value").reset_index().rename(
            columns={"target_end_date": "Forecast week", .025: "2.5%", .25: "25%", .5: "Median", .75: "75%", .975: "97.5%"})
    doc = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIGHTE-Linear audit · Oct 17, 2026</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#243a45;background:#f8fafb;margin:0}}main{{max-width:1100px;margin:auto;padding:32px 24px 70px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:23px;margin-top:34px}}h3{{font-size:18px}}a{{color:#006b7a}}code,pre{{font:13px/1.5 ui-monospace,monospace}}pre{{white-space:pre-wrap}}.lead{{font-size:19px}}.callout{{border-left:4px solid #db8744;background:#fff6ec;padding:16px 20px}}.pass{{border-color:#007f89;background:#edf8f7}}.muted{{color:#536873}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{text-align:left;vertical-align:top;padding:9px 11px;border-bottom:1px solid #dbe4e9}}th{{background:#edf2f5}}figure{{margin:20px 0 45px;background:white}}figcaption{{padding:0 18px 14px;font-size:14px}}li{{margin:5px 0}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}figure{{break-inside:avoid}}}} </style><main>
<p class="muted">Read-only scientific diagnostic · 15 October 2026 · Reference Saturday 17 October 2026</p>
<h1>MIGHTE-Linear: why the national forecast flattens</h1>
<p class="lead"><strong>The national curve is flat in the submitted CSV, and there is a reproducible structural reason in the model design.</strong> The strongest identified mechanism is negative mean reversion tied to absolute admission counts and shared across locations, with a larger penalty at longer leads. Targeted checks found no stale anchor, lost trend from indexing, broken inverse transform, unintended fallback, rounding distortion, or review-plot mismatch for this run.</p>
<div class="callout"><strong>Interpretation:</strong> this is a model-specification concern, not evidence that the future must rise or that the submitted forecast is already known to be wrong. Synthetic rising series are forecast upward. Controlled changes restore national growth, but their prospective accuracy and calibration remain untested. Production code, settings, submitted forecasts, and published review were preserved.</div>

<h2>Observed rise and the curves that are actually flat</h2>
<p>At the run's data vintage, US weekly admissions were 1,554 (Sep 12), 2,070 (Sep 19), 2,660 (Sep 26), 3,420 (Oct 3), and 3,835 (Oct 10). The last increment is +415 admissions (+12.13%); Sep 19 to Oct 10 adds 588.3 admissions/week by endpoint slope. Linear submits medians 4,058, 4,049, 3,987, 3,871: −4.61% from Oct 17 to Nov 7 and −62.33 admissions/week by endpoint slope. From the Oct 10 anchor to Nov 7, the median is +0.94%, not a prediction of a large decline.</p>
<p>For an explicit descriptive threshold, “flat” here means the range of four forecast medians is at most 10% of the first median. Five of 53 locations meet it. All 52 subnational locations have a higher final than first forecast median; only US falls. Many smaller states still project substantial percentage growth. OLS and endpoint slopes and the recent observed values for every location are in <a href="jurisdiction_slopes.csv">jurisdiction_slopes.csv</a>.</p>
<div class="scroll">{table(flat_display)}</div>
<figure>{fig_html}<figcaption>Only observations available at the October 15 run are shown. Shading is the actual submitted 50% and 95% predictive interval. The expanding bands make the median visually flat; the underlying medians confirm that it is flat.</figcaption></figure>
<div class="scroll">{table(intervals, 0)}</div>
<p>The plotted center is a <strong>median</strong>. A Gaussian log-change model does not make its inverse-transformed center the mean admissions count. The implied unfloored count means are {', '.join(f'{x:,.0f}' for x in us_sigma.implied_count_mean_before_zero_floor)}; they rise modestly as uncertainty expands. This does not eliminate the central-trajectory concern. See <a href="national_mean_vs_median.csv">mean versus median calculations</a>.</p>

<h2>Data vintage, preliminary observations, and revisions</h2>
<p>The preserved snapshot <code>20261015T043834801440Z</code> completed at 04:38:45 UTC Oct 15. Fitting began immediately afterward and the full run completed at 06:27:14 UTC. Its observation anchor is Oct 10, seven days before the reference Saturday. Every hospitalization location had that anchor. There were 36 interior interpolated training values; none of the latest anchors was carried forward.</p>
<p>The hub hospitalization file was updated Oct 14 at 14:52:09 UTC and took priority over CDC preliminary data updated at 14:32:44 UTC that day. They agree on the latest US sequence. The separately retained <em>published</em> CDC dataset was updated Oct 9 at 14:09 UTC and ended at Oct 3 with 3,334 admissions; it had no Oct 10 value. “Published” does not imply permanently final. Oct 10 = 3,835 was preliminary and could revise.</p>
<p>The preceding Oct 7 preliminary vintage had Oct 3 = 3,246. That rose to 3,334 in the Oct 9 published data and 3,420 in the run's Oct 14 data (+174, or 5.36%, from the first vintage). Sep 26 also revised from 2,621 → 2,633 → 2,660. The newest observed growth slowed from 28.57% (Sep 26 to Oct 3) to 12.13% (Oct 3 to Oct 10); a reporting-delay explanation is plausible but cannot be quantified as a correction from these vintages alone. No learned backfill nowcast is in this model.</p>
<p>In that newest week, among the 52 subnational jurisdictions, 22 increased, 22 fell, and eight were unchanged. Their unweighted mean log1p change is −0.00945, while the US log1p change is +0.11450. The model's “national” growth features are unweighted cross-jurisdiction summaries, not the growth of the admission total. The frozen-coefficient latest-growth counterfactual below isolates why provisional slowdown is not sufficient to explain the flattening.</p>
<p>Evidence: <a href="raw_preliminary_vs_published_US.csv">raw CDC preliminary versus published values</a>; <a href="national_truth_vintages.csv">saved acquisition vintages</a>; {source('src/mighte/data.py', '109–123, 161–169, 228–247, 386–449')}.</p>

<h2>How the flattening is produced</h2>
<p>The response is <code>log1p(y[t+h]) − log1p(y[t])</code>, fitted directly for leads 1–4. Predictor histories end at <code>t</code>. There is no autoregressive recursion over predicted future values. The mean design uses a globally standardized absolute <code>log1p(count)</code> predictor, location intercepts, shared lead interactions, and location-specific deviations for level and the two newest changes. The location deviations have <em>no lead interaction</em>, so they shift all four leads together and cannot cancel a location's lead-dependent level penalty.</p>
<p>At the US anchor, <code>log1p(3835) = 8.2522</code>, versus a pooled training mean 3.3740 and SD 1.9120: a standardized level of 2.5514. This largely reflects the national count scale. The shared standardized level coefficients added at leads 2, 3, and 4 are −0.03086, −0.07530, and −0.11958. At US level they contribute −0.07874, −0.19211, and −0.30510 log units. The final term is the largest negative change from the first to last lead. The combined target-calendar contribution actually rises over these weeks; seasonal timing alone does not force this downturn.</p>
<p>Population sizes do not enter this Linear mean design. A high-count state or national series is thus partly treated as if its absolute count level indicates an advanced epidemic phase. This is a structural vulnerability of pooling absolute levels with shared lead effects. It is not a proof that all mean reversion is undesirable. Ridge shrinkage and Gaussian weighting determine the magnitude, but no coefficient or floor mechanically requires a flat forecast.</p>
<p>Source: {source('src/mighte/panel_ar.py', '92–145, 237–324, 418–461, 482–547')}. Full <a href="mean_coefficients.csv">coefficients</a>, <a href="anchor_features.csv">anchor rows</a>, <a href="standardization.csv">training scalers</a>, <a href="grouped_contributions.csv">grouped contributions</a>, and <a href="mean_contributions.csv">individual contributions</a> are inspectable. Individual coefficients share correlated predictors; the controlled tests below provide stronger evidence than coefficient magnitude alone.</p>

<h2>Controlled tests of the mechanism</h2>
<p>All scenarios use the preserved observation vintage. The first two perturb only fitted mean contributions. The next refits change only the named mean-design feature construction or training-era restriction; frozen penalties and iteration limits remain unchanged. These are falsification and attribution checks, not selected replacements or scored competitors.</p>
<figure>{cf_html}<figcaption>Diagnostic medians only; dashed/dotted lines are unvalidated counterfactuals. Intervals for those alternatives were not evaluated. The production forecast is unchanged.</figcaption></figure>
<div class="scroll">{table(cf_display)}</div>
<p>Zeroing the three lead-by-level contributions under frozen coefficients restores growth, and a refit removing those three columns also restores growth. Centering the level predictor within each location's training history preserves the current count anchor and future log-change targets and yields 4,071 → 4,403 → 4,921 → 5,413 (+32.96%). Restricting training origins to NHSN weeks gives 4,073 → 3,999 → 3,925 → 3,809: reconstructed seed history is not necessary for the flat result.</p>
<p>The last scenario replaces only the newest US observation by continuation of the preceding log-growth rate (a hypothetical anchor of {hypothetical_anchor:,.2f}), retaining frozen coefficients and original training scalers. It gives 4,430 → 4,416 → 4,341 → 4,200. This is not a proposed backfill estimate; it shows that the observed deceleration alone does not account for the forecast's shape.</p>
<p>Independent tests also verified exact lag construction, direct targets, one-time anchor integration, correct target dates, no future-feature leakage, missing-week target handling, and fitted continuation of flat and log-linear rising series. These capacity checks do not establish epidemic forecast performance. Reproduce with <a href="test_linear_diagnostics.py">isolated tests</a> and <a href="counterfactuals.py">counterfactuals.py</a>.</p>

<h2>Export, ensemble, and review visualization</h2>
<p>All 4,876 final Linear quantiles exactly equal rounded internal Linear quantiles. All 4,876 Nsemble quantiles exactly equal the rounded equal-third average of unrounded WW-only, NSSP-only, and Linear quantiles. Linear itself receives hospitalization history and seasonality, without wastewater or NSSP predictors. The directly modeled US median declines across leads even though the sum of subnational medians rises from 3,874 to 4,530 (+16.93%, including Puerto Rico). That sum is a diagnostic of lack of reconciliation, not the median or interval of a valid aggregate predictive distribution.</p>
<p>The three CSVs at merged hub commit <code>53e86e2</code> match the local run byte for byte. The live review HTML has SHA-256 <code>{remote['live_review']['sha256']}</code> and matches the preserved local review byte for byte. Every displayed current Linear median and interval (1,060 values) equals the submitted CSV; all 17,119 hospitalization truth rows agree with the run snapshot. The review uses a later 14:05 snapshot, but hospitalization values did not change. Its label “Observed · revised” does not identify the newest point as preliminary. Visually inspecting Linear selected for US confirms the same flat median and expanding intervals.</p>
<p>Evidence: <a href="remote_verification.json">remote byte verification</a>, <a href="export_and_plot_checks.json">per-key export and plot checks</a>; {source('src/mighte/models.py', '155–174')}; {source('src/mighte/pipeline.py', '266–297, 350–384')}; {source('src/mighte/report.py', '51–88, 251–271')}; <a href="https://github.com/cdcepi/FluSight-forecast-hub/pull/3838">merged PR #3838</a>; <a href="https://www.meyerlab.io/flusight_2026/">live review</a>.</p>

<h2>A secondary numerical-definition issue</h2>
<p>The fit routine says it coordinate-minimizes penalized Gaussian NLL, but it normalizes inverse-variance weights to mean one before each penalized mean solve. For fixed nonzero ridge penalties, that changes the mean-coordinate optimum relative to literal NLL plus those fixed penalties. Here the raw weight mean is {objective['raw_inverse_variance_weight_mean']:.5f}; the normalized objective's mean gradient is {objective['normalized_weight_mean_gradient_max_abs']:.3g}, while the literal fixed-penalty NLL gradient is {objective['fixed_penalty_NLL_mean_gradient_max_abs']:.5f}. Holding the fitted scale fixed and using unnormalized weights still gives a nearly flat 4,224 → 4,214 → 4,205 → 4,136, so this discrepancy is not sufficient to explain the current curve.</p>
<p>Treat this as a specification/documentation issue unless a literal joint penalized likelihood was intended. Changing it would alter scientific behavior and requires separate validation. In this run, every scale optimization reported convergence and final mean iterations stabilized. Production checks only test finite scale coefficients; adding convergence diagnostics would improve auditability. Source: {source('src/mighte/panel_ar.py', '367–415, 426–460')}; <a href="objective_check.json">objective diagnostic</a>.</p>

<h2>Smallest reviewable candidate and validation before any change</h2>
<p><a href="candidate_centering_UNAPPLIED.patch">candidate_centering_UNAPPLIED.patch</a> shows a six-line mean-design change: center the level predictor by its location's training-only mean before global scaling. It leaves the original origin count and response untouched and was not applied. It represents the centered-level counterfactual above; it is a candidate to validate, not a recommendation to deploy from one forecast. A fixed population offset is another principled option, but should be chosen in a frozen design before comparison.</p>
<ol><li>Approve the scientific comparison and freeze a candidate before testing. Keep the original model as the primary reference; do not tune on 2026–27 errors. Any eventual deployment changes the frozen model specification and should receive a new documented model version.</li><li>Add invariance tests for proportional series of different population scales, forecast-time centering, unknown-location handling, unchanged log1p integration, and complete export keys. Preserve the synthetic rising/flat tests.</li><li>Run a small, predetermined set of historical rolling origins spanning early rises, peaks, declines, and summers. Use saved as-of vintages where available; label revised-truth emulation explicitly elsewhere. Rebuild historical transforms at each origin.</li><li>Compare WIS and log-WIS, MAE, 50/80/95% coverage, and calibration by horizon and US versus states, with a persistence and hub baseline. Assess whether stronger growth improves early-rise behavior without worsening peaks/declines or overconfidence.</li><li>Because Linear is one-third of Nsemble, recompute and validate the ensemble in an isolated preview before any approved release. Do not obtain national intervals by summing state quantiles.</li><li>Optionally improve diagnostics by saving coefficients/scalers, optimizer status, and training-feature summaries with each run, and label preliminary observations in review. Those changes do not require changing forecasts.</li></ol>

<h2>Verification, limits, and reproducibility</h2>
<div class="scroll">{table(pd.DataFrame(notes, columns=['Type', 'Check', 'Status', 'Evidence / limit']))}</div>
<p>Project tests for data/model behavior, golden numerical fits, and contract validation: <strong>41 passed in 3.36 s</strong>. Isolated synthetic/invariant tests: <strong>7 passed in 1.93 s</strong>. Workflow and PR-review checks:</p><pre>{test_text}</pre>
<p>The isolated full Linear refit used 143,006 training examples, 316 mean terms, ten scale terms, the recorded runtime settings, and the same numerical package versions; it took 5.1 seconds. Maximum discrepancy over 4,876 saved internal quantiles: 4.67 × 10⁻⁸ admissions. No boosted model was refit. {len(before)} source/settings/run/review files were hashed before and after and remained unchanged; the original snapshot's internal hashes were verified. The working tree already contained edits in README, CLI, evaluation, report, and tests. This audit did not edit those files. The whole-package hash differs from the run, which lacks a frozen source bundle; exact numerical reproduction is the primary evidence for the Linear path.</p>
<p>October 17 and later target outcomes were not available as of this audit. No replacement accuracy claim, calibration claim, or absolute correctness guarantee is justified. No task blocker remains for this diagnostic; longer rolling-origin validation remains necessary before an approved model change.</p>
<pre>.venv/bin/python diagnostics/linear_audit_20261015/audit_linear.py
.venv/bin/python diagnostics/linear_audit_20261015/counterfactuals.py
.venv/bin/python -m pytest -p no:cacheprovider diagnostics/linear_audit_20261015/test_linear_diagnostics.py -q
.venv/bin/python diagnostics/linear_audit_20261015/verify_public.py
.venv/bin/python diagnostics/linear_audit_20261015/build_evidence.py</pre>
<p>The public verification script requires read-only network access. Other diagnostics use the preserved local snapshot. Relevant sources and CSVs are linked above; <a href="fit_summary.json">fit summary</a>, <a href="test_results.json">test evidence</a>, <a href="claim_ledger.csv">claim ledger</a>, <a href="final_protected_file_verification.json">protected-file hashes</a>.</p>
</main></html>"""
    (OUT / "AUDIT.html").write_text(doc)
    print("Built AUDIT.html, slope/vintage/export tables, claim ledger, and unapplied candidate patch.")
    print(flat_display.to_string(index=False))
    print("Implied US count means", us_sigma.implied_count_mean_before_zero_floor.round(2).tolist())


if __name__ == "__main__":
    main()
