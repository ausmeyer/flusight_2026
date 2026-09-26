# Implementation verification

Verified September 25, 2026. These checks establish implementation behavior; they do not establish prospective forecast accuracy.

## Model port

Original and standalone feature tables and predictions were compared on a deterministic, nondegenerate synthetic three-location history in the pinned environment. The maximum absolute prediction difference was 0.0 for both the distributional LightGBM fit and partially pooled autoregression. Golden forecasts from the original implementation are committed under `tests/fixtures/`; tests need no access to the original repository. Linux CI retains a strict numerical tolerance. The initial CI failure was traced to a symmetric synthetic fixture and resolved by using a deterministic, more varied fixture, without relaxing the tolerance or changing the model.

## Historical training adjustment

An independent base-R implementation checked the original pooled regression and legacy adjustment equations against the standalone implementation using the September 25 input snapshot. The original regression coefficients agreed to floating-point precision. All 39,160 hospitalization training values matched exactly; 53 location-specific shifts agreed within 4.8e-16, with identical calibration counts. CSV parsing preserves the stored ED fractions exactly before count rounding. Raw observed truth is unchanged.

The operational pipeline fills internal training gaps linearly and records their count. This check uses that documented interpolation policy; it does not claim that linear interpolation reproduces the earlier R spline interpolation. Latest anchors are never extrapolated.

## Workflow and scoring

Automated checks cover calendar alignment, covariate cutoffs, future-data invariance, full-history revision replacement, suppression, ED unit conversion, historical scaling, fixed ensemble thirds, incomplete components, serialization, hub constraints, deadlines including daylight saving time, and exclusion of previews from prospective scoring. Complete synthetic workflows exercise all three models with and without a Base-only ordinal plugin. Serial and two-process runs produce byte-identical submission files.

WIS is checked against an independent weighted-interval formula. Tests cover matched skill denominators, coverage, revised truth, and undefined zero-baseline scores. The interactive report was exercised with synthetic scored forecasts for hospitalization and ED targets; displayed values were checked against Python calculations. Synthetic scores remain outside the prospective archive. The report was checked in the browser with target, location, baseline and horizon filters; explanatory chart notes were removed.

The comparison-model update passes 45 local tests and linting. Tests verify current-season discovery from actual hub submissions, unchanged-file caching, offline use, and rejection of incomplete catalogs. Browser checks confirm searchable model selection, Select all/Clear, static colors, and unchanged score values when other rows are hidden. The forecast-week buttons, slider keyboard controls and mouse wheel were checked against multiple synthetic forecast weeks: forecast curves change while the rendered observed curve and axis positions stay fixed, including after manual zoom. Expanded history was checked for both targets in the fixture and against actual observed history in the current report. The four summary statistics and header timestamp are removed.

FluSight-ensemble, UMass-flusion and Google_SAI-FluEns remain available as comparison curves and scoring baselines, with browser values matching Python scores. Fetching a comparison for a preview never adds that week to prospective accuracy. Unpublished comparison options remain visible and disabled in the current report. An isolated import check also read the actual May 30, 2026 files for all four public comparisons; those files were not fitted or scored and remain outside the prospective archive.

Registration and submission are separate, explicitly confirmed commands. Tests exercise cancellation without a GitHub request, upload allowlists, preview rejection, and a final deadline check before PR creation. CI only runs local tests and validation; it never submits forecasts or opens a hub PR.

## Initial current-input rehearsal

The September 26, 2026 preview completed using the September 25 revised-data snapshot and the preserved historical training adjustment. All four boosted components completed 100 fits, and the linear fit completed. Runtime was about 46 minutes on the development computer with two component workers. The three serialized files passed local hub-contract validation: 9,568 rows for Base and 4,876 each for Linear and Nsemble. All 53 hospitalization locations and the 51 ED locations with current observations were covered. Recomputing the fixed one-third ensemble from its saved components reproduced all 4,876 final rounded quantiles exactly.

This was one forecast origin using current inputs, not a retrospective performance study. It is stored under `runs/previews/2026-09-26/20260925T193624628497Z`, cannot be submitted, and contributes no prospective accuracy rows. The ED fit used observed history; pre-pandemic reconstruction still requires the weekly historical NSSP input described in `DATA.md`.

## ED response-link correction

The production ED response now uses changes in log-odds of canonical proportions. Its input boundary is fixed at 0.00005, or 0.005 percentage points, half the reporting increment observed in the current NSSP data. This replaces the one-percentage-point additive offset implied by log1p of percentage points. Input boundary handling does not change truth or predictor histories, and inverse-link forecast quantiles are not floored at that boundary. Optional ED reconstruction uses the same link. The initial preview above retains the earlier transformation and remains unchanged.

All 58 local tests pass, including the original hospitalization golden forecasts. New checks cover finite boundary handling, interior round trips, explicit unit conversion, log-odds changes and anchor integration against an independent formula, proportion export, quantile ordering, and rejection of ED checkpoints with missing or incompatible transformation metadata. Existing synthetic workflows exercise both serial and parallel orchestration with the corrected ED fit. A full-round, single-fit check on the current ED data produced 4,692 valid quantiles. These checks establish implementation behavior; they do not establish prospective interval coverage or an optimal boundary convention.

The complete rerun is `runs/previews/2026-09-26/20260925T212404979138Z`, using a fresh acquisition at `20260925T212341670563Z`. All four boosted components completed 100 fits, plus Linear, in about 46 minutes. The three exported files passed hub-contract validation with the same row and location coverage as the initial rehearsal. The refreshed input values matched the previous snapshot, and all 14,628 final hospitalization quantiles across the three models matched exactly. ED quantiles reconstructed independently from all 100 checkpoints matched the export within 2.2e-13 in proportion units. There were 74 boundary observations among 10,608 ED training values and no boundary observations at the current anchors.

The national ED horizon-3 median is 0.96%, with a 95% prediction interval of 0.38–2.27%, compared with the initial preview's 1.11% and 0.66–1.65%. The report payload agrees with the exported quantiles, and the regenerated national ED chart was checked in the browser. The run remains a non-submittable preview with no prospective accuracy rows.

## Clean installation and CI

Commit `c5f603f` was cloned from GitHub into a new temporary directory. The documented `./mighte setup` command created its own environment; all 39 tests, linting and metadata checks passed with the package loaded from that clone. The same commit passed [Linux GitHub CI](https://github.com/ausmeyer/flusight_2026/actions/runs/36180915580). No neighboring research directory was needed.

## Direct categorical component

The prospective multiclass port reproduces all 60 probabilities from the independent study fixture exactly in the pinned environment. The fixture uses the shared Base feature table, synthetic covariates and populations, and exercises all five classes. Original source and fixture hashes are recorded in `port-provenance.json`.

All 100 local tests pass. Checks cover the CDC count/rate boundaries at every horizon, the reference-minus-seven baseline, identical Base/ordinal feature columns including both covariate lag families, future-outcome and unreleased-covariate invariance, checkpoint reuse, invalid probability rejection and complete categorical location/horizon coverage. Full synthetic pipeline runs fit the actual classifier, append its output only to Base and produce identical serial/parallel files. Quantitative golden forecasts remain unchanged.

RPS, Brier and log scores are checked against numerical examples, including missing and revised observed truth, fixed forecast populations, tied categories, zero-probability outcomes and matched skill denominators. Categorical-only peer models are loaded from the season catalog. Neither their preview weeks nor local previews enter accuracy. Browser checks confirm probability grids, model selection and horizon filtering; displayed scores agree with the Python calculations. The synthetic dashboard fixture remains outside the prospective archive and is not published.

A fresh temporary standalone copy was installed with `./mighte setup`; all 100 tests, linting and metadata validation also passed there, with package imports resolved inside that copy and no neighboring study directory required.
