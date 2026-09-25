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

The comparison-model update passes 42 local tests and linting. Browser checks confirm selectable FluSight-ensemble, UMass-flusion and Google_SAI-FluEns curves and baseline choices, with matching Python scores. Tests verify cached offline comparisons and that fetching a comparison for a preview never adds that week to prospective accuracy. Unpublished comparison options remain visible and disabled in the current report. An isolated import check also read the actual May 30, 2026 files for all four public comparisons; those files were not fitted or scored and remain outside the prospective archive.

Registration and submission are separate, explicitly confirmed commands. Tests exercise cancellation without a GitHub request, upload allowlists, preview rejection, and a final deadline check before PR creation. CI only runs local tests and validation; it never submits forecasts or opens a hub PR.

## Full current-input rehearsal

The September 26, 2026 preview completed using the September 25 revised-data snapshot and the preserved historical training adjustment. All four boosted components completed 100 fits, and the linear fit completed. Runtime was about 46 minutes on the development computer with two component workers. The three serialized files passed local hub-contract validation: 9,568 rows for Base and 4,876 each for Linear and Nsemble. All 53 hospitalization locations and the 51 ED locations with current observations were covered. Recomputing the fixed one-third ensemble from its saved components reproduced all 4,876 final rounded quantiles exactly.

This was one forecast origin using current inputs, not a retrospective performance study. It is stored under `runs/previews/2026-09-26/20260925T193624628497Z`, cannot be submitted, and contributes no prospective accuracy rows. The ED fit used observed history; pre-pandemic reconstruction still requires the weekly historical NSSP input described in `DATA.md`.

## Clean installation and CI

Commit `c5f603f` was cloned from GitHub into a new temporary directory. The documented `./mighte setup` command created its own environment; all 39 tests, linting and metadata checks passed with the package loaded from that clone. The same commit passed [Linux GitHub CI](https://github.com/ausmeyer/flusight_2026/actions/runs/36180915580). No neighboring research directory was needed.
