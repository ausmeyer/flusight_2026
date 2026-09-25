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

Registration and submission are separate, explicitly confirmed commands. Tests exercise cancellation without a GitHub request, upload allowlists, preview rejection, and a final deadline check before PR creation. CI only runs local tests and validation; it never submits forecasts or opens a hub PR.
