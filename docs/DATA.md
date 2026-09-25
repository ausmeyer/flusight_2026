# Data and vintage policy

Every forecast acquisition downloads the complete available histories. Appending only recent weeks would miss older backfill. Each acquisition has a timestamped directory, raw responses, URLs, dataset update times and SHA-256 hashes. Failed or truncated acquisitions never replace the latest successful snapshot. CDC datasets are checked before and after pagination to detect updates during a download.

Hub rules and target data are retrieved from a single commit pinned when acquisition begins. File-update timestamps are queried at that same commit, so a simultaneous hub update cannot attach a new timestamp to older downloaded values. The snapshot records the commit and every source URL.

## Sources

| Stream | Public source | Processing |
|---|---|---|
| Hospitalizations | Hub `target-hospital-admissions.csv`; CDC `mpgq-jmmr` preliminary and `ua7e-t2fy` published | Compare hub file commit time and CDC data-update times; newest source wins on overlapping keys. Retain suppression rather than replacing a new missing value with an older observed value. |
| ED proportion | Hub `target-ed-visits-prop.csv`; CDC `rdmq-nq56`, county='All' | Convert CDC influenza visit percentage to a proportion exactly once. Preserve current source availability by jurisdiction. |
| National NSSP covariate | National observation from the same ED truth retrieval | Convert proportion to percentage points for the frozen hospitalization recipe. |
| WastewaterSCAN | CDC `ymmh-divb`, source='WastewaterSCAN', pcr_target='fluav' | Complete raw history; equal-site weekly aggregation and frozen lag rules. |
| Locations and submission rules | FluSight `auxiliary-data/locations.csv` and `hub-config/` | Snapshot with every acquisition and check again before a PR. |

CDC metadata paths are `https://data.cdc.gov/api/views/DATASET.json`; observation paths are `https://data.cdc.gov/resource/DATASET.json`. Hub sources are at `https://github.com/cdcepi/FluSight-forecast-hub`. The hub data may be fresher during the season; it is automatically preferred when its file update is more recent. Both source copies remain available for inspection. Source priority and counts appear in each snapshot manifest.

## Historical training data

The hospitalization seed preserves the original ILINet/FluSurv regression, rate rounding, historical population, count rounding and 728-day shift through June 2021. It is stored before regime adjustment so the adjustment is applied exactly once. Sources, hashes and coefficients are recorded in `data/historical/PROVENANCE.json`. From July 2021 onward, training starts from the latest revised observed NHSN values each week.

The study's historical adjustment is preserved: before May 2024, counts are multiplied by the influenza fraction of combined respiratory ED visits, with the original within-location carry-forward and leading-missing default of one. A state-specific log shift then maps the legacy period before November 2024 to the current regime. The shift is the difference in median log1p(observed) minus log1p(proxy) residuals between regimes, requires 26 legacy and eight current paired weeks, and is bounded to a multiplier of 0.5–2.0. The unshifted ILINet/FluSurv calibration predictor is frozen from the original April 2026 input; observed outcomes and ED fractions are refreshed. Every run records its actual state shifts and calibration counts. These transforms affect training only; evaluation truth remains unadjusted.

Observed gaps strictly between observed dates are linearly interpolated for training on a weekly calendar. Their count is recorded. Latest anchors are never extrapolated or fabricated; absent hospitalization anchors stop production. Scoring always uses the original observed values, with missing/suppressed truth left missing.

`ilinet_normalized.csv` is the frozen per-location transformed ILINet input produced by the existing `bestNormalize(ILI + 1)` preprocessing. The recorded source CSV contains those transformed values, not raw ILI percentages. Bundling it preserves the exact historical input without requiring an R environment or a neighboring checkout.

The public NSSP/Delphi sources checked on September 25, 2026 begin in October 2022. No pre-pandemic national NSSP influenza observations were found. The ED reconstruction implementation consequently has explicit observed-only, post-2022-calibration, and user-supplied pre-pandemic-calibration modes. The latter two regress log1p national ED percentage on normalized national ILINet, apply the fitted relationship to location ILINet through June 30, 2019, and shift dates forward by 728 days, matching the hospitalization proxy's historical cutoff. The national-to-state transfer and any transfer across eras are assumptions, not validated conclusions. No pseudo-observation is used for evaluation.

## Prospective safeguards

Forecast generation uses only target observations dated on or before the anchor. External measurements are cut off by the frozen lag rules. The complete bytes were actually downloaded before forecast generation; nominal covariate `available_date` values reproduce the original recipe's reference-week indexing and are not represented as actual publication timestamps. Actual acquisition times are recorded separately. Current revisions may inform a current fit; they never rewrite an earlier run's inputs or predictions.

Production requires Wednesday freshness and completion inside the Sunday-to-Wednesday submission window. A resumed fit keeps its original snapshot; obtaining more recent revisions requires a new run. The same checks apply at submission. Out-of-window rehearsals have separate directories and are never scored as prospective performance.
