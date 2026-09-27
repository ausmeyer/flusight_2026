# Data and vintage policy

Every forecast acquisition downloads the complete available histories. Appending only recent weeks would miss older backfill. Each acquisition has a timestamped directory, raw responses, URLs, dataset update times and SHA-256 hashes. Failed or truncated acquisitions never replace the latest successful snapshot. CDC datasets are checked before and after pagination to detect updates during a download.

Hub rules and target data are retrieved from a single commit pinned when acquisition begins. File-update timestamps are queried at that same commit, so a simultaneous hub update cannot attach a new timestamp to older downloaded values. The snapshot records the commit and every source URL.

## Sources

| Stream | Public source | Processing |
|---|---|---|
| Hospitalizations | Hub `target-hospital-admissions.csv`; CDC `mpgq-jmmr` preliminary and `ua7e-t2fy` published | Compare hub file commit time and CDC data-update times; newest source wins on overlapping keys. Retain suppression rather than replacing a new missing value with an older observed value. |
| ED proportion | Hub `target-ed-visits-prop.csv`; CDC `rdmq-nq56`, county='All' | Convert CDC influenza visit percentage to a proportion exactly once. Preserve current source availability by jurisdiction. |
| National NSSP covariate | National observation from the same ED truth retrieval | Convert proportion to percentage points for the frozen hospitalization recipe. |
| WastewaterSCAN | CDC `ymmh-divb`, source='WastewaterSCAN', pcr_target='fluav' | Complete raw history; equal-site weekly mean and frozen lag rules. |
| Locations and submission rules | FluSight `auxiliary-data/locations.csv` and `hub-config/` | Snapshot with every acquisition and check again before a PR. |

CDC metadata paths are `https://data.cdc.gov/api/views/DATASET.json`; observation paths are `https://data.cdc.gov/resource/DATASET.json`. The [2026–27 hub documentation](https://github.com/cdcepi/FluSight-forecast-hub#emergency-department-visits) says ED target data will be updated Wednesday by midday, pending availability, ahead of the Friday public release. The pipeline automatically prefers hub values when their file update is more recent. Both source copies remain available for inspection. Source priority and counts appear in each snapshot manifest.

## Historical training data

The hospitalization seed preserves the original ILINet/FluSurv regression, rate rounding, historical population, count rounding and 728-day shift through June 2021. Sources and hashes are recorded in `data/historical/PROVENANCE.json`. From July 2021 onward, training uses the latest revised NHSN counts as reported.

The earlier stitch multiplied October 2022–April 2024 counts by influenza's share of respiratory ED visits and applied a legacy-regime multiplier, capped at 2× for 51 of 53 locations. Both created artificial steps (US training 1,591 → 313 on 2022-10-01 while reported admissions rose; 2,175 → 1,016 on 2024-11-02 while they were flat) and were removed in September 2026. The ILINet proxy and current NHSN counts are already on the same scale (median log ratio 0.00 since November 2024), so no rescaling is applied.

Observed gaps strictly between observed dates are linearly interpolated for training on a weekly calendar. Their count is recorded. Latest anchors are never extrapolated or fabricated; absent hospitalization anchors stop production. Scoring always uses the original observed values, with missing/suppressed truth left missing.

`ilinet_normalized.csv` is the frozen per-location transformed ILINet input produced by the existing `bestNormalize(ILI + 1)` preprocessing. The recorded source CSV contains those transformed values, not raw ILI percentages. Bundling it preserves the exact historical input without requiring an R environment or a neighboring checkout.

Observed NSSP ED data begin in October 2022, and no machine-readable pre-pandemic series is public. The ED history is therefore assembled like the hospitalization seed: one pooled regression of observed ED log-odds on normalized ILINet, over all locations and weeks where both exist, is applied to each location's ILINet through June 30, 2019; dates shift forward 728 days and values are rounded to NSSP's 0.01-point precision. The gap before observed NSSP stays missing rather than interpolated. This assumes the recent ILI-to-ED relationship held before 2019. Proxy values are used for training only, never for evaluation.

## Prospective safeguards

Forecast generation uses only target observations dated on or before the anchor. External measurements are cut off by the frozen lag rules. The complete bytes were actually downloaded before forecast generation; nominal covariate `available_date` values reproduce the original recipe's reference-week indexing and are not represented as actual publication timestamps. Actual acquisition times are recorded separately. Current revisions may inform a current fit; they never rewrite an earlier run's inputs or predictions.

The national NSSP predictor uses the anchor week (the Saturday before the forecast reference date) and wastewater the week before it; each may be one week staler under the frozen recipe. Newer wastewater samples are still downloaded and preserved, but the model does not use them beyond this fixed cutoff. These lags preserve the study recipe; they are not a finding that these are the optimal prospective lags. Backfilled source values are incorporated when refitting; no learned backfill correction is applied to unrevised observations.

No location or week is left out. The fallback is MIGHTE-Base without wastewater and NSSP (all other standard features retained). It replaces any part, trends included, whose covariate the model would not receive (checked with the model's own covariate lookup), and it forecasts and classifies a location without an anchor value from its last reported value. Quantiles above the hub's plausibility limits (ED 0.25, admissions 30% of population) are left unchanged and flagged for review. Each case appears as a notice in the terminal, the report and the PR description. A failed download stops the run.

Production requires Wednesday freshness and completion inside the Sunday-to-Wednesday submission window. A resumed fit keeps its original snapshot; obtaining more recent revisions requires a new run. The same checks apply at submission. Out-of-window rehearsals have separate directories and are never scored as prospective performance.
