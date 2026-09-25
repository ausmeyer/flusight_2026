# Prospective scoring

Each target is scored separately. The default table excludes the national series to avoid mixing aggregate and state burdens, matching the previous study's default geography. Location and horizon filters are interactive.

Forecasts enter scoring only if their run was completed within its original FluSight submission window. A week contributes one run: the submitted run if present, otherwise the latest completed on-time run. Previews never enter scoring. Truth is the latest retrieved observed value for the same target/date/location. Suppressed or unavailable outcomes are unscored, not imputed. Scores remain provisional while backfill continues.

- **WIS**: mean quantile score over the 23 CDC quantiles, equivalent to the weighted sum of the 11 central interval scores and median absolute error with denominator 11.5. Report arithmetic mean WIS and geometric mean WIS.
- **Log WIS**: the same score after log1p transformation of observed and forecast values; show its geometric mean. This matches the study's transformation convention.
- **MAE**: absolute error of the median forecast. Per-forecast artifacts also include squared error and signed median error.
- **WIS skill**: `1 - sum(model WIS) / sum(baseline WIS)` on exactly overlapping forecast keys. Positive values are better than the selected baseline; zero is a tie. Missing overlap or zero baseline score yields an undefined value, displayed as a dash. This is a simple paired skill measure, not identical to the pairwise geometric-mean relative score.
- **Relative WIS / relative log-WIS**: scoringutils-style geometric mean of pairwise sum-score ratios across models with shared forecast units, scaled to the selected baseline. Lower is better and the baseline is one. Undefined comparisons remain undefined.
- **Coverage**: fraction of observations inside the inclusive central 50%, 80% and 95% intervals. Nominal coverage appears in the column name. With little prospective data these estimates can be unstable.
- **N / Matched**: number of scored model forecasts and number sharing the selected baseline's keys. Different availability is shown explicitly.

The **Local-persistence** baseline is available immediately because it is generated and frozen in the same run. Its median persists the current observed level; its symmetric uncertainty uses the last 104 observed weekly changes, widened by sqrt(lead). It is named explicitly to avoid confusing it with the official FluSight baseline. Public **FluSight-baseline**, **FluSight-ensemble**, **UMass-flusion** and **Google_SAI-FluEns** forecasts are retrieved when available. All can be selected for curves or as a scoring baseline. Unavailable curves are disabled for the selected week, target and location; missing publications are listed in the run details. Offline review uses cached comparison forecasts. Relative metrics are unavailable until the selected public comparator exists. Forecasts from the study's retrospective outputs are never imported into this prospective table.

Comparison forecasts are fetched for the displayed week even during a preview, but only weeks with an eligible prospective MIGHTE run enter the accuracy table. Displaying or downloading a public forecast does not add a scored prospective week.

ED errors are computed in proportion units, while the time-series plot displays ED percentages. Raw WIS from the hospitalization and ED targets should not be compared or averaged together.
