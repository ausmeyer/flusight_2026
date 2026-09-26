# Prospective scoring

Each target is scored separately. The default table includes states and DC; Puerto Rico and the national series can be selected separately. Location and horizon filters are interactive.

Forecasts enter scoring only if their run was completed within its original FluSight submission window. A week contributes one run: the submitted run if present, otherwise the latest completed on-time run. Previews never enter scoring. Truth is the latest retrieved observed value for the same target/date/location. Suppressed or unavailable outcomes are unscored, not imputed. Scores remain provisional while backfill continues.

- **WIS**: mean quantile score over the 23 CDC quantiles, equivalent to the weighted sum of the 11 central interval scores and median absolute error with denominator 11.5. Report arithmetic mean WIS and geometric mean WIS.
- **Log WIS**: the same score after log1p transformation of observed and forecast values; show its geometric mean. This matches the study's transformation convention.
- **MAE**: absolute error of the median forecast. Per-forecast artifacts also include squared error and signed median error.
- **WIS skill**: `1 - sum(model WIS) / sum(baseline WIS)` on exactly overlapping forecast keys. Positive values are better than the selected baseline; zero is a tie. Missing overlap or zero baseline score yields an undefined value, displayed as a dash. This is a simple paired skill measure, not identical to the pairwise geometric-mean relative score.
- **Relative WIS / relative log-WIS**: scoringutils-style geometric mean of pairwise sum-score ratios across models with shared forecast units, scaled to the selected baseline. Lower is better and the baseline is one. Undefined comparisons remain undefined.
- **Coverage**: fraction of observations inside the inclusive central 50%, 80% and 95% intervals. Nominal coverage appears in the column name. With little prospective data these estimates can be unstable.
- **N / Matched**: number of scored model forecasts and number sharing the selected baseline's keys. Different availability is shown explicitly.

The **Local-persistence** baseline is generated and frozen in the same run. Its median persists the current observed level; its symmetric uncertainty uses the last 104 observed weekly changes, widened by sqrt(lead). It is distinct from the official FluSight baseline. Report refreshes discover models with forecast files in the hub's accepted reference weeks for the current season, including **FluSight-ensemble**, **UMass-flusion** and **Google_SAI-FluEns**. Registration alone does not qualify a model. Forecast files are retrieved for the displayed week and archived prospective MIGHTE weeks; local MIGHTE files remain authoritative. Hub file hashes avoid downloading unchanged comparisons. Offline review uses the cached catalog and files.

The searchable Models control selects any combination of available models for curves and accuracy rows. A model is disabled when it has no loaded forecasts for the selected target. No curve is drawn for a missing week or location. Colors depend only on model identity. Hiding a row does not change the comparison cohort used to calculate relative scores or remove the selected scoring baseline. Missing publications are listed in the run details; undefined relative metrics remain unavailable. Forecasts from the study's retrospective outputs are never imported into this prospective table.

The forecast-week slider, arrow buttons, mouse wheel over the slider, and reference-week dropdown all navigate the same saved forecasts. The latest revised ground-truth curve and both axes remain fixed as the forecast week changes; manual zoom is preserved. Recent history starts 120 days before the latest forecast and extends to include every saved forecast week. Past year and All available extend the observed history without importing reconstructed training values or changing scoring.

Comparison forecasts are fetched for the displayed week even during a preview, but only weeks with an eligible prospective MIGHTE run enter the accuracy table. Displaying or downloading a public forecast does not add a scored prospective week.

ED errors are computed in proportion units, while the time-series plot displays ED percentages. Raw WIS from the hospitalization and ED targets should not be compared or averaged together.

## Hospitalization trends

Five-class PMFs are scored only when both the target week and the Saturday before the reference date have observed hospitalization counts. Labels follow [ORDINAL.md](ORDINAL.md). The forecast's frozen population applies to its comparisons as well. Revised observed values at either week can change the category and scores; training adjustments and proxies never enter this evaluation.

- **RPS:** sum of squared cumulative probability errors over the four ordered category boundaries, without division by four (range 0–4), matching the study.
- **Brier:** sum of squared errors across all five category probabilities (range 0–2).
- **Log score:** negative log probability assigned to the observed category, evaluated with a numerical floor of 1e-15. The zero-probability outcome rate is reported separately.
- **Category accuracy / error:** frequency of the most-probable category matching truth, and mean absolute distance in category steps. Ties favor stable, decrease, increase, large decrease, then large increase.
- **RPS skill:** `1 - sum(model RPS) / sum(baseline RPS)` on exactly matched forecast keys. Missing overlap or a zero baseline sum remains undefined. The preferred baseline is the hub's `FluSight-baseline_cat`, once published for the season.

Quantile interval coverage remains specific to the quantitative targets. The trend plot shows all five probabilities for each forecast horizon, using a fixed color for each model. Models publishing only categorical forecasts are included in current-season discovery.
