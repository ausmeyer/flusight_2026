# MIGHTE-Base-Ordinal

The prospective pipeline fits the study's multiclass MIGHTE-Base-Ordinal component each week and appends its probabilities to the MIGHTE-Base CSV. It is a separate fitted component within the Base submission, with no fourth hub model or retrospective forecast runner. Its entry point is `mighte.ordinal:predict` in `config/settings.json`.

## Construction

The shared Base feature builder supplies hospitalization lags, rolling and seasonal features, location and horizon features, national WastewaterSCAN influenza A level and lags 1, 2, 4, and national NSSP influenza ED level and lags 1, 2, 4. It uses the same frozen snapshot, availability indicators, source delays and staleness limits as Base. No quantile forecasts enter the classifier.

Training targets end on or before the anchor Saturday (`reference_date - 7 days`); predictor histories also stop at that anchor. Exact calendar joins construct the future training outcomes. A bag samples 80% of seasons without replacement. The 100 fits use the Base tree settings, 250 boosting rounds and multiclass log loss. Their five-class probability vectors are averaged arithmetically. The prospective seed is Base's seed, 20360429, with the same anchor timestamp and bag offsets as the study kernel; no retrospective origin-index offset is needed.

The term “ordinal” identifies the ordered outcome. This selected implementation uses nominal multiclass loss, without a proportional-odds link or an explicit ordering penalty. Its probabilities form a valid ordered distribution but are not required to agree with the independently fitted hospitalization quantiles. No post-fit calibration, recent-performance weighting or tuning is added.

## Labels

The [hub's written rules and examples](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/README.md#weekly-flu-hospitalization-rate-change) define change relative to the Saturday **before** the reference date. Horizon 0–3 target dates equal the reference Saturday plus 0–3 weeks. Population comes from the frozen hub locations file.

| Horizon | Stable rate boundary per 100,000 | Large-change boundary per 100,000 |
| --- | ---: | ---: |
| 0 | 0.3 | 1.7 |
| 1 | 0.5 | 3.0 |
| 2 | 0.7 | 4.0 |
| 3 | 1.0 | 5.0 |

Change is stable when its absolute rate is strictly below the stable boundary **or** its absolute count is less than ten admissions. Large changes meet or exceed the large boundary after applying that count rule. The remaining nonstable changes are increases or decreases. Categories are `large_decrease`, `decrease`, `stable`, `increase`, `large_increase`.

Historical training counts retain the study's reconstructed ILINet/FluSurv-NET segment and historical scale adjustment. Categories made from those training counts are consequently proxy labels in that period, not official observed historical trend truth. Modern anchors use observed admissions. Evaluation derives labels only from raw observed hospitalization truth at both required weeks, with the population frozen for that forecast. Missing observations are not reconstructed for scoring.

## Output and checks

Every location/horizon has all five `pmf` rows for `wk flu hosp rate change`, with finite probabilities in [0,1] summing to one. The pipeline validates complete hospitalization-location coverage and the combined eight-column Base file. Linear and Nsemble remain hospitalization-only. Per-bag checkpoints and `fit.json` record features, cutoffs, class counts, populations, thresholds and fit settings; normal run hashes prevent resuming against changed code, settings or inputs.

The dashboard's Hospitalization trend target shows probabilities and, once observed outcomes exist, RPS, Brier score, log score, classification accuracy, category error and matched-baseline RPS skill. Previews never enter prospective accuracy. These implementation checks do not establish prospective calibration or improvement over the quantitative Base model.
