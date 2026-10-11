# Conditional peak timing

The weekly MIGHTE-Base submission includes the Conditional peak timing model when `peak_timing` is enabled (the default). It uses Base's reconstructed and historically adjusted hospitalization series, seasonal predictors, and wastewater/NSSP levels and lags.

## Construction

- A label is the date of the unique maximum in a complete October–May location-season. Labels become available only after the season ends. Tied maxima are excluded from timing training, consistent with the hub not scoring tied peak weeks. Peak-height training retains those seasons. Labels and predictors use the same prepared hospitalization series.
- Each issue-week example uses features ending at the preceding Saturday (the observed anchor). Its final peak date, lead time, peak fraction and tie count are labels/audit fields and never predictors.
- A binary LightGBM classifier estimates whether the final peak is on or before the observed anchor. On training examples with a later final peak, a median LightGBM fit predicts the logarithm of positive lead time in weeks. A second fit learns Gaussian spread on that log scale from forward, whole-season median holdouts. All labels used by a holdout fit precede its validation season. These internal fits train uncertainty; they do not add retrospective forecast products.
- Each of 100 fits samples 80% of complete seasons, keeping all locations/origins together. The classifier and median use 250 rounds; spread uses 120. All fits use 7 leaves, depth 3, minimum leaf size 50 and L2 penalty 5. Each location-season has equal total weight within each stage.
- Future peak probabilities integrate the resulting lognormal lead-time distribution over weekly bins and normalize to the remaining hub dates. Past probability is split equally over the reported seasonal maximum's week(s). When no seasonal observations exist yet, all mass goes to future weeks. After the last eligible week is observed, all mass goes to the reported maximum. Missing past weeks receive no separate probability; reported observations are treated as fixed within a run and revisions update the next run.
- Final probabilities are the arithmetic mean of the 100 complete past/future mixtures, with no fitted recalibration or performance-based weighting. Output uses `peak week inc flu hosp`, `pmf`, all week-ending dates in the current hub contract, and blank `horizon`/`target_end_date`. Probabilities must be finite, nonnegative and sum to one. These rows are included only in the MIGHTE-Base file.

Each run freezes its inputs and records training-label hashes, sampled seasons, forward folds, gate probabilities and log-lead median/spread parameters. Checkpoints are versioned by model design.

## Local review

Normal `./mighte forecast` and `./mighte preview` include peak timing. To regenerate only timing while preserving a frozen run's other forecast values:

```sh
./mighte preview --reuse-run runs/previews/REFERENCE_DATE/RUN_ID --peak-timing
```

This creates a separate preview. A frozen official run can also be used. In **Peak timing**, select MIGHTE-Base to compare it with the default FluSight-ensemble. Nothing is submitted or published by this command.
