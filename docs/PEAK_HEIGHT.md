# MIGHTE-Base peak height

This prospective component directly forecasts the largest weekly hospitalization count in the season. It shares Base's reconstructed, historically adjusted predictor history and wastewater/NSSP availability rules.

## Design

- Labels are maxima of Base's reconstructed and historically adjusted hospitalization training series in completed October–May seasons, restricted to locations with every seasonal week present. This includes the reconstructed historical seasons. Labels and predictors share the same series and scale; raw pandemic observations are not paired with shifted pre-pandemic predictors. Historical windows begin on the first Saturday on or after October 4 and end on the last Saturday in May; the current window is read from the hub's peak-week contract.
- One example per location and issue week, using the preceding Saturday as the feature anchor. Only completed seasons enter training. A season's label becomes available at its end, irrespective of when its maximum occurred.
- Predictors reuse Base's feature construction, including wastewater and NSSP lags, and add current log hospitalization level, population, season progress, the maximum so far in that same training series, and time since that maximum. No weekly forecast-horizon features enter this model. The final seasonal label never enters the predictors.
- Response: `(peak hospitalization count + 1) / population * 100000`, with population fixed to the run's hub location table. Each of 100 fits samples 80% of completed seasons without replacement, with a minimum of three seasons per fit. Shallow LightGBM trees (7 leaves, depth 3, minimum 50 rows per leaf, L2 penalty 5) estimate the center. A second shallow LightGBM fit estimates Gaussian log-spread using Base's conditional negative-log-likelihood objective. Both fits operate on the population-rate response; the peak response is not log-transformed.
- Spread is learned only from forward predictions of entire withheld seasons, fitted on earlier seasons in the bag. No location or week from a withheld season enters its center fit. Insufficient forward predictions fail the fit; there is no in-sample fallback. These internal predictions estimate uncertainty, not a retrospective forecast product.
- Quantiles are converted back to counts using `rate * population / 100000 - 1`, floored at the current seasonal maximum in Base's history, aggregated by pointwise median across bags, and rounded to integers. Current-season counts are unadjusted observed hospitalizations. This treats the latest reported seasonal maximum as fixed within a run; downward revisions are incorporated in subsequent runs, not modeled explicitly.
- Output: `peak inc flu hosp`, 23 quantiles, with blank `horizon` and `target_end_date`, appended only to MIGHTE-Base.

Each fit saves its unique location-season labels in `training-labels.csv` alongside the checkpoints. Features and labels use the run's frozen current data vintage. This is not an exact historical-vintage backtest: historical reporting revisions and the existing reconstruction's fitted adjustments are not undone. Reconstructed seasonal peaks are training proxies, not observed outcomes for scoring. Successful fitting and valid output do not establish seasonal forecast calibration.

## Local review

Peak height is enabled by default. Normal `./mighte preview` and `./mighte forecast` append it to the MIGHTE-Base file alongside hospitalization, ED and hospitalization-trend forecasts; no additional command is required for submission. To inspect the new component against an existing frozen run without refitting its other components:

```sh
./mighte preview --reuse-run runs/official_submissions/REFERENCE_DATE/RUN_ID
```

This creates a separate, non-submittable preview, reuses that run's data snapshot and other forecasts, and fits only peak height. Select **Peak height** in the resulting review. The source run and its submitted files remain intact.
