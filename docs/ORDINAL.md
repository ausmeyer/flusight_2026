# Adding the ordinal model

The ordinal model remains disabled until its testing is finished. The runner never derives trend probabilities from the hospitalization quantiles as a substitute.

1. Implement `predict(context)` in `src/mighte/ordinal.py` or another module inside `mighte`.
2. Set `ordinal_plugin` to `mighte.ordinal:predict` in `config/settings.json`.
3. Update MIGHTE-Base's metadata to describe the added categorical method, and submit that metadata change.
4. Run a preview and tests before the next prospective forecast.

The context dictionary contains `reference_date`, `snapshot_path`, `hospitalization_history`, `base_hospitalization_quantiles`, and `settings`. The history ends at the anchor (`reference_date - 7 days`). Reuse the frozen inputs rather than fetching another vintage inside the plugin. All model code and calibration artifacts must live inside this repository.

Return a nonempty pandas DataFrame with precisely the eight hub columns. Every row has target `wk flu hosp rate change` and output type `pmf`. Allowed horizons here are 0–3. Every forecast unit needs all five categories, in the hub spelling:

`large_decrease`, `decrease`, `stable`, `increase`, `large_increase`.

Values must be nonnegative, at most one, and sum to one for each location/horizon/reference date. Target end date still equals reference date plus horizon weeks. The hub's 2026–27 written definition compares the target hospitalization rate with the week **before** the reference date; note that this detail must be checked when integrating the categorical model. The hub's task metadata description and README have differed in wording, so use the current written examples and confirm any unresolved interpretation with CDC.

The output is appended only to the MIGHTE-Base CSV and validated with the quantile rows. Base's hospitalization and ED models, Linear and Nsemble remain unchanged. Adding ordinal-specific accuracy metrics and a probability plot should accompany the final validated ordinal model; the current accuracy table evaluates quantile forecasts only.
