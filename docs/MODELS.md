# Frozen model design

The prospective study window is the 2026–27 FluSight season. Models are fixed before this window; prospective errors do not change features, model selection, quantile weights or calibration. The original study implementations are carried into `src/mighte/core.py` and `src/mighte/panel_ar.py`; their source hashes are in `port-provenance.json`. No runtime import or file reference points to the study directory.

## Boosted components

MIGHTE-Base hospitalization, the wastewater-only component and the NSSP-only component preserve the selected **joint-base / no-donor / corrected conditional Gaussian NLL** recipe. The ED component applies the same recipe to percentage-point ED history, with wastewater as the external predictor. It is a separate fit; it is not trained as an auxiliary response in the hospitalization model.

- Own-history lags: 1, 2, 3, 4, 5, 6, 8, 12, 26, 52 weeks; differences, percentage changes, rolling means/spreads, seasonal harmonics, regime indicators, location indicators, and lagged national summaries.
- No individual donor-state features. The legacy `donor_lags` field also enters the shared national lag-set definition, so it is retained in the ported runtime record; it does not enable donor features.
- Wastewater: within-site weekly median log10(PMMoV-normalized influenza A + 1e-5), then an equal-site national median; at least ten sites. Level and 1-, 2-, 4-week lags; lookup ends one week before the observation anchor and allows at most one further week of staleness.
- NSSP hospitalization covariate: national influenza ED percentage level and 1-, 2-, 4-week lags with availability indicators. The ED target's own history replaces this covariate block in its model.
- Hospitalization Gaussian response: log1p(y[t+h]) − log1p(y[t]); direct pooled horizons h=1..4. Integration uses only the observed anchor value.
- ED Gaussian response: logit(p[t+h]) − logit(p[t]), where p is the ED proportion. Predictor histories remain in percentage points. Input proportions are clipped to [0.00005, 0.99995] before taking log-odds; the lower boundary is half the observed 0.01-percentage-point reporting increment. The boundary is fixed in `ed_target_transform`, recorded with each fit, and does not alter raw truth. Quantiles use the inverse logit after adding the anchor log-odds; output quantiles are not floored at the input boundary.
- Stage 1: 250 LightGBM L2 boosting rounds. Stage 2: 120 rounds of conditional Gaussian log-sigma NLL using rolling out-of-fold Stage 1 centers. No model-scale spread cap. Numerical clipping only protects exponential calculations.
- Each fit samples 80% of observed seasons without replacement; 100 fits. The output is the pointwise median of their quantiles, exactly as in the chosen study implementation. Missing bags fail the run rather than silently changing the model.
- Seed: 20260429 + 100000. Season selection also incorporates the anchor timestamp; tree seeds match the study's joint-base profile.

The operational port matches target observations by exact calendar date, and inputs are reindexed to complete weekly grids before constructing lags. It rejects in-sample fallback when out-of-fold spread fitting cannot be completed. These are operational checks; on complete, supported inputs the original numerical fit is unchanged. Regression fixtures compare both feature tables and forecasts directly with the original implementation.

## MIGHTE-Base-Ordinal

A separate pooled five-class LightGBM classifier supplies the hospitalization rate-change probabilities within the MIGHTE-Base submission. It uses the same hospitalization, seasonality, wastewater level/lags 1, 2, 4 and national NSSP level/lags 1, 2, 4 feature table as Base, with identical source availability rules. Each of 100 season-resampled fits minimizes multiclass log loss for 250 boosting rounds. Final probabilities are the arithmetic mean across fits. This component does not use or categorize Base's quantitative forecasts. See [ORDINAL.md](ORDINAL.md) for the label construction and audit.

## MIGHTE-Linear

Direct partially pooled Gaussian autoregression of log1p changes. Shared features include origin log level; differences at 1, 2, 3, 4, 8, 12, 26, 52 weeks; rolling moments; cross-state hospitalization summaries; seasonal harmonics and regime indicators. The mean design includes location intercepts, horizon interactions and ridge-shrunk state deviations for current level and the two most recent differences. A linear log-scale model uses current level, absolute change, rolling spread, national spread, horizon and season indicators. Five coordinate iterations alternate penalized mean and scale fits, followed by a final scale fit. All penalties and iteration limits are preserved in `config/settings.json`. No June–August exclusion or new recalibration is introduced.

## MIGHTE-Nsemble

At every location, horizon and quantile: `(WW-only + NSSP-only + Linear) / 3`. All three components must contain identical keys. This uses the internal unrounded quantiles; integer rounding occurs only on final hospitalization output. It does not include the combined-covariate Base model and never adapts weights from recent WIS.

## Target conventions

The anchor is reference Saturday minus seven days. Horizons 0–3 correspond to anchor leads 1–4. Prospective runs require fresh Wednesday inputs and observed hospitalization anchors for all 53 jurisdictions including US. ED locations without an observed anchor are omitted and named in the report. National forecasts are model outputs, not a sum of state quantiles. Training proxies/interpolated gaps are never scored as observed truth.

ED forecasts use a Gaussian working distribution for changes in log-odds and are exported as proportions bounded to [0,1] by the inverse link. This replaces log1p of percentage points, whose additive offset was one percentage point. The boundary convention handles rounded zeros; it is not an estimated correction for reporting or backfill. A bounded link and a successful numerical fit do not establish interval calibration, which requires observed forecast outcomes. The hub's written plausibility limit of 0.25 is checked and triggers a failure if exceeded; forecasts are not silently truncated to that plausibility threshold. Hospitalization forecasts are integer-valued, nonnegative and checked against the written 30%-of-population bound.

The first center fit estimates an L2 conditional mean on transformed changes. Because the working conditional distribution is Gaussian, that location parameter also defines its median. The concise metadata therefore describes median and spread without claiming joint optimization of both parameters or adding bagging details to the public synopsis.
