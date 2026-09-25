# MIGHTE · FluSight 2026–27

A standalone prospective forecasting pipeline for **MIGHTE-Base**, **MIGHTE-Linear**, and **MIGHTE-Nsemble**. All code, fixed model settings, historical training inputs and hub metadata live in this repository. No neighboring project, R installation, study outputs, or retrospective forecast engine is required.

## Weekly workflow

Install once from the downloaded repository:

```bash
./mighte setup
```

On **Wednesday, after the weekly data release**:

```bash
./mighte forecast
```

This refreshes the complete revised histories, fits the models, validates the three files and opens a local interactive report. Review the medians and prediction intervals by location, target and week, then inspect prospective WIS, MAE, skill and coverage. The report works offline and can be reopened with:

```bash
./mighte review
```

When the forecasts look right:

```bash
./mighte submit
```

Type `submit` to confirm review of the displayed run. The command opens **one PR containing the three model files**, using a branch based on current upstream `main`. It does not merge the PR. It requires [GitHub CLI](https://cli.github.com/) with `gh auth login`. Hub CI is the final authority on acceptance.

**Deadline: Wednesday 11:00 p.m. America/New_York.** The reference date is the following Saturday. The 2026–27 first reference date is October 10, 2026. The pipeline rejects late submissions and past-date production runs. Start sufficiently early: production uses 100 season-resampled fits for each of four boosted components, plus the linear model.

## First-season registration

The two new models require metadata registration; the existing ensemble description is updated for its fixed weights:

```bash
./mighte register
```

This creates a metadata-only PR using [these descriptions](model-metadata/) and [this PR text](docs/REGISTRATION_PR.md). Wait for CDC to merge it before submitting weekly forecasts. Austin Meyer and Mauricio Santillana are listed on all three models, with contact details from the existing public MIGHTE-Nsemble metadata. Base and Nsemble are designated by default; Linear remains submitted and evaluated. The hub normally allows two designated models per team. Historical MIGHTE-Joint metadata is left intact; the registration PR explains the intended active lineup.

## Before the season / quick installation check

```bash
./mighte preview --quick
```

This exercises the complete pipeline using current real inputs with two bags and reduced boosting rounds. The report is prominently marked **PREVIEW** and these forecasts cannot be submitted or counted in prospective accuracy. Use `./mighte preview` for a full 100-bag rehearsal. Reopen its report with `./mighte review --preview`.

## Models and targets

| Submission | Hospital admissions | ED visit proportion | Ordinal hospitalization trends |
|---|---|---|---|
| MIGHTE-Base | Two-stage pooled LightGBM; history, seasonality, national wastewater and NSSP with lags | Separate instance of the same fitting template; ED history, seasonality and wastewater | Disabled integration point for the forthcoming model |
| MIGHTE-Linear | Partially pooled distributional autoregression; hospitalization history and seasonality | — | — |
| MIGHTE-Nsemble | Fixed equal-third quantile average of WW-only LightGBM, NSSP-only LightGBM and Linear | — | — |

All models forecast horizons 0–3 with the hub's 23 quantiles. Hospitalization quantiles are rounded to integers **after** component combination; ED quantiles are exported as proportions. Base's targets share a single CSV. No peak timing, peak height, sample trajectories or retrospective forecast runs are implemented.

The preserved LightGBM center uses L2 loss on log-scale change. Under its Gaussian working distribution this center also defines the conditional median. The spread uses conditional Gaussian negative log-likelihood against rolling out-of-fold center predictions. This is a two-stage fit, not simultaneous optimization of both parameters. See [model details and port verification](docs/MODELS.md).

## ED historical reconstruction

The public national NSSP influenza series starts **October 2022**, so a pre-pandemic NSSP calibration cannot currently be obtained from the checked public sources. The safe initial setting is `ed_history.mode: observed`, which fits the ED model on observed NSSP history. Two explicit reconstruction paths are implemented:

- `post2022_proxy`: fit normalized national ILINet to log1p national ED percentage on available post-2022 overlap; apply that mapping to older location ILINet, then shift it forward **728 days**. This is an unvalidated assumption that the mapping transfers across eras and locations.
- `prepandemic_proxy`: use a supplied repository-local CSV of pre-pandemic national NSSP, with `date,value` where `value` is a proportion. Set `historical_nssp_file` to its relative path. At least 52 paired weeks are required.

Choose the policy explicitly in [config/settings.json](config/settings.json) before operational use. The chosen method and mapping coefficients are recorded in every run and shown in the report. No proxy values enter evaluation truth. See [data handling](docs/DATA.md).

## Other useful commands

```bash
./mighte refresh                       # Refresh all revised data without fitting
./mighte review --refresh              # Update truth and recalculate accuracy
./mighte validate                      # Verify actual CSV bytes and hashes
./mighte resume runs/.../RUN_TIMESTAMP  # Resume an interrupted fit, keeping its original inputs
./mighte check                         # Validate metadata and installed environment
.venv/bin/pytest -q                    # Scientific and operational regression tests
```

Run paths are printed in the terminal and report. Successful bag predictions are checkpointed. Resuming requires the same code, settings, numerical environment and historical input hashes. Start a new `forecast` run to incorporate a later data revision; a resume deliberately retains the original snapshot.

## Installation requirements

- Python 3.11 or 3.12, an internet connection for data refresh, and GitHub CLI only for PRs. The launcher uses a local `.venv`; [uv](https://docs.astral.sh/uv/) is recommended and respects the committed `uv.lock`.
- On macOS, install LightGBM's OpenMP runtime if needed: `brew install libomp`.
- Without uv, `./mighte setup` creates a standard virtual environment with pinned direct dependencies. On Windows use `python -m venv .venv`, `.venv\Scripts\python -m pip install -e ".[dev]"`, then `.venv\Scripts\mighte forecast` from the repository directory.
- The launcher caps native math threads at four. Change `threads` in the settings if needed; this is a runtime control, not a model-selection option.

## Files and reproducibility

- `data/historical/`: small, committed historical training inputs with provenance.
- `data/snapshots/`: immutable full source responses, current rules, transformed covariates and unmodified truth. A failed refresh is never silently replaced by cached data.
- `runs/prospective/`: original weekly forecasts, settings, numerical versions, input/output hashes, checkpoints and submission receipts.
- `runs/previews/`: clearly separated rehearsals; excluded from prospective scoring.
- `reports/`: self-contained HTML reports, per-forecast score CSVs, accuracy CSVs and report data.
- `model-metadata/`, `hub-contract/`: concise model descriptions and the reviewed hub contract. Live rules are fetched again during refresh and submission.

Generated snapshots, runs, reports and credentials are excluded from Git. Back up `runs/` and `data/snapshots/` locally: they are needed to resume and evaluate your actual prospective forecasts. The public hub retains submitted forecasts, but it does not retain these input vintages. A fresh clone can generate the next forecast immediately after setup; it will have an empty private accuracy history until prospective runs are accumulated.

The ordinal integration contract is in [docs/ORDINAL.md](docs/ORDINAL.md). Scoring definitions are in [docs/SCORING.md](docs/SCORING.md). Authoritative requirements: [FluSight hub](https://github.com/cdcepi/FluSight-forecast-hub), [submission rules](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/README.md), and [metadata rules](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/README.md).
