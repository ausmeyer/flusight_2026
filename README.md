# MIGHTE · FluSight

Standalone influenza hospitalization and emergency-department forecasting with MIGHTE-Base, MIGHTE-Linear, and MIGHTE-Nsemble. MIGHTE-Base also forecasts hospitalization trends.

## Requirements

- Python 3.11 or 3.12 and an internet connection to download public surveillance data.
- On macOS, install the LightGBM runtime: `brew install libomp`.

## Run locally

```bash
git clone https://github.com/ausmeyer/flusight_2026.git
cd flusight_2026
./mighte setup
./mighte preview --quick
```

Setup creates a local `.venv` and installs the dependencies in [pyproject.toml](pyproject.toml). If [uv](https://docs.astral.sh/uv/) is installed, it uses [uv.lock](uv.lock). No separate requirements file or neighboring project is needed.

The quick run downloads current data, fits reduced models, and opens an interactive forecast report in your browser. Without `--reference-date`, a preview anchors to the most recent week with released data for both targets and forecasts the following Saturday; a location missing that week is carried forward. `./mighte forecast` uses the round open for submission (Wednesday noon to Thursday 8 AM Eastern) and stops if that week's data are not out yet. To run the full models:

```bash
./mighte preview
```

To reopen the latest report:

```bash
./mighte review --preview
```

To compare an open FluSight hub PR locally, add `--include-pr NUMBER` to `review` (repeat it for multiple PRs). Pending forecasts are labeled and excluded from accuracy scores; the comparison report is saved separately and cannot be published. This option requires an internet connection.

Data snapshots, forecasts, and reports are saved locally under `data/snapshots/`, `runs/` (`previews/` and `official_submissions/`), and `reports/`; `./mighte publish` also commits a reviewed run's forecast files to `forecasts/` on `main` ([docs/PUBLISHING.md](docs/PUBLISHING.md)). Rerunning a preview or forecast for the same reference date replaces the earlier run and its report once the new run completes; a submitted run is kept as the record of what the hub received.

### Windows

After cloning the repository, use PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\mighte preview --quick
```
