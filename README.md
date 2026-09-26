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

The quick run downloads current data, fits reduced models, and opens an interactive forecast report in your browser. To run the full models:

```bash
./mighte preview
```

To reopen the latest report:

```bash
./mighte review --preview
```

Data snapshots, forecasts, and reports are saved locally under `data/snapshots/`, `runs/`, and `reports/`.

### Windows

After cloning the repository, use PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\mighte preview --quick
```
