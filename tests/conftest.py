from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mighte.contract import COLUMNS, HOSP, QUANTILES


@pytest.fixture
def root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def forecast():
    return pd.DataFrame([{"reference_date": "2026-10-10", "target": HOSP, "horizon": h,
                          "target_end_date": (pd.Timestamp("2026-10-10") + pd.Timedelta(weeks=h)).date().isoformat(),
                          "location": "01", "output_type": "quantile", "output_type_id": float(q),
                          "value": int(i + 10 + h)}
                         for h in range(4) for i, q in enumerate(QUANTILES)], columns=COLUMNS)


@pytest.fixture
def history():
    dates = pd.date_range("2016-01-02", "2026-10-03", freq="W-SAT")
    weeks = np.arange(len(dates))
    return pd.concat([pd.DataFrame({"location_name": name, "date": dates,
                                    "total_hosp": 50 + i * 20 + 15 * np.sin(weeks / 8) + .05 * weeks})
                      for i, name in enumerate(["Alabama", "Alaska", "US"])], ignore_index=True)
