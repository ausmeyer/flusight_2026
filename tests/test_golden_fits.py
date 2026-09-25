"""Fixtures were generated independently with the original study implementation."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mighte.contract import HOSP, UNIT
from mighte.data import NSSP, WW
from mighte.models import distributional, linear


def test_ported_models_match_study_golden_values(root, tmp_path, history):
    settings = json.loads((root / "tests/fixtures/settings.json").read_text())
    dates = pd.date_range("2016-01-02", "2026-10-03", freq="W-SAT")
    weeks = np.arange(len(dates))
    wastewater = pd.DataFrame({"date": dates, WW: np.round(np.sin(weeks / 5) - 3, 6),
                                "available_date": dates + pd.Timedelta(weeks=2)})
    for lag in [1, 2, 4]:
        wastewater[WW + "_lag" + str(lag)] = wastewater[WW].shift(lag)
    wastewater.to_csv(tmp_path / "wastewater.csv", index=False)
    pd.DataFrame({"date": dates, NSSP: np.round(2 + np.sin(weeks / 9), 6),
                   "available_date": dates + pd.Timedelta(weeks=1)}).to_csv(tmp_path / "nssp.csv", index=False)
    locations = {"Alabama": "01", "Alaska": "02", "US": "US"}
    with threadpool_limits(limits=2):
        boosted = distributional(history, tmp_path, "2026-10-10", settings, ("ww", "nssp"), HOSP,
                                  "fixture", tmp_path / "checkpoints", locations)
        ar = linear(history, "2026-10-10", settings, locations)
    for actual, name in [(boosted, "distributional"), (ar, "linear")]:
        expected = pd.read_csv(root / f"tests/fixtures/{name}-golden.csv", dtype={"location": str})
        keys = UNIT + ["output_type", "output_type_id"]
        pd.testing.assert_frame_equal(actual[keys].reset_index(drop=True), expected[keys], check_dtype=False)
        np.testing.assert_allclose(actual.value, expected.value, rtol=1e-7, atol=1e-8)
    # Reusing bag checkpoints must reproduce the same outputs without refitting.
    repeat = distributional(history, tmp_path, "2026-10-10", settings, ("ww", "nssp"), HOSP,
                             "fixture", tmp_path / "checkpoints", locations)
    pd.testing.assert_frame_equal(boosted, repeat)
