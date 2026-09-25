from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from mighte.contract import Contract, ED, HOSP, TREND, CATEGORIES, check_window


def test_hub_contract_and_metadata(root, forecast):
    contract = Contract(root / "hub-contract")
    assert contract.validate(forecast, "MIGHTE-Base", "2026-10-10")["rows"] == 92
    assert len(contract.validate_metadata(root / "model-metadata")) == 3


@pytest.mark.parametrize("corruption", ["fractional", "crossing", "missing_quantile", "duplicate", "date", "fips", "infinite", "column", "peak"])
def test_reject_invalid_files(root, forecast, corruption):
    if corruption == "fractional":
        forecast["value"] = forecast.value + .5
    elif corruption == "crossing":
        forecast.loc[0, "value"] = 1000
    elif corruption == "missing_quantile":
        forecast = forecast.iloc[1:]
    elif corruption == "duplicate":
        forecast = pd.concat([forecast, forecast.iloc[:1]])
    elif corruption == "date":
        forecast.loc[0, "target_end_date"] = "2026-10-17"
    elif corruption == "fips":
        forecast["location"] = "1"
    elif corruption == "infinite":
        forecast["value"] = forecast.value.astype(float)
        forecast.loc[0, "value"] = np.inf
    elif corruption == "column":
        forecast["model_id"] = "MIGHTE-Base"
    elif corruption == "peak":
        forecast["target"] = "peak inc flu hosp"
    with pytest.raises(ValueError):
        Contract(root / "hub-contract").validate(forecast, "MIGHTE-Base", "2026-10-10")


def test_ed_units_and_combined_file(root, forecast):
    ed = forecast.assign(target=ED, value=forecast.value / 1000)
    combined = pd.concat([forecast, ed])
    Contract(root / "hub-contract").validate(combined, "MIGHTE-Base", "2026-10-10")
    with pytest.raises(ValueError):
        Contract(root / "hub-contract").validate(combined, "MIGHTE-Linear", "2026-10-10")
    with pytest.raises(ValueError, match="plausibility"):
        Contract(root / "hub-contract").validate(pd.concat([forecast, ed.assign(value=.26)]), "MIGHTE-Base", "2026-10-10")


def test_ordinal_pmf(root, forecast):
    rows = [dict(forecast.iloc[0], target=TREND, output_type="pmf", output_type_id=c, value=.2) for c in CATEGORIES]
    pmf = pd.DataFrame(rows)
    Contract(root / "hub-contract").validate(pd.concat([forecast, pmf]), "MIGHTE-Base", "2026-10-10")
    pmf.loc[0, "value"] = .3
    with pytest.raises(ValueError, match="sum to one"):
        Contract(root / "hub-contract").validate(pd.concat([forecast, pmf]), "MIGHTE-Base", "2026-10-10")


@pytest.mark.parametrize("reference,deadline", [("2026-10-10", "2026-10-07T23:00:00-04:00"),
                                                ("2026-11-07", "2026-11-04T23:00:00-05:00")])
def test_eastern_deadline_with_dst(reference, deadline):
    check_window(reference, datetime.fromisoformat(deadline))
    with pytest.raises(ValueError):
        check_window(reference, datetime.fromisoformat(deadline.replace("23:00:00", "23:00:01")))
