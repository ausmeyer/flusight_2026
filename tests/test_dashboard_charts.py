"""The dashboard's packed data and the chart geometry that runs in the page."""
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mighte import evaluate, report
from mighte.contract import HOSP
from mighte.util import digest, write_json

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is needed to execute dashboard JavaScript")
LEVELS = [.01, .025, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5,
          .55, .6, .65, .7, .75, .8, .85, .9, .95, .975, .99]


def js(expression, data=None):
    """Evaluate an expression after the page's pure chart helpers; `input` holds `data`."""
    helpers = report.TEMPLATE[report.TEMPLATE.index("// Ridge and interval geometry"):
                              report.TEMPLATE.index("function probabilityPlot(")]
    program = ("const input=JSON.parse(require('fs').readFileSync(0,'utf8'));\n" + helpers
               + f"\nprocess.stdout.write(JSON.stringify({expression}));")
    result = subprocess.run([NODE, "-e", program], input=json.dumps(data), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def unpack(page: str):
    """Run the page's own loader in Node and return the data it unpacks."""
    start = page.index("const packed=")
    end = page.index("\n", page.index("const D=", start))
    program = "(async()=>{" + page[start:end] + "\nprocess.stdout.write(JSON.stringify(D))})()"
    return json.loads(subprocess.run([NODE, "-e", program], text=True, capture_output=True, check=True).stdout)


def test_page_unpacks_the_reviewed_data_with_every_quantile_level(root, tmp_path, monkeypatch, forecast, peak_forecasts):
    run = tmp_path / "runs/official_submissions/2026-10-10/fixture"
    filename = "model-output/MIGHTE-Base/2026-10-10-MIGHTE-Base.csv"
    (run / filename).parent.mkdir(parents=True)
    pd.concat([forecast, peak_forecasts]).to_csv(run / filename, index=False)
    manifest = {"run_id": "official_submissions/2026-10-10/fixture", "reference_date": "2026-10-10",
                "preview": False, "quick": False, "settings": {"season": "2026-2027", "runtime": {"num_bags": 100}},
                "notices": ["Using older data for the fixture forecast."],
                "output_hashes": {filename: digest(run / filename)}}
    write_json(run / "manifest.json", manifest)
    write_json(run / "data-audit.json", {})
    snapshot = tmp_path / "data/snapshots/fixture"
    shutil.copytree(root / "hub-contract", snapshot / "contract")
    pd.DataFrame({"target": [HOSP], "location": ["01"], "date": ["2026-10-10"], "value": [20]}).to_csv(
        snapshot / "truth.csv", index=False)
    monkeypatch.setattr(report, "latest_snapshot", lambda root: snapshot)
    monkeypatch.setattr(report, "verify_snapshot", lambda path: None)
    monkeypatch.setattr(report, "verify_run", lambda *a: manifest)
    monkeypatch.setattr(report, "preview_runs", lambda root: {})
    monkeypatch.setattr(report, "discover_benchmarks", lambda *a, **k: (None, {}))
    monkeypatch.setattr(evaluate, "load_archive", lambda root: pd.DataFrame())
    monkeypatch.setattr(evaluate, "fetch_benchmarks",
                        lambda *a, **k: (forecast.assign(model_id="FluSight-ensemble"), []))
    path = report.build_report(tmp_path, run, online=False)
    payload = json.loads((path.parent / "report-data.json").read_text())
    page = path.read_text()
    assert unpack(page) == payload
    assert '"model_id"' not in page  # The data are not also embedded uncompressed.
    assert 'id="notices"' not in page
    assert payload["manifest"]["notices"] == manifest["notices"]
    assert payload["quantile_levels"] == LEVELS
    weekly = {r["model_id"]: r["q"] for r in payload["forecasts"] if r["horizon"] == 0}
    assert weekly == {"MIGHTE-Base": list(range(10, 33)), "FluSight-ensemble": list(range(10, 33))}
    assert [r["q"] for r in payload["peak_height_forecasts"]] == [list(range(1000, 3300, 100))]
    assert payload["default_models"] == ["FluSight-ensemble", "MIGHTE-Base"]


def test_dashboard_opens_on_the_ensemble_only_when_it_covers_the_opening_week():
    rows = [{"model_id": model, "target": HOSP, "reference_date": week, "location": "US"}
            for model, week in [("FluSight-ensemble", "2026-11-07"), ("MIGHTE-Base", "2026-11-07"),
                                ("MIGHTE-Base", "2026-11-14")]]

    def opening(week):
        return js("openingModel(input.preferred,input.rows,input.view)",
                  {"preferred": ["FluSight-ensemble", "MIGHTE-Base"], "rows": rows,
                   "view": {"target": HOSP, "reference": week, "location": "US"}})

    assert opening("2026-11-07") == "FluSight-ensemble"
    assert opening("2026-11-14") == "MIGHTE-Base"  # Built before the hub posts the ensemble.


def test_central_intervals_pair_symmetric_levels_up_to_95_percent():
    assert js("centralIntervals(input)", LEVELS) == [
        [1, 21, 95], [2, 20, 90], [3, 19, 80], [4, 18, 70], [5, 17, 60],
        [6, 16, 50], [7, 15, 40], [8, 14, 30], [9, 13, 20], [10, 12, 10]]


def test_narrower_intervals_are_shaded_darker():
    shades = js("centralIntervals(input).map((_,k)=>mix('#ae4861',bandOpacity(k+1)))", LEVELS)
    lightness = [sum(int(shade[i:i + 2], 16) for i in (1, 3, 5)) for shade in shades]
    assert lightness == sorted(lightness, reverse=True) and len(set(lightness)) == len(shades)
    assert js("[mix('#ae4861',0),mix('#ae4861',1)]") == ["#ffffff", "#ae4861"]


def test_quantile_density_recovers_a_normal_forecast():
    grid = np.linspace(20, 180, 1601)
    density = np.array(js("quantileDensity(input.values,input.levels,input.grid,2)",
                          {"values": (100 + 15 * norm.ppf(LEVELS)).tolist(), "levels": LEVELS,
                           "grid": grid.tolist()}))
    assert np.trapz(density, grid) == pytest.approx(.98, abs=.005)  # Mass between the 1% and 99% quantiles.
    assert grid[density.argmax()] == pytest.approx(100, abs=1)
    assert density[800] == pytest.approx(norm.pdf(0) / 15, rel=.05)


def test_tied_quantiles_keep_a_finite_density_and_their_mass():
    values = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 12, 14]
    grid = np.linspace(-5, 20, 2501)
    density = np.array(js("quantileDensity(input.values,input.levels,input.grid,.5)",
                          {"values": values, "levels": LEVELS, "grid": grid.tolist()}))
    assert np.isfinite(density).all()
    assert np.trapz(density, grid) == pytest.approx(.98, abs=.005)


@pytest.mark.parametrize("xs,ys,p,expected", [
    ([0, 10], [1, 1], .3, 3),              # Flat: proportional to distance.
    ([0, 1, 2, 3], [1, 1, 1, 1], .5, 1.5),  # Crosses segment boundaries.
    ([0, 1, 2], [0, 1, 0], .5, 1),         # Symmetric triangle: the apex.
    ([0, 1, 2], [0, 1, 0], .125, .5),      # Area x²/2 of a total 1.
    ([0, 1], [0, 2], .25, .5),             # Rising ramp: area x² of a total 1.
])
def test_area_quantile_inverts_the_area_under_a_piecewise_linear_curve(xs, ys, p, expected):
    assert js("areaQuantile(input.xs,input.ys,input.p)", {"xs": xs, "ys": ys, "p": p}) == pytest.approx(expected)


def test_band_outline_follows_the_curve_and_closes_on_the_baseline():
    assert js("areaBetween([0,1,2,3],[0,2,2,0],.5,2.5)") == [[.5, .5, 1, 2, 2.5, 2.5], [0, 1, 2, 2, 1, 0]]


def test_ridges_are_ordered_by_median_on_one_shared_scale():
    curves = [{"model": "B", "median": 5, "ys": [0, 1, 0]}, {"model": "A", "median": 2, "ys": [0, 4, 0]},
              {"model": "C", "median": 9, "ys": [0, 2, 0]}]
    stacked = js("stackRidges(input).map(c=>[c.model,c.base,c.scale])", curves)
    assert [row[:2] for row in stacked] == [["A", 2], ["B", 1], ["C", 0]]  # Lowest median on top.
    assert [row[2] for row in stacked] == pytest.approx([.35, .35, .35])  # The tallest ridge spans 1.4 rows.
