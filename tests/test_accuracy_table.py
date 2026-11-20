import json
import re
import shutil
import subprocess

import pytest

from mighte.contract import ED, HOSP, TREND
from mighte.report import TEMPLATE


NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is needed to execute dashboard JavaScript")


def render(scores, *, target=HOSP, horizon="all", sort="Model", direction=1, models=None):
    selection = TEMPLATE[TEMPLATE.index("const selectedModels="):TEMPLATE.index("let selectionGroup=")]
    table = TEMPLATE[TEMPLATE.index("const mean=rows=>"):TEMPLATE.index("['reference','target','location','history'].forEach")]
    harness = r'''
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const D={scores:input.scores,comparison_baseline:'FluSight-baseline',trend_baseline:'FluSight-baseline_cat'};
const TREND='wk flu hosp rate change',PEAK_WEEK='peak week inc flu hosp',PEAK_HEIGHT='peak inc flu hosp';
const inputs=[...new Set(D.scores.map(r=>r.model_id))].map(value=>({value,checked:input.models===null||input.models.includes(value)}));
const controls={target:{value:input.target},scope:{value:'states'},horizon:{value:input.horizon},accuracy:{},location:{value:'01'},'accuracy-card':{},
models:{querySelectorAll:()=>[{value:'Other',checked:true}]},
'accuracy-models':{getAttribute:()=> 'Accuracy models',querySelectorAll:selector=>inputs.filter(box=>
    (!selector.includes(':checked')||box.checked)&&(!selector.includes(':disabled')||!box.disabled))},
'accuracy-model-picker':{querySelector:()=>({setAttribute(){}})}};
const el=id=>controls[id],esc=value=>String(value);
'''
    result = subprocess.run([NODE, "-e", harness + selection + table +
                             "accuracySort={label:input.sort,direction:input.direction};table();console.log(el('accuracy').innerHTML);"],
                            input=json.dumps({"scores": scores, "target": target, "horizon": horizon,
                                              "sort": sort, "direction": direction, "models": models}),
                            text=True, capture_output=True, check=True)
    html = result.stdout
    headers = re.findall(r'data-sort="([^"]+)"', html)
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    rows = [re.findall(r'<td class="([^"]*)">([^<]*)</td>', row) for row in re.findall(r"<tr>(.*?)</tr>", body)]
    return headers, rows, html


def weekly_scores(target=HOSP, scale=1):
    return [{"model_id": model, "target": target, "location": "01", "reference_date": "2026-10-10",
             "target_end_date": f"2026-10-{10 + 7 * h}", "horizon": h, "ae": error * scale, "wis": error * scale,
             "wis_log1p": error * scale / 10, "coverage_50": 1, "coverage_80": 1, "coverage_95": 1}
            for model, errors in [("MIGHTE-Base", [2., 10.]), ("FluSight-baseline", [4.]), ("Other", [4., 1.])]
            for h, error in enumerate(errors)]


@pytest.mark.parametrize("target,scale,mae,filtered_mae", [(HOSP, 1, "6", "2"), (ED, 1e-5, "0.00006", "0.00002")])
def test_weekly_table_order_metrics_and_sorting(target, scale, mae, filtered_mae):
    scores = weekly_scores(target, scale)
    headers, rows, html = render(scores, target=target, sort="MAE Skill", direction=-1)
    assert headers == ["Model", "N", "Matched", "MAE", "Rel MAE", "MAE Skill", "Mean WIS",
                       "Rel WIS", "Rel log-WIS", "WIS Skill", "50% coverage", "80% coverage", "95% coverage"]
    assert rows[0][0][1] == "MIGHTE-Base"
    assert [cell[1] for cell in rows[0][1:7]] == ["2", "1", mae, "0.843", "50.0%", mae]
    assert rows[0][5][0] == rows[0][9][0] == "positive"
    assert 'aria-sort="descending"><button type="button" data-sort="MAE Skill"' in html
    _, filtered, _ = render(scores, target=target, horizon="0")
    base = next(row for row in filtered if row[0][1] == "MIGHTE-Base")
    assert [cell[1] for cell in base[1:7]] == ["1", "1", filtered_mae, "0.5", "50.0%", filtered_mae]


@pytest.mark.parametrize("baseline_errors", [None, 0])
def test_mae_baseline_missing_or_zero_is_not_reported_as_skill(baseline_errors):
    scores = [row for row in weekly_scores() if row["model_id"] == "MIGHTE-Base"]
    if baseline_errors is not None:
        scores += [{**row, "model_id": "FluSight-baseline", "ae": 0, "wis": 0, "wis_log1p": 0}
                   for row in list(scores)]
    _, rows, _ = render(scores)
    base = next(row for row in rows if row[0][1] == "MIGHTE-Base")
    assert base[4] == ("", "—") and base[5] == ("", "—")


def test_accuracy_selection_filters_rows_without_changing_baseline_comparisons():
    scores = weekly_scores()
    _, all_rows, _ = render(scores)
    _, selected_rows, _ = render(scores, models=["MIGHTE-Base"])
    assert selected_rows == [row for row in all_rows if row[0][1] == "MIGHTE-Base"]
    assert selected_rows[0][5] == ("positive", "50.0%")


def test_dashboard_scores_models_that_only_overlap_via_the_baseline():
    scores = [{**weekly_scores()[0], "model_id": model, "horizon": horizon,
               "target_end_date": ["2026-10-10", "2026-10-17"][horizon],
               "ae": error, "wis": error, "wis_log1p": error / 10}
              for model, horizon, error in [("Early", 0, 2), ("Late", 1, 8),
                                            ("FluSight-baseline", 0, 4), ("FluSight-baseline", 1, 4)]]
    headers, rows, _ = render(scores)
    for column in ["Rel MAE", "Rel WIS", "Rel log-WIS"]:
        assert {row[0][1]: row[headers.index(column)][1] for row in rows} == {
            "Early": "0.707", "Late": "1.414", "FluSight-baseline": "1"}
    assert "Comparison baseline" not in TEMPLATE


def test_trend_table_retains_categorical_metrics_and_skill_color():
    scores = [{**weekly_scores()[0], "target": TREND, "model_id": model, "rps": rps,
               "brier": .3, "log_score": .4, "accuracy": .5, "absolute_category_error": .6,
               "zero_observed_probability": 0}
              for model, rps in [("MIGHTE-Base", .2), ("FluSight-baseline_cat", .4)]]
    headers, rows, _ = render(scores, target=TREND)
    assert headers == ["Model", "N", "Matched", "RPS", "RPS skill", "Brier", "Log score",
                       "Category accuracy", "Category error", "Zero-probability outcomes"]
    base = next(row for row in rows if row[0][1] == "MIGHTE-Base")
    assert [cell[1] for cell in base[3:]] == ["0.2", "50.0%", "0.3", "0.4", "50.0%", "0.6", "0.0%"]
    assert base[headers.index("RPS skill")] == ("positive", "50.0%")
