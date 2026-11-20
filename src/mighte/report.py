"""A portable, offline HTML review of forecasts, provenance, and accuracy."""
from __future__ import annotations

import base64
import colorsys
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

from .contract import CATEGORIES, ED, HOSP, PEAK_HEIGHT, PEAK_WEEK, QUANTILES, TREND, UNIT, Contract, read_forecast
from .data import latest_snapshot, verify_snapshot
from .evaluate import FLUSIGHT_BASELINE, TREND_BASELINE, discover_benchmarks, evaluate, fetch_pr_forecasts, preview_runs, summarize
from .pipeline import verify_run
from .util import digest, utc_now, write_json


def records(frame: pd.DataFrame) -> list:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


LEVELS = [float(level) for level in QUANTILES]


def quantile_records(frame: pd.DataFrame, keys: list, *, complete=False) -> list:
    """One record per forecast, holding its values at every hub quantile level (LEVELS order) in "q"."""
    wide = frame.assign(output_type_id=pd.to_numeric(frame.output_type_id)).pivot(
        index=keys, columns="output_type_id", values="value").reindex(columns=LEVELS)
    wide = wide.dropna(how="any" if complete else "all").reset_index()
    return [{**row, "q": values} for row, values in
            zip(records(wide[keys]), json.loads(wide[LEVELS].to_json(orient="values")))]


def model_color(name: str) -> str:
    fixed = {"MIGHTE-Base": "#007f89", "MIGHTE-Linear": "#db8744", "MIGHTE-Nsemble": "#7555ac",
             "Local-persistence": "#7c8a96", "FluSight-baseline": "#a5a08d",
             "FluSight-ensemble": "#ae4861", "UMass-flusion": "#4c73b7", "Google_SAI-FluEns": "#428449"}
    if name in fixed:
        return fixed[name]
    # Model identity alone determines color; additions, order and filters cannot change it.
    seed = hashlib.sha256(name.encode()).digest()
    rgb = colorsys.hls_to_rgb(int.from_bytes(seed[:4], "big") / 2**32,
                             .40 + seed[4] / 255 * .12, .58 + seed[5] / 255 * .18)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)


def build_report(root: Path, run: Path, *, online=True, include_prs=()) -> Path:
    if include_prs and not online:
        raise ValueError("--include-pr requires an online review")
    manifest = verify_run(root, run)
    snapshot = latest_snapshot(root)
    verify_snapshot(snapshot)
    # A preview report also shows the other previews on disk, one per reference date, for browsing.
    previews = {day: path for day, path in preview_runs(root).items()
                if manifest["preview"] and day != manifest["reference_date"]}
    season_references = Contract(snapshot / "contract").by_target[HOSP]["task_ids"]["reference_date"]["optional"]
    catalog, catalog_status = discover_benchmarks(root, season_references,
                                                 season=manifest["settings"]["season"], online=online)
    comparison_models = sorted({f["model"] for f in catalog["files"]} if catalog else [])
    scores, benchmark_status, archive = evaluate(root, snapshot, online=online, catalog=catalog,
                                                comparison_references=[manifest["reference_date"], *previews])
    current = pd.concat([read_forecast(run / filename).assign(model_id=Path(filename).parent.name)
                         for filename in manifest["output_hashes"]], ignore_index=True)
    official = set()
    if not archive.empty:
        archive = archive[~(archive.reference_date.eq(manifest["reference_date"])
                            & archive.model_id.isin(current.model_id.unique()))]
        official = set(archive.reference_date[archive.model_id.isin(current.model_id.unique())])
    # A week with an official forecast shows that forecast instead of a preview.
    shown = [read_forecast(path / filename).assign(model_id=Path(filename).parent.name)
             for day, path in previews.items() if day not in official
             for filename in verify_run(root, path)["output_hashes"]]
    forecasts = pd.concat(([archive] if not archive.empty else []) + shown + [current], ignore_index=True)
    included_prs = []
    if include_prs:
        pending, included_prs = fetch_pr_forecasts(root, include_prs, set(forecasts.reference_date))
        # Scores and baselines were already computed using only the official archive.
        forecasts = pd.concat([forecasts, pending], ignore_index=True)
    chart = quantile_records(forecasts[forecasts.target.isin([HOSP, ED]) & forecasts.output_type.eq("quantile")],
                             ["model_id", *UNIT])
    categorical = forecasts[forecasts.target.eq(TREND) & forecasts.output_type.eq("pmf")]
    category_chart = categorical.pivot(index=["model_id", *UNIT], columns="output_type_id", values="value").reindex(
        columns=CATEGORIES).dropna().reset_index() if not categorical.empty else pd.DataFrame()
    peak_keys = ["model_id", "reference_date", "target", "location"]
    height_chart = quantile_records(forecasts[forecasts.target.eq(PEAK_HEIGHT) & forecasts.output_type.eq("quantile")],
                                    peak_keys, complete=True)
    timing_chart = forecasts[forecasts.target.eq(PEAK_WEEK) & forecasts.output_type.eq("pmf")][
        peak_keys + ["output_type_id", "value"]].rename(columns={"output_type_id": "peak_week", "value": "probability"})
    truth = pd.read_csv(snapshot / "truth.csv", dtype={"location": str})
    locations = pd.read_csv(snapshot / "contract/locations.csv", dtype={"location": str})
    payload = {"manifest": manifest, "generated_at": utc_now(), "truth_snapshot": snapshot.name,
               "data_audit": json.loads((run / "data-audit.json").read_text()),
               "forecasts": chart, "truth": records(truth[["date", "target", "location", "value"]]),
               "quantile_levels": LEVELS, "categorical_forecasts": records(category_chart), "categories": CATEGORIES,
               "peak_height_forecasts": height_chart, "peak_week_forecasts": records(timing_chart),
               "scores": records(scores.drop(columns=[c for c in scores.columns if isinstance(c, float) or c in CATEGORIES], errors="ignore")),
               "locations": records(locations[["location", "location_name"]]),
               "benchmarks": [catalog_status, *benchmark_status],
               "comparison_models": comparison_models, "comparison_baseline": FLUSIGHT_BASELINE,
               "trend_baseline": TREND_BASELINE,
               # The weekly views open on the first of these with a forecast in the opening week.
               "default_models": ["FluSight-ensemble", "MIGHTE-Base"],
               "model_colors": {m: model_color(m) for m in set(forecasts.model_id) | set(comparison_models)}}
    output = root / "reports" / manifest["run_id"]
    if included_prs:
        payload["included_prs"] = included_prs
        payload["model_colors"].update({f["model_id"]: model_color(f["model"])
                                        for pr in included_prs for f in pr["files"]})
        output = root / "reports/local-comparisons" / manifest["run_id"] / (
            "pr-" + "-".join(str(pr["number"]) for pr in included_prs))
    output.mkdir(parents=True, exist_ok=True)
    if not scores.empty:
        scores.to_csv(output / "scores.csv", index=False)
        summarize(scores).to_csv(output / "accuracy.csv", index=False)
    write_json(output / "report-data.json", payload)
    # The page carries its data gzip-compressed, far below GitHub's 100 MB file limit.
    packed = gzip.compress(json.dumps(payload, allow_nan=False).encode(), compresslevel=9, mtime=0)
    html = TEMPLATE.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", base64.b64encode(packed).decode())
    path = output / "index.html"
    path.write_text(html)
    write_json(output / "review.json", {"run_id": manifest["run_id"],
               "manifest_sha256": digest(run / "manifest.json"), "report_sha256": digest(path),
               "data_sha256": digest(output / "report-data.json")})
    write_json(root / "reports/latest.json", {"file": str(path.relative_to(root))})
    return path


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIGHTE · Weekly review</title><script>__PLOTLY__</script>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#1b3040;background:#f2f5f6}
body{margin:0}main{max-width:1450px;margin:auto;padding:28px 32px 50px}h1{font-size:30px;letter-spacing:-1px;margin:4px 0}
.eyebrow{color:#367b80;font-size:12px;font-weight:750;letter-spacing:2px}
.head{display:flex;justify-content:space-between;gap:18px;align-items:center}
.card{margin-top:22px;border:1px solid #dce4e8;border-radius:13px;background:white;padding:22px;box-shadow:0 2px 3px #19333e04}
h2{font-size:18px;margin:0 0 15px}.controls{display:flex;gap:16px;align-items:end;flex-wrap:wrap}label{font-size:12px;font-weight:650;display:flex;flex-direction:column;gap:7px}
select{font:inherit;min-width:160px;border:1px solid #c5d2d9;border-radius:6px;padding:8px 30px 8px 12px;appearance:none;color:#213f50;
background:white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%23213f50' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat right 10px center}
.model-control{font-size:12px;font-weight:650;display:flex;flex-direction:column;gap:7px}.model-picker{position:relative;margin:0;min-width:190px}
.model-picker summary{border:1px solid #c5d2d9;border-radius:6px;padding:8px 12px;background:white;color:#213f50;font-size:12px}
.model-panel{position:absolute;z-index:20;right:0;top:calc(100% + 7px);width:min(320px,calc(100vw - 48px));box-sizing:border-box;background:white;border:1px solid #c5d2d9;border-radius:8px;padding:12px;box-shadow:0 8px 24px #19333e24}
#accuracy-model-picker .model-panel{left:0;right:auto}
.model-search{box-sizing:border-box;width:100%;padding:8px;border:1px solid #c5d2d9;border-radius:5px;font:inherit}.model-actions{display:flex;gap:8px;margin:10px 0}
.model-actions button{font:inherit;color:#216b75;border:1px solid #c5d2d9;border-radius:5px;background:white;padding:5px 9px;cursor:pointer}
.checks{max-height:270px;overflow-y:auto}.checks label{display:flex;flex-direction:row;align-items:center;font-weight:500;padding:7px 0;gap:8px}.checks label[hidden]{display:none}.checks label:has(input:disabled){opacity:.45}.checks input{accent-color:#167f87;margin:0}.swatch{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.timeline{display:flex;gap:24px;align-items:end;margin-top:18px}.week-navigation{flex:1;min-width:0}.week-navigation>label{margin-bottom:7px}.week-slider{display:flex;align-items:center;gap:10px}.week-slider input{flex:1;min-width:0;accent-color:#167f87;cursor:ew-resize}.week-slider button{font:inherit;border:1px solid #c5d2d9;border-radius:6px;background:white;color:#213f50;width:34px;height:34px;cursor:pointer}.week-slider button:disabled{opacity:.35;cursor:default}
.timeline>label[hidden]{display:none}
#chart{width:100%;height:440px}.tablewrap{overflow-x:auto;margin-top:15px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
#chart .hoverlayer .legendlines{display:none}
.chart-note{margin:10px 0 0;font-size:12px;color:#5c707b}.chart-note:empty{display:none}
th{text-align:right;padding:11px 10px;background:#f0f5f7;color:#46606e;font-size:11px}td{text-align:right;padding:12px 10px;border-top:1px solid #e4ecef}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:white}
th button{border:0;background:none;color:inherit;font:inherit;font-weight:inherit;padding:0;cursor:pointer}th button:focus-visible{outline:2px solid #167f87;outline-offset:4px}
.positive{color:#157267}.negative{color:#a24141}.empty{padding:30px;text-align:center;color:#6f808a}
details{margin-top:14px;font-size:13px}summary{cursor:pointer;color:#396575}
details.model-picker{margin-top:0}@media(max-width:900px){.model-panel{left:0;right:auto}}
@media(max-width:720px){main{padding:18px 12px}.card{padding:13px}.head{display:block}#chart{height:350px}.timeline{gap:12px}.timeline select{min-width:125px}}
</style></head><body><main>
<div class="head"><div><div class="eyebrow">MIGHTE / FLUSIGHT 2026–27</div><h1>Weekly forecast review</h1></div></div>
<section class="card"><h2>Prospective forecasts</h2><div class="controls">
<label>Reference week<select id="reference"></select></label><label>Target<select id="target"><option value="wk inc flu hosp">Hospital admissions</option><option value="wk inc flu prop ed visits">ED visits</option><option value="wk flu hosp rate change">Hospitalization trend</option><option value="peak week inc flu hosp">Peak timing</option><option value="peak inc flu hosp">Peak height</option></select></label>
<label>Location<select id="location"></select></label>
<div class="model-control"><span>Models</span><details id="model-picker" class="model-picker"><summary aria-label="Forecast models">1 selected</summary>
<div class="model-panel"><input class="model-search" type="search" aria-label="Search forecast models" placeholder="Search models">
<div class="model-actions"><button data-select-all type="button">Select all</button><button data-clear type="button">Clear</button></div>
<div id="models" class="checks" role="group" aria-label="Forecast models"></div><div class="no-model-matches" hidden>No matching models.</div></div>
</details></div></div>
<div class="timeline"><div class="week-navigation"><label for="week-slider">Browse forecast weeks</label><div class="week-slider">
<button id="previous-week" type="button" aria-label="Previous forecast week">‹</button><input id="week-slider" type="range" min="0" step="1"><button id="next-week" type="button" aria-label="Next forecast week">›</button>
</div></div><label>Ground-truth history<select id="history"><option value="recent">Year to date</option><option value="year" selected>Past year</option><option value="two_years">Past two years</option><option value="all">All available</option></select></label></div>
<div id="chart"></div><p id="chart-note" class="chart-note"></p>
</section>
<section id="accuracy-card" class="card"><h2>Prospective accuracy to date</h2><div class="controls">
<div class="model-control"><span>Models</span><details id="accuracy-model-picker" class="model-picker"><summary aria-label="Accuracy models">0 selected</summary>
<div class="model-panel"><input class="model-search" type="search" aria-label="Search accuracy models" placeholder="Search models">
<div class="model-actions"><button data-select-all type="button">Select all</button><button data-clear type="button">Clear</button></div>
<div id="accuracy-models" class="checks" role="group" aria-label="Accuracy models"></div><div class="no-model-matches" hidden>No matching models.</div></div>
</details></div>
<label>Scoring locations<select id="scope"><option value="states">States + DC</option><option value="states_pr">States + DC + Puerto Rico</option><option value="US">United States</option><option value="all">All (includes national)</option><option value="selected">Selected location above</option></select></label>
<label>Horizon<select id="horizon"><option value="all">All horizons</option><option value="0">0 · nowcast</option><option value="1">1 week ahead</option><option value="2">2 weeks ahead</option><option value="3">3 weeks ahead</option></select></label>
</div><div id="accuracy" class="tablewrap"></div>
</section></main><script type="module">
const packed=atob('__DATA__'),bytes=new Uint8Array(packed.length);for(let i=0;i<packed.length;i++)bytes[i]=packed.charCodeAt(i);
const D=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
const el=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const options=(id,items)=>{el(id).innerHTML=items.map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('')};
const TREND='wk flu hosp rate change',PEAK_WEEK='peak week inc flu hosp',PEAK_HEIGHT='peak inc flu hosp';
const LEVELS=D.quantile_levels,MID=LEVELS.indexOf(.5),HI95=LEVELS.indexOf(.975),BANDS=centralIntervals(LEVELS);
const allForecasts=[...D.forecasts,...D.categorical_forecasts,...D.peak_height_forecasts,...D.peak_week_forecasts];
const references=[...new Set(allForecasts.map(r=>r.reference_date))].sort();
const seasonYear=Number(D.manifest.settings.season.split('-')[0]);
options('reference',[...references].reverse().map(x=>[x,x]));el('reference').value=D.manifest.reference_date;
el('week-slider').max=references.length-1;el('week-slider').disabled=references.length<2;
options('location',D.locations.filter(x=>allForecasts.some(r=>r.location===x.location)).sort((a,b)=>(b.location==='US')-(a.location==='US')||a.location_name.localeCompare(b.location_name)).map(x=>[x.location,x.location_name]));el('location').value='US';
const modelNames=[...new Set([...allForecasts.map(r=>r.model_id),...D.comparison_models])].sort();
const colors=D.model_colors;
const opening=openingModel(D.default_models,D.forecasts,{target:el('target').value,reference:el('reference').value,location:el('location').value})||D.default_models.find(m=>modelNames.includes(m));
const modelOptions=(names,checked)=>names.map(m=>`<label><input type="checkbox" value="${esc(m)}" ${checked.has(m)?'checked':''}><span class="swatch" aria-hidden="true" style="background:${colors[m]||'#658091'}"></span>${esc(m)}</label>`).join('');
el('models').innerHTML=modelOptions(modelNames,new Set([opening]));
const scoredModels=new Set(D.scores.map(r=>r.model_id));
el('accuracy-models').innerHTML=modelOptions([...scoredModels].sort(),scoredModels);
const selectedModels=id=>[...el(id).querySelectorAll('input:checked')].map(x=>x.value);
function updateModelSummary(id,list){const summary=el(id).querySelector('summary');
const count=el(list).querySelectorAll('input:checked:not(:disabled)').length;
summary.textContent=`${count} selected`;summary.setAttribute('aria-label',`${el(list).getAttribute('aria-label')}: ${count} selected`);
}
let selectionGroup='weekly';const selections={};
const rgba=(hex,a)=>`rgba(${parseInt(hex.slice(1,3),16)},${parseInt(hex.slice(3,5),16)},${parseInt(hex.slice(5,7),16)},${a})`;
const shiftDate=(date,days)=>new Date(Date.parse(date)+days*86400000).toISOString().slice(0,10);
// Ridge and interval geometry: pure functions, also run by the tests in Node.
// The first preferred model with a forecast in the opening view; the hub posts the ensemble after its deadline.
function openingModel(preferred,forecasts,view){return preferred.find(m=>forecasts.some(r=>r.model_id===m&&r.target===view.target&&r.reference_date===view.reference&&r.location===view.location))}
// Central intervals up to 95% as [lower level index, upper level index, percent], widest first.
function centralIntervals(levels){return levels.map((p,i)=>[i,levels.length-1-i,Math.round(100*(levels[levels.length-1-i]-p))]).filter(([i,j,c])=>i<j&&c<=95)}
// Each nested band adds the same opacity, so the k-th band inward reaches 1-(1-a)^k. Ridges use these steps as solid tints.
const BAND_ALPHA=.065,bandOpacity=k=>1-(1-BAND_ALPHA)**k;
function mix(hex,t){return '#'+[1,3,5].map(i=>Math.round(255-(255-parseInt(hex.slice(i,i+2),16))*t).toString(16).padStart(2,'0')).join('')}
const erf=x=>{const t=1/(1+.3275911*Math.abs(x)),y=1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x);return x<0?-y:y};
const normalCdf=z=>.5*(1+erf(z/Math.SQRT2)),normalPdf=z=>Math.exp(-z*z/2)/Math.sqrt(2*Math.PI);
// Density implied by quantiles: each step in probability spread evenly between neighbouring quantiles (a linear CDF),
// smoothed by a normal kernel of width h so that tied quantiles stay finite.
function quantileDensity(values,levels,xs,h){return xs.map(x=>{let d=0;for(let i=1;i<values.length;i++){const a=values[i-1],b=values[i],w=levels[i]-levels[i-1];
d+=b-a<h/100?w*normalPdf((x-(a+b)/2)/h)/h:w*(normalCdf((x-a)/h)-normalCdf((x-b)/h))/(b-a)}return d})}
function curveAt(xs,ys,x){const n=xs.length-1;if(x<=xs[0])return ys[0];if(x>=xs[n])return ys[n];let i=1;while(xs[i]<x)i++;return ys[i-1]+(ys[i]-ys[i-1])*(x-xs[i-1])/(xs[i]-xs[i-1])}
// The x below which a piecewise-linear curve holds the fraction p of its area.
function areaQuantile(xs,ys,p){const areas=xs.slice(1).map((x,i)=>(x-xs[i])*(ys[i]+ys[i+1])/2);let need=p*areas.reduce((a,b)=>a+b,0);
for(let i=0;i<areas.length;i++){if(need>areas[i]&&i<areas.length-1){need-=areas[i];continue}
const f=ys[i],slope=(ys[i+1]-f)/(xs[i+1]-xs[i]),r=Math.min(need,areas[i]),root=f+Math.sqrt(Math.max(0,f*f+2*slope*r));
return xs[i]+(root>0?Math.min(xs[i+1]-xs[i],2*r/root):0)}return xs[0]}
// Outline of the area under the curve from lo to hi, closed along the baseline.
function areaBetween(xs,ys,lo,hi){lo=Math.max(lo,xs[0]);hi=Math.min(hi,xs[xs.length-1]);const px=[lo,lo],py=[0,curveAt(xs,ys,lo)];
xs.forEach((x,i)=>{if(x>lo&&x<hi){px.push(x);py.push(ys[i])}});px.push(hi,hi);py.push(curveAt(xs,ys,hi),0);return [px,py]}
// Ridges sorted by median, lowest or earliest at the top, on one vertical scale on which the tallest spans 1.4 rows.
function stackRidges(curves){const sorted=[...curves].sort((a,b)=>a.median-b.median||a.model.localeCompare(b.model)),top=Math.max(0,...sorted.flatMap(c=>c.ys));
return sorted.map((c,row)=>({...c,base:sorted.length-1-row,scale:top>0?1.4/top:0}))}
function probabilityPlot(ref,loc,chosen){
const rows=D.categorical_forecasts.filter(r=>r.reference_date===ref&&r.location===loc&&chosen.includes(r.model_id));
const models=chosen.filter(m=>rows.some(r=>r.model_id===m)),n=Math.max(1,models.length),traces=[];
const layout={margin:{t:45,b:45,l:125,r:20},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',color:'#375260'},annotations:[],showlegend:false};
const height=Math.max(400,n*300),plotHeight=height-layout.margin.t-layout.margin.b,gap=72,panelHeight=(plotHeight-gap*(n-1))/n;
models.forEach((model,i)=>{const values=rows.filter(r=>r.model_id===model).sort((a,b)=>a.horizon-b.horizon),suffix=i?String(i+1):'',c=colors[model];
const domain=[(n-i-1)*(panelHeight+gap)/plotHeight,1-i*(panelHeight+gap)/plotHeight];
traces.push({type:'heatmap',x:values.map(r=>r.target_end_date),y:D.categories.map(c=>c.replaceAll('_',' ')),z:D.categories.map(c=>values.map(r=>r[c])),
xaxis:'x'+suffix,yaxis:'y'+suffix,zmin:0,zmax:1,colorscale:[[0,'#f4f7f8'],[1,c]],showscale:false,xgap:4,ygap:4,texttemplate:'%{z:.1%}',textfont:{size:13},
hovertemplate:`${esc(model.replace(/^MIGHTE-/,''))} · %{x}<br>%{y}: %{z:.1%}<extra></extra>`});
layout['xaxis'+suffix]={type:'category',anchor:'y'+suffix,tickmode:'array',tickvals:values.map(r=>r.target_end_date),ticktext:values.map(r=>`${r.target_end_date}<br>Horizon ${r.horizon}`),showgrid:false};
layout['yaxis'+suffix]={type:'category',domain,anchor:'x'+suffix,showgrid:false,ticks:''};});
if(!models.length)layout.annotations.push({text:'No selected forecasts for this week and location.',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false});
el('chart').style.height=height+'px';Plotly.react('chart',traces,layout,{responsive:true,displaylogo:false}).then(()=>Plotly.Plots.resize('chart'));
}
function peakPlot(ref,loc,target,chosen){
const timing=target===PEAK_WEEK,source=timing?D.peak_week_forecasts:D.peak_height_forecasts;
const rows=source.filter(r=>r.reference_date===ref&&r.location===loc&&chosen.includes(r.model_id)),curves=[];
const layout={margin:{t:20,b:65,l:10,r:25},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',color:'#375260'},annotations:[],showlegend:false,hovermode:'closest'};
if(timing){
// Weekly probabilities joined by straight lines; interval ends split the area under that line.
chosen.forEach(model=>{const weeks=rows.filter(r=>r.model_id===model).sort((a,b)=>a.peak_week.localeCompare(b.peak_week));
const xs=weeks.map(r=>Date.parse(r.peak_week)),ys=weeks.map(r=>r.probability),at=p=>areaQuantile(xs,ys,p);
if(xs.length>1&&ys.some(y=>y>0))curves.push({model,xs,ys,median:at(.5),bounds:BANDS.map(([i,j])=>[at(LEVELS[i]),at(LEVELS[j])])})});
const weeks=[...new Set(source.map(r=>r.peak_week))].sort(),every=el('chart').clientWidth<600?8:4;
layout.xaxis={type:'date',title:{text:'Peak week ending'},tickformat:'%b %-d<br>%Y',tickmode:'array',tickvals:weeks.filter((_,i)=>i%every===0),gridcolor:'#edf1f3',...(weeks.length?{range:[shiftDate(weeks[0],-3),shiftDate(weeks[weeks.length-1],3)]}:{})};
}else{
// The axis ends just past the widest 95% interval; interval ends are the submitted quantiles themselves.
const shown=chosen.map(m=>rows.find(r=>r.model_id===m)).filter(Boolean),end=Math.max(1,...shown.map(r=>r.q[HI95]))*1.05,grid=Array.from({length:241},(_,i)=>end*i/240);
shown.forEach(r=>{const q=r.q,xs=[...new Set([...grid,...q.filter(v=>v>0&&v<end)])].sort((a,b)=>a-b);
curves.push({model:r.model_id,xs,ys:quantileDensity(q,LEVELS,xs,Math.max(.08*(q[LEVELS.indexOf(.9)]-q[LEVELS.indexOf(.1)]),end/400)),median:q[MID],bounds:BANDS.map(([i,j])=>[q[i],q[j]])})});
layout.xaxis={title:{text:'Peak weekly hospital admissions'},range:[0,end],gridcolor:'#edf1f3',zeroline:false};
}
const ridges=stackRidges(curves),traces=[],i50=BANDS.findIndex(b=>b[2]===50),i90=BANDS.findIndex(b=>b[2]===90);
const day=new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'});
ridges.forEach(c=>{const color=colors[c.model]||'#658091',y=v=>c.base+c.scale*v,x=v=>timing?v.map(ms=>new Date(ms).toISOString().slice(0,19)):v;
// Timing intervals name the week holding each end.
const label=timing?v=>day.format(c.xs.reduce((a,b)=>Math.abs(b-v)<Math.abs(a-v)?b:a)):v=>Math.round(v).toLocaleString();
const summary=`<b>${esc(c.model)}</b><br>Median: ${label(c.median)}<br>50% interval: ${label(c.bounds[i50][0])} – ${label(c.bounds[i50][1])}<br>90% interval: ${label(c.bounds[i90][0])} – ${label(c.bounds[i90][1])}`;
const area=(lo,hi,fill)=>{const [px,py]=areaBetween(c.xs,c.ys,lo,hi);traces.push({x:x(px),y:py.map(y),mode:'lines',line:{width:0},fill:'toself',fillcolor:fill,hoverinfo:'skip',showlegend:false})};
// Ridges lower on the page are drawn later, so they hide the ones behind them.
area(c.xs[0],c.xs[c.xs.length-1],'white');
c.bounds.forEach(([lo,hi],k)=>area(lo,hi,mix(color,bandOpacity(k+1))));
traces.push({x:x(c.xs),y:c.ys.map(y),mode:'lines',line:{color,width:1.5},hoverinfo:'skip',showlegend:false});
traces.push({x:x([c.median,c.median]),y:[c.base,y(curveAt(c.xs,c.ys,c.median))],mode:'lines',line:{color,width:2},hoverinfo:'skip',showlegend:false});
const points=c.xs.map((_,k)=>k).filter(k=>timing||k%4===0);
traces.push({x:x(points.map(k=>c.xs[k])),y:points.map(k=>y(c.ys[k])),mode:'markers',marker:{size:12,color,opacity:0},showlegend:false,
customdata:points.map(k=>c.ys[k]),hovertemplate:summary+(timing?'<br>Week ending %{x|%b %-d, %Y}: %{customdata:.1%}':'')+'<extra></extra>'});
});
layout.yaxis={tickmode:'array',tickvals:ridges.map(c=>c.base),ticktext:ridges.map(c=>c.model),range:[-.3,Math.max(1,...ridges.map(c=>c.base+c.scale*Math.max(...c.ys)))+.15],gridcolor:'#e3eaee',zeroline:false,fixedrange:true,automargin:true,ticklabelstandoff:12};
if(!ridges.length)layout.annotations.push({text:'No selected peak forecasts for this week and location.',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false});
layout.height=Math.max(360,85+50*(layout.yaxis.range[1]-layout.yaxis.range[0]));el('chart').style.height=layout.height+'px';
Plotly.react('chart',traces,layout,{responsive:true,displaylogo:false}).then(()=>Plotly.Plots.resize('chart'));
}
function plot(){const ref=el('reference').value,target=el('target').value,loc=el('location').value,mult=target.includes('prop')?100:1;
const peak=[PEAK_WEEK,PEAK_HEIGHT].includes(target),group=peak?target:'weekly';
if(group!==selectionGroup){
selections[selectionGroup]=selectedModels('models');
const available=modelNames.filter(m=>allForecasts.some(r=>r.target===target&&r.model_id===m&&r.reference_date===ref&&r.location===loc));
const defaults=target===PEAK_HEIGHT?modelNames.filter(m=>D.peak_height_forecasts.some(r=>r.model_id===m)):
target===PEAK_WEEK?['MIGHTE-Base','FluSight-ensemble'].filter(m=>D.peak_week_forecasts.some(r=>r.model_id===m)):
[D.default_models.find(m=>available.includes(m))||(available.includes('FluSight-ensemble')?'FluSight-ensemble':available[0])].filter(Boolean);
const chosen=new Set(selections[group]||defaults);
el('models').querySelectorAll('input').forEach(input=>input.checked=chosen.has(input.value));selectionGroup=group;
}
const index=references.indexOf(ref);el('week-slider').value=index;el('week-slider').setAttribute('aria-valuetext',ref);
el('previous-week').disabled=index<=0;el('next-week').disabled=index>=references.length-1;
el('models').querySelectorAll('input').forEach(input=>{input.disabled=!allForecasts.some(r=>r.target===target&&r.model_id===input.value)});
updateModelSummary('model-picker','models');
const traces=[],chosen=selectedModels('models');
el('history').parentElement.hidden=target===TREND||peak;
const shading='central prediction intervals from 10% to 90% in steps of 10, plus 95%; darker is narrower.';
el('chart-note').textContent=target===TREND?'':peak?`Each ridge is one model's forecast, sorted by median with the ${target===PEAK_WEEK?'earliest':'lowest'} at the top. Shading marks ${shading}${target===PEAK_HEIGHT?' Curves are smoothed from the submitted quantiles.':''}`:`Bands are ${shading}`;
if(peak){peakPlot(ref,loc,target,chosen);return;}
if(target===TREND){probabilityPlot(ref,loc,chosen);return;}
el('chart').style.height='';
const observed=D.truth.filter(r=>r.target===target&&r.location===loc&&r.value!==null).sort((a,b)=>a.date.localeCompare(b.date));
const latest=references[references.length-1],history=el('history').value;
const start=history==='all'?(observed[0]?.date||references[0]):`${seasonYear-({recent:0,year:1,two_years:2}[history])}-07-01`;
// Include every saved forecast week and hold the view fixed while moving between them.
const begin=[start,shiftDate(references[0],-7)].sort()[0],end=[shiftDate(latest,28),observed[observed.length-1]?.date||latest].sort().pop();
const truth=observed.filter(r=>r.date>=begin&&r.date<=end);
const comparison=D.forecasts.filter(r=>r.target===target&&r.location===loc&&chosen.includes(r.model_id));
const ymax=Math.max(1e-6,...truth.map(r=>r.value*mult),...comparison.filter(r=>r.target_end_date>=begin&&r.target_end_date<=end).map(r=>r.q[HI95]*mult));
// The hover reads one invisible point per week that lists only that week's values. Unified hover over the
// drawn series would add each series' nearest point within 20 px, e.g. the first forecast on the last observation.
const hover={},note=(date,color,label,value)=>(hover[date]=hover[date]||[]).push(`<span style="color:${color}">\u25CF</span> ${label}: ${mult===100?value.toFixed(2):+value.toFixed(2)}`);
truth.forEach(r=>note(r.date,'#253e4c','Observed',r.value*mult));
traces.push({x:truth.map(r=>r.date),y:truth.map(r=>r.value*mult),name:'Observed · revised',mode:'lines+markers',line:{color:'#253e4c',width:2},marker:{size:6},hoverinfo:'skip'});
chosen.forEach(model=>{const rows=comparison.filter(r=>r.reference_date===ref&&r.model_id===model&&r.q[MID]!==null).sort((a,b)=>a.horizon-b.horizon);if(!rows.length)return;
const c=colors[model]||'#658091',x=rows.map(r=>r.target_end_date);
rows.forEach(r=>note(r.target_end_date,c,esc(model.replace(/^MIGHTE-/,'')),r.q[MID]*mult));
// Nested central intervals, widest first: the overlapping fills darken toward the median.
BANDS.forEach(([i,j])=>{const band=rows.filter(r=>r.q[i]!==null&&r.q[j]!==null),dates=band.map(r=>r.target_end_date);
if(band.length)traces.push({x:[...dates,...[...dates].reverse()],y:[...band.map(r=>r.q[i]*mult),...band.map(r=>r.q[j]*mult).reverse()],mode:'lines',line:{width:0},fill:'toself',fillcolor:rgba(c,BAND_ALPHA),showlegend:false,hoverinfo:'skip',legendgroup:model})});
traces.push({x,y:rows.map(r=>r.q[MID]*mult),mode:'lines+markers',name:model,legendgroup:model,line:{color:c,width:2.5},marker:{size:8},hoverinfo:'skip'})});
const weeks=Object.keys(hover).sort();
traces.push({x:weeks,y:weeks.map(()=>0),text:weeks.map(d=>hover[d].join('<br>')),mode:'markers',marker:{opacity:0},showlegend:false,hovertemplate:'%{text}<extra></extra>'});
Plotly.react('chart',traces,{hoverdistance:-1,margin:{t:25,b:45,l:65,r:20},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',color:'#375260'},uirevision:[target,loc,history,...chosen].join('|'),xaxis:{type:'date',range:[begin,end],gridcolor:'#edf1f3',hoverformat:'%b %-d, %Y'},yaxis:{title:{text:mult===100?'Influenza ED visits (%)':'Hospital admissions'},range:[0,ymax*1.08],gridcolor:'#edf1f3'},legend:{orientation:'h',y:1.14,maxheight:.25},hovermode:'x unified',shapes:[{type:'line',x0:ref,x1:ref,y0:0,y1:1,yref:'paper',line:{color:'#95a7ae',dash:'dot',width:1}}]},{responsive:true,displaylogo:false}).then(()=>Plotly.Plots.resize('chart'));
}
const mean=rows=>rows.length?rows.reduce((a,b)=>a+b,0)/rows.length:null;
const key=r=>[r.reference_date,r.target,r.horizon,r.target_end_date,r.location].join('|');
const fmt=(v,n=3)=>v===null||!Number.isFinite(v)?'—':v.toLocaleString(undefined,{maximumFractionDigits:n});
const pct=v=>v===null||!Number.isFinite(v)?'—':(100*v).toFixed(1)+'%';
let accuracySort={label:'Model',direction:1};
function renderAccuracy(headers,rows,formatCell,skillColumns){
if(!headers.includes(accuracySort.label))accuracySort={label:'Model',direction:1};
const column=headers.indexOf(accuracySort.label),missing=v=>v===null||typeof v==='number'&&!Number.isFinite(v);
rows.sort((a,b)=>{const av=a[column],bv=b[column],am=missing(av),bm=missing(bv);
return am!==bm?(am?1:-1):am?a[0].localeCompare(b[0]):(typeof av==='number'?av-bv:av.localeCompare(bv))*accuracySort.direction||a[0].localeCompare(b[0]);});
const head=headers.map((label,i)=>`<th scope="col" aria-sort="${i===column?(accuracySort.direction===1?'ascending':'descending'):'none'}"><button type="button" data-sort="${esc(label)}">${esc(label)}<span aria-hidden="true">${i===column?(accuracySort.direction===1?' ↑':' ↓'):' ↕'}</span></button></th>`).join('');
const body=rows.map(row=>'<tr>'+row.map((value,i)=>`<td class="${skillColumns.includes(i)&&Number.isFinite(value)?(value>=0?'positive':'negative'):''}">${formatCell(value,i)}</td>`).join('')+'</tr>').join('');
el('accuracy').innerHTML='<table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table>';
}
function table(){const target=el('target').value,card=el('accuracy-card');
card.hidden=target===PEAK_WEEK||target===PEAK_HEIGHT;if(card.hidden)return;
let rows=D.scores.filter(r=>r.target===target);const scope=el('scope').value,h=el('horizon').value,base=target===TREND?D.trend_baseline:D.comparison_baseline,digits=target.includes('prop')?6:3;
rows=rows.filter(r=>(scope==='all'||(scope==='states'?!['US','72'].includes(r.location):scope==='states_pr'?r.location!=='US':r.location===(scope==='selected'?el('location').value:scope)))&&(h==='all'||r.horizon===Number(h)));
const available=new Set(rows.map(r=>r.model_id));
el('accuracy-models').querySelectorAll('input').forEach(input=>input.disabled=!available.has(input.value));
updateModelSummary('accuracy-model-picker','accuracy-models');
if(!rows.length){el('accuracy').innerHTML='<div class="empty">No scored prospective forecasts yet.</div>';return;}
const names=[...available].sort(),groups=Object.fromEntries(names.map(m=>[m,rows.filter(r=>r.model_id===m)])),maps=Object.fromEntries(names.map(m=>[m,new Map(groups[m].map(r=>[key(r),r]))]));
const selected=new Set(selectedModels('accuracy-models')),visible=names.filter(m=>selected.has(m));
if(!visible.length){el('accuracy').innerHTML='<div class="empty">No selected models have scored forecasts.</div>';return;}
function theta(metric){
  return Object.fromEntries(names.map(model=>{
    const logRatios=[0]; // log(1) for the model's self-comparison
    for(const other of names){
      if(other===model)continue;
      let ownSum=0,otherSum=0;
      for(const row of groups[model]){
        const match=maps[other].get(key(row));
        if(match && Number.isFinite(row[metric]) && Number.isFinite(match[metric])){
          ownSum+=row[metric];otherSum+=match[metric];
        }
      }
      const ratio=ownSum/otherSum;
      if(!Number.isNaN(ratio))logRatios.push(Math.log(ratio));
    }
    return [model,names.length>1?Math.exp(mean(logRatios)):NaN];
  }));
}
if(target===TREND){
const data=visible.map(m=>{const g=groups[m],bm=maps[base]||new Map();let a=0,b=0,n=0;g.forEach(r=>{const br=bm.get(key(r));if(br){a+=r.rps;b+=br.rps;n++}});const skill=n&&b>0?1-a/b:null;
return [m,g.length,n,mean(g.map(r=>r.rps)),skill,mean(g.map(r=>r.brier)),mean(g.map(r=>r.log_score)),mean(g.map(r=>r.accuracy)),mean(g.map(r=>r.absolute_category_error)),mean(g.map(r=>r.zero_observed_probability))];});
renderAccuracy(['Model','N','Matched','RPS','RPS skill','Brier','Log score','Category accuracy','Category error','Zero-probability outcomes'],data,(value,i)=>i===0?esc(value):[4,7,9].includes(i)?pct(value):fmt(value),[4]);return;}
const tr=theta('wis'),tl=theta('wis_log1p'),ta=theta('ae');
const data=visible.map(m=>{const g=groups[m],bm=maps[base]||new Map();let a=0,b=0,ma=0,mb=0,n=0;g.forEach(r=>{const br=bm.get(key(r));if(br){a+=r.wis;b+=br.wis;ma+=r.ae;mb+=br.ae;n++}});const skill=n&&b>0?1-a/b:null,maeSkill=n&&mb>0?1-ma/mb:null;
return [m,g.length,n,mean(g.map(r=>r.ae)),ta[base]>0?ta[m]/ta[base]:null,maeSkill,mean(g.map(r=>r.wis)),tr[base]>0?tr[m]/tr[base]:null,tl[base]>0?tl[m]/tl[base]:null,skill,mean(g.map(r=>r.coverage_50)),mean(g.map(r=>r.coverage_80)),mean(g.map(r=>r.coverage_95))];});
renderAccuracy(['Model','N','Matched','MAE','Rel MAE','MAE Skill','Mean WIS','Rel WIS','Rel log-WIS','WIS Skill','50% coverage','80% coverage','95% coverage'],data,(value,i)=>i===0?esc(value):i===5||i>=9?pct(value):fmt(value,[3,6].includes(i)?digits:3),[5,9]);}
['reference','target','location','history'].forEach(id=>el(id).addEventListener('change',plot));
function browseWeek(index){el('reference').value=references[Math.max(0,Math.min(references.length-1,index))];plot()}
el('week-slider').addEventListener('input',()=>browseWeek(Number(el('week-slider').value)));
el('previous-week').addEventListener('click',()=>browseWeek(references.indexOf(el('reference').value)-1));
el('next-week').addEventListener('click',()=>browseWeek(references.indexOf(el('reference').value)+1));
el('week-slider').addEventListener('wheel',event=>{const delta=event.deltaX||event.deltaY;if(!delta||el('week-slider').disabled)return;event.preventDefault();const now=performance.now();if(now-(el('week-slider').lastWheel||0)<150)return;el('week-slider').lastWheel=now;browseWeek(references.indexOf(el('reference').value)+(delta>0?1:-1))},{passive:false});
function bindModelPicker(id,listId,onChange){const picker=el(id),list=el(listId),search=picker.querySelector('.model-search');
list.addEventListener('change',onChange);
search.addEventListener('input',()=>{const query=search.value.toLowerCase();let count=0;list.querySelectorAll('label').forEach(label=>{label.hidden=!label.textContent.toLowerCase().includes(query);if(!label.hidden)count++});picker.querySelector('.no-model-matches').hidden=count>0});
picker.querySelector('[data-select-all]').addEventListener('click',()=>{list.querySelectorAll('input:not(:disabled)').forEach(input=>input.checked=true);onChange()});
picker.querySelector('[data-clear]').addEventListener('click',()=>{list.querySelectorAll('input').forEach(input=>input.checked=false);onChange()});
document.addEventListener('click',event=>{if(!picker.contains(event.target))picker.open=false});
picker.addEventListener('keydown',event=>{if(event.key==='Escape'){picker.open=false;picker.querySelector('summary').focus()}});
}
bindModelPicker('model-picker','models',plot);
bindModelPicker('accuracy-model-picker','accuracy-models',table);
el('accuracy').addEventListener('click',event=>{const button=event.target.closest('button[data-sort]');if(!button)return;const label=button.dataset.sort;accuracySort={label,direction:accuracySort.label===label?-accuracySort.direction:1};table();
[...el('accuracy').querySelectorAll('button[data-sort]')].find(button=>button.dataset.sort===label)?.focus()});
['target','location','scope','horizon'].forEach(id=>el(id).addEventListener('change',table));plot();table();
</script></body></html>'''
