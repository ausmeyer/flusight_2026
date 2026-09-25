"""A portable, offline HTML review of forecasts, provenance, and accuracy."""
from __future__ import annotations

import colorsys
import hashlib
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

from .contract import HOSP, MODELS, UNIT, Contract, read_forecast
from .data import latest_snapshot, verify_snapshot
from .evaluate import BENCHMARKS, LOCAL_BASELINE, discover_benchmarks, evaluate, summarize
from .pipeline import verify_run
from .util import utc_now, write_json


def records(frame: pd.DataFrame) -> list:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


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


def build_report(root: Path, run: Path, *, online=True) -> Path:
    manifest = verify_run(root, run)
    snapshot = latest_snapshot(root)
    verify_snapshot(snapshot)
    season_references = Contract(snapshot / "contract").by_target[HOSP]["task_ids"]["reference_date"]["optional"]
    catalog, catalog_status = discover_benchmarks(root, season_references,
                                                 season=manifest["settings"]["season"], online=online)
    comparison_models = sorted(set(BENCHMARKS) | ({f["model"] for f in catalog["files"]} if catalog else set()))
    scores, benchmark_status, archive = evaluate(root, snapshot, online=online,
                                                comparison_references=[manifest["reference_date"]], catalog=catalog)
    current = pd.concat([read_forecast(run / filename).assign(model_id=Path(filename).parent.name)
                         for filename in manifest["output_hashes"]], ignore_index=True)
    if not archive.empty:
        archive = archive[~(archive.reference_date.eq(manifest["reference_date"])
                            & archive.model_id.isin(current.model_id.unique()))]
        forecasts = pd.concat([archive, current], ignore_index=True)
    else:
        forecasts = current
    quantiles = forecasts[forecasts.output_type.eq("quantile")].copy()
    quantiles["output_type_id"] = pd.to_numeric(quantiles.output_type_id)
    quantiles = quantiles[quantiles.output_type_id.isin([.025, .25, .5, .75, .975])]
    chart = quantiles.pivot(index=["model_id", *UNIT], columns="output_type_id", values="value").reset_index()
    chart = chart.rename(columns={.025: "lo95", .25: "lo50", .5: "median", .75: "hi50", .975: "hi95"})
    truth = pd.read_csv(snapshot / "truth.csv", dtype={"location": str})
    locations = pd.read_csv(snapshot / "contract/locations.csv", dtype={"location": str})
    payload = {"manifest": manifest, "generated_at": utc_now(), "truth_snapshot": snapshot.name,
               "data_audit": json.loads((run / "data-audit.json").read_text()),
               "forecasts": records(chart), "truth": records(truth[["date", "target", "location", "value"]]),
               "scores": records(scores.drop(columns=[c for c in scores.columns if isinstance(c, float)], errors="ignore")),
               "locations": records(locations[["location", "location_name"]]),
               "benchmarks": [catalog_status, *benchmark_status],
               "comparison_models": comparison_models, "baseline_models": [LOCAL_BASELINE, *comparison_models],
               "default_models": list(MODELS),
               "model_colors": {m: model_color(m) for m in set(forecasts.model_id) | set(comparison_models)}}
    output = root / "reports" / manifest["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    if not scores.empty:
        scores.to_csv(output / "scores.csv", index=False)
        summarize(scores).to_csv(output / "accuracy.csv", index=False)
    write_json(output / "report-data.json", payload)
    html = TEMPLATE.replace("__PLOTLY__", get_plotlyjs()).replace(
        "__DATA__", json.dumps(payload, allow_nan=False).replace("</", "<\\/"))
    path = output / "index.html"
    path.write_text(html)
    write_json(root / "reports/latest.json", {"file": str(path.relative_to(root))})
    return path


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIGHTE · Weekly review</title><script>__PLOTLY__</script>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#1b3040;background:#f2f5f6}
body{margin:0}main{max-width:1450px;margin:auto;padding:28px 32px 50px}h1{font-size:30px;letter-spacing:-1px;margin:4px 0}
.eyebrow{color:#367b80;font-size:12px;font-weight:750;letter-spacing:2px}.muted{color:#5b6f7c;font-size:13px;line-height:1.65}
.head{display:flex;justify-content:space-between;gap:18px;align-items:center}.badge{padding:8px 14px;border-radius:20px;background:#d6ede7;color:#215949;font-size:12px;font-weight:750}
.preview{background:#fff1cf;color:#805908}.card{margin-top:22px;border:1px solid #dce4e8;border-radius:13px;background:white;padding:22px;box-shadow:0 2px 3px #19333e04}
h2{font-size:18px;margin:0 0 15px}.controls{display:flex;gap:16px;align-items:end;flex-wrap:wrap}label{font-size:12px;font-weight:650;display:flex;flex-direction:column;gap:7px}
select{font:inherit;min-width:160px;border:1px solid #c5d2d9;border-radius:6px;padding:8px;background:white;color:#213f50}
.model-control{font-size:12px;font-weight:650;display:flex;flex-direction:column;gap:7px}.model-picker{position:relative;margin:0;min-width:190px}
.model-picker summary{border:1px solid #c5d2d9;border-radius:6px;padding:8px;background:white;color:#213f50;font-size:12px}
.model-panel{position:absolute;z-index:20;right:0;top:calc(100% + 7px);width:min(320px,calc(100vw - 48px));box-sizing:border-box;background:white;border:1px solid #c5d2d9;border-radius:8px;padding:12px;box-shadow:0 8px 24px #19333e24}
.model-search{box-sizing:border-box;width:100%;padding:8px;border:1px solid #c5d2d9;border-radius:5px;font:inherit}.model-actions{display:flex;gap:8px;margin:10px 0}
.model-actions button{font:inherit;color:#216b75;border:1px solid #c5d2d9;border-radius:5px;background:white;padding:5px 9px;cursor:pointer}
.checks{max-height:270px;overflow-y:auto}.checks label{display:flex;flex-direction:row;align-items:center;font-weight:500;padding:7px 0;gap:8px}.checks label[hidden]{display:none}.checks label:has(input:disabled){opacity:.45}.checks input{accent-color:#167f87;margin:0}.swatch{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.timeline{display:flex;gap:24px;align-items:end;margin-top:18px}.week-navigation{flex:1;min-width:0}.week-navigation>label{margin-bottom:7px}.week-slider{display:flex;align-items:center;gap:10px}.week-slider input{flex:1;min-width:0;accent-color:#167f87;cursor:ew-resize}.week-slider button{font:inherit;border:1px solid #c5d2d9;border-radius:6px;background:white;color:#213f50;width:34px;height:34px;cursor:pointer}.week-slider button:disabled{opacity:.35;cursor:default}
#chart{width:100%;height:440px}.tablewrap{overflow-x:auto;margin-top:15px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
th{text-align:right;padding:11px 10px;background:#f0f5f7;color:#46606e;font-size:11px}td{text-align:right;padding:12px 10px;border-top:1px solid #e4ecef}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:white}
.positive{color:#157267}.negative{color:#a24141}.empty{padding:30px;text-align:center;color:#6f808a}
details{margin-top:14px;font-size:13px}summary{cursor:pointer;color:#396575}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:16px;max-height:400px;overflow:auto;font-size:11px}
details.model-picker{margin-top:0}@media(max-width:900px){.model-panel{left:0;right:auto}}
@media(max-width:720px){main{padding:18px 12px}.card{padding:13px}.head{display:block}.badge{display:inline-block;margin-top:12px}#chart{height:350px}.timeline{gap:12px}.timeline select{min-width:125px}}
</style></head><body><main>
<div class="head"><div><div class="eyebrow">MIGHTE / FLUSIGHT 2026–27</div><h1>Weekly forecast review</h1></div><span id="status" class="badge"></span></div>
<section class="card"><h2>Prospective forecasts</h2><div class="controls">
<label>Reference week<select id="reference"></select></label><label>Target<select id="target"><option value="wk inc flu hosp">Hospital admissions</option><option value="wk inc flu prop ed visits">Influenza ED visits</option></select></label>
<label>Location<select id="location"></select></label>
<div class="model-control"><span>Models</span><details id="model-picker" class="model-picker"><summary id="model-summary" aria-label="Models">3 selected</summary>
<div class="model-panel"><input id="model-search" class="model-search" type="search" aria-label="Search models" placeholder="Search models">
<div class="model-actions"><button id="select-all" type="button">Select all</button><button id="clear-models" type="button">Clear</button></div>
<div id="models" class="checks" role="group" aria-label="Forecast models"></div><div id="no-model-matches" hidden>No matching models.</div></div>
</details></div></div>
<div class="timeline"><div class="week-navigation"><label for="week-slider">Browse forecast weeks</label><div class="week-slider">
<button id="previous-week" type="button" aria-label="Previous forecast week">‹</button><input id="week-slider" type="range" min="0" step="1"><button id="next-week" type="button" aria-label="Next forecast week">›</button>
</div></div><label>Ground-truth history<select id="history"><option value="recent">Recent</option><option value="year">Past year</option><option value="all">All available</option></select></label></div>
<div id="chart"></div>
</section>
<section class="card"><h2>Prospective accuracy to date</h2><div class="controls">
<label>Comparison baseline<select id="baseline"></select></label>
<label>Scoring locations<select id="scope"><option value="states">States + DC + Puerto Rico</option><option value="US">United States</option><option value="all">All (includes national)</option><option value="selected">Selected location above</option></select></label>
<label>Horizon<select id="horizon"><option value="all">All horizons</option><option value="0">0 · nowcast</option><option value="1">1 week ahead</option><option value="2">2 weeks ahead</option><option value="3">3 weeks ahead</option></select></label>
</div><div id="accuracy" class="tablewrap"></div>
</section><section class="card"><h2>Run details</h2><div id="provenance" class="muted"></div>
<details><summary>Input freshness, historical reconstruction, and omitted ED locations</summary><pre id="audit"></pre></details>
<details><summary>Hub baseline and comparison-model availability</summary><pre id="benchmarks"></pre></details>
<details><summary>Model settings and validation</summary><pre id="manifest"></pre></details>
</section></main><script>
const D=__DATA__;
const el=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const options=(id,items)=>{el(id).innerHTML=items.map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('')};
el('status').textContent=D.manifest.preview?'PREVIEW · CANNOT SUBMIT':'VALIDATED · READY FOR REVIEW';
el('status').classList.toggle('preview',D.manifest.preview);
const references=[...new Set(D.forecasts.map(r=>r.reference_date))].sort();
options('reference',[...references].reverse().map(x=>[x,x]));el('reference').value=D.manifest.reference_date;
el('week-slider').max=references.length-1;el('week-slider').disabled=references.length<2;
options('location',D.locations.filter(x=>D.forecasts.some(r=>r.location===x.location)).sort((a,b)=>(b.location==='US')-(a.location==='US')||a.location_name.localeCompare(b.location_name)).map(x=>[x.location,x.location_name]));el('location').value='US';
options('baseline',D.baseline_models.map(m=>[m,m]));
const modelNames=[...new Set([...D.forecasts.map(r=>r.model_id),...D.comparison_models])].sort();
const colors=D.model_colors;
el('models').innerHTML=modelNames.map(m=>`<label><input type="checkbox" value="${esc(m)}" ${D.default_models.includes(m)?'checked':''}><span class="swatch" aria-hidden="true" style="background:${colors[m]}"></span>${esc(m)}</label>`).join('');
const selectedModels=()=>[...el('models').querySelectorAll('input:checked')].map(x=>x.value);
const rgba=(hex,a)=>`rgba(${parseInt(hex.slice(1,3),16)},${parseInt(hex.slice(3,5),16)},${parseInt(hex.slice(5,7),16)},${a})`;
const shiftDate=(date,days)=>new Date(Date.parse(date)+days*86400000).toISOString().slice(0,10);
function plot(){const ref=el('reference').value,target=el('target').value,loc=el('location').value,mult=target.includes('prop')?100:1;
const index=references.indexOf(ref);el('week-slider').value=index;el('week-slider').setAttribute('aria-valuetext',ref);
el('previous-week').disabled=index<=0;el('next-week').disabled=index>=references.length-1;
el('models').querySelectorAll('input').forEach(input=>{input.disabled=!D.forecasts.some(r=>r.target===target&&r.model_id===input.value)});
const selectedCount=el('models').querySelectorAll('input:checked:not(:disabled)').length;
el('model-summary').textContent=`${selectedCount} selected`;el('model-summary').setAttribute('aria-label',`Models: ${selectedCount} selected`);
const traces=[],chosen=selectedModels();
const observed=D.truth.filter(r=>r.target===target&&r.location===loc&&r.value!==null).sort((a,b)=>a.date.localeCompare(b.date));
const latest=references[references.length-1],history=el('history').value;
const start=history==='all'?(observed[0]?.date||references[0]):shiftDate(latest,history==='year'?-365:-120);
// Include every saved forecast week and hold the view fixed while moving between them.
const begin=[start,shiftDate(references[0],-7)].sort()[0],end=[shiftDate(latest,28),observed[observed.length-1]?.date||latest].sort().pop();
const truth=observed.filter(r=>r.date>=begin&&r.date<=end);
const comparison=D.forecasts.filter(r=>r.target===target&&r.location===loc&&chosen.includes(r.model_id));
const ymax=Math.max(1e-6,...truth.map(r=>r.value*mult),...comparison.filter(r=>r.target_end_date>=begin&&r.target_end_date<=end).map(r=>r.hi95*mult));
traces.push({x:truth.map(r=>r.date),y:truth.map(r=>r.value*mult),name:'Observed · revised',mode:'lines+markers',line:{color:'#253e4c',width:2},marker:{size:4}});
chosen.forEach(model=>{const rows=comparison.filter(r=>r.reference_date===ref&&r.model_id===model).sort((a,b)=>a.horizon-b.horizon);if(!rows.length)return;
const c=colors[model]||'#658091',x=rows.map(r=>r.target_end_date);
[['lo95','hi95',.08],['lo50','hi50',.15]].forEach(([lo,hi,a])=>{traces.push({x,y:rows.map(r=>r[lo]*mult),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip',legendgroup:model});traces.push({x,y:rows.map(r=>r[hi]*mult),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:rgba(c,a),showlegend:false,hoverinfo:'skip',legendgroup:model})});
traces.push({x,y:rows.map(r=>r.median*mult),mode:'lines+markers',name:model,legendgroup:model,line:{color:c,width:2.5},marker:{size:6}})});
Plotly.react('chart',traces,{margin:{t:25,b:45,l:65,r:20},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',color:'#375260'},uirevision:[target,loc,history,...chosen].join('|'),xaxis:{type:'date',range:[begin,end],gridcolor:'#edf1f3'},yaxis:{title:{text:mult===100?'Influenza ED visits (%)':'Hospital admissions'},range:[0,ymax*1.08],gridcolor:'#edf1f3'},legend:{orientation:'h',y:1.14,maxheight:.25},hovermode:'x unified',shapes:[{type:'line',x0:ref,x1:ref,y0:0,y1:1,yref:'paper',line:{color:'#95a7ae',dash:'dot',width:1}}]},{responsive:true,displaylogo:false});
table();}
const mean=rows=>rows.length?rows.reduce((a,b)=>a+b,0)/rows.length:null;
const geom=arr=>arr.length?(arr.includes(0)?0:Math.exp(mean(arr.map(Math.log)))):null;
const key=r=>[r.reference_date,r.target,r.horizon,r.target_end_date,r.location].join('|');
const fmt=(v,n=3)=>v===null||!Number.isFinite(v)?'—':v.toLocaleString(undefined,{maximumFractionDigits:n});
const pct=v=>v===null||!Number.isFinite(v)?'—':(100*v).toFixed(1)+'%';
function table(){let rows=D.scores.filter(r=>r.target===el('target').value);const scope=el('scope').value,h=el('horizon').value,base=el('baseline').value,digits=el('target').value.includes('prop')?6:3;
rows=rows.filter(r=>(scope==='all'||(scope==='states'?r.location!=='US':r.location===(scope==='selected'?el('location').value:scope)))&&(h==='all'||r.horizon===Number(h)));
if(!rows.length){el('accuracy').innerHTML='<div class="empty">No scored prospective forecasts yet.</div>';return;}
const names=[...new Set(rows.map(r=>r.model_id))].sort(),groups=Object.fromEntries(names.map(m=>[m,rows.filter(r=>r.model_id===m)])),maps=Object.fromEntries(names.map(m=>[m,new Map(groups[m].map(r=>[key(r),r]))]));
const selected=new Set(selectedModels()),visible=names.filter(m=>selected.has(m));
if(!visible.length){el('accuracy').innerHTML='<div class="empty">No selected models have scored forecasts.</div>';return;}
const theta=metric=>Object.fromEntries(names.map(m=>[m,geom(names.map(n=>{let a=0,b=0,count=0;groups[m].forEach(r=>{const other=maps[n].get(key(r));if(other){a+=r[metric];b+=other[metric];count++}});return count&&b>0?a/b:NaN}))]));
const tr=theta('wis'),tl=theta('wis_log1p');
let html='<table><thead><tr>'+['Model','N','Matched','Mean WIS','Geo WIS','Geo log-WIS','MAE','Rel WIS','Rel log-WIS','WIS skill','50% coverage','80% coverage','95% coverage'].map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
visible.forEach(m=>{const g=groups[m],bm=maps[base]||new Map();let a=0,b=0,n=0;g.forEach(r=>{const br=bm.get(key(r));if(br){a+=r.wis;b+=br.wis;n++}});const skill=n&&b>0?1-a/b:null;const cells=[esc(m),g.length,n,fmt(mean(g.map(r=>r.wis)),digits),fmt(geom(g.map(r=>r.wis)),digits),fmt(geom(g.map(r=>r.wis_log1p)),Math.max(digits,5)),fmt(mean(g.map(r=>r.ae)),digits),fmt(tr[base]>0?tr[m]/tr[base]:null),fmt(tl[base]>0?tl[m]/tl[base]:null),pct(skill),pct(mean(g.map(r=>r.coverage_50))),pct(mean(g.map(r=>r.coverage_80))),pct(mean(g.map(r=>r.coverage_95)))];html+='<tr>'+cells.map((x,i)=>`<td class="${i===9&&skill!==null?(skill>=0?'positive':'negative'):''}">${x}</td>`).join('')+'</tr>'});
el('accuracy').innerHTML=html+'</tbody></table>';}
el('provenance').textContent=`Forecast inputs: ${D.manifest.snapshot_id}. Scoring truth: ${D.truth_snapshot}. Report: ${new Date(D.generated_at).toLocaleString()}. Run: ${D.manifest.run_id}.`;
['audit','benchmarks','manifest'].forEach(id=>el(id).textContent=JSON.stringify(id==='audit'?D.data_audit:id==='benchmarks'?D.benchmarks:D.manifest,null,2));
['reference','target','location','history'].forEach(id=>el(id).addEventListener('change',plot));el('models').addEventListener('change',plot);
function browseWeek(index){el('reference').value=references[Math.max(0,Math.min(references.length-1,index))];plot()}
el('week-slider').addEventListener('input',()=>browseWeek(Number(el('week-slider').value)));
el('previous-week').addEventListener('click',()=>browseWeek(references.indexOf(el('reference').value)-1));
el('next-week').addEventListener('click',()=>browseWeek(references.indexOf(el('reference').value)+1));
el('week-slider').addEventListener('wheel',event=>{const delta=event.deltaX||event.deltaY;if(!delta||el('week-slider').disabled)return;event.preventDefault();const now=performance.now();if(now-(el('week-slider').lastWheel||0)<150)return;el('week-slider').lastWheel=now;browseWeek(references.indexOf(el('reference').value)+(delta>0?1:-1))},{passive:false});
el('model-search').addEventListener('input',()=>{const query=el('model-search').value.toLowerCase();let count=0;el('models').querySelectorAll('label').forEach(label=>{label.hidden=!label.textContent.toLowerCase().includes(query);if(!label.hidden)count++});el('no-model-matches').hidden=count>0});
el('select-all').addEventListener('click',()=>{el('models').querySelectorAll('input:not(:disabled)').forEach(input=>input.checked=true);plot()});
el('clear-models').addEventListener('click',()=>{el('models').querySelectorAll('input').forEach(input=>input.checked=false);plot()});
document.addEventListener('click',event=>{if(!el('model-picker').contains(event.target))el('model-picker').open=false});
el('model-picker').addEventListener('keydown',event=>{if(event.key==='Escape'){el('model-picker').open=false;el('model-summary').focus()}});
['baseline','scope','horizon'].forEach(id=>el(id).addEventListener('change',table));plot();
</script></body></html>'''
