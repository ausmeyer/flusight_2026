"""A portable, offline HTML review of forecasts, provenance, and accuracy."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

from .contract import UNIT, read_forecast
from .data import latest_snapshot, verify_snapshot
from .evaluate import evaluate, summarize
from .pipeline import verify_run
from .util import utc_now, write_json


def records(frame: pd.DataFrame) -> list:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def build_report(root: Path, run: Path, *, online=True) -> Path:
    manifest = verify_run(root, run)
    snapshot = latest_snapshot(root)
    verify_snapshot(snapshot)
    scores, benchmark_status, archive = evaluate(root, snapshot, online=online)
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
               "locations": records(locations[["location", "location_name"]]), "benchmarks": benchmark_status}
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
.checks{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 0}.checks label{display:block}.checks input{accent-color:#167f87}
#chart{width:100%;height:440px}.tablewrap{overflow-x:auto;margin-top:15px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
th{text-align:right;padding:11px 10px;background:#f0f5f7;color:#46606e;font-size:11px}td{text-align:right;padding:12px 10px;border-top:1px solid #e4ecef}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:white}
.positive{color:#157267}.negative{color:#a24141}.empty{padding:30px;text-align:center;color:#6f808a}.stats{display:flex;gap:35px;flex-wrap:wrap;margin:20px 0 5px}.stat strong{font-size:23px;display:block}.stat span{font-size:12px;color:#637984}
details{margin-top:14px;font-size:13px}summary{cursor:pointer;color:#396575}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:16px;max-height:400px;overflow:auto;font-size:11px}
@media(max-width:720px){main{padding:18px 12px}.card{padding:13px}.head{display:block}.badge{display:inline-block;margin-top:12px}#chart{height:350px}}
</style></head><body><main>
<div class="head"><div><div class="eyebrow">MIGHTE / FLUSIGHT 2026–27</div><h1>Weekly forecast review</h1><div id="subtitle" class="muted"></div></div><span id="status" class="badge"></span></div>
<div class="stats" id="stats"></div>
<section class="card"><h2>Prospective forecasts</h2><div class="controls">
<label>Reference week<select id="reference"></select></label><label>Target<select id="target"><option value="wk inc flu hosp">Hospital admissions</option><option value="wk inc flu prop ed visits">Influenza ED visits</option></select></label>
<label>Location<select id="location"></select></label></div><div id="models" class="checks"></div><div id="chart"></div>
</section>
<section class="card"><h2>Prospective accuracy to date</h2><div class="controls">
<label>Comparison baseline<select id="baseline"><option>Local-persistence</option><option>FluSight-baseline</option><option>FluSight-ensemble</option></select></label>
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
el('subtitle').textContent=`Reference ${D.manifest.reference_date} · generated ${new Date(D.manifest.completed_at).toLocaleString()}`;
el('status').textContent=D.manifest.preview?'PREVIEW · CANNOT SUBMIT':'VALIDATED · READY FOR REVIEW';
el('status').classList.toggle('preview',D.manifest.preview);
el('stats').innerHTML=`<div class="stat"><strong>3</strong><span>submission models</span></div><div class="stat"><strong>0–3</strong><span>forecast horizons</span></div><div class="stat"><strong>${D.manifest.settings.runtime.num_bags}</strong><span>fits per boosted component</span></div><div class="stat"><strong>${new Set(D.scores.map(r=>r.reference_date)).size}</strong><span>prospective weeks with truth</span></div>`;
options('reference',[...new Set(D.forecasts.map(r=>r.reference_date))].sort().reverse().map(x=>[x,x]));el('reference').value=D.manifest.reference_date;
options('location',D.locations.filter(x=>D.forecasts.some(r=>r.location===x.location)).sort((a,b)=>a.location_name.localeCompare(b.location_name)).map(x=>[x.location,x.location_name]));el('location').value='US';
const modelNames=[...new Set(D.forecasts.map(r=>r.model_id))].sort();
el('models').innerHTML=modelNames.map((m,i)=>`<label><input type="checkbox" value="${esc(m)}" ${m.startsWith('MIGHTE')?'checked':''}> ${esc(m)}</label>`).join('');
const colors={'MIGHTE-Base':'#007f89','MIGHTE-Linear':'#db8744','MIGHTE-Nsemble':'#7555ac','Local-persistence':'#7c8a96','FluSight-baseline':'#a5a08d','FluSight-ensemble':'#ae4861','UMass-flusion':'#4c73b7'};
const rgba=(hex,a)=>`rgba(${parseInt(hex.slice(1,3),16)},${parseInt(hex.slice(3,5),16)},${parseInt(hex.slice(5,7),16)},${a})`;
function plot(){const ref=el('reference').value,target=el('target').value,loc=el('location').value,mult=target.includes('prop')?100:1;
const traces=[],chosen=[...el('models').querySelectorAll('input:checked')].map(x=>x.value);
const begin=new Date(ref);begin.setDate(begin.getDate()-120);const end=new Date(ref);end.setDate(end.getDate()+28);
const truth=D.truth.filter(r=>r.target===target&&r.location===loc&&new Date(r.date)>=begin&&new Date(r.date)<=end&&r.value!==null).sort((a,b)=>a.date.localeCompare(b.date));
traces.push({x:truth.map(r=>r.date),y:truth.map(r=>r.value*mult),name:'Observed · revised',mode:'lines+markers',line:{color:'#253e4c',width:2},marker:{size:4}});
chosen.forEach(model=>{const rows=D.forecasts.filter(r=>r.reference_date===ref&&r.target===target&&r.location===loc&&r.model_id===model).sort((a,b)=>a.horizon-b.horizon);if(!rows.length)return;
const c=colors[model]||'#658091',x=rows.map(r=>r.target_end_date);
[['lo95','hi95',.08],['lo50','hi50',.15]].forEach(([lo,hi,a])=>{traces.push({x,y:rows.map(r=>r[lo]*mult),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip',legendgroup:model});traces.push({x,y:rows.map(r=>r[hi]*mult),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:rgba(c,a),showlegend:false,hoverinfo:'skip',legendgroup:model})});
traces.push({x,y:rows.map(r=>r.median*mult),mode:'lines+markers',name:model,legendgroup:model,line:{color:c,width:2.5},marker:{size:6}})});
Plotly.react('chart',traces,{margin:{t:25,b:45,l:65,r:20},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',color:'#375260'},xaxis:{type:'date',range:[begin.toISOString().slice(0,10),end.toISOString().slice(0,10)],gridcolor:'#edf1f3'},yaxis:{title:{text:mult===100?'Influenza ED visits (%)':'Hospital admissions'},rangemode:'tozero',gridcolor:'#edf1f3'},legend:{orientation:'h',y:1.14},hovermode:'x unified',shapes:[{type:'line',x0:ref,x1:ref,y0:0,y1:1,yref:'paper',line:{color:'#95a7ae',dash:'dot',width:1}}]},{responsive:true,displaylogo:false});
table();}
const mean=rows=>rows.length?rows.reduce((a,b)=>a+b,0)/rows.length:null;
const geom=arr=>arr.length?(arr.includes(0)?0:Math.exp(mean(arr.map(Math.log)))):null;
const key=r=>[r.reference_date,r.target,r.horizon,r.target_end_date,r.location].join('|');
const fmt=(v,n=3)=>v===null||!Number.isFinite(v)?'—':v.toLocaleString(undefined,{maximumFractionDigits:n});
const pct=v=>v===null||!Number.isFinite(v)?'—':(100*v).toFixed(1)+'%';
function table(){let rows=D.scores.filter(r=>r.target===el('target').value);const scope=el('scope').value,h=el('horizon').value,base=el('baseline').value;
rows=rows.filter(r=>(scope==='all'||(scope==='states'?r.location!=='US':r.location===(scope==='selected'?el('location').value:scope)))&&(h==='all'||r.horizon===Number(h)));
if(!rows.length){el('accuracy').innerHTML='<div class="empty">No eligible prospective forecasts have observed truth yet. Accuracy will appear after the season begins and outcomes are reported.</div>';return;}
const names=[...new Set(rows.map(r=>r.model_id))].sort(),groups=Object.fromEntries(names.map(m=>[m,rows.filter(r=>r.model_id===m)])),maps=Object.fromEntries(names.map(m=>[m,new Map(groups[m].map(r=>[key(r),r]))]));
const theta=metric=>Object.fromEntries(names.map(m=>[m,geom(names.map(n=>{let a=0,b=0,count=0;groups[m].forEach(r=>{const other=maps[n].get(key(r));if(other){a+=r[metric];b+=other[metric];count++}});return count&&b>0?a/b:NaN}))]));
const tr=theta('wis'),tl=theta('wis_log1p');
let html='<table><thead><tr>'+['Model','N','Matched','Mean WIS','Geo WIS','Geo log-WIS','MAE','Rel WIS','Rel log-WIS','WIS skill','50% coverage','80% coverage','95% coverage'].map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
names.forEach(m=>{const g=groups[m],bm=maps[base]||new Map();let a=0,b=0,n=0;g.forEach(r=>{const br=bm.get(key(r));if(br){a+=r.wis;b+=br.wis;n++}});const skill=n&&b>0?1-a/b:null;const cells=[esc(m),g.length,n,fmt(mean(g.map(r=>r.wis))),fmt(geom(g.map(r=>r.wis))),fmt(geom(g.map(r=>r.wis_log1p)),5),fmt(mean(g.map(r=>r.ae))),fmt(tr[base]>0?tr[m]/tr[base]:null),fmt(tl[base]>0?tl[m]/tl[base]:null),pct(skill),pct(mean(g.map(r=>r.coverage_50))),pct(mean(g.map(r=>r.coverage_80))),pct(mean(g.map(r=>r.coverage_95)))];html+='<tr>'+cells.map((x,i)=>`<td class="${i===9&&skill!==null?(skill>=0?'positive':'negative'):''}">${x}</td>`).join('')+'</tr>'});
el('accuracy').innerHTML=html+'</tbody></table>';}
el('provenance').textContent=`Forecast inputs: ${D.manifest.snapshot_id}. Scoring truth: ${D.truth_snapshot}. Report: ${new Date(D.generated_at).toLocaleString()}. Run: ${D.manifest.run_id}.`;
['audit','benchmarks','manifest'].forEach(id=>el(id).textContent=JSON.stringify(id==='audit'?D.data_audit:id==='benchmarks'?D.benchmarks:D.manifest,null,2));
['reference','target','location'].forEach(id=>el(id).addEventListener('change',plot));el('models').addEventListener('change',plot);
['baseline','scope','horizon'].forEach(id=>el(id).addEventListener('change',table));plot();
</script></body></html>'''
