"""Download only public, read-only evidence into this isolated diagnostic folder."""
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/official_submissions/2026-10-17/20261015T043845714774Z"
summary = {}
for model in ["MIGHTE-Linear", "MIGHTE-Nsemble", "MIGHTE-Base"]:
    url = ("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/53e86e2/"
           f"model-output/{model}/2026-10-17-{model}.csv")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    (OUT / f"merged-{model}.csv").write_bytes(response.content)
    local = RUN / f"model-output/{model}/2026-10-17-{model}.csv"
    summary[model] = {"url": url, "sha256": hashlib.sha256(response.content).hexdigest(),
                      "byte_identical": local.read_bytes() == response.content}
response = requests.get("https://www.meyerlab.io/flusight_2026/", timeout=30)
response.raise_for_status()
local = ROOT / "reports/official_submissions/2026-10-17/20261015T043845714774Z/index.html"
summary["live_review"] = {"url": response.url, "sha256": hashlib.sha256(response.content).hexdigest(),
                          "bytes": len(response.content), "byte_identical": local.read_bytes() == response.content}
(OUT / "remote_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
