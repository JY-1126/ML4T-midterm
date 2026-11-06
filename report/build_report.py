"""產生期中報告：report/proposal.html（圖片內嵌、可離線開啟）→ report/proposal.pdf（Edge 無頭列印）。

所有數字由 results/ 讀出；敘述文字在 report/narrative.py。
用法：python report/build_report.py
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIGS = R / "figures"
sys.path.insert(0, str(ROOT / "report"))
CFG = yaml.safe_load(open(ROOT / "setup.yaml", encoding="utf-8"))
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]


def csv(name, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(R / f"{name}.csv", encoding="utf-8-sig", **kw)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame({"說明": [f"（缺 {name}）"]})


def kv(name) -> dict:
    out = {}
    for k, v in pd.read_csv(R / f"{name}.csv", encoding="utf-8-sig", index_col=0)["值"].items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def fmt(v, digits=3):
    if isinstance(v, (bool, np.bool_)):
        return "✔" if v else "✘"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        if np.isinf(v):
            return "∞"
        if float(v).is_integer() and abs(v) < 1e6:
            return f"{int(v):,}" if abs(v) >= 1e4 else f"{int(v)}"
        return f"{v:,.{digits}f}"
    if isinstance(v, str) and v in ("True", "False"):
        return "✔" if v == "True" else "✘"
    return "—" if v is None or (isinstance(v, str) and v == "nan") else str(v)


def tbl(df, cols=None, digits=3, pct=(), small=False, rename=None) -> str:
    df = df[cols] if cols else df
    head = "".join(f"<th>{(rename or {}).get(c, c)}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            s = f"{v:.1%}" if c in pct and isinstance(v, (float, np.floating)) and np.isfinite(v) else fmt(v, digits)
            cells.append(f"<td>{s}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{"small" if small else ""}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def fig(name, caption, width="100%") -> str:
    p = FIGS / name
    if not p.exists():
        return f"<p class='miss'>（缺圖 {name}）</p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


CSS = """
@page { size: A4; margin: 15mm 14mm 15mm 14mm; }
:root { --ink:#1b1b1b; --accent:#1c7ed6; --muted:#555; --bg:#fff; }
body { font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif; font-size: 9.6pt; line-height: 1.5; color: var(--ink);
       max-width: 900px; margin: 0 auto; padding: 0 16px; background: var(--bg); }
h1 { font-size: 18pt; margin: 0 0 4px; } h2 { font-size: 13pt; border-bottom: 2px solid var(--accent); padding-bottom: 2px; margin: 20px 0 8px; break-after: avoid; }
h3 { font-size: 10.8pt; margin: 12px 0 4px; break-after: avoid; }
.meta { color: var(--muted); font-size: 9pt; } .abstract { background: #f1f3f5; padding: 8px 12px; border-left: 4px solid var(--accent); }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.2pt; break-inside: avoid; }
table.small { font-size: 7.4pt; } th, td { border: 1px solid #ced4da; padding: 2px 5px; text-align: left; vertical-align: top; }
th { background: #e7f5ff; } figure { margin: 8px 0; break-inside: avoid; text-align: center; } img { max-width: 100%; }
figcaption { font-size: 8.4pt; color: #444; text-align: left; }
.wcb { background: #fff4e6; border-left: 4px solid #f08c00; padding: 6px 10px; margin: 10px 0 4px; font-size: 9.2pt; break-inside: avoid; }
.wcb b { color: #d9480f; } .key { background: #ebfbee; border-left: 4px solid #2b8a3e; padding: 6px 10px; } .miss { color: #c92a2a; }
code { font-size: 8.6pt; background: #f1f3f5; padding: 0 3px; } .pb { break-before: page; }
"""


def build():
    from narrative import sections
    body = sections(csv=csv, kv=kv, tbl=tbl, fig=fig, R=R, CFG=CFG)
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ML4T 期中報告</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    out = ROOT / "report" / "proposal.html"
    out.write_text(html, encoding="utf-8")
    edge = next((e for e in EDGE if e.exists()), None)
    if edge:
        pdf = ROOT / "report" / "proposal.pdf"
        subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", out.as_uri()], check=False, timeout=180,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("PDF:", pdf, pdf.exists())
    print("HTML:", out)


if __name__ == "__main__":
    build()
