"""產生簡報：report/slides.html → report/slides.pdf（16:9，12 頁，Edge 無頭列印）。

內容：機制、設定、資料、triage、模型、回測與成本、總表、3 張必附圖（權益曲線、毛→淨、回撤）、診斷、下一步。
數字由 results/ 讀出；文字在 report/narrative.py 的 SLIDES。
用法：python report/build_slides.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))
from build_report import EDGE, csv, fig, kv, tbl, CFG, R  # noqa: E402

CSS = """
@page { size: 338.7mm 190.5mm; margin: 0; }
:root { --ink:#1b1b1b; --accent:#1c7ed6; --muted:#555; --bg:#fff; }
body { margin: 0; font-family: "Microsoft JhengHei", sans-serif; color: var(--ink); background: var(--bg); }
.s { width: 338.7mm; height: 190.5mm; box-sizing: border-box; padding: 11mm 15mm; break-after: page; overflow: hidden; position: relative; }
.s h1 { font-size: 25pt; margin: 0 0 5mm; color: var(--accent); } .s h2 { font-size: 20pt; margin: 0 0 4mm; border-bottom: 3px solid var(--accent); }
.s p, .s li { font-size: 13.5pt; line-height: 1.45; } .s ul { margin: 2mm 0; padding-left: 7mm; }
.s table { border-collapse: collapse; width: 100%; font-size: 10.5pt; margin-top: 2mm; }
.s th, .s td { border: 1px solid #ced4da; padding: 2px 6px; text-align: left; vertical-align: top; } .s th { background: #e7f5ff; }
.s figure { margin: 0; text-align: center; } .s img { max-height: 112mm; max-width: 100%; }
.s figcaption { font-size: 10.5pt; color: #444; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 8mm; } .key { background: #ebfbee; border-left: 5px solid #2b8a3e; padding: 3mm 5mm; font-size: 14pt; }
.n { position: absolute; right: 12mm; bottom: 7mm; color: #868e96; font-size: 10pt; }
"""


def build():
    from narrative import slides
    pages = slides(csv=csv, kv=kv, tbl=tbl, fig=fig, R=R, CFG=CFG)
    body = "".join(f'<section class="s">{p}<div class="n">{i}/{len(pages)}</div></section>' for i, p in enumerate(pages, 1))
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ML4T 期中簡報</title><style>{CSS}</style></head><body>{body}</body></html>"""
    out = ROOT / "report" / "slides.html"
    out.write_text(html, encoding="utf-8")
    edge = next((e for e in EDGE if e.exists()), None)
    if edge:
        pdf = ROOT / "report" / "slides.pdf"
        subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", out.as_uri()], check=False, timeout=180,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("PDF:", pdf, pdf.exists())
    print(f"{len(pages)} slides →", out)


if __name__ == "__main__":
    build()
