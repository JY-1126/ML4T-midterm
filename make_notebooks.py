"""產生並執行 notebooks/01_setup … 07_results.ipynb（每階段一本，從 repo 根目錄執行）。

每本 notebook 預設 RUN = False：只讀 results/ 已存的表格與圖；改成 True 會重新執行該階段。
07_results 在 RUN = True 時，若 holdout 已封存，會以「reproduction run」為理由重跑（不改任何設定）。
用法：python make_notebooks.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

HEAD = """import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)
R = ROOT / "results"
def show(name, n=60):
    p = R / f"{name}.csv"
    display(Markdown(f"**{name}**")); display(pd.read_csv(p, encoding="utf-8-sig").head(n)) if p.exists() else print("缺", name)
def fig(name, w=800):
    p = R / "figures" / name
    display(Image(filename=str(p), width=w)) if p.exists() else print("缺圖", name)"""

STAGES = [
    ("01_setup", "交易設定 v1 與驗證協定", "s01_setup",
     'print((ROOT / "setup.yaml").read_text(encoding="utf-8")[:4000])\nshow("01_protocol"); show("01_splits")'),
    ("02_data", "資料：時間點檢查與品質報告", "s02_data",
     'show("02_summary"); show("02_pit_checklist"); show("02_coverage"); show("02_survivorship"); show("02_ohlc"); '
     'show("02_missing")\nfig("02_coverage.png"); fig("02_corporate_action.png")'),
    ("03_features", "標籤、特徵與 triage", "s03_features",
     'show("03_triage"); show("03_truncation"); show("03_mechanism"); show("03_regime")\n'
     'print(json.loads((R / "03_model_features.json").read_text(encoding="utf-8")))\nfig("03_triage_ic.png"); fig("03_dag.png")'),
    ("04_models", "模型階梯、驗證、SHAP 與 conformal", "s04_models",
     'show("04_ladder"); show("04_paired"); show("04_ridge_alpha"); show("04_lgbm_grid"); show("04_shap"); show("04_conformal")\n'
     'print(json.loads((R / "04_deployed.json").read_text(encoding="utf-8")))\nfig("04_fold_ic.png"); fig("04_rolling_ic.png"); fig("04_shap.png", 650)'),
    ("05_backtest", "回測、成本、投組配置與覆蓋（驗證期）", "s05_backtest",
     'for t in ("main", "cost_sweep", "breakeven", "phases", "allocator", "overlay", "realism"): show(f"05_val_{t}")\nfig("05_val_cost.png")'),
    ("06_risk", "風險清單（驗證期）", "s06_risk",
     'for t in ("var", "regime", "risk_summary", "stress", "killswitch"): show(f"06_val_{t}")\nshow("06_risk_matrix")\n'
     'fig("06_val_regime.png"); fig("06_val_drawdown.png")'),
    ("07_results", "Holdout（只跑一次）、總表、推論與診斷", "s07_holdout",
     'print(json.loads((R / "HOLDOUT_TOUCHED.json").read_text(encoding="utf-8")))\nshow("07_headline")\n'
     'print((R / "07_pattern.txt").read_text(encoding="utf-8"))\n'
     'for t in ("ic", "inference", "attrition", "drift", "regime_mix", "psi", "yearly", "ho_main", "ho_breakeven", "ho_phases", '
     '"ho_stress", "ho_killswitch"): show(f"07_{t}")\n'
     'show("08_feature_ic"); show("08_holdout_regime_ic")\n'
     'fig("07_equity.png"); fig("07_rolling_ic.png"); fig("07_ho_cost.png"); fig("07_ho_regime.png"); fig("04_shap.png", 650); '
     'fig("07_ho_drawdown.png")'),
]


def make(name, title, module, body, execute=True):
    nb = nbf.v4.new_notebook()
    run = (f'if RUN:\n    from src import {module}\n    {module}.main("reproduction run" if (R / "HOLDOUT_TOUCHED.json").exists() else None)'
           if module == "s07_holdout" else f"if RUN:\n    from src import {module}\n    {module}.main()")
    nb.cells = [nbf.v4.new_markdown_cell(f"# {name}：{title}\n\n設定與協定見 `setup.yaml`；程式在 `src/{module}.py`。"
                                         "預設 `RUN = False` 只讀已存的結果。"),
                nbf.v4.new_code_cell(HEAD), nbf.v4.new_code_cell("RUN = False\n" + run), nbf.v4.new_code_cell(body)]
    path = NB / f"{name}.ipynb"
    if execute:
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    for s in STAGES:
        make(*s)
