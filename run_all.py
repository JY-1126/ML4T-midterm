"""從 repo 根目錄重現全部結果。

python run_all.py --fetch            下載原始資料（yfinance 約 10 分鐘；FinMind 免費額度約 2 小時）
python run_all.py                    s02 資料 → s01 切分日期 → s03 特徵/triage → s04 模型 → s05 回測 → s06 風險
python run_all.py --holdout          再加 s07 holdout 與 s08 事後診斷（已封存時自動以「reproduction run」為理由重跑，不改任何設定）
"""
import argparse
import importlib
import json

from src.config import SEAL

ap = argparse.ArgumentParser()
ap.add_argument("--fetch", action="store_true")
ap.add_argument("--holdout", action="store_true")
a = ap.parse_args()
if a.fetch:
    from src import data_fetch
    data_fetch.fetch_yf()
    data_fetch.fetch_finmind()
for m in ("s02_data", "s01_setup", "s03_features", "s04_models", "s05_backtest", "s06_risk"):
    print(f"==== {m}", flush=True)
    importlib.import_module(f"src.{m}").main()
if a.holdout:
    from src import s07_holdout
    s07_holdout.main("reproduction run" if SEAL.exists() and json.loads(SEAL.read_text(encoding="utf-8")) else None)
    from src import s08_diagnosis  # holdout 之後的診斷（只解釋、不改設定）
    s08_diagnosis.main()
