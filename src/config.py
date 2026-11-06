"""共用設定：路徑、setup.yaml、切分日期、holdout 封存檢查、執行紀錄與研究帳本。

所有階段程式都從 repo 根目錄執行（python -m src.xxx），讀同一份 setup.yaml。
SMOKE=1 時紀錄寫到 results/cache/（程式除錯用，不算試驗、不進正式帳本）。
"""
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "setup.yaml", encoding="utf-8"))
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = ROOT / "results"
FIG = OUT / "figures"
CACHE = OUT / "cache"
for p in (RAW, OUT, FIG, CACHE):
    p.mkdir(parents=True, exist_ok=True)

P = CFG["evaluation_protocol"]
DATA_START = pd.Timestamp(P["data_start"])
TRAIN_START = pd.Timestamp(P["train_start"])
VAL_START, VAL_END = (pd.Timestamp(x) for x in P["validation"])
HO_START, HO_END = (pd.Timestamp(x) for x in P["holdout"])
HORIZON = CFG["label"]["horizon"]
PURGE = P["purge_days"]
FOLD = P["fold_length"]
TD = 252
SMOKE = os.environ.get("SMOKE") == "1"
SEAL = OUT / "HOLDOUT_TOUCHED.json"


def git_commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return (h or "no-commit") + ("-dirty" if dirty else "")
    except OSError:
        return "no-git"


def snapshot(path: Path) -> str:
    """資料檔 sha256 前 12 碼（依大小與修改時間快取）。"""
    if not path.exists():
        return "missing"
    key = f"{path.name}:{path.stat().st_size}:{int(path.stat().st_mtime)}"
    f = CACHE / "snapshots.json"
    known = json.loads(f.read_text()) if f.exists() else {}
    if key not in known:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        known[key] = h.hexdigest()[:12]
        f.write_text(json.dumps(known, indent=1))
    return f"{path.name}@{known[key]}"


def _num(v):
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v)


def _js(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=_num)


def log_run(stage: str, config, window: str, metrics, decision: str = "", gate: str = "") -> str:
    run_id = f"{stage}-{datetime.now():%Y%m%d%H%M%S%f}"[:-3]
    f = (CACHE / "smoke_run_log.csv") if SMOKE else (ROOT / "run_log.csv")
    with open(f, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([run_id, f"{datetime.now():%Y-%m-%d %H:%M:%S}", stage, git_commit(),
                                 snapshot(DATA / "prices.parquet"), CFG["setup_version"], _js(config), window,
                                 _js(metrics), decision, gate])
    return run_id


def ledger(family: str, trial: str, run_id: str, choice: str, window: str, metric: str, value,
           decision: str, gate: str = "", notes: str = ""):
    f = (CACHE / "smoke_ledger.csv") if SMOKE else (ROOT / "research_ledger.csv")
    with open(f, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([CFG["strategy"], family, trial, run_id, f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                                 choice, window, metric, "" if value is None else round(float(value), 6),
                                 decision, gate, notes])


def save(df: pd.DataFrame, name: str, index: bool = False, floatfmt: str = ".3f"):
    df.to_csv(OUT / f"{name}.csv", index=index, encoding="utf-8-sig")
    (OUT / f"{name}.md").write_text(df.to_markdown(index=index, floatfmt=floatfmt), encoding="utf-8")


def dev_end() -> pd.Timestamp:
    """開發階段可讀到的最後日期。holdout 執行前一律截在驗證期結束。"""
    return VAL_END
