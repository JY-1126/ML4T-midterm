"""下載原始資料（只下載、不加工）。加工見 src/data_build.py。

來源
  現存股名單  TWSE OpenAPI STOCK_DAY_ALL、TPEx OpenAPI 上櫃收盤行情 → data/raw/listed.csv
  現存股日 K  yfinance（.TW / .TWO），auto_adjust=False：Close 已還原分割、Adj Close 另還原股利
              → data/raw/yf/*.parquet（每批暫存，中斷可續跑）
  下市股      FinMind TaiwanStockDelisting（名單）、TaiwanStockPrice（未還原日 K）、
              TaiwanStockDividendResult（除權息前後參考價）→ data/raw/finmind/
  基準        FinMind 0050 日 K + 除權息；TAIEX 發行量加權股價報酬指數（TaiwanStockTotalReturnIndex）
FinMind 免費額度約每小時 300 次；被限制時安靜等 65 分鐘再試（提早重試會延長封鎖）。
可設環境變數 FINMIND_TOKEN 提高額度。

用法（repo 根目錄）：python -m src.data_fetch [--only yf|finmind]
"""
import argparse
import os
import re
import time
from datetime import date

import pandas as pd
import requests
import yfinance as yf

from src.config import RAW

API = "https://api.finmindtrade.com/api/v4/data"
START = "2005-01-01"
END = date.today().isoformat()  # yfinance 的 end 不含當天，避免盤中 K 棒
HEADERS = {"User-Agent": "Mozilla/5.0"}
YF, FM = RAW / "yf", RAW / "finmind"


def _json(url):
    for i in range(5):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def listed() -> pd.DataFrame:
    f = RAW / "listed.csv"
    if f.exists():
        return pd.read_csv(f, dtype=str)
    twse = _json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL")
    tpex = _json("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes")
    rows = [(d["Code"], "TWSE", ".TW", d["Name"]) for d in twse]
    rows += [(d["SecuritiesCompanyCode"], "TPEx", ".TWO", d["CompanyName"]) for d in tpex]
    m = pd.DataFrame(rows, columns=["stock_id", "market", "suffix", "name"]).drop_duplicates("stock_id")
    m = m[m.stock_id.str.fullmatch(r"[1-9]\d{3}") & ~m.stock_id.str.startswith("91")]
    m["download_date"] = date.today().isoformat()
    m.to_csv(f, index=False, encoding="utf-8-sig")
    return m


def _yf_batch(tickers: list[str], sid_of: dict) -> pd.DataFrame:
    raw = yf.download(tickers, start=START, end=END, auto_adjust=False, group_by="ticker",
                      threads=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    raw.columns = raw.columns.set_names(["Ticker", "Price"])
    df = raw.stack(level=0, future_stack=True).reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["stock_id"] = df.ticker.map(sid_of)
    df = df.dropna(subset=["close", "adj_close"])
    df["date"] = pd.to_datetime(df.date).dt.tz_localize(None)
    return df[["date", "stock_id", "open", "high", "low", "close", "adj_close", "volume"]]


def fetch_yf():
    YF.mkdir(parents=True, exist_ok=True)
    m = listed()
    sid_of = dict(zip(m.stock_id + m.suffix, m.stock_id))
    syms = sorted(sid_of)
    print(f"現存普通股 {len(syms)} 檔", flush=True)
    for i in range(0, len(syms), 80):
        f = YF / f"b{i:05d}.parquet"
        if not f.exists():
            _yf_batch(syms[i:i + 80], sid_of).to_parquet(f, index=False)
            print(f"  yfinance {min(i + 80, len(syms))}/{len(syms)}", flush=True)
    for attempt in range(3):  # yfinance 常因限流誤報 possibly delisted：缺的以小批次重試
        got = set(pd.concat([pd.read_parquet(f, columns=["stock_id"]) for f in YF.glob("*.parquet")]).stock_id)
        miss = [s for s in syms if sid_of[s] not in got]
        if not miss:
            break
        print(f"  重試 {attempt + 1}：{len(miss)} 檔", flush=True)
        time.sleep(15)
        for i in range(0, len(miss), 20):
            df = _yf_batch(miss[i:i + 20], sid_of)
            if len(df):
                df.to_parquet(YF / f"r{attempt}_{i:05d}.parquet", index=False)


def fm(dataset: str, **params) -> list[dict]:
    tok = os.environ.get("FINMIND_TOKEN")
    for i in range(30):
        try:
            j = requests.get(API, params={"dataset": dataset, **params}, timeout=60,
                             headers={"Authorization": f"Bearer {tok}"} if tok else None).json()
        except (requests.RequestException, ValueError):
            time.sleep(10 * (i + 1))
            continue
        if j.get("msg") == "success":
            return j.get("data", [])
        print(f"  FinMind {dataset} {params.get('data_id', '')}: {j.get('msg')} → 等 65 分鐘", flush=True)
        time.sleep(3900)
    raise RuntimeError(f"{dataset} {params}")


def _fm_save(f, dataset, **params):
    if not f.exists():
        pd.DataFrame(fm(dataset, **params)).to_parquet(f, index=False)
        time.sleep(1.5)


def fetch_finmind():
    FM.mkdir(parents=True, exist_ok=True)
    end = END
    _fm_save(FM / "TAIEX_tri.parquet", "TaiwanStockTotalReturnIndex", data_id="TAIEX", start_date=START, end_date=end)
    _fm_save(FM / "0050_price.parquet", "TaiwanStockPrice", data_id="0050", start_date=START, end_date=end)
    _fm_save(FM / "0050_div.parquet", "TaiwanStockDividendResult", data_id="0050", start_date=START, end_date=end)
    f = RAW / "delisted_list.csv"
    if f.exists():
        lst = pd.read_csv(f, dtype=str)
    else:
        d = pd.DataFrame(fm("TaiwanStockDelisting"))
        d = d[d.stock_id.str.fullmatch(r"[1-9]\d{3}") & ~d.stock_id.str.startswith("91")
              & ~d.stock_name.str.contains("DR", na=False) & (d.date >= "2006-01-01")]
        lst = (d.sort_values("date").drop_duplicates("stock_id", keep="last")
               .rename(columns={"date": "delist_date"})[["stock_id", "stock_name", "delist_date"]])
        lst["download_date"] = date.today().isoformat()
        lst.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"下市普通股 {len(lst)} 檔", flush=True)
    for k, r in enumerate(lst.itertuples(), 1):
        stop = (pd.Timestamp(r.delist_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        _fm_save(FM / f"{r.stock_id}_price.parquet", "TaiwanStockPrice", data_id=r.stock_id, start_date=START, end_date=stop)
        _fm_save(FM / f"{r.stock_id}_div.parquet", "TaiwanStockDividendResult", data_id=r.stock_id, start_date=START, end_date=stop)
        if k % 20 == 0:
            print(f"  FinMind {k}/{len(lst)}", flush=True)
    print("FinMind 完成", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["yf", "finmind"])
    a = ap.parse_args()
    if a.only != "finmind":
        fetch_yf()
    if a.only != "yf":
        fetch_finmind()
