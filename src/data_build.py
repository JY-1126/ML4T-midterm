"""把 data/raw 加工成時間點正確的價格表（不含任何特徵）。

輸出（Parquet）
  data/prices.parquet  date, stock_id, open, high, low, close（還原權息）, close_raw（未還原股利）, volume, source
  data/meta.csv        每檔：來源、市場、首末交易日、下市日、是否代號重複使用
  data/bench.parquet   date, tri（加權股價報酬指數）, etf0050（0050 還原權息）
  results/02_build_log.json  每一步刪除／調整的筆數（資料品質報告用）
規則
  1. yfinance：open/high/low/close 乘上 Adj Close / Close（股利還原；Close 已還原分割），close_raw = Close
  2. FinMind 下市股：除權息日前的價格乘上 reference_price / before_price（同為等比例還原）
  3. 代號重複使用：下市股代號若也在現存名單，下市股改名為「代號_d年份」，各自只保留生效期間
  4. 刪除：成交量為 0（沒有成交的 K 棒，不補值）、任一 OHLC <= 0、(日期, 代號) 重複
     high / low 擴到開收盤範圍；殘缺日（當日檔數 < 前 60 日中位數一半）整日刪除
  5. 不可能的跳動（超過漲跌幅，停牌期間依天數累積放寬，新股前 5 日不限）視為減資等非經濟事件，
     以等比例調整「之前」的價格抹平；跳動日以後的價格維持實際成交價
時間：台灣收盤 13:30；所有日期為交易日（Asia/Taipei），無時區轉換。
"""
import json

import numpy as np
import pandas as pd

from src.config import DATA, OUT, RAW

LIMIT_CHANGE = pd.Timestamp("2015-06-01")
PX = ["open", "high", "low", "close"]


def _yf() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "yf").glob("*.parquet"))], ignore_index=True)
    df = df.drop_duplicates(["date", "stock_id"], keep="last")
    f = df.adj_close / df.close
    df["close_raw"] = df.close
    for c in PX:
        df[c] = df[c] * f
    df["source"] = "yfinance"
    return df[["date", "stock_id"] + PX + ["close_raw", "volume", "source"]]


def adjust_finmind(price: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    p = price.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    p["date"] = pd.to_datetime(p.date)
    p = p[["date", "stock_id"] + PX + ["volume"]].sort_values("date").reset_index(drop=True)
    p["close_raw"] = p.close
    f = np.ones(len(p))
    if len(div):
        div = div[(div.before_price > 0) & (div.reference_price > 0)]
        for ex, ratio in zip(pd.to_datetime(div.date), div.reference_price / div.before_price):
            f[(p.date < ex).values] *= ratio
    for c in PX:
        p[c] = p[c].values * f
    return p


def _finmind(listed_ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    lst = pd.read_csv(RAW / "delisted_list.csv", dtype=str)
    parts, meta = [], []
    for r in lst.itertuples():
        fp, fd = RAW / "finmind" / f"{r.stock_id}_price.parquet", RAW / "finmind" / f"{r.stock_id}_div.parquet"
        if not fp.exists():
            meta.append((r.stock_id, r.stock_id, r.delist_date, False, "未下載"))
            continue
        price = pd.read_parquet(fp)
        if price.empty:
            meta.append((r.stock_id, r.stock_id, r.delist_date, False, "無價格"))
            continue
        a = adjust_finmind(price, pd.read_parquet(fd) if fd.exists() else pd.DataFrame())
        a = a[a.date <= pd.Timestamp(r.delist_date)]
        reused = r.stock_id in listed_ids
        uid = f"{r.stock_id}_d{r.delist_date[:4]}" if reused else r.stock_id
        a["stock_id"] = uid
        a["source"] = "finmind"
        parts.append(a)
        meta.append((uid, r.stock_id, r.delist_date, reused, "ok"))
    m = pd.DataFrame(meta, columns=["stock_id", "code", "delist_date", "code_reused", "status"])
    return pd.concat(parts, ignore_index=True), m


def clean_jumps(close: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """每格價格調整係數（只調整跳動日之前）與被抹平的跳動數。"""
    lim = np.where(close.index < LIMIT_CHANGE, 0.07, 0.10)[:, None]
    valid = close.notna().values
    prev = close.ffill().shift(1).values
    pos = np.arange(len(close), dtype=float)[:, None].repeat(close.shape[1], 1)
    last = pd.DataFrame(np.where(valid, pos, np.nan)).ffill().shift(1).values
    nd = np.clip(pos - last, 1, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        up, dn = (1 + lim) ** nd - 1 + 0.005, 1 - (1 - lim) ** nd + 0.005
        ratio = close.values / prev
        bad = valid & ~np.isnan(prev) & (valid.cumsum(0) > 5) & ((ratio - 1 > up) | (1 - ratio > dn))
    jr = pd.DataFrame(np.where(bad, ratio, 1.0), index=close.index, columns=close.columns)
    return jr.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0), int(bad.sum())


def bench() -> pd.DataFrame:
    tri = pd.read_parquet(RAW / "finmind" / "TAIEX_tri.parquet")
    tri = tri.assign(date=pd.to_datetime(tri.date)).set_index("date").price.rename("tri")
    e = adjust_finmind(pd.read_parquet(RAW / "finmind" / "0050_price.parquet"),
                       pd.read_parquet(RAW / "finmind" / "0050_div.parquet")).set_index("date")
    e = e[e.volume > 0]
    fac, _ = clean_jumps(e[["close"]])  # 2025-06 的 1 拆 4 分割（FinMind 未還原）
    out = pd.concat([tri, (e.close * fac.close).rename("etf0050")], axis=1).sort_index()
    out.index.name = "date"
    return out.reset_index()


def build():
    log = {}
    y = _yf()
    listed = pd.read_csv(RAW / "listed.csv", dtype=str)
    f, fmeta = _finmind(set(listed.stock_id))
    # 代號重複使用：現存股 yfinance 價格只保留舊公司下市之後
    for r in fmeta[fmeta.code_reused].itertuples():
        y = y[~((y.stock_id == r.code) & (y.date <= pd.Timestamp(r.delist_date)))]
    p = pd.concat([y, f], ignore_index=True)
    p["volume"] = p.volume.astype(float)
    log["原始筆數"] = p.groupby("source").size().to_dict()
    log["成交量為 0（刪除）"] = p[p.volume <= 0].groupby("source").size().to_dict()
    p = p[p.volume > 0]
    nonpos = (p[PX] <= 0).any(axis=1) | (p.close_raw <= 0) | p[PX].isna().any(axis=1)
    log["OHLC <= 0 或缺值（刪除）"] = p[nonpos].groupby("source").size().to_dict()
    p = p[~nonpos]
    log["high < max(open, close)"] = p[p.high < p[["open", "close"]].max(axis=1) - 1e-9].groupby("source").size().to_dict()
    log["low > min(open, close)"] = p[p.low > p[["open", "close"]].min(axis=1) + 1e-9].groupby("source").size().to_dict()
    p["high"], p["low"] = p[PX].max(axis=1), p[PX].min(axis=1)
    dup = p.duplicated(["date", "stock_id"])
    log["(日期, 代號) 重複（刪除）"] = int(dup.sum())
    p = p[~dup]
    cnt = p.groupby("date").size()
    ref = cnt.rolling(60, min_periods=5).median().shift(1).fillna(cnt)
    bad_days = cnt.index[cnt < 0.5 * ref]
    log["殘缺日（整日刪除）"] = [str(d.date()) for d in bad_days]
    p = p[~p.date.isin(bad_days)]

    close = p.pivot(index="date", columns="stock_id", values="close").sort_index()
    fac, n_jump = clean_jumps(close)
    log["超過漲跌幅的跳動（等比例抹平）"] = n_jump
    fl = fac.stack().rename("fac").reset_index()
    p = p.merge(fl, on=["date", "stock_id"], how="left")
    for c in PX:
        p[c] = p[c] * p.fac
    p = p.drop(columns="fac").sort_values(["stock_id", "date"]).reset_index(drop=True)
    p.to_parquet(DATA / "prices.parquet", index=False)

    g = p.groupby("stock_id")
    meta = pd.DataFrame({"source": g.source.first(), "first_trade": g.date.min(), "last_trade": g.date.max(),
                         "n_days": g.size()})
    meta = meta.join(listed.set_index("stock_id")[["market", "name"]])
    meta = meta.join(fmeta.set_index("stock_id")[["code", "delist_date", "code_reused", "status"]], how="outer")
    meta.index.name = "stock_id"
    meta.to_csv(DATA / "meta.csv", encoding="utf-8-sig")
    bench().to_parquet(DATA / "bench.parquet", index=False)
    log["期間"] = f"{p.date.min().date()} ~ {p.date.max().date()}"
    log["股票數"] = int(p.stock_id.nunique())
    log["下市股（有價格）"] = int((meta.source == "finmind").sum())
    (OUT / "02_build_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    build()
