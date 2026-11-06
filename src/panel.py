"""讀取價格成寬表、建立每日可投資池與基準。

load(end) 只讀到 end（含）。開發階段一律 end = 驗證期結束（2024-10-31），holdout 的價格讀不到；
只有 src/s07_holdout.py 會以 holdout 結束日呼叫（並寫入封存紀錄）。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CFG, DATA, DATA_START, VAL_END

U = CFG["universe"]["eligibility"]


@dataclass
class Data:
    dates: pd.DatetimeIndex
    stocks: pd.Index
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    close_raw: pd.DataFrame
    volume: pd.DataFrame
    univ: pd.DataFrame
    adv20: pd.DataFrame
    last_trade: pd.Series      # 下市股的最後交易日（現存股為 NaT）
    bench: pd.DataFrame        # tri, etf0050（以交易日對齊、累積報酬指數）
    end: pd.Timestamp


def load(end=VAL_END) -> Data:
    end = pd.Timestamp(end)
    p = pd.read_parquet(DATA / "prices.parquet")
    p = p[(p.date >= DATA_START) & (p.date <= end)]
    w = {c: p.pivot(index="date", columns="stock_id", values=c).sort_index()
         for c in ["open", "high", "low", "close", "close_raw", "volume"]}
    close = w["close"]
    dates, stocks = close.index, close.columns
    adv20 = (w["close_raw"] * w["volume"]).rolling(20, min_periods=15).mean()
    age = close.notna().cumsum()
    univ = (close.notna() & (w["close_raw"] >= U["min_raw_close_twd"]) & (adv20 >= U["min_adv20_twd"])
            & (age >= U["min_age_days"]))
    meta = pd.read_csv(DATA / "meta.csv", dtype={"stock_id": str}, parse_dates=["last_trade"]).set_index("stock_id")
    last = meta.last_trade.where(meta.source == "finmind").reindex(stocks)
    last = last.where(last <= end)  # 在 end 之後才下市的股票，以 end 的角度看仍在交易
    b = pd.read_parquet(DATA / "bench.parquet").set_index("date").reindex(dates).ffill()
    b = b / b.iloc[0]
    return Data(dates, stocks, w["open"], w["high"], w["low"], close, w["close_raw"], w["volume"],
                univ, adv20, last, b, end)


def exit_open(d: Data) -> pd.DataFrame:
    """可出場價格：下市股在最後交易日之後，以最後收盤價代替開盤價（標籤與回測共用）。"""
    o = d.open.copy()
    for s in d.last_trade.dropna().index:
        after = d.dates > d.last_trade[s]
        lc = d.close.loc[:d.last_trade[s], s].dropna()
        if after.any() and len(lc):
            o.loc[after, s] = lc.iloc[-1]
    return o
