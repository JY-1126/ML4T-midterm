"""事件驅動回測。排名緩衝、現金、漲跌停與下市都會影響下一步動作，所以不用向量化。

時點  t 日收盤分數 → t+1 開盤成交；持股以開盤、收盤逐日評價
進場  檢視日：空位由前 n 名中未持有者依序補上；ew 每檔 1/n，invvol 依 60 日波動倒數（上限 10%）
出場  檢視日排名 > exit_mult × n 或不在可投資池（分數為缺值）→ 隔日開盤賣出
限制  開盤漲停不買（最多重試 5 日）、開盤跌停不賣（隔日重試）；當日沒有 K 棒不成交
下市  最後交易日之後仍持有 → 以最後收盤價 × (1 − haircut) 出清
成本  每邊（手續費 + 半價差 × spread_mult）× cost_mult；賣出另加證交稅 × cost_mult
所有限制都在模擬迴圈內、以當時可得的資訊檢查。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CFG

CM = CFG["cost_model"]
LIMIT_CHANGE = pd.Timestamp("2015-06-01")


@dataclass(frozen=True)
class Spec:
    n: int = 20
    exit_mult: float = 3.0
    review: int = CFG["decision_schedule"]["review_every_days"]
    allocator: str = "ew"
    cost_mult: float = 1.0
    spread_mult: float = 1.0
    haircut: float = 0.0
    limit_rule: bool = True
    phase: int = 0


def costs(spec: Spec) -> tuple[float, float]:
    leg = (CM["commission_bps"] + CM["half_spread_bps"] * spec.spread_mult) / 1e4
    return leg * spec.cost_mult, (leg + CM["sell_tax_bps"] / 1e4) * spec.cost_mult


_VOL: dict = {}


def simulate(d, S: np.ndarray, spec: Spec, start, end) -> dict:
    O, C = d.open.values, d.close.values
    if id(d) not in _VOL:
        _VOL[id(d)] = d.close.pct_change(fill_method=None).rolling(60, min_periods=40).std().values
    vol = _VOL[id(d)]
    lim = np.where(d.dates < LIMIT_CHANGE, 0.07, 0.10)
    last = d.last_trade.values.astype("datetime64[ns]")
    buy_c, sell_c = costs(spec)
    t0 = int(np.searchsorted(d.dates, np.datetime64(pd.Timestamp(start))))
    t1 = int(np.searchsorted(d.dates, np.datetime64(pd.Timestamp(end)), side="right"))
    cap = CFG["constraints"]["max_weight"]

    cash, pos, sells, buys, wts = 1.0, {}, {}, [], {}
    eq, opre, opost, trades, hhi, nh = [], [], [], [], [], []
    traded, n_delist, n_block = 0.0, 0, 0
    for t in range(t0, t1):
        today = d.dates[t].to_datetime64()
        for p in pos.values():
            j = p["j"]
            if not np.isnan(O[t, j]):
                p["value"] *= O[t, j] / p["mark"]
                p["mark"] = O[t, j]
        opre.append(cash + sum(p["value"] for p in pos.values()))
        for j in [j for j in pos if not np.isnat(last[j]) and last[j] < today]:
            p = pos.pop(j)
            got = p["value"] * (1 - spec.haircut) * (1 - sell_c)
            cash += got
            traded += p["value"]
            trades.append((d.stocks[j], d.dates[p["entry"]], d.dates[t], t - p["entry"], got / p["cost"] - 1, "下市"))
            n_delist += 1
            sells.pop(j, None)
        for j, why in list(sells.items()):
            if j not in pos or np.isnan(O[t, j]):
                continue
            if spec.limit_rule and not np.isnan(C[t - 1, j]) and O[t, j] / C[t - 1, j] - 1 <= -lim[t] + 0.002:
                n_block += 1
                continue
            p = pos.pop(j)
            sells.pop(j)
            cash += p["value"] * (1 - sell_c)
            traded += p["value"]
            trades.append((d.stocks[j], d.dates[p["entry"]], d.dates[t], t - p["entry"],
                           p["value"] * (1 - sell_c) / p["cost"] - 1, why))
        total = cash + sum(p["value"] for p in pos.values())
        retry = []
        for j, tries in buys:
            if j in pos or len(pos) >= spec.n:
                continue
            if np.isnan(O[t, j]) or (spec.limit_rule and not np.isnan(C[t - 1, j])
                                     and O[t, j] / C[t - 1, j] - 1 >= lim[t] - 0.002):
                n_block += not np.isnan(O[t, j])
                if tries < 5:
                    retry.append((j, tries + 1))
                continue
            amt = min(total * min(wts.get(j, 1 / spec.n), cap), cash)
            if amt < 0.2 * total / spec.n:
                continue
            cash -= amt
            traded += amt
            pos[j] = {"j": j, "value": amt * (1 - buy_c), "mark": O[t, j], "cost": amt, "entry": t}
        opost.append(cash + sum(p["value"] for p in pos.values()))
        for p in pos.values():
            j = p["j"]
            if not np.isnan(C[t, j]):
                p["value"] *= C[t, j] / p["mark"]
                p["mark"] = C[t, j]
        e = cash + sum(p["value"] for p in pos.values())
        eq.append(e)
        v = np.array([p["value"] for p in pos.values()])
        hhi.append(float(((v / v.sum()) ** 2).sum()) if len(v) else np.nan)
        nh.append(len(pos))

        buys = retry
        k = t - t0 - spec.phase
        if k >= 0 and k % spec.review == 0:
            s = S[t]
            ok = np.flatnonzero(~np.isnan(s))
            order = ok[np.argsort(-s[ok], kind="stable")]
            rank = np.full(len(s), np.inf)
            rank[order] = np.arange(1, len(order) + 1)
            for j in pos:
                if j not in sells and rank[j] > spec.exit_mult * spec.n:
                    sells[j] = "排名掉出" if np.isfinite(rank[j]) else "離開可投資池"
            keep = set(pos) - set(sells)
            target = order[:spec.n]
            if spec.allocator == "invvol":
                vv = vol[t, target]
                vv = np.where(np.isnan(vv) | (vv <= 0), np.nanmedian(vv), vv)
                wts = dict(zip(target, (1 / vv) / (1 / vv).sum()))
            else:
                wts = {}
            queued = {j for j, _ in buys}
            buys += [(j, 0) for j in target if j not in keep and j not in queued][:spec.n - len(keep)]
    idx = d.dates[t0:t1]
    equity = pd.Series(eq, index=idx)
    tr = pd.DataFrame(trades, columns=["stock_id", "entry", "exit", "days", "ret", "reason"])
    yrs = len(idx) / 252
    return {"equity": equity, "open_pre": np.array(opre), "open_post": np.array(opost), "trades": tr,
            "turnover_2way": traded / equity.mean() / yrs, "hhi": pd.Series(hhi, index=idx),
            "n_hold": pd.Series(nh, index=idx), "delist_events": n_delist, "limit_blocks": n_block,
            "avg_hold_days": float(tr.days.mean()) if len(tr) else np.nan}


def overlay(res: dict, target: np.ndarray, spec: Spec, band: float = 0.10) -> pd.Series:
    """整體持股比例 target[t−1]（t−1 收盤決定）在 t 日開盤生效；調整部位按成本收費。"""
    buy_c, sell_c = costs(spec)
    E, Op, Oq = res["equity"].values, res["open_pre"], res["open_post"]
    prev = np.r_[1.0, E[:-1]]
    r_on, r_tr, r_in = Op / prev - 1, Oq / Op - 1, E / Oq - 1
    V, e = np.ones(len(E)), 1.0
    for t in range(1, len(E)):
        v = V[t - 1] * (1 + e * r_on[t])
        new = target[t - 1] if not np.isnan(target[t - 1]) else e
        if abs(new - e) >= band:
            v -= abs(new - e) * v * (sell_c if new < e else buy_c)
            e = new
        v *= 1 + e * r_tr[t]
        V[t] = v * (1 + e * r_in[t])
    return pd.Series(V, index=res["equity"].index)


def vol_target(eq: pd.Series) -> np.ndarray:
    """持股比例 = min(1, 過去 252 日的 20 日波動中位數 / 最近 20 日波動)；只用到當日收盤。"""
    v20 = eq.pct_change().rolling(20, min_periods=15).std()
    return np.clip(v20.rolling(252, min_periods=120).median() / v20, 0, 1).values


def benchmarks(d, idx) -> pd.DataFrame:
    r = d.close.pct_change(fill_method=None)
    ew = r.where(d.univ.shift(1, fill_value=False)).mean(axis=1).reindex(idx).fillna(0)
    ew.iloc[0] = 0
    out = pd.DataFrame({"0050": d.bench.etf0050.reindex(idx), "加權報酬指數": d.bench.tri.reindex(idx),
                        "可投資池等權": (1 + ew).cumprod()})
    return out / out.iloc[0]


def score_matrix(pred: pd.DataFrame, d) -> np.ndarray:
    s = pred.pivot(index="date", columns="stock_id", values="score").reindex(index=d.dates, columns=d.stocks)
    return s.where(d.univ).values
