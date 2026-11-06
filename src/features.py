"""特徵與標籤（全部只用 t 日收盤以前的價量）。

族群（同伴股）
  每 20 個交易日（固定錨點：資料第 120 列起每 20 列）用過去 120 日「個股日報酬 − 加權報酬指數日報酬」
  的相關係數，替當天可投資池中的每檔股票找相關最高的 10 檔同伴；同伴名單在下一次更新前不變。
  不使用任何產業或概念分類（那些只有今日快照，會有前視偏誤）。
表示方式
  signal 特徵：每日在可投資池內的百分位排名（0~1）
  市場狀態：mkt_vol20、disp20 為相對自身歷史的擴張百分位；mkt_trend 為水準值
標籤
  fwd_h = 可出場開盤價[t+1+h] / 可出場開盤價[t+1] − 1（h = 5, 20, 60）；fwd20 是主要標籤
  y = fwd20 的每日橫斷面百分位排名（訓練目標）
"""
import numpy as np
import pandas as pd

from src.config import CFG, TRAIN_START
from src.panel import Data, exit_open

K_PEERS = 10
PEER_WIN = CFG["admissible_information"]["peer_window_days"]
PEER_EVERY = CFG["admissible_information"]["peer_update_days"]
CANDS = list(CFG["features"]["candidates"])
MARKET = ["mkt_vol20", "mkt_trend", "disp20"]
SIGNALS = [c for c in CANDS if c not in MARKET]


def _expanding_pct(s: pd.Series, min_periods: int = 250) -> pd.Series:
    """t 日的值在 t 以前（含）歷史中的百分位，只用過去資料。"""
    v = s.values
    out = np.full(len(v), np.nan)
    hist = []
    import bisect
    for i, x in enumerate(v):
        if np.isnan(x):
            continue
        bisect.insort(hist, x)
        if len(hist) >= min_periods:
            out[i] = bisect.bisect_right(hist, x) / len(hist)
    return pd.Series(out, index=s.index)


def peers(d: Data, e: pd.DataFrame) -> tuple[dict, dict]:
    """回傳 {更新列: (候選股欄位索引, 同伴索引 n×K, 平均相關 n)}。"""
    E = e.values
    U = d.univ.values
    out = {}
    for u in range(PEER_WIN, len(d.dates), PEER_EVERY):
        W = E[u - PEER_WIN + 1:u + 1]
        ok = np.isfinite(W).sum(0) >= int(PEER_WIN * 0.8)
        cand = np.flatnonzero(U[u] & ok)
        if len(cand) <= K_PEERS:
            continue
        X = W[:, cand]
        X = X - np.nanmean(X, 0)
        X = np.nan_to_num(X)
        X /= np.linalg.norm(X, axis=0) + 1e-12
        C = X.T @ X
        np.fill_diagonal(C, -np.inf)
        idx = np.argpartition(-C, K_PEERS, axis=1)[:, :K_PEERS]
        tight = np.take_along_axis(C, idx, 1).mean(1)
        out[u] = (cand, cand[idx], tight)
    return out


def build(d: Data, start=TRAIN_START) -> tuple[pd.DataFrame, dict]:
    c, o, v = d.close, d.open, d.volume
    r = c.pct_change(fill_method=None)
    m = d.bench.tri.pct_change()
    ret = {k: c / c.shift(k) - 1 for k in (5, 20, 60)}
    adv60 = np.log((d.close_raw * v).rolling(60, min_periods=40).mean())
    F = {
        "ret20": ret[20], "rev5": ret[5],
        "mom120_20": c.shift(20) / c.shift(120) - 1,
        "near_high250": c / c.rolling(250, min_periods=200).max(),
        "vol60": r.rolling(60, min_periods=40).std(),
        "max_ret20": r.rolling(20, min_periods=15).max(),
        "overnight20": (o / c.shift(1) - 1).rolling(20, min_periods=15).mean(),
        "volume_surge": v.rolling(5, min_periods=4).mean() / v.rolling(60, min_periods=40).mean(),
        "adv60": adv60,
    }
    rm = r.mul(m, axis=0).rolling(60, min_periods=40).mean()
    F["beta60"] = (rm - r.rolling(60, min_periods=40).mean().mul(m.rolling(60, min_periods=40).mean(), axis=0)) \
        .div(m.rolling(60, min_periods=40).var(ddof=0), axis=0)

    # 族群特徵
    e = r.sub(m, axis=0)
    P = peers(d, e)
    T, N = c.shape
    pk = {k: np.full((T, N), np.nan) for k in ("peer_ret20", "peer_ret60", "peer_ret5", "peer_breadth20",
                                               "leader_ret20", "peer_tight")}
    R5, R20, R60, A = ret[5].values, ret[20].values, ret[60].values, adv60.values
    ups = sorted(P)
    for a, u in enumerate(ups):
        rows = np.arange(u, ups[a + 1] if a + 1 < len(ups) else T)
        cand, idx, tight = P[u]
        x20 = R20[rows][:, idx]                         # (rows, n, K)
        with np.errstate(invalid="ignore"):
            pk["peer_ret20"][np.ix_(rows, cand)] = np.nanmean(x20, 2)
            pk["peer_ret60"][np.ix_(rows, cand)] = np.nanmean(R60[rows][:, idx], 2)
            pk["peer_ret5"][np.ix_(rows, cand)] = np.nanmean(R5[rows][:, idx], 2)
            pk["peer_breadth20"][np.ix_(rows, cand)] = np.nanmean(
                np.where(np.isfinite(x20), (x20 > 0).astype(float), np.nan), 2)
            a3 = np.nan_to_num(A[rows][:, idx], nan=-np.inf)
            top = np.argsort(-a3, axis=2)[:, :, :3]
            pk["leader_ret20"][np.ix_(rows, cand)] = np.nanmean(np.take_along_axis(x20, top, 2), 2)
        pk["peer_tight"][np.ix_(rows, cand)] = np.broadcast_to(tight, (len(rows), len(cand)))
    for k, a in pk.items():
        F[k] = pd.DataFrame(a, index=c.index, columns=c.columns)
    F["peer_gap20"] = F["peer_ret20"] - ret[20]

    # 市場狀態
    tri = d.bench.tri
    disp = ret[20].where(d.univ).std(axis=1)
    S = {"mkt_vol20": _expanding_pct(m.rolling(20, min_periods=15).std()),
         "mkt_trend": tri / tri.rolling(200, min_periods=150).mean() - 1,
         "disp20": _expanding_pct(disp)}

    # 標籤
    eo = exit_open(d)
    L = {f"fwd{h}": eo.shift(-(h + 1)) / eo.shift(-1) - 1 for h in (5, 20, 60)}

    keep = d.univ & (d.dates >= pd.Timestamp(start))[:, None]
    ranked = {k: F[k].where(d.univ).rank(axis=1, pct=True) for k in SIGNALS}
    cols = {k: ranked[k].where(keep).stack() for k in SIGNALS}
    panel = pd.DataFrame(cols)
    for k, lab in L.items():
        panel[k] = lab.where(keep).stack().reindex(panel.index)
    panel.index.names = ["date", "stock_id"]
    panel = panel.reset_index()
    for k, s in S.items():
        panel[k] = panel.date.map(s)
    panel["y"] = panel.groupby("date").fwd20.rank(pct=True)
    num = panel.columns.difference(["date", "stock_id"])
    panel[num] = panel[num].astype("float32")
    return panel, {"F": F, "S": S, "ranked": ranked, "peers": P, "L": L}
