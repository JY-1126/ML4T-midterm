"""統計工具：Newey–West 平均、每日 IC、Sharpe 推論、Deflated Sharpe、最短紀錄長度、績效、fold 切分。"""
import numpy as np
import pandas as pd
from scipy import stats as st

from src.config import FOLD, HORIZON, TD

NW_LAG = HORIZON


def nw_mean(x, lags: int = NW_LAG) -> dict:
    """平均值與 Newey–West（Bartlett 核）標準誤；20 日重疊標籤用 lag = 20。"""
    x = pd.Series(x).dropna().values.astype(float)
    n = len(x)
    if n < 5:
        return {"mean": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan, "t": np.nan, "p": np.nan, "n": n}
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(s, 0) / n)
    t = x.mean() / se if se > 0 else np.nan
    return {"mean": x.mean(), "se": se, "lo": x.mean() - 1.96 * se, "hi": x.mean() + 1.96 * se,
            "t": t, "p": 2 * (1 - st.norm.cdf(abs(t))), "n": n}


def daily_ic(df: pd.DataFrame, col: str, label: str = "fwd20", min_n: int = 30) -> pd.Series:
    """向量化每日 Spearman IC（在成對完整的資料內重新排名）。"""
    s = df[["date", col, label]].dropna()
    r = s.groupby("date")[[col, label]].rank()
    x, y = r[col].values.astype(float), r[label].values.astype(float)
    k = pd.DataFrame({"g": s.date.values, "x": x, "y": y, "xy": x * y, "xx": x * x, "yy": y * y}).groupby("g")
    mu, n = k.mean(), k.size()
    ic = (mu.xy - mu.x * mu.y) / np.sqrt((mu.xx - mu.x ** 2) * (mu.yy - mu.y ** 2))
    return ic[n >= min_n].dropna()


def ic_summary(ic: pd.Series, folds: list | None = None) -> dict:
    s = nw_mean(ic)
    out = {"IC": s["mean"], "lo": s["lo"], "hi": s["hi"], "t": s["t"], "p": s["p"],
           "ICIR": ic.mean() / ic.std() if ic.std() > 0 else np.nan, "days": len(ic), "N_eff": len(ic) / HORIZON}
    if folds:
        fm = [ic[(ic.index >= a) & (ic.index <= b)].mean() for a, b in folds]
        fm = [x for x in fm if np.isfinite(x)]
        out["folds"] = len(fm)
        out["正 IC fold 比例"] = np.mean(np.sign(fm) == np.sign(s["mean"])) if fm else np.nan
    return out


def sharpe_inference(r: pd.Series, sr0_ann: float = 0.0) -> dict:
    """Sharpe 標準誤含偏態、峰態修正（Mertens 2002）；PSR 對 sr0。"""
    r = pd.Series(r).dropna()
    n = len(r)
    sr = r.mean() / r.std()
    g3, g4 = st.skew(r), st.kurtosis(r, fisher=False)
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (n - 1))
    a = np.sqrt(TD)
    return {"SR": sr * a, "SE": se * a, "lo": (sr - 1.96 * se) * a, "hi": (sr + 1.96 * se) * a,
            "skew": g3, "kurt": g4, "n_days": n, "PSR": st.norm.cdf((sr - sr0_ann / a) / se)}


def deflated_sharpe(r: pd.Series, n_trials: int, var_trials_ann: float) -> dict:
    """Bailey & López de Prado：N 個試驗、年化 Sharpe 變異數 var 下的期望最大 Sharpe SR0，DSR = PSR(SR0)。"""
    g = 0.5772156649
    n_trials = max(int(n_trials), 2)
    sr0 = np.sqrt(var_trials_ann) * ((1 - g) * st.norm.ppf(1 - 1 / n_trials) + g * st.norm.ppf(1 - 1 / (n_trials * np.e)))
    return {"SR0": sr0, "DSR": sharpe_inference(r, sr0)["PSR"], "N": n_trials, "var_SR": var_trials_ann}


def min_track_record(sr_ann: float, alpha: float = 0.05, power: float = 0.80) -> float:
    return np.inf if not sr_ann > 0 else ((st.norm.ppf(1 - alpha) + st.norm.ppf(power)) / sr_ann) ** 2


def perf(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna()
    yrs = len(r) / TD
    dd = eq / eq.cummax() - 1
    return {"CAGR": (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(TD),
            "Sharpe": r.mean() / r.std() * np.sqrt(TD), "MDD": dd.min()}


def folds(dates: pd.DatetimeIndex, start, end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """從 start 起每 FOLD 個交易日一段 [起, 迄]，最後一段截在 end。"""
    i0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(start))))
    i1 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(end)), side="right"))
    return [(dates[k], dates[min(k + FOLD, i1) - 1]) for k in range(i0, i1, FOLD)]
