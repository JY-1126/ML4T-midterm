"""第 6 節（下）：風險清單——VaR/CVaR（依狀態）、回撤、滾動 beta（HAC）、壓力測試（價差 3 倍）、
停損階梯、風險控制矩陣、市場狀態切片（波動 × 趨勢，門檻只用過去資料）。

risk_report() 也給 s07 在 holdout 使用。用法：python -m src.s06_risk
"""
import json
import pickle
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import CACHE, CFG, DATA, FIG, OUT, VAL_END, VAL_START, log_run, save
from src.engine import Spec, score_matrix, simulate
from src.features import _expanding_pct
from src.panel import load
from src.stats import perf

warnings.simplefilter("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
KS = CFG["risk"]["kill_switch"]


def episodes(start, end) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out = []
    for s in CFG["risk"]["stress_episodes"]:
        a, b = (pd.Timestamp(x) for x in re.findall(r"\d{4}-\d{2}-\d{2}", s))
        name = s.split()[-1]
        if a >= pd.Timestamp(start) and b <= pd.Timestamp(end):
            out.append((name, a, b))
    return out


def market_state(d) -> pd.DataFrame:
    """波動三分位用擴張百分位（只用 t 以前）；趨勢 = 加權報酬指數 > 200 日均線。"""
    m = d.bench.tri.pct_change()
    vp = _expanding_pct(m.rolling(20, min_periods=15).std())
    vol = pd.cut(vp, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["低波動", "中波動", "高波動"])
    tr = np.where(d.bench.tri > d.bench.tri.rolling(200, min_periods=150).mean(), "多頭", "空頭")
    return pd.DataFrame({"vol": vol, "trend": tr}, index=d.dates).shift(1)  # t−1 的狀態決定 t 日歸屬


def drawdown_stats(eq: pd.Series) -> dict:
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[peak]]
    rdate = rec.index[0] if len(rec) else None
    under = (dd < 0).astype(int)
    runs = under.groupby((under == 0).cumsum()).sum()
    return {"MDD": dd.min(), "高點": peak.date(), "谷底": trough.date(), "回撤天數（高點→谷底）": int(eq.loc[peak:trough].size - 1),
            "恢復日": rdate.date() if rdate is not None else "未恢復",
            "恢復天數（谷底→恢復）": int(eq.loc[trough:rdate].size - 1) if rdate is not None else None,
            "最長水下天數": int(runs.max())}


def rolling_beta(r: pd.Series, b: pd.Series, win: int = 120, step: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(win, len(r) + 1, step):
        y, x = r.iloc[i - win:i], b.iloc[i - win:i]
        m = sm.OLS(y.values, sm.add_constant(x.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({"date": r.index[i - 1], "beta": m.params[1], "lo": m.conf_int()[1][0], "hi": m.conf_int()[1][1]})
    return pd.DataFrame(rows).set_index("date")


def risk_report(d, preds: pd.DataFrame, bt: dict, start, end, tag: str) -> dict:
    eq = bt["res"][bt["deployed"]]["equity"]
    r = eq.pct_change().dropna()
    st = market_state(d).reindex(r.index)
    rows = [{"狀態": "全部", "日數": len(r), "VaR95": -r.quantile(0.05), "CVaR95": -r[r <= r.quantile(0.05)].mean()}]
    for col in ("vol", "trend"):
        for k, g in r.groupby(st[col]):
            rows.append({"狀態": k, "日數": len(g), "VaR95": -g.quantile(0.05), "CVaR95": -g[g <= g.quantile(0.05)].mean()})
    var = pd.DataFrame(rows)
    save(var, f"{tag}_var", floatfmt=".4f")

    sl = []
    for (v, t), g in r.groupby([st.vol, st.trend], observed=True):
        e = (1 + g).cumprod()
        sl.append({"波動": v, "趨勢": t, "日數": len(g), "年化報酬": g.mean() * 252,
                   "Sharpe": g.mean() / g.std() * np.sqrt(252), "MDD（狀態內串接）": (e / e.cummax() - 1).min()})
    sl = pd.DataFrame(sl)
    save(sl, f"{tag}_regime")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    lab = sl.波動.astype(str) + "\n" + sl.趨勢
    ax[0].bar(lab, sl.Sharpe, color="#1c7ed6"); ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title("Sharpe", fontsize=9)
    ax[1].bar(lab, sl["MDD（狀態內串接）"], color="#e03131"); ax[1].set_title("回撤（狀態內串接）", fontsize=9)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.suptitle("策略在哪一種市場狀態失效？（波動 × 趨勢，門檻只用過去資料）", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"{tag}_regime.png", dpi=150); plt.close(fig)

    dds = drawdown_stats(eq)
    b = bt["bench"]["0050"].pct_change().reindex(r.index)
    rb = rolling_beta(r, b.fillna(0))
    rb.to_csv(OUT / f"{tag}_rolling_beta.csv", encoding="utf-8-sig")
    full = sm.OLS(r.values, sm.add_constant(b.fillna(0).values)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    S = score_matrix(preds, d)
    stress = []
    for name, a, z in episodes(start, end):
        for lab_, sp in (("正常成本", Spec()), ("價差 3 倍", Spec(spread_mult=CFG["cost_model"]["stress_spread_mult"]))):
            e = simulate(d, S, sp, start, end)["equity"].loc[a:z]
            stress.append({"事件": name, "期間": f"{a.date()}~{z.date()}", "成本": lab_, "期間報酬": e.iloc[-1] / e.iloc[0] - 1,
                           "期間 MDD": (e / e.cummax() - 1).min(),
                           "0050 期間報酬": bt["bench"]["0050"].loc[a:z].iloc[-1] / bt["bench"]["0050"].loc[a:z].iloc[0] - 1})
    stress = pd.DataFrame(stress)
    save(stress, f"{tag}_stress")

    dd = eq / eq.cummax() - 1
    ks = pd.DataFrame([{"門檻": f"{v:.0%}", "動作": k, "負責人": "研究者本人（期中）",
                        "首次觸發": (dd[dd <= v].index[0].date() if (dd <= v).any() else "未觸發"),
                        "觸發天數": int((dd <= v).sum())} for k, v in KS.items()])
    save(ks, f"{tag}_killswitch")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(dd.index, dd, 0, color="#e03131", alpha=0.35, label="策略回撤（淨）")
    z = bt["bench"]["0050"]
    ax.plot(z.index, z / z.cummax() - 1, color="gray", lw=0.8, label="0050 回撤")
    for (k, v), c in zip(KS.items(), ["#fab005", "#f08c00", "#e8590c", "#c92a2a"]):
        ax.axhline(v, color=c, ls="--", lw=0.9, label=f"{k} {v:.0%}")
    ax.legend(fontsize=7, ncol=3, loc="lower left"); ax.set_title("回撤什麼時候會觸發停損階梯？", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"{tag}_drawdown.png", dpi=150); plt.close(fig)

    out = {"var": var, "regime": sl, "drawdown": dds, "beta_full": (full.params[1], *full.conf_int()[1]),
           "rolling_beta": rb, "stress": stress, "killswitch": ks}
    pd.Series({**dds, "全期 beta": full.params[1], "beta lo": full.conf_int()[1][0], "beta hi": full.conf_int()[1][1]}) \
        .to_frame("值").to_csv(OUT / f"{tag}_risk_summary.csv", encoding="utf-8-sig")
    return out


RISK_MATRIX = pd.DataFrame([
    ("市場", "滾動 120 日 beta（對 0050）", "beta > 1.3 持續 60 日 → 檢查是否集中在高 beta 股", "每週"),
    ("因子", "持股的 vol60、adv60 平均排名", "偏離可投資池中位數 0.3 以上 → 列入檢討", "每次檢視"),
    ("槓桿", "總曝險 = 持股市值 / 淨值", "上限 1.0（模擬內強制）", "每日"),
    ("集中度", "單檔權重、1/HHI", "單檔 ≤ 10%（模擬內強制）；1/HHI < 12 警示", "每次檢視"),
    ("流動性", "持股金額 / 20 日均成交額", "> 5% 不進場；可投資池門檻 2,000 萬", "每次檢視"),
    ("模型", "30/60/90 日滾動 IC、PSI", "60 日 IC < 0 且 PSI > 0.25 → 重訓；90 日 IC < 0 → 暫停", "每週"),
    ("營運", "資料下載檢查（檔數、殘缺日）、執行對帳", "當日檔數 < 前 60 日中位數一半 → 不下單", "每日"),
], columns=["風險", "代理指標", "控制", "頻率"])


def main():
    spec = json.loads((OUT / "04_deployed.json").read_text(encoding="utf-8"))
    dep = spec["deployed"]
    bt = pickle.load(open(CACHE / "05_val_backtest.pkl", "rb"))
    pv = pd.read_parquet(DATA / "preds_val.parquet")
    d = load(VAL_END)
    rk = risk_report(d, pv[pv.model == dep], bt, VAL_START, VAL_END, "06_val")
    save(RISK_MATRIX, "06_risk_matrix")
    log_run("06_risk", {"deployed": dep, "kill_switch": KS}, f"{VAL_START.date()}~{VAL_END.date()}",
            {"drawdown": rk["drawdown"], "beta": rk["beta_full"]}, "risk report")
    print(rk["var"].round(4).to_string()); print(rk["regime"].round(3).to_string())
    print(rk["drawdown"]); print(rk["stress"].round(3).to_string()); print(rk["killswitch"].to_string())


if __name__ == "__main__":
    main()
