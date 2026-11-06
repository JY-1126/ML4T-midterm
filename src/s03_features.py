"""第 4 節：特徵、標籤與 triage（五道關卡）、去重、機制檢查。只用驗證期（2012-01-02 ~ 2024-10-31）。

輸出
  data/panel_dev.parquet          特徵（排名）+ 標籤，2007 起、至驗證期結束
  results/03_triage.csv           每個候選：五關數字與 PROCEED / REVISE / STOP
  results/03_model_features.json  進入模型的特徵（去重後）
  results/03_truncation.csv       截斷重算測試
  results/03_mechanism.csv、03_regime.csv、figures/03_*.png
用法：python -m src.s03_features
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.config import CFG, DATA, FIG, HORIZON, OUT, VAL_END, VAL_START, ledger, log_run, save
from src.features import CANDS, MARKET, SIGNALS, build
from src.panel import load
from src.stats import daily_ic, folds, ic_summary, nw_mean

warnings.simplefilter("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
CM = CFG["cost_model"]
ROUND_TRIP = (2 * (CM["commission_bps"] + CM["half_spread_bps"]) + CM["sell_tax_bps"]) / 1e4
TRUNC = pd.Timestamp("2016-06-30")


def quintiles(v: pd.DataFrame, col: str) -> pd.Series:
    s = v[["date", col, "fwd20"]].dropna()
    q = s.groupby("date")[col].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    return s.groupby([s.date, q]).fwd20.mean().groupby(level=1).mean()


def rank_autocorr(ranked: pd.DataFrame, lag: int = HORIZON) -> float:
    a, b = ranked.iloc[::lag], ranked.shift(-lag).iloc[::lag]
    return float(a.corrwith(b, axis=1, method="spearman").mean())


def residual_ic(v: pd.DataFrame, col: str, controls: list[str]) -> float:
    """每日橫斷面把特徵對控制變數做 OLS，取殘差的 IC（共同驅動控制）。"""
    s = v[["date", col, "fwd20"] + controls].dropna()

    def res(g):
        X = np.c_[np.ones(len(g)), g[controls].values]
        b, *_ = np.linalg.lstsq(X, g[col].values, rcond=None)
        return pd.Series(g[col].values - X @ b, index=g.index)
    s["res"] = s.groupby("date", group_keys=False).apply(res)
    return nw_mean(daily_ic(s, "res"))


def dag_figure():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.axis("off")
    nodes = {"共同消息衝擊\n（需求、供應鏈、政策）": (0.12, 0.55), "同伴過去報酬\npeer_ret20 / leader": (0.42, 0.85),
             "本股過去報酬\nret20、near_high": (0.42, 0.25), "本股未來報酬\nfwd20": (0.82, 0.55),
             "市場 beta、規模\n（干擾：要控制）": (0.12, 0.08), "注意力／流動性\n（中介：不控制）": (0.62, 0.55),
             "進入可投資池\n（碰撞：不可再篩）": (0.82, 0.12)}
    for k, (x, y) in nodes.items():
        ax.text(x, y, k, ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round", fc="#e7f5ff", ec="#1c7ed6"))
    arrows = [(0, 1), (0, 2), (1, 5), (5, 3), (2, 3), (4, 1), (4, 2), (4, 3), (2, 6), (3, 6)]
    keys = list(nodes)
    for a, b in arrows:
        (x0, y0), (x1, y1) = nodes[keys[a]], nodes[keys[b]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#495057", lw=1, shrinkA=28, shrinkB=28))
    ax.set_title("族群擴散機制的 DAG：控制 beta／規模、不控制中介、不以結果相關的條件再篩選", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "03_dag.png", dpi=150); plt.close(fig)


def main():
    d = load(VAL_END)
    panel, W = build(d)
    panel.to_parquet(DATA / "panel_dev.parquet", index=False)

    # 截斷重算：資料只讀到 TRUNC，重疊日期的特徵必須完全相同
    p2, _ = build(load(TRUNC))
    a = panel[panel.date <= TRUNC].set_index(["date", "stock_id"])[CANDS]
    b = p2.set_index(["date", "stock_id"])[CANDS].reindex(a.index)
    diff = (a - b).abs()
    trunc = pd.DataFrame({"特徵": CANDS, "比對格數": a.notna().sum().values,
                          "最大絕對差": diff.max().values, "不一致格數": (diff > 1e-6).sum().values})
    save(trunc, "03_truncation", floatfmt=".2e")

    v = panel[(panel.date >= VAL_START) & (panel.date <= VAL_END)]
    F = folds(d.dates, VAL_START, VAL_END)
    rows, ics = [], {}
    for f in SIGNALS:
        ic = daily_ic(v, f)
        ics[f] = ic
        s = ic_summary(ic, F)
        pl = W["ranked"][f].shift(HORIZON + 1).where(d.univ).stack().rename("pl")
        vp = v.set_index(["date", "stock_id"]).join(pl, how="inner").reset_index()
        spl = nw_mean(daily_ic(vp, "pl"))
        dec = {h: daily_ic(v, f, h).mean() for h in ("fwd5", "fwd60")}
        q = quintiles(v, f)
        mono = float(pd.Series(q.values).corr(pd.Series(range(5)), method="spearman"))
        spread = abs(q.iloc[-1] - q.iloc[0])
        ac = rank_autocorr(W["ranked"][f].loc[VAL_START:VAL_END])
        trunc_ok = bool(trunc.set_index("特徵").loc[f, "不一致格數"] == 0)
        placebo_kills = bool((spl["lo"] <= 0 <= spl["hi"]) or abs(spl["mean"]) < 0.5 * abs(s["IC"]))
        rows.append({"特徵": f, "角色": CFG["features"]["candidates"][f]["role"],
                     "G1 截斷相同": trunc_ok, "G1 安慰劑 IC": spl["mean"], "G1 安慰劑消失": placebo_kills,
                     "IC": s["IC"], "lo": s["lo"], "hi": s["hi"], "p": s["p"], "ICIR": s["ICIR"],
                     "同號 fold 比例": s["正 IC fold 比例"], "IC fwd5": dec["fwd5"], "IC fwd60": dec["fwd60"],
                     "Q1": q.iloc[0], "Q5": q.iloc[-1], "五分位單調 ρ": mono,
                     "頂底價差/20日": spread, "20 日排名自相關": ac})
    t = pd.DataFrame(rows)
    # G5：BH
    m = len(t)
    order = t.p.rank(method="first").astype(int)
    bh = (t.p * m / order).values
    srt = np.argsort(t.p.values)
    adj = np.empty(m)
    adj[srt] = np.minimum.accumulate(bh[srt][::-1])[::-1]
    t["BH q"] = np.minimum(adj, 1)
    t["G1"] = t["G1 截斷相同"] & t["G1 安慰劑消失"]
    t["G2"] = (np.sign(t.lo) == np.sign(t.hi)) & (t["同號 fold 比例"] >= 0.6)
    t["G3"] = t["五分位單調 ρ"].abs() >= 0.9
    t["G4"] = t["頂底價差/20日"] > 2 * ROUND_TRIP
    t["G5"] = t["BH q"] < 0.05
    ci0 = np.sign(t.lo) != np.sign(t.hi)
    t["決定"] = np.where(t[["G1", "G2", "G3", "G4", "G5"]].all(1), "PROCEED",
                       np.where(ci0 & ~t.G5, "STOP", "REVISE"))
    save(t, "03_triage")

    # 去重：|Spearman ρ| > 0.8 分群，每群留 |IC| 最大者
    keep_c = t[t["決定"] != "STOP"]["特徵"].tolist()
    smp = v[v.date.isin(v.date.unique()[::10])]
    rho = smp[keep_c].corr(method="spearman").abs().fillna(0)
    if len(keep_c) > 1:
        Z = linkage(squareform(1 - rho.values, checks=False), "complete")
        cl = fcluster(Z, 0.2, "distance")
    else:
        cl = np.ones(len(keep_c), int)
    groups = pd.DataFrame({"特徵": keep_c, "群": cl}).merge(t[["特徵", "IC"]], on="特徵")
    groups["|IC|"] = groups.IC.abs()
    chosen = groups.sort_values("|IC|", ascending=False).drop_duplicates("群").特徵.tolist()
    dropped = {r.特徵: groups[(groups.群 == r.群)].sort_values("|IC|").特徵.iloc[-1]
               for r in groups.itertuples() if r.特徵 not in chosen}
    model_feats = [f for f in SIGNALS if f in chosen] + MARKET
    (OUT / "03_model_features.json").write_text(json.dumps(
        {"features": model_feats, "dropped_as_duplicate": dropped,
         "stopped": t[t["決定"] == "STOP"].特徵.tolist()}, ensure_ascii=False, indent=1), encoding="utf-8")
    rho.round(2).to_csv(OUT / "03_feature_corr.csv", encoding="utf-8-sig")

    # 機制檢查：|t| 前三名 + 族群主特徵 peer_ret20
    top = t.assign(at=(t.IC / ((t.hi - t.lo) / 3.92)).abs()).sort_values("at", ascending=False).特徵.head(3).tolist()
    checks = list(dict.fromkeys(top + ["peer_ret20", "peer_gap20"]))
    mech, reg = [], []
    terc = pd.cut(v.mkt_vol20, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["低波動", "中波動", "高波動"])
    trend = np.where(v.mkt_trend > 0, "多頭", "空頭")
    for f in checks:
        base = t.set_index("特徵").loc[f]
        ctrl = [c for c in ("beta60", "adv60") if c != f]
        r1 = residual_ic(v, f, ctrl)
        r2 = residual_ic(v, f, ctrl + ["ret20"]) if f.startswith(("peer", "leader")) else {"mean": np.nan, "lo": np.nan, "hi": np.nan}
        mech.append({"特徵": f, "原始 IC": base.IC, "安慰劑 IC（移 21 日）": base["G1 安慰劑 IC"],
                     "控制 beta、規模後 IC": r1["mean"], "控制後 lo": r1["lo"], "控制後 hi": r1["hi"],
                     "再控制本股 ret20 後 IC": r2["mean"], "lo ": r2["lo"], "hi ": r2["hi"]})
        for lab, g in list(v.groupby(terc, observed=True)) + list(v.groupby(trend)):
            s = nw_mean(daily_ic(g, f))
            reg.append({"特徵": f, "狀態": lab, "IC": s["mean"], "lo": s["lo"], "hi": s["hi"], "日數": s["n"]})
    save(pd.DataFrame(mech), "03_mechanism")
    save(pd.DataFrame(reg), "03_regime")
    dag_figure()

    # 圖：每個候選的 IC 與 CI，標出決定
    tt = t.sort_values("IC")
    col = tt["決定"].map({"PROCEED": "#2b8a3e", "REVISE": "#f08c00", "STOP": "#adb5bd"})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(tt.特徵, tt.IC, xerr=[tt.IC - tt.lo, tt.hi - tt.IC], color=col, capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("驗證期每日 Spearman IC（NW 95% CI）；綠 PROCEED、橘 REVISE、灰 STOP")
    ax.set_title("哪些候選特徵在扣掉多重檢定後仍有訊號？", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "03_triage_ic.png", dpi=150); plt.close(fig)

    run = log_run("03_features", {"candidates": SIGNALS, "market_state": MARKET},
                  f"{VAL_START.date()}~{VAL_END.date()}",
                  {"decisions": t.set_index("特徵")["決定"].to_dict(), "model_features": model_feats,
                   "truncation_mismatch": int(trunc["不一致格數"].sum())}, "triage done", "5 gates + BH")
    for r in t.to_dict("records"):
        ledger("feature triage", r["特徵"], run, "include feature in model", f"{VAL_START.date()}~{VAL_END.date()}",
               "daily IC", r["IC"], r["決定"], "G1-G5", f"BH q={r['BH q']:.3g}")
    print(t[["特徵", "IC", "lo", "hi", "BH q", "五分位單調 ρ", "頂底價差/20日", "G1", "G2", "G3", "G4", "G5", "決定"]]
          .round(4).to_string())
    print("model features:", model_feats, "dropped:", dropped)
    print(trunc.to_string())
    print(pd.DataFrame(mech).round(4).to_string())


if __name__ == "__main__":
    main()
