"""第 7 節：唯一一次的 holdout（2024-11-01 ~ 2026-10-30）、總表、推論、型態歸類、診斷、PSI。

封存規則
  第一次執行先寫 results/HOLDOUT_TOUCHED.json（時間、git commit、setup.yaml 雜湊），之後才讀 holdout 價格。
  已封存後再執行必須加 --rerun-reason（只允許修 bug 或重現），理由會附加在封存紀錄裡。
  SMOKE=1：假 holdout（2022-11-01 ~ 2024-10-31，仍在驗證期內），封存紀錄寫到 results/cache，不碰真正的 holdout。
用法：python -m src.s07_holdout
"""
import argparse
import hashlib
import json
import pickle
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (CACHE, CFG, DATA, FIG, HO_END, HO_START, OUT, ROOT, SEAL, SMOKE, VAL_END, VAL_START,
                        git_commit, ledger, log_run, save)
from src.features import build
from src.models import walk_forward
from src.panel import load
from src.s05_backtest import PER_LEG_BP, run_backtests
from src.s06_risk import market_state, risk_report
from src.stats import daily_ic, deflated_sharpe, folds, ic_summary, min_track_record, nw_mean, perf, sharpe_inference

warnings.simplefilter("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
N_PRIOR = 270  # 期中之前的同主題探索（約略試驗數，只用於敏感度分析）

if SMOKE:
    HO_START, HO_END = pd.Timestamp("2022-11-01"), pd.Timestamp("2024-10-31")
    SEAL = CACHE / "SMOKE_HOLDOUT_TOUCHED.json"


def seal(reason: str | None):
    rec = json.loads(SEAL.read_text(encoding="utf-8")) if SEAL.exists() else []
    if rec and not reason:
        sys.exit(f"holdout 已於 {rec[0]['time']} 使用過。只有修 bug 或重現才可重跑：--rerun-reason '...'")
    rec.append({"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "git_commit": git_commit(),
                "setup_sha256": hashlib.sha256((ROOT / "setup.yaml").read_bytes()).hexdigest()[:12],
                "holdout": [str(HO_START.date()), str(HO_END.date())], "reason": reason or "first and only planned run"})
    SEAL.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(((pb - pa) * np.log(pb / pa)).sum())


def classify(v: dict, h: dict, net_h: float, bench_h: float, be_h: float) -> str:
    """Chapter 20 的五種型態（依序判斷）。"""
    v_sig, h_sig = v["lo"] > 0, h["lo"] > 0
    if not v_sig and not h_sig:
        return "never resolved（兩段 IC 的 CI 都含 0）"
    if v_sig and (h["hi"] < 0 or not h_sig):
        return "decay or sign-flip（holdout IC 的 CI 含 0 或為負）"
    if h["mean"] > v["hi"]:
        return "strengthen（holdout IC 高於驗證期 CI 上界）"
    if net_h <= bench_h or be_h < 2 * PER_LEG_BP:
        return "marginal behind a cost screen（訊號維持，但扣成本後不贏基準／兩平成本 < 2 倍假設成本）"
    return "hold（訊號與淨績效都維持）"


def ledger_trials() -> tuple[int, float]:
    L = pd.read_csv(ROOT / "research_ledger.csv", encoding="utf-8", dtype=str)
    L = L[~L.notes.fillna("").str.contains("SMOKE")]
    # 只算 holdout 之前的搜尋：排除確認性檢定本身；重現執行會重複寫入同樣的試驗，每個試驗只算第一次
    L = L[L.trial_family != "confirmatory test"].drop_duplicates(["trial_family", "trial"])
    n = L.shape[0]
    sr = pd.to_numeric(L[L.metric.str.contains("Sharpe", na=False)].value, errors="coerce").dropna()
    return n, float(sr.var()) if len(sr) > 2 else 0.25


def main(reason: str | None):
    spec = json.loads((OUT / "04_deployed.json").read_text(encoding="utf-8"))
    dep, feats, ov_on = spec["deployed"], spec["features"], spec.get("overlay", "無覆蓋")
    rec = seal(reason)
    tag = "07_ho"

    d = load(HO_END)
    panel, W = build(d)
    dev = pd.read_parquet(DATA / "panel_dev.parquet")
    chk = panel[panel.date <= VAL_END].set_index(["date", "stock_id"])[feats]
    ref = dev.set_index(["date", "stock_id"])[feats].reindex(chk.index)
    pit_diff = int(((chk - ref).abs() > 1e-6).sum().sum())

    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    F = folds(dates, HO_START, HO_END)
    preds, info = walk_forward(panel, dates, feats, F)
    preds.to_parquet(DATA / "preds_holdout.parquet", index=False)
    lab = panel[["date", "stock_id", "fwd20"]]
    ic_h = {m: daily_ic(preds[preds.model == m].merge(lab, on=["date", "stock_id"]), "score") for m in preds.model.unique()}
    pv = pd.read_parquet(DATA / "preds_val.parquet")
    dv = load(VAL_END)
    labv = pd.read_parquet(DATA / "panel_dev.parquet", columns=["date", "stock_id", "fwd20"])
    ic_v = {m: daily_ic(pv[pv.model == m].merge(labv, on=["date", "stock_id"]), "score") for m in pv.model.unique()}
    Fv = folds(dv.dates, VAL_START, VAL_END)
    lad = pd.DataFrame({f"{m}｜{p}": ic_summary(s, Fv if p == "驗證" else F)
                        for p, D in (("驗證", ic_v), ("holdout", ic_h)) for m, s in D.items()}).T
    save(lad.reset_index().rename(columns={"index": "模型｜期間"}), "07_ic", floatfmt=".4f")

    order = [dep] + [m for m in ("rule", "ridge", "lgbm", "ensemble") if m != dep]
    bt = run_backtests(d, {m: preds[preds.model == m] for m in order}, dep, HO_START, HO_END, tag, overlay_on=ov_on)
    rk = risk_report(d, preds[preds.model == dep], bt, HO_START, HO_END, tag)
    btv = pickle.load(open(CACHE / "05_val_backtest.pkl", "rb"))

    # 推論
    n_trials, var_sr = ledger_trials()
    inf = []
    for lab_, b in (("驗證", btv), ("holdout", bt)):
        r = b["res"][dep]["equity"].pct_change().dropna()
        rg = b["gross"].pct_change().dropna()
        ex = r - b["bench"]["0050"].pct_change().reindex(r.index).fillna(0)
        si, sg, se = sharpe_inference(r), sharpe_inference(rg), sharpe_inference(ex)
        ds = deflated_sharpe(r, n_trials, var_sr)
        dp = deflated_sharpe(r, n_trials + N_PRIOR, var_sr)
        inf.append({"期間": lab_, "淨 Sharpe": si["SR"], "lo": si["lo"], "hi": si["hi"], "毛 Sharpe": sg["SR"],
                    "毛 lo": sg["lo"], "毛 hi": sg["hi"], "偏態": si["skew"], "峰態": si["kurt"],
                    "超額 Sharpe（對 0050）": se["SR"], "超額 lo": se["lo"], "超額 hi": se["hi"],
                    f"DSR（N={n_trials}）": ds["DSR"], "SR0": ds["SR0"], f"DSR（N={n_trials + N_PRIOR}，含事前探索）": dp["DSR"],
                    "最短紀錄（年）": min_track_record(si["SR"]), "最短紀錄－超額（年）": min_track_record(se["SR"]),
                    "日數": si["n_days"]})
    inf = pd.DataFrame(inf)
    save(inf, "07_inference")

    # 總表
    def ic_cell(s, Fl):
        x = ic_summary(s, Fl)
        return f"{x['IC']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]；ICIR {x['ICIR']:.2f}；正 fold {x['正 IC fold 比例']:.0%}"

    def pf(b, key="gross"):
        e = b["gross"] if key == "gross" else b["res"][dep]["equity"]
        s = sharpe_inference(e.pct_change().dropna())
        return f"{s['SR']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]；MDD {perf(e)['MDD']:.0%}；換手 {b['res'][dep]['turnover_2way']:.1f}×"

    def bench_cell(b):
        z, e = perf(b["bench"]["0050"]), perf(b["bench"]["可投資池等權"])
        return f"0050 {z['Sharpe']:.2f}（MDD {z['MDD']:.0%}）；池等權 {e['Sharpe']:.2f}"

    def ov_cell(b):
        e = b["overlays"][ov_on] if ov_on in b["overlays"] else b["res"][dep]["equity"]
        p = perf(e)
        return f"{ov_on}：{p['Sharpe']:.2f}，MDD {p['MDD']:.0%}"

    iv, ih = inf.iloc[0], inf.iloc[1]
    head = pd.DataFrame([
        ("訊號", "每日 IC [HAC 95% CI]、ICIR、IC 為正的 fold 比例", ic_cell(ic_v[dep], Fv), ic_cell(ic_h[dep], F), "—"),
        ("投組", "毛 Sharpe [CI]、MDD、年換手（雙向）", pf(btv), pf(bt), bench_cell(bt) + "（holdout）；" + bench_cell(btv) + "（驗證）"),
        ("成本", "假設成本下淨 Sharpe；兩平成本（bp／邊，對零／對 0050）",
         f"{iv['淨 Sharpe']:.2f}；{btv['breakeven']['兩平成本_bp_對零']:.0f}／{btv['breakeven']['兩平成本_bp_對0050']:.0f}",
         f"{ih['淨 Sharpe']:.2f}；{bt['breakeven']['兩平成本_bp_對零']:.0f}／{bt['breakeven']['兩平成本_bp_對0050']:.0f}", "—"),
        ("風險覆蓋", "有覆蓋時淨 Sharpe 與 MDD", ov_cell(btv), ov_cell(bt), "—"),
        ("推論", f"Deflated Sharpe（N = {n_trials}）、最短紀錄（年）",
         f"{iv[f'DSR（N={n_trials}）']:.2f}；{iv['最短紀錄（年）']:.1f}", f"{ih[f'DSR（N={n_trials}）']:.2f}；{ih['最短紀錄（年）']:.1f}", "—"),
    ], columns=["階段", "指標", "驗證（2012-01 ~ 2024-10）", "holdout（2024-11 ~ 2026-10）", "基準"])
    save(head, "07_headline")
    sv, sh = nw_mean(ic_v[dep]), nw_mean(ic_h[dep])
    z_h = perf(bt["bench"]["0050"])["Sharpe"]
    pattern = classify(sv, sh, ih["淨 Sharpe"], z_h, bt["breakeven"]["兩平成本_bp_對零"])
    (OUT / "07_pattern.txt").write_text(pattern, encoding="utf-8")
    primary = bool(sh["lo"] > 0)
    secondary = bool(ih[f"DSR（N={n_trials}）"] > 0.95)

    # 攤損：每一關
    p_v, p_h = perf(btv["res"][dep]["equity"]), perf(bt["res"][dep]["equity"])
    attr = pd.DataFrame([
        ("訊號關", "驗證 IC CI 下界 > 0", sv["lo"] > 0, f"{sv['mean']:.3f} [{sv['lo']:.3f}, {sv['hi']:.3f}]"),
        ("投組關", "驗證毛 Sharpe > 池等權 Sharpe", perf(btv["gross"])["Sharpe"] > perf(btv["bench"]["可投資池等權"])["Sharpe"],
         f"{perf(btv['gross'])['Sharpe']:.2f} vs {perf(btv['bench']['可投資池等權'])['Sharpe']:.2f}"),
        ("成本關", "驗證淨 Sharpe > 0050 且兩平成本（對零）> 2 倍假設", (p_v["Sharpe"] > perf(btv["bench"]["0050"])["Sharpe"])
         and btv["breakeven"]["兩平成本_bp_對零"] > 2 * PER_LEG_BP,
         f"淨 {p_v['Sharpe']:.2f} vs 0050 {perf(btv['bench']['0050'])['Sharpe']:.2f}；{btv['breakeven']['兩平成本_bp_對零']:.0f} bp"),
        ("holdout 關（主要）", "holdout IC CI 下界 > 0（事先登記）", primary, f"{sh['mean']:.3f} [{sh['lo']:.3f}, {sh['hi']:.3f}]"),
        ("holdout 關（次要）", f"holdout DSR > 0.95（N = {n_trials}）", secondary, f"{ih[f'DSR（N={n_trials}）']:.2f}"),
        ("holdout 對基準（描述）", "holdout 淨 Sharpe vs 0050", p_h["Sharpe"] > z_h, f"{p_h['Sharpe']:.2f} vs {z_h:.2f}"),
    ], columns=["關卡", "條件", "通過", "數字"])
    save(attr, "07_attrition")

    # 漂移診斷
    stv, sth = market_state(dv).loc[VAL_START:VAL_END], market_state(d).loc[HO_START:HO_END]
    reg_mix = pd.concat([(stv.vol.astype(str) + "/" + stv.trend).value_counts(normalize=True).rename("驗證"),
                         (sth.vol.astype(str) + "/" + sth.trend).value_counts(normalize=True).rename("holdout")], axis=1)
    reg_mix.index.name = "狀態"
    save(reg_mix.reset_index(), "07_regime_mix")
    drift = pd.DataFrame([
        ("預測品質", "部署模型平均 IC", sv["mean"], sh["mean"]),
        ("投組轉換", "淨 Sharpe ÷ 平均 IC", p_v["Sharpe"] / sv["mean"], p_h["Sharpe"] / sh["mean"]),
        ("投組轉換", "毛年化 − 池等權年化", perf(btv["gross"])["CAGR"] - perf(btv["bench"]["可投資池等權"])["CAGR"],
         perf(bt["gross"])["CAGR"] - perf(bt["bench"]["可投資池等權"])["CAGR"]),
        ("市場狀態", "池等權年化 − 0050 年化", perf(btv["bench"]["可投資池等權"])["CAGR"] - perf(btv["bench"]["0050"])["CAGR"],
         perf(bt["bench"]["可投資池等權"])["CAGR"] - perf(bt["bench"]["0050"])["CAGR"]),
        ("市場狀態", "高波動日比例", float((stv.vol == "高波動").mean()), float((sth.vol == "高波動").mean())),
        ("成本", "年化成本拖累（毛 − 淨）", perf(btv["gross"])["CAGR"] - p_v["CAGR"], perf(bt["gross"])["CAGR"] - p_h["CAGR"]),
    ], columns=["類型", "指標", "驗證", "holdout"])
    save(drift, "07_drift", floatfmt=".3f")

    # PSI：SHAP 前 5 名特徵（原始值），驗證期最後 252 日 vs holdout
    top5 = pd.read_csv(OUT / "04_shap.csv", encoding="utf-8-sig").特徵.head(5).tolist()
    ps = []
    iv0 = dv.dates[dv.dates <= VAL_END][-252]
    for f in top5:
        if f in W["F"]:
            X = W["F"][f].where(d.univ)
            a, b = X.loc[iv0:VAL_END].values.ravel(), X.loc[HO_START:HO_END].values.ravel()
        else:
            s = W["S"][f]
            a, b = s.loc[iv0:VAL_END].values, s.loc[HO_START:HO_END].values
        v_ = psi(a, b)
        ps.append({"特徵": f, "PSI": v_, "判讀": "穩定" if v_ < 0.10 else ("觀察" if v_ < 0.25 else "需重訓")})
    save(pd.DataFrame(ps), "07_psi")

    figures(btv, bt, ic_v[dep], ic_h[dep], dep)
    yearly(btv, bt, dep)
    run = log_run("07_holdout", {"deployed": dep, "overlay": ov_on, "seal": rec[-1]}, f"{HO_START.date()}~{HO_END.date()}",
                  {"ic": sh, "net_sharpe": ih["淨 Sharpe"], "dsr": ih[f"DSR（N={n_trials}）"], "pattern": pattern,
                   "pit_diff_vs_dev": pit_diff}, f"primary={'PASS' if primary else 'FAIL'}; secondary={'PASS' if secondary else 'FAIL'}",
                  "confirmatory test")
    if not reason:
        ledger("confirmatory test", "holdout", run, "none (single sealed run)", f"{HO_START.date()}~{HO_END.date()}",
               "holdout daily IC", sh["mean"], "PASS" if primary else "FAIL", "IC CI lower > 0",
               f"DSR={ih[f'DSR（N={n_trials}）']:.3f}; pattern={pattern}")
    print(head.to_string()); print(pattern); print(attr.to_string()); print(inf.round(3).T.to_string())
    print(drift.round(3).to_string()); print(pd.DataFrame(ps).to_string()); print("PIT diff vs dev:", pit_diff)


def equity_figure(btv, bt, dep):
    import matplotlib.dates as mdates
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 1.3]})
    for a, b, lab in ((ax[0], btv, "驗證 2012-01 ~ 2024-10"), (ax[1], bt, "holdout 2024-11 ~ 2026-10")):
        a.plot(b["res"][dep]["equity"], lw=1.3, color="#1c7ed6", label=f"{dep}（淨）")
        if "rule" in b["res"]:
            a.plot(b["res"]["rule"]["equity"], lw=0.9, color="#868e96", label="規則基準（淨）")
        a.plot(b["bench"]["0050"], lw=0.9, color="#e03131", label="0050 含息")
        a.plot(b["bench"]["可投資池等權"], lw=0.8, color="#f08c00", ls="--", label="池等權")
        a.set_yscale("log")
        a.yaxis.set_major_formatter(ScalarFormatter()); a.yaxis.set_minor_formatter(NullFormatter())
        a.set_title(lab, fontsize=9); a.tick_params(labelsize=7)
    ax[1].set_yticks([0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
    ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[0].legend(fontsize=7)
    fig.suptitle("扣成本後，策略在驗證期與 holdout 是否都贏過規則基準與 0050？（兩段各自從 1 起算，對數刻度）", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "07_equity.png", dpi=150); plt.close(fig)


def figures(btv, bt, icv, ich, dep):
    equity_figure(btv, bt, dep)

    fig, ax = plt.subplots(figsize=(10, 3))
    s = pd.concat([icv, ich]).sort_index()
    ax.plot(s.rolling(63, min_periods=40).mean(), color="#1c7ed6", lw=1)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(HO_START, color="k", ls="--", lw=0.8)
    ax.text(HO_START, ax.get_ylim()[1] * 0.9, " holdout", fontsize=8)
    ax.set_title("滾動 3 個月 IC：預測力在 holdout 有沒有維持在 0 以上？", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "07_rolling_ic.png", dpi=150); plt.close(fig)


def yearly(btv, bt, dep):
    rows = []
    for lab, b in (("驗證", btv), ("holdout", bt)):
        e, z = b["res"][dep]["equity"], b["bench"]["0050"]
        for y in sorted(set(e.index.year)):
            m, mp = e.index.year == y, e.index.year < y
            rows.append({"期間": lab, "年": y, "策略（淨）": e[m].iloc[-1] / (e[mp].iloc[-1] if mp.any() else e.iloc[0]) - 1,
                         "0050": z[m].iloc[-1] / (z[mp].iloc[-1] if mp.any() else z.iloc[0]) - 1})
    yr = pd.DataFrame(rows)
    yr["差"] = yr["策略（淨）"] - yr["0050"]
    save(yr, "07_yearly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-reason")
    ap.add_argument("--figures-only", action="store_true", help="只用已存的回測結果重畫權益曲線，不讀 holdout、不動封存")
    a = ap.parse_args()
    if a.figures_only:
        dep_ = json.loads((OUT / "04_deployed.json").read_text(encoding="utf-8"))["deployed"]
        equity_figure(pickle.load(open(CACHE / "05_val_backtest.pkl", "rb")),
                      pickle.load(open(CACHE / "07_ho_backtest.pkl", "rb")), dep_)
    else:
        main(a.rerun_reason)
