"""第 6 節（上）：回測協定、基準、成本掃描與兩平成本、投組配置、覆蓋（驗證期）。

run_backtests() 也給 s07 在 holdout 使用（同一協定，只換期間與預測）。
用法：python -m src.s05_backtest
"""
import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CACHE, CFG, DATA, FIG, OUT, VAL_END, VAL_START, ledger, log_run, save
from src.engine import Spec, benchmarks, overlay, score_matrix, simulate, vol_target
from src.panel import load
from src.stats import perf

warnings.simplefilter("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
CM = CFG["cost_model"]
PER_LEG_BP = (2 * (CM["commission_bps"] + CM["half_spread_bps"]) + CM["sell_tax_bps"]) / 2
PHASES = (0, 5, 10, 15)
KS = CFG["risk"]["kill_switch"]


def regime(mult: float) -> str:
    return "硬地板（< 2× 假設成本）" if mult < 2 else ("灰色地帶（2–4×）" if mult < 4 else "安全（> 4×）")


def kill_target(eq: pd.Series) -> np.ndarray:
    """停損階梯（只用到當日收盤）：回撤 ≤ halve → 持股 50%，≤ unwind → 0；回撤回到 watch 以上才恢復 100%。"""
    dd = (eq / eq.cummax() - 1).values
    out, e = np.ones(len(dd)), 1.0
    for t, x in enumerate(dd):
        if x <= KS["unwind"]:
            e = 0.0
        elif x <= KS["halve"]:
            e = min(e, 0.5)
        elif x > KS["watch"]:
            e = 1.0
        out[t] = e
    return out


def run_backtests(d, preds: dict, deployed: str, start, end, tag: str, overlay_on: str | None = None) -> dict:
    S = {k: score_matrix(p, d) for k, p in preds.items()}
    base = Spec()
    res = {k: simulate(d, s, base, start, end) for k, s in S.items()}
    eq = res[deployed]["equity"]
    bench = benchmarks(d, eq.index)
    gross = simulate(d, S[deployed], Spec(cost_mult=0), start, end)

    rows = [{"策略": f"{k}（淨）", **perf(r["equity"]), "年換手(雙向)": r["turnover_2way"],
             "平均持有天數": r["avg_hold_days"], "下市出清": r["delist_events"]} for k, r in res.items()]
    rows.append({"策略": f"{deployed}（毛）", **perf(gross["equity"]), "年換手(雙向)": gross["turnover_2way"]})
    rows += [{"策略": b, **perf(bench[b])} for b in bench]
    main_tbl = pd.DataFrame(rows)

    sweep = []
    for m in CM["sweep_multipliers"]:
        r = gross if m == 0 else (res[deployed] if m == 1 else simulate(d, S[deployed], Spec(cost_mult=m), start, end))
        sweep.append({"成本倍數": m, "每邊成本_bp": PER_LEG_BP * m, **perf(r["equity"])})
    sweep = pd.DataFrame(sweep)
    g, z, ew = perf(gross["equity"]), perf(bench["0050"]), perf(bench["可投資池等權"])
    T2 = gross["turnover_2way"]
    be = {"毛年化": g["CAGR"], "0050 年化": z["CAGR"], "池等權年化": ew["CAGR"], "年換手(雙向)": T2,
          "假設每邊成本_bp": PER_LEG_BP, "兩平成本_bp_對零": g["CAGR"] / T2 * 1e4,
          "兩平成本_bp_對0050": (g["CAGR"] - z["CAGR"]) / T2 * 1e4,
          "兩平成本_bp_對池等權": (g["CAGR"] - ew["CAGR"]) / T2 * 1e4}
    be["成本區間（對 0050）"] = regime(be["兩平成本_bp_對0050"] / PER_LEG_BP) if be["兩平成本_bp_對0050"] > 0 else "毛報酬已輸 0050"
    be["成本區間（對零）"] = regime(be["兩平成本_bp_對零"] / PER_LEG_BP) if be["兩平成本_bp_對零"] > 0 else "毛報酬為負"

    ph = pd.DataFrame([{"相位（起始日延後）": p, **perf(simulate(d, S[deployed], Spec(phase=p), start, end)["equity"])}
                       for p in PHASES])

    alloc = []
    for a in ("ew", "invvol"):
        rs = [simulate(d, S[deployed], Spec(allocator=a, phase=p), start, end) for p in PHASES]
        sh = [perf(r["equity"])["Sharpe"] for r in rs]
        alloc.append({"配置": a, "Sharpe（4 相位平均）": np.mean(sh), "最好−最差": max(sh) - min(sh),
                      "MDD": np.mean([perf(r["equity"])["MDD"] for r in rs]),
                      "年換手(雙向)": np.mean([r["turnover_2way"] for r in rs]),
                      "有效檔數 1/HHI": np.mean([1 / r["hhi"].dropna().mean() for r in rs])})
    alloc = pd.DataFrame(alloc)

    ovs = {"無覆蓋": eq, "波動目標": overlay(res[deployed], vol_target(eq), base),
           "停損階梯": overlay(res[deployed], kill_target(eq), base, band=0.0)}
    ovt = pd.DataFrame([{"版本": k, **perf(v)} for k, v in ovs.items()])
    if overlay_on is None:  # 事先寫定：Sharpe 提升且 MDD 不變差才部署
        p0 = perf(eq)
        ok = [k for k in ("波動目標",) if perf(ovs[k])["Sharpe"] > p0["Sharpe"] and perf(ovs[k])["MDD"] >= p0["MDD"]]
        overlay_on = ok[0] if ok else "無覆蓋"

    real = pd.DataFrame([{"情境": "基準（下市以最後價、漲跌停限制）", **perf(eq)},
                         {"情境": "下市折價 −50%", **perf(simulate(d, S[deployed], Spec(haircut=0.5), start, end)["equity"])},
                         {"情境": "不考慮漲跌停（樂觀）", **perf(simulate(d, S[deployed], Spec(limit_rule=False), start, end)["equity"])}])

    for name, t in [("main", main_tbl), ("cost_sweep", sweep), ("phases", ph), ("allocator", alloc),
                    ("overlay", ovt), ("realism", real)]:
        save(t, f"{tag}_{name}")
    pd.Series(be).to_frame("值").to_csv(OUT / f"{tag}_breakeven.csv", encoding="utf-8-sig")
    out = {"res": {k: {kk: v for kk, v in r.items() if kk != "trades"} for k, r in res.items()},
           "gross": gross["equity"], "bench": bench, "overlays": ovs, "overlay_on": overlay_on, "breakeven": be,
           "sweep": sweep, "main": main_tbl, "phases": ph, "alloc": alloc, "trades": res[deployed]["trades"],
           "deployed": deployed}
    with open(CACHE / f"{tag}_backtest.pkl", "wb") as f:
        pickle.dump(out, f)
    cost_figure(out, tag)
    return out


def cost_figure(out, tag):
    sw, be = out["sweep"], out["breakeven"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(sw.每邊成本_bp, sw.Sharpe, marker="o", label="策略淨 Sharpe")
    ax[0].axvline(PER_LEG_BP, color="gray", ls="--", lw=0.8, label="假設成本")
    ax[0].axhline(perf(out["bench"]["0050"])["Sharpe"], color="C3", ls=":", label="0050 Sharpe")
    ax[0].set_xlabel("每邊成本（bp）"); ax[0].set_ylabel("Sharpe"); ax[0].legend(fontsize=7)
    ax[0].set_title("成本加到多少，策略就不如 0050？", fontsize=9)
    gs, ns = sw.loc[sw.成本倍數 == 0, "Sharpe"].iloc[0], sw.loc[sw.成本倍數 == 1, "Sharpe"].iloc[0]
    ax[1].bar(["毛 Sharpe", "淨 Sharpe", "0050"], [gs, ns, perf(out["bench"]["0050"])["Sharpe"]],
              color=["#adb5bd", "#1c7ed6", "#e03131"])
    ax[1].set_title(f"毛→淨；兩平成本 對零 {be['兩平成本_bp_對零']:.0f} bp、對 0050 {be['兩平成本_bp_對0050']:.0f} bp（每邊）",
                    fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / f"{tag}_cost.png", dpi=150); plt.close(fig)


def main():
    spec = json.loads((OUT / "04_deployed.json").read_text(encoding="utf-8"))
    dep = spec["deployed"]
    pv = pd.read_parquet(DATA / "preds_val.parquet")
    order = [dep] + [m for m in ("rule", "ridge", "lgbm", "ensemble") if m != dep]
    preds = {m: pv[pv.model == m] for m in order}
    d = load(VAL_END)
    out = run_backtests(d, preds, dep, VAL_START, VAL_END, "05_val")
    win = f"{VAL_START.date()}~{VAL_END.date()}"
    run = log_run("05_backtest", {"deployed": dep, "spec": Spec().__dict__}, win,
                  {"main": out["main"].set_index("策略").Sharpe.to_dict(), "breakeven": out["breakeven"]},
                  f"overlay={out['overlay_on']}", "pre-registered overlay rule")
    for r in out["main"].to_dict("records"):
        if "（淨）" in r["策略"]:
            ledger("portfolio backtest", r["策略"], run, "model → portfolio", win, "net Sharpe", r["Sharpe"],
                   "deployed" if r["策略"].startswith(dep) else "comparison")
    for r in out["alloc"].to_dict("records"):
        ledger("allocator", r["配置"], run, "allocator", win, "net Sharpe (phase avg)", r["Sharpe（4 相位平均）"],
               "deployed (pre-registered EW)" if r["配置"] == "ew" else "comparison only")
    for k, v in out["overlays"].items():
        if k != "無覆蓋":
            ledger("risk overlay", k, run, "overlay on/off", win, "net Sharpe", perf(v)["Sharpe"],
                   "deploy" if k == out["overlay_on"] else "do not deploy")
    spec["overlay"] = out["overlay_on"]
    (OUT / "04_deployed.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out["main"].round(3).to_string()); print(pd.Series(out["breakeven"]).to_string())
    print(out["alloc"].round(3).to_string()); print(out["phases"].round(3).to_string())
    print(pd.DataFrame([{"版本": k, **perf(v)} for k, v in out["overlays"].items()]).round(3).to_string())


if __name__ == "__main__":
    main()
