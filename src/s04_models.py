"""第 5 節：模型階梯（rule → Ridge → LightGBM → 集成）、成對比較、選擇規則、SHAP、conformal。驗證期 walk-forward。

輸出 data/preds_val.parquet、results/04_*.csv、results/04_deployed.json、figures/04_*.png
用法：python -m src.s04_models
"""
import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import CACHE, CFG, DATA, FIG, OUT, SMOKE, VAL_END, VAL_START, ledger, log_run, save
from src.models import ALPHAS, GRID, walk_forward
from src.stats import daily_ic, folds, ic_summary, nw_mean

warnings.simplefilter("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
LADDER = ["rule", "ridge", "lgbm", "ensemble"]
NAMES = {"rule": "規則（族群動能）", "ridge": "Ridge", "lgbm": "LightGBM", "ensemble": "集成"}


def ic_series(preds: pd.DataFrame, panel: pd.DataFrame, model: str) -> pd.Series:
    p = preds[preds.model == model].merge(panel[["date", "stock_id", "fwd20"]], on=["date", "stock_id"])
    return daily_ic(p, "score")


def select(lad: pd.DataFrame, paired: pd.DataFrame) -> str:
    """setup.yaml 的選擇規則：沿階梯往上，只有 IC CI 不含 0 且相對前一階 ΔIC CI 不含 0（正）才升級。"""
    sig = lambda m: lad.loc[m, "lo"] > 0
    dep = "rule" if sig("rule") else None
    for prev, cur in (("rule", "ridge"), ("ridge", "lgbm")):
        if not sig(cur):
            continue
        if dep is None:
            dep = cur
            continue
        row = paired[(paired.比較 == f"{cur} − {dep}")]
        if len(row) and row.lo.iloc[0] > 0:
            dep = cur
    if dep in ("ridge", "lgbm"):
        row = paired[paired.比較 == f"ensemble − {dep}"]
        if len(row) and row.lo.iloc[0] > 0 and sig("ensemble"):
            dep = "ensemble"
    return dep


def main():
    panel = pd.read_parquet(DATA / "panel_dev.parquet")
    feats = json.loads((OUT / "03_model_features.json").read_text(encoding="utf-8"))["features"]
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    F = folds(dates, VAL_START, VAL_END)
    F = F[::8] if SMOKE else F  # 除錯只跑少數 fold
    preds, info = walk_forward(panel, dates, feats, F)
    preds.to_parquet(DATA / "preds_val.parquet", index=False)

    ics = {m: ic_series(preds, panel, m) for m in LADDER}
    lad = pd.DataFrame({m: ic_summary(ics[m], F) for m in LADDER}).T
    lad.index.name = "模型"
    pairs = [("ridge", "rule"), ("lgbm", "ridge"), ("lgbm", "rule"), ("ensemble", "lgbm"), ("ensemble", "ridge")]
    rows = []
    for a, b in pairs:
        dlt = (ics[a] - ics[b]).dropna()
        s = nw_mean(dlt)
        rows.append({"比較": f"{a} − {b}", "ΔIC": s["mean"], "lo": s["lo"], "hi": s["hi"], "p": s["p"]})
    paired = pd.DataFrame(rows)
    dep = select(lad, paired) or "none"
    save(lad.reset_index(), "04_ladder", floatfmt=".4f")
    save(paired, "04_paired", floatfmt=".4f")

    # α 掃描（內部驗證 IC，跨 fold 平均）與 LightGBM 格點
    sw = pd.DataFrame([i["ridge_sweep"] for i in info]).mean().rename("內部驗證 IC（fold 平均）")
    sw.index.name = "alpha"
    chosen = pd.Series([i["ridge_alpha"] for i in info]).value_counts().rename("被選中次數")
    save(pd.concat([sw, chosen], axis=1).reset_index(), "04_ridge_alpha", floatfmt=".4f")
    lg = pd.DataFrame([i["lgbm_grid"] for i in info]).mean().rename("內部驗證 IC（fold 平均）").to_frame()
    lg["被選中次數"] = pd.Series([str({k: v for k, v in i["lgbm_params"].items() if k != "n_estimators"})
                                  for i in info]).value_counts().reindex(lg.index).fillna(0).values
    lg.index.name = "LightGBM 格點"
    save(lg.reset_index(), "04_lgbm_grid", floatfmt=".4f")

    # 每 fold IC 與 HAC 區間
    fr = []
    for k, (a, b) in enumerate(F):
        for m in LADDER:
            s = nw_mean(ics[m][(ics[m].index >= a) & (ics[m].index <= b)])
            fr.append({"fold": k + 1, "起": a.date(), "迄": b.date(), "模型": m, "IC": s["mean"], "lo": s["lo"], "hi": s["hi"]})
    fr = pd.DataFrame(fr)
    save(fr, "04_fold_ic", floatfmt=".4f")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for m, c in zip(LADDER, ["#adb5bd", "#4dabf7", "#1c7ed6", "#f08c00"]):
        g = fr[fr.模型 == m]
        ax.plot(g.fold, g.IC, marker="o", ms=3, color=c, label=NAMES[m], lw=1.6 if m == dep else 1)
        if m == dep:
            ax.fill_between(g.fold, g.lo, g.hi, color=c, alpha=0.15, label=f"{NAMES[m]} HAC 95%")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("驗證 fold（每段 126 交易日，2012 → 2024-10）"); ax.set_ylabel("每日 Spearman IC")
    ax.set_title("每個 fold 的預測力是否穩定為正？", fontsize=10); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "04_fold_ic.png", dpi=150); plt.close(fig)

    # 滾動 3 個月（63 日）IC
    fig, ax = plt.subplots(figsize=(9, 3))
    for m, c in (("rule", "#adb5bd"), (dep if dep != "none" else "lgbm", "#1c7ed6")):
        ax.plot(ics[m].rolling(63, min_periods=40).mean(), color=c, lw=1, label=NAMES[m])
    ax.axhline(0, color="k", lw=0.8); ax.legend(fontsize=8)
    ax.set_title("滾動 3 個月 IC：預測力有沒有長期跌破 0？（驗證期）", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "04_rolling_ic.png", dpi=150); plt.close(fig)

    # SHAP（部署模型；若為 ridge 用線性 SHAP，集成則以 lgbm 成分解讀）
    rng = np.random.default_rng(0)
    shap_rows, beeswarm = [], []
    for i, (a, b) in zip(info, F):
        te = panel[(panel.date >= a) & (panel.date <= b)]
        smp = te.iloc[rng.choice(len(te), min(1500, len(te)), replace=False)]
        if dep == "ridge":
            X = i["ridge_prep"].transform(smp[feats])
            sv = X * np.array([i["ridge_coef"][f] for f in feats])
        else:
            sv = shap.TreeExplainer(i["lgbm_model"]).shap_values(smp[feats])
        for j, f in enumerate(feats):
            x = smp[f].values
            ok = np.isfinite(x)
            sign = np.corrcoef(x[ok], sv[ok, j])[0, 1] if ok.sum() > 10 else np.nan
            shap_rows.append({"fold": i["fold"] + 1, "特徵": f, "mean|SHAP|": np.abs(sv[:, j]).mean(), "方向": sign})
        beeswarm.append((smp[feats], sv))
    sh = pd.DataFrame(shap_rows)
    imp = sh.groupby("特徵").agg(**{"mean|SHAP|": ("mean|SHAP|", "mean"), "方向（平均）": ("方向", "mean"),
                                    "方向一致 fold 比例": ("方向", lambda s: (np.sign(s) == np.sign(s.mean())).mean())})
    rk = sh.pivot(index="fold", columns="特徵", values="mean|SHAP|").rank(axis=1, ascending=False)
    imp["排名標準差"] = rk.std()
    imp = imp.sort_values("mean|SHAP|", ascending=False)
    save(imp.reset_index(), "04_shap", floatfmt=".4f")
    X = pd.concat([x for x, _ in beeswarm]); V = np.vstack([v for _, v in beeswarm])
    plt.figure()
    shap.summary_plot(V, X, show=False, max_display=len(feats), plot_size=(7, 4.6))
    plt.title(f"SHAP：{NAMES.get(dep, dep)} 的預測靠哪些特徵、方向是否符合機制？", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "04_shap.png", dpi=150); plt.close("all")

    # conformal 覆蓋率
    cf = pd.DataFrame([{"fold": i["fold"] + 1, **{k: v for k, v in i.items() if k.startswith(("cov_", "width_"))}}
                       for i in info])
    m_ = "ridge" if dep == "ridge" else "lgbm"
    cf["低於 80% 標記"] = cf[f"cov_{m_}_80"] < 0.8
    save(cf, "04_conformal", floatfmt=".3f")

    spec = {"deployed": dep, "features": feats, "selection_rule": CFG["models"]["selection_rule"]}
    (OUT / "04_deployed.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(CACHE / "04_info.pkl", "wb") as f:
        pickle.dump([{k: v for k, v in i.items() if not k.endswith(("_model", "_prep"))} for i in info], f)

    win = f"{VAL_START.date()}~{VAL_END.date()}"
    run = log_run("04_models", {"features": feats, "alpha_grid": ALPHAS, "lgbm_grid": GRID, "folds": len(F)}, win,
                  {"ladder": lad[["IC", "lo", "hi"]].to_dict("index"), "paired": paired.to_dict("records"),
                   "deployed": dep}, f"deploy {dep}", "selection rule")
    for m in LADDER:
        ledger("model class", m, run, "model rung", win, "daily IC", lad.loc[m, "IC"],
               "deploy" if m == dep else "not deployed", "IC CI excludes 0 & paired ΔIC")
    for al, v in sw.items():
        ledger("ridge alpha (nested)", f"alpha={al:g}", run, "hyperparameter", "inner validation", "inner IC", v,
               f"chosen {int(chosen.get(al, 0))}/{len(F)} folds")
    for g, v in lg["內部驗證 IC（fold 平均）"].items():
        ledger("lgbm grid (nested)", g, run, "hyperparameter", "inner validation", "inner IC", v,
               f"chosen {int(lg.loc[g, '被選中次數'])}/{len(F)} folds")
    print(lad.round(4).to_string()); print(paired.round(4).to_string()); print("deployed:", dep)
    print(imp.round(4).to_string()); print(cf.mean().round(3).to_string())


if __name__ == "__main__":
    main()
