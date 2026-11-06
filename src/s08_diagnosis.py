"""holdout 之後的診斷（Chapter 20）：只解釋、不改任何設定，也不再產生任何策略版本。

1. 每個進模特徵在驗證期 vs holdout 的 IC（哪些訊號衰退）
2. holdout 期間依市場狀態分組的部署模型 IC（是不是集中在某種狀態失效）
用法：python -m src.s08_diagnosis
"""
import json

import pandas as pd

from src.config import DATA, HO_END, HO_START, OUT, VAL_END, VAL_START, log_run, save
from src.features import build
from src.panel import load
from src.s06_risk import market_state
from src.stats import daily_ic, nw_mean


def main():
    feats = json.loads((OUT / "04_deployed.json").read_text(encoding="utf-8"))["features"]
    d = load(HO_END)
    panel, _ = build(d)
    rows = []
    for f in [f for f in feats if f not in ("mkt_vol20", "mkt_trend", "disp20")]:
        r = {"特徵": f}
        for lab, a, b in (("驗證", VAL_START, VAL_END), ("holdout", HO_START, HO_END)):
            s = nw_mean(daily_ic(panel[(panel.date >= a) & (panel.date <= b)], f))
            r |= {f"{lab} IC": s["mean"], f"{lab} lo": s["lo"], f"{lab} hi": s["hi"]}
        rows.append(r)
    fi = pd.DataFrame(rows)
    fi["變化"] = fi["holdout IC"] - fi["驗證 IC"]
    save(fi, "08_feature_ic", floatfmt=".4f")

    ph = pd.read_parquet(DATA / "preds_holdout.parquet")
    p = ph[ph.model == "lgbm"].merge(panel[["date", "stock_id", "fwd20"]], on=["date", "stock_id"])
    ic = daily_ic(p, "score")
    st = market_state(d).reindex(ic.index)
    rg = []
    for col in ("vol", "trend"):
        for k, g in ic.groupby(st[col], observed=True):
            s = nw_mean(g)
            rg.append({"狀態": k, "日數": s["n"], "IC": s["mean"], "lo": s["lo"], "hi": s["hi"]})
    rg = pd.DataFrame(rg)
    save(rg, "08_holdout_regime_ic", floatfmt=".4f")
    log_run("08_diagnosis", {"note": "post-holdout diagnosis only; no decision"}, f"{HO_START.date()}~{HO_END.date()}",
            {"feature_ic": fi.set_index("特徵")[["驗證 IC", "holdout IC"]].to_dict("index")}, "no decision", "diagnosis")
    print(fi.round(4).to_string()); print(rg.round(4).to_string())


if __name__ == "__main__":
    main()
