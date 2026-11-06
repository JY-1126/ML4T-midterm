"""第 2 節：協定表與實際切分日期（寫入 run_log）。只用清理後價格檔的交易日曆，不讀任何報酬。

協定本身在 setup.yaml（Week 1 提交）；這裡把它換算成實際日期，所以要在 s02 建好價格檔之後執行。

用法：python -m src.s01_setup
"""
import pandas as pd

from src.config import CFG, DATA, FOLD, HO_END, HO_START, HORIZON, P, PURGE, TRAIN_START, VAL_END, VAL_START, log_run, save
from src.stats import folds


def main():
    cal = pd.DatetimeIndex(pd.read_parquet(DATA / "prices.parquet", columns=["date"]).date.unique()).sort_values()
    proto = pd.DataFrame([
        ("訓練視窗", f"擴張視窗，自 {TRAIN_START.date()} 起"),
        ("步長／重訓頻率", f"{FOLD} 個交易日；每個 fold 重訓一次"),
        ("標籤期間", f"{HORIZON} 個交易日（t+1 開盤 → t+21 開盤）"),
        ("最長特徵回看", f"{P['longest_feature_lookback']} 個交易日（near_high250）"),
        ("Purge／Embargo", f"{PURGE} 個交易日／0（訓練永遠早於測試）"),
        ("超參數", "巢狀：每個 fold 訓練資料最後 126 日（再 purge 21 日）當內部驗證"),
        ("驗證（樣本外 walk-forward）", f"{VAL_START.date()} ~ {VAL_END.date()}"),
        ("Holdout（封存，只跑一次）", f"{HO_START.date()} ~ {HO_END.date()}（最後 24 個月）"),
        ("確認性檢定", CFG["confirmatory_test"]["primary"]),
    ], columns=["項目", "承諾"])
    save(proto, "01_protocol")
    rows = []
    for ph, (a, b) in (("驗證", (VAL_START, VAL_END)), ("holdout", (HO_START, HO_END))):
        for k, (x, y) in enumerate(folds(cal, a, b), 1):
            ix = int(cal.searchsorted(x))
            rows.append({"期間": ph, "fold": k, "測試起": x.date(), "測試迄": y.date(),
                         "訓練迄（purge 後）": cal[ix - PURGE - 1].date(),
                         "內部驗證": f"{cal[ix - PURGE - FOLD].date()} ~ {cal[ix - PURGE - 1].date()}"})
    sp = pd.DataFrame(rows)
    save(sp, "01_splits")
    log_run("01_setup", {"protocol": proto.set_index("項目").承諾.to_dict()}, "calendar only",
            {"splits": sp.astype(str).to_dict("records")}, "protocol committed", "setup v1")
    print(proto.to_string()); print(sp.to_string())


if __name__ == "__main__":
    main()
