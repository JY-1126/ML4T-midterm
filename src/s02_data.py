"""第 3 節：資料。加工價格 → 時間點檢查表（數值／股票池／代號）→ 資料品質報告。

品質報告只用驗證期結束以前的資料（holdout 的報酬不在這裡計算）。
用法：python -m src.s02_data
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CFG, DATA, FIG, OUT, RAW, VAL_END, log_run, save
from src.data_build import build
from src.panel import load

warnings.simplefilter("ignore", RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    build()
    blog = json.loads((OUT / "02_build_log.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(DATA / "meta.csv", dtype={"stock_id": str, "code": str}, parse_dates=["last_trade", "first_trade"])
    d = load(VAL_END)
    dl = d.stocks.isin(meta.stock_id[meta.source == "finmind"])

    # 1. 每期可投資檔數
    cnt = pd.DataFrame({"現存股": d.univ.loc[:, ~dl].sum(1), "後來下市": d.univ.loc[:, dl].sum(1)})
    cov = cnt.groupby(cnt.index.year).median().astype(int)
    cov["下市股占比"] = cov["後來下市"] / cov.sum(1)
    cov.index.name = "年"
    save(cov.reset_index(), "02_coverage")
    fig, ax = plt.subplots(figsize=(8, 3))
    mm = cnt.resample("ME").median()
    ax.stackplot(mm.index, mm["現存股"], mm["後來下市"], labels=["現存股", "後來下市的股票"], alpha=0.85)
    ax.axvline(pd.Timestamp(CFG["evaluation_protocol"]["validation"][0]), color="k", ls="--", lw=0.8)
    ax.set_title("每月可投資檔數（中位數，至驗證期結束）"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "02_coverage.png", dpi=150); plt.close(fig)

    # 2. 存活者偏誤
    fm = meta[meta.source == "finmind"].copy()
    leavers = fm.groupby(fm.last_trade.dt.year).size()
    r = d.close.pct_change(fill_method=None)
    lu = d.univ.shift(1, fill_value=False)
    surv_mask = lu & ~pd.DataFrame(np.broadcast_to(dl, lu.shape), index=lu.index, columns=lu.columns)
    yr = lambda s: (1 + s.fillna(0)).groupby(s.index.year).prod() - 1
    surv = pd.DataFrame({"離開樣本檔數": leavers, "池等權（含下市）": yr(r.where(lu).mean(1)),
                         "池等權（只有現存）": yr(r.where(surv_mask).mean(1))})
    surv = surv[(surv.index >= 2007) & (surv.index <= VAL_END.year)].fillna({"離開樣本檔數": 0})
    surv["偏誤（百分點）"] = (surv.iloc[:, 2] - surv.iloc[:, 1]) * 100
    surv.index.name = "年"
    save(surv.reset_index(), "02_survivorship")

    # 3. 公司行動：原始 vs 還原
    raw_p = pd.read_parquet(DATA / "prices.parquet", columns=["date", "stock_id", "close", "close_raw", "source"])
    ndiv = {s: len(pd.read_parquet(RAW / "finmind" / f"{c}_div.parquet"))
            for s, c in zip(fm.stock_id, fm.code) if (RAW / "finmind" / f"{c}_div.parquet").exists()}
    pick = max(ndiv, key=ndiv.get)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, s in zip(axes, ["2330", pick]):
        g = raw_p[(raw_p.stock_id == s) & (raw_p.date <= VAL_END)].set_index("date").sort_index()
        ax.plot(g.index, g.close_raw, lw=0.9, label="原始收盤")
        ax.plot(g.index, g.close, lw=0.9, label="還原權息收盤")
        ax.set_yscale("log"); ax.set_title(f"{s}（{g.source.iloc[0]}）", fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("公司行動檢查：除權息日前的價格等比例下調", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "02_corporate_action.png", dpi=150); plt.close(fig)

    # 4. OHLC、重複
    srcs = ["yfinance", "finmind"]
    chk = pd.DataFrame({k: [blog.get(k, {}).get(s, 0) if isinstance(blog.get(k), dict) else "" for s in srcs]
                        for k in ["原始筆數", "成交量為 0（刪除）", "OHLC <= 0 或缺值（刪除）",
                                  "high < max(open, close)", "low > min(open, close)"]}, index=srcs)
    chk.index.name = "來源"
    save(chk.reset_index(), "02_ohlc")

    # 5. 缺值：存續期間內沒有 K 棒的交易日（不補值；特徵視窗用 min_periods，成交日沒有 K 棒就不交易）
    alive = (d.close.notna().cumsum() > 0) & (d.close.notna()[::-1].cumsum()[::-1] > 0)
    gap = alive & d.close.isna()
    miss = pd.DataFrame({"存續格數": alive.groupby(d.dates.year).sum().sum(1),
                         "缺 K 棒格數": gap.groupby(d.dates.year).sum().sum(1)})
    miss["缺值比例"] = miss["缺 K 棒格數"] / miss["存續格數"]
    miss.index.name = "年"
    save(miss.reset_index(), "02_missing")

    reused = meta[meta.code_reused.fillna(False).astype(bool)]
    no_px = meta[meta.status.isin(["無價格", "未下載"])]
    summary = {
        "價格期間（檔案）": blog["期間"], "開發階段讀取至": str(VAL_END.date()),
        "股票數（含下市）": blog["股票數"], "下市股（有價格）": blog["下市股（有價格）"],
        "下市股曾進入可投資池": int(d.univ.loc[:, dl].any().sum()),
        "下市名單中無價格": len(no_px), "代號重複使用": len(reused),
        "(日期, 代號) 重複（刪除）": blog["(日期, 代號) 重複（刪除）"],
        "殘缺日（整日刪除）": len(blog["殘缺日（整日刪除）"]),
        "超過漲跌幅的跳動（抹平）": blog["超過漲跌幅的跳動（等比例抹平）"],
        "公司行動檢查（下市股）": pick,
        "下載日": pd.read_csv(RAW / "listed.csv", dtype=str).download_date.iloc[0],
    }
    pit = pd.DataFrame([
        ("數值", "特徵只用 t 日收盤以前的價量", "PASS", "s03 截斷重算：資料截在較早日期重算，重疊日期的特徵完全相同"),
        ("數值", "非價格資料的公布延遲", "N/A", "只用價量；除權息在除權息日前公告，還原為等比例係數"),
        ("數值", "還原價不引入未來資訊", "PASS", "視窗內價格比值只受視窗內的除權息影響；篩選用未還原價"),
        ("數值", "yfinance 的 Close 已還原分割", "揭露", "未還原價篩選（>= 10 元）對日後分割的股票偏低；台股分割少見"),
        ("股票池", "可投資條件只用 t 日資訊", "PASS", "未還原收盤 >= 10、20 日均成交額 >= 2,000 萬、上市櫃滿 120 日、t 日有成交"),
        ("股票池", "包含後來下市的股票", "PASS", f"FinMind 補入 {blog['下市股（有價格）']} 檔下市股"),
        ("股票池", "下市名單本身的完整性", "FAIL→揭露", "FinMind 名單無法與官方逐年核對；漏掉的下市股會讓結果偏樂觀"),
        ("代號", "代號重複使用", "PASS", f"下市代號與現存代號重複 {len(reused)} 檔；重複者以「代號_d年份」分開，各自只用生效期間"),
        ("代號", "股票種類變更（ETF、TDR、特別股）", "PASS", "只取 4 碼普通股，排除 00xx、91xx、名稱含 DR"),
        ("代號", "更名", "N/A", "以代號為主鍵，名稱不參與計算"),
    ], columns=["層次", "檢查項目", "結果", "證據"])
    save(pit, "02_pit_checklist")
    pd.Series(summary).to_frame("值").to_csv(OUT / "02_summary.csv", encoding="utf-8-sig")
    log_run("02_data", {"build": "src.data_build"}, f"{d.dates[0].date()}~{VAL_END.date()}", summary, "dataset built")
    print(pd.Series(summary).to_string()); print(cov.to_string()); print(surv.round(3).to_string())


if __name__ == "__main__":
    main()
