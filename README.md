# 台股族群擴散 × 個股時間序列（ML4T 期中）

一次完整的 ML4T 工作流程：機制 → 交易設定 v1 → 時間點正確的資料 → 標籤與特徵 triage → 模型階梯 →
事件驅動回測（成本、投組、風險）→ 只執行一次的 holdout → 診斷與下一步。

| 繳交物 | 位置 |
|---|---|
| Proposal（PDF）與簡報 | 另外繳交，不放在 repo |
| 原始資料快照（2026-11-07 下載） | [`data/raw/`](data/raw/)（見 [`data/README.md`](data/README.md)） |
| 交易設定與協定（Week 1 提交，看資料之前） | [`setup.yaml`](setup.yaml) |
| 研究帳本（Strategy → Trial family → Trial → Run） | [`research_ledger.csv`](research_ledger.csv) |
| 執行紀錄（git commit、資料快照、設定、評估區間、指標、決策） | [`run_log.csv`](run_log.csv) |
| Holdout 封存紀錄 | [`results/HOLDOUT_TOUCHED.json`](results/HOLDOUT_TOUCHED.json) |
| 各階段 notebook | [`notebooks/01_setup … 07_results`](notebooks/) |

## 切分

| | 期間 |
|---|---|
| 訓練起點（擴張視窗） | 2007-01-02 |
| 驗證（walk-forward 樣本外，126 日一個 fold、purge 21 日） | 2012-01-02 ~ 2024-10-31 |
| Holdout（最後 24 個月，Week 1 封存，只執行一次） | 2024-11-01 ~ 2026-10-30 |

開發階段的程式只讀得到 2024-10-31 以前的價格（`src/panel.py` 的 `load(end)`）；
`src/s07_holdout.py` 先寫入封存紀錄才讀 holdout，之後重跑必須寫理由（`--rerun-reason`）。

## 重現（從 repo 根目錄）

原始資料已放在 `data/raw/`（報告所用的同一份快照，2026-11-07 下載），**不需要重新下載**：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # 精確版本見 requirements-lock.txt（Python 3.14）
.venv/Scripts/python run_all.py --holdout                   # data/raw → s02 → s01 → s03 … s07 → s08（約 1 小時）
.venv/Scripts/python make_notebooks.py                      # 重新執行 notebooks
```

- 只想看結果：`results/` 的表格與圖、`notebooks/07_results.ipynb` 都已附上執行結果，不必重跑。
- 用 repo 內的原始資料重跑，總表數字應與報告相同（LightGBM 固定 `random_state = 42`）。
- 想從頭重新下載：`run_all.py --fetch --holdout`（yfinance 約 10 分鐘；FinMind 免費額度約每小時 300 次，下市股約需 2 小時）。
  新下載的資料可能和快照不同（yfinance 會回溯修正還原價、FinMind 名單會更新），數字因此可能小幅改變。
- holdout 已封存：`run_all.py --holdout` 會以「reproduction run」為理由重跑，並附加在封存紀錄，不改任何設定。

## Repo 結構

```
setup.yaml              交易設定 v1、驗證協定、確認性檢定（Week 1）
research_ledger.csv     研究帳本        run_log.csv   執行紀錄
run_all.py              一鍵重現        make_notebooks.py  產生 notebooks
src/
  config.py             路徑、切分日期、帳本與紀錄
  data_fetch.py         下載原始資料（yfinance、FinMind、TWSE/TPEx OpenAPI）
  data_build.py         還原權息、代號重複、清理 → data/prices.parquet
  panel.py              寬表、可投資池、基準；load(end) 控制可讀到的最後日期
  features.py           同伴股（相關係數）與個股特徵、標籤
  stats.py              Newey–West、IC、Sharpe 推論、DSR、最短紀錄
  models.py             rule / Ridge / LightGBM / 集成、walk-forward、conformal
  engine.py             事件驅動回測
  s01_setup.py … s07_holdout.py   各階段（對應 notebooks/01 … 07）
results/                表格（csv / md）與 figures/
data/raw/               原始資料快照（進 git）；加工檔（prices.parquet 等，> 100 MB）由 s02 重建、不進 git
```

## 事前接觸資料的揭露

研究者在本期中之前做過同主題的探索（不同程式碼與設定），看過 2010–2026 台股走勢與部分特徵的表現。
這無法消除。本 repo 的處理方式：設定在下載資料前提交（commit `75212d2`）、holdout 由程式隔離且只執行一次、
Deflated Sharpe 另以「本次試驗數 + 事前約 270 次」做敏感度分析。詳見報告第 2 節。
