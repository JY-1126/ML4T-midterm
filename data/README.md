# data/

原始資料與加工後資料都不進 git；從 repo 根目錄重新產生：

```
python -m src.data_fetch      # 下載到 data/raw/（yfinance、FinMind；FinMind 免費額度約 2 小時）
python -m src.s02_data        # 加工成 data/prices.parquet、meta.csv、bench.parquet 並產生品質報告
```

| 檔案 | 內容 |
|---|---|
| raw/listed.csv | 現存上市櫃 4 碼普通股（TWSE / TPEx OpenAPI，含下載日） |
| raw/yf/*.parquet | yfinance 日 K（Close 已還原分割、Adj Close 另還原股利） |
| raw/delisted_list.csv | FinMind 下市名單（下市日 >= 2006-01-01，含下載日） |
| raw/finmind/{id}_price / _div.parquet | 下市股未還原日 K、除權息結果；0050、TAIEX 報酬指數 |
| prices.parquet | 長表 date, stock_id, open, high, low, close（還原）, close_raw, volume, source |
| meta.csv | 每檔來源、首末交易日、下市日、代號是否重複使用 |
| bench.parquet | 加權股價報酬指數 tri、0050 還原權息 etf0050 |
| panel_dev.parquet | 特徵＋標籤（2007 ~ 2024-10-31，開發階段） |
| preds_val / preds_holdout.parquet | 各模型 walk-forward 預測 |
