"""模型階梯與 walk-forward（擴張視窗、每 fold 重訓、purge 21 日、巢狀內部驗證）。

每個 fold [a, b]：
  訓練    TRAIN_START ≤ t ≤ a 前第 22 個交易日（purge 21 日：標籤用到 t+21 的開盤，必須早於 a），每 5 日取一列
  內部驗證 訓練期最後 126 日；內部訓練再往前 purge 21 日
  rule    score = peer_ret20 的排名（事先由機制決定，不訓練）
  ridge   α 在內部驗證上依每日 IC 選，再用整段訓練資料重配；winsorize 1%/99% 與標準化只用訓練資料
  lgbm    4 組格點在內部驗證上依每日 IC 選（l1、early stopping），再以最佳樹數在整段訓練資料重配
  ensemble 三者每日 z 分數平均
  conformal 內部驗證的 |y − ŷ| 分位數（80%、90%）→ 測試 fold 的實際覆蓋率
"""
import itertools
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import CFG, FOLD, PURGE, TRAIN_START
from src.stats import daily_ic

warnings.simplefilter("ignore")
M = CFG["models"]["ladder"]
ALPHAS = [float(a) for a in M["ridge"]["alpha_grid"]]
LG = M["lightgbm"]
GRID = [dict(zip(LG["grid"], v)) for v in itertools.product(*LG["grid"].values())]
EVERY = CFG["evaluation_protocol"]["train_sample_every_days"]
SEED = 42


class Prep:
    """winsorize + 標準化 + 缺值補中位數；所有參數只從訓練資料估計。"""

    def fit(self, X: pd.DataFrame):
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)
        Xc = X.clip(self.lo, self.hi, axis=1)
        self.med = Xc.median()
        Xc = Xc.fillna(self.med)
        self.mu, self.sd = Xc.mean(), Xc.std().replace(0, 1)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return ((X.clip(self.lo, self.hi, axis=1).fillna(self.med) - self.mu) / self.sd).values


def _ic(df: pd.DataFrame, pred: np.ndarray) -> float:
    return daily_ic(df.assign(_p=pred), "_p").mean()


def split(panel: pd.DataFrame, dates: pd.DatetimeIndex, a: pd.Timestamp):
    ia = int(np.searchsorted(dates, np.datetime64(a)))
    tr_end = dates[ia - PURGE - 1]
    in_start = dates[ia - PURGE - FOLD]
    in_tr_end = dates[ia - PURGE - FOLD - PURGE - 1]
    pos = pd.Series(np.arange(len(dates)), index=dates)
    sampled = set(dates[(pos.values % EVERY) == 0])
    base = panel[(panel.date >= TRAIN_START) & panel.y.notna()]
    tr = base[(base.date <= tr_end) & base.date.isin(sampled)]
    inner_tr = tr[tr.date <= in_tr_end]
    inner_va = base[(base.date >= in_start) & (base.date <= tr_end)]  # 內部驗證用每日（評估 IC）
    return tr, inner_tr, inner_va


def fit_fold(panel, dates, feats, a, b) -> dict:
    tr, itr, iva = split(panel, dates, a)
    te = panel[(panel.date >= a) & (panel.date <= b)]
    out = {"test": te[["date", "stock_id"]].copy(), "n_train": len(tr), "train_end": tr.date.max()}

    # ridge
    pi = Prep().fit(itr[feats])
    Xi, Xv = pi.transform(itr[feats]), pi.transform(iva[feats])
    sweep = {al: _ic(iva, Ridge(alpha=al).fit(Xi, itr.y).predict(Xv)) for al in ALPHAS}
    al = max(sweep, key=lambda k: (np.nan_to_num(sweep[k], nan=-9), k))
    pf = Prep().fit(tr[feats])
    rg = Ridge(alpha=al).fit(pf.transform(tr[feats]), tr.y)
    out["ridge"] = rg.predict(pf.transform(te[feats]))
    out["ridge_alpha"], out["ridge_sweep"], out["ridge_coef"] = al, sweep, dict(zip(feats, rg.coef_))
    q_r = np.abs(iva.y.values - Ridge(alpha=al).fit(Xi, itr.y).predict(Xv))

    # lightgbm
    best, scores = None, {}
    for g in GRID:
        m = lgb.LGBMRegressor(objective=LG["objective"], random_state=SEED, verbose=-1, n_jobs=-1, **LG["fixed"], **g)
        m.fit(itr[feats], itr.y, eval_set=[(iva[feats], iva.y)], eval_metric="l1",
              callbacks=[lgb.early_stopping(LG["early_stopping_rounds"], verbose=False)])
        ic = _ic(iva, m.predict(iva[feats]))
        scores[str(g)] = ic
        if best is None or ic > best[0]:
            best = (ic, g, max(m.best_iteration_ or 1, 20), m)
    _, g, n_iter, m_in = best
    fixed = {**LG["fixed"], "n_estimators": n_iter}
    lm = lgb.LGBMRegressor(objective=LG["objective"], random_state=SEED, verbose=-1, n_jobs=-1, **fixed, **g).fit(tr[feats], tr.y)
    out["lgbm"] = lm.predict(te[feats])
    out["lgbm_params"], out["lgbm_grid"], out["lgbm_model"] = {**g, "n_estimators": n_iter}, scores, lm
    q_l = np.abs(iva.y.values - m_in.predict(iva[feats]))

    # conformal（以內部驗證殘差校準，套用到重配後的模型）
    for name, q in (("ridge", q_r), ("lgbm", q_l)):
        for lvl in (0.8, 0.9):
            w = np.quantile(q, lvl)
            ok = te.y.notna().values
            out[f"cov_{name}_{int(lvl * 100)}"] = float((np.abs(te.y.values[ok] - out[name][ok]) <= w).mean())
            out[f"width_{name}_{int(lvl * 100)}"] = float(2 * w)
    out["ridge_prep"], out["ridge_model"] = pf, rg
    return out


def zscore(s: pd.Series, by: pd.Series) -> pd.Series:
    g = s.groupby(by)
    return (s - g.transform("mean")) / g.transform("std")


def walk_forward(panel, dates, feats, fold_list, log=print) -> tuple[pd.DataFrame, list]:
    preds, info = [], []
    for k, (a, b) in enumerate(fold_list):
        o = fit_fold(panel, dates, feats, a, b)
        t = o["test"]
        te = panel.loc[t.index]
        t = t.assign(rule=te["peer_ret20"].values, ridge=o["ridge"], lgbm=o["lgbm"])
        preds.append(t)
        info.append({k2: v for k2, v in o.items() if k2 not in ("test", "ridge", "lgbm")} | {"fold": k, "a": a, "b": b})
        log(f"  fold {k + 1}/{len(fold_list)} {a.date()}~{b.date()} n_train={o['n_train']} "
            f"alpha={o['ridge_alpha']:g} lgbm={o['lgbm_params']}")
    p = pd.concat(preds, ignore_index=True)
    z = [zscore(p[m], p.date) for m in ("rule", "ridge", "lgbm")]
    p["ensemble"] = pd.concat(z, axis=1).mean(axis=1)
    long = p.melt(id_vars=["date", "stock_id"], value_vars=["rule", "ridge", "lgbm", "ensemble"],
                  var_name="model", value_name="score")
    return long, info
