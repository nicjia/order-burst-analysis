#!/usr/bin/env python3
"""burst-defs-raw-v1 stage 2: forecasting power of seven burst definitions, with order-book features.

Input: results/burst_forecasting/burst_defs_raw/RAW_all.csv.gz (one row per sampled burst; see burst_defs_raw.py).
Split (3-year window): train = 60 names, 2022-23; test = 60 different names, 2024.
Target: decision-to-close mid move in excess of the equal-weight intraday market (bps, signed by the burst side);
also the 30-minute move after T_dec. Nested sets per definition:
  CTRL   move since the open, 30-min pre-move, time of day, spread          (known non-burst predictors)
  IMPACT peak impact, displacement at 1 and 10 min, mean displacement, ratio (the burst's own price path)
  STRUCT children, size/ADV, duration, modal-clip count                      (burst structure)
  BOOK   queue imbalance at the first packet and over the burst, trade-flow imbalance over the prior 60 s,
         quote order-flow imbalance before and during the burst               (order-book flow)
Gradient boosting (depth 3, fixed). Metrics on the test names: daily rank IC (NW t), top-minus-bottom decile
(gross, bps), and the paired daily IC gain of each richer set. Plus the reversal benchmark: daily rank correlation
of the market-excess move since the open with the market-excess move to the close, at burst decision times vs at
random times (random_time_reversal.py).
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import p4_analyze as PA

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "results" / "burst_forecasting"
CTRL = ["own_open_bps", "pre30_bps", "tod", "spread_dec"]
IMPACT = ["peak_bps", "d60_bps", "d600_bps", "dmean_bps", "ratio"]
STRUCT = ["log_n", "log_q_adv", "log_dur", "mode_count"]
BOOK = ["imb_first", "imb_mean", "tfi60", "qofi60", "qofi_burst"]
SETS = {"CTRL": CTRL, "CTRL+IMPACT": CTRL + IMPACT, "+STRUCT": CTRL + IMPACT + STRUCT,
        "+STRUCT+BOOK": CTRL + IMPACT + STRUCT + BOOK, "BOOK alone": BOOK}


def market_excess(d):
    z = np.load(D / "market_index" / "TEST_mkt.npz"); idx = dict(zip(z["dates"], z["idx"]))
    k = np.clip(np.ceil((d.t_dec.to_numpy() - 34200.0) / 60.0).astype(int) - 1, 0, 389)
    k30 = np.clip(np.ceil((np.minimum(d.t_dec.to_numpy() + 1800, 57600.0) - 34200.0) / 60.0).astype(int) - 1, 0, 389)
    dates = d.date.astype(str).to_numpy()
    mc = np.array([idx[x][389] - idx[x][a] if x in idx else np.nan for x, a in zip(dates, k)])
    m30 = np.array([idx[x][b] - idx[x][a] if x in idx else np.nan for x, a, b in zip(dates, k, k30)])
    mo = np.array([idx[x][a] if x in idx else np.nan for x, a in zip(dates, k)])     # 9:31 -> T_dec
    d["d_close_x"] = d.d_close - d.side * mc * 1e4
    d["d30_x"] = d.d30 - d.side * m30 * 1e4
    d["open_x_raw"] = d.side * d.own_open_bps - mo * 1e4                               # unsigned, market-excess
    d["close_x_raw"] = d.side * d.d_close_x
    return d


def metrics(te, pred, y):
    t = pd.DataFrame(dict(date=te.date.to_numpy(), p=pred, y=te[y].to_numpy())).dropna()
    ic, sp = [], []
    for _, g in t.groupby("date"):
        if len(g) < 20:
            continue
        ic.append(g.p.rank().corr(g.y.rank()))
        r = g.p.rank(method="first"); k = max(1, len(g) // 10)
        sp.append(g.y[r > len(g) - k].mean() - g.y[r <= k].mean())
    return dict(ic=PA.nw_t(np.array(ic)), spread=PA.nw_t(np.array(sp)), _ic=pd.Series(ic))


def main():
    d = pd.read_csv(D / "burst_defs_raw" / "RAW_all.csv.gz", dtype={"date": str})
    nd = pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str},
                     usecols=["permno", "date", "family", "adv20"])
    d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
    d["log_q_adv"] = np.log(d.vol / d.adv20); d["log_n"] = np.log(d.n); d["log_dur"] = np.log1p(d.t_e - d.t_b)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    for c in ("d_close", "d30"):
        d.loc[d[c].abs() > 1000, c] = np.nan
    d = market_excess(d)
    train_names = set(pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    tr_all = d[d.permno.isin(train_names) & d.date.str[:4].isin(["2022", "2023"])]
    te_all = d[~d.permno.isin(train_names) & (d.date.str[:4] == "2024")]
    rnd = pd.read_csv(D / "random_rev" / "TEST_random.csv.gz", dtype={"date": str}) if (D / "random_rev" / "TEST_random.csv.gz").exists() else None
    res = {}
    print("bursts: train %d (%d names, 2022-23) | test %d (%d names, 2024)" % (len(tr_all), tr_all.permno.nunique(), len(te_all), te_all.permno.nunique()))
    for defn in sorted(d.defn.unique()):
        tr, te = tr_all[tr_all.defn == defn], te_all[te_all.defn == defn]
        if len(tr) < 5000 or len(te) < 2000:
            print("\n%s: too few bursts (train %d, test %d)" % (defn, len(tr), len(te))); continue
        r = {"n_train": int(len(tr)), "n_test": int(len(te))}
        print("\n=== %s  train %d  test %d" % (defn, len(tr), len(te)))
        for y in ("d_close_x", "d30_x"):
            t = tr[tr[y].notna()]; lo, hi = t[y].quantile([0.01, 0.99])
            e = te[te[y].notna()].copy(); e[y] = e[y].clip(lo, hi)
            out = {}
            for k, cols in SETS.items():
                m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=200,
                                                  early_stopping=False, random_state=1).fit(t[cols].to_numpy(float), t[y].clip(lo, hi).to_numpy())
                out[k] = metrics(e, m.predict(e[cols].to_numpy(float)), y)
            line = "  %-10s" % y + " ".join("%s IC %+.4f (t %5.2f)" % (k, v["ic"]["mean"], v["ic"]["t"]) for k, v in out.items())
            print(line)
            g_book = PA.nw_t((out["+STRUCT+BOOK"]["_ic"] - out["+STRUCT"]["_ic"]).to_numpy())
            g_burst = PA.nw_t((out["+STRUCT+BOOK"]["_ic"] - out["CTRL"]["_ic"]).to_numpy())
            print("  %-10s IC gain from BOOK over CTRL+IMPACT+STRUCT %+.4f (t %.2f) | all burst+book over CTRL %+.4f (t %.2f) | D10-D1 full %+.2f bps (t %.2f)"
                  % (y, g_book["mean"], g_book["t"], g_burst["mean"], g_burst["t"], out["+STRUCT+BOOK"]["spread"]["mean"], out["+STRUCT+BOOK"]["spread"]["t"]))
            for v in out.values():
                v.pop("_ic", None)
            r[y] = dict(sets=out, gain_book=g_book, gain_all_burst=g_burst)
        # reversal at burst times (unsigned, market-excess)
        ics = [g.open_x_raw.rank().corr(g.close_x_raw.rank()) for _, g in te.dropna(subset=["open_x_raw", "close_x_raw"]).groupby("date") if len(g) >= 20]
        r["reversal_ic_burst_times"] = PA.nw_t(np.array(ics))
        res[defn] = r
    if rnd is not None:
        rnd = rnd[~rnd.permno.isin(train_names) & (rnd.date.str[:4] == "2024")]
        rnd["x"] = rnd.x_bps - rnd.mx_bps; rnd["y"] = rnd.y_bps - rnd.my_bps
        ics = [g.x.rank().corr(g.y.rank()) for _, g in rnd.groupby("date") if len(g) >= 20]
        res["reversal_ic_random_times"] = PA.nw_t(np.array(ics))
        print("\nREVERSAL IC (market-excess move since open vs move to close; test names, 2024): random times %+.4f (t %.2f)"
              % (res["reversal_ic_random_times"]["mean"], res["reversal_ic_random_times"]["t"]))
    for defn, r in res.items():
        if isinstance(r, dict) and "reversal_ic_burst_times" in r:
            print("  at %-8s decision times %+.4f (t %.2f)" % (defn, r["reversal_ic_burst_times"]["mean"], r["reversal_ic_burst_times"]["t"]))
    (D / "burst_defs_raw" / "forecast_by_definition.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
