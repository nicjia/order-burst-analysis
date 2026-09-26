#!/usr/bin/env python3
"""burst-defs-raw-v2 stage 2: real-time forecasting by burst definition and horizon, and the 30-minute bucket panel.

Per-burst (fill at the mid, no costs): target = signed mid move from the real-time decision to +10 s, +60 s,
+300 s, +1800 s and the close; +1800 s and the close also in excess of the equal-weight intraday market.
Feature groups, known at the decision:
  CTRL   move since the open, 30-min pre-move, time of day, spread at decision    (known non-burst predictors)
  PATH   the burst's own move so far, 60-s pre-move                               (price path)
  STRUCT children, duration, size / ADV
  REG    size and timing regularity: modal-clip share, non-round clip, child-size CV, child size / touch depth,
         inter-arrival CV, median inter-arrival, sub-second phase concentration
  BOOK   queue imbalance at the first and last packet, quote OFI before and during, trade-flow imbalance before
Reported per definition x horizon: IC of CTRL, of each group alone, of everything, and paired IC gains
(BOOK over the rest; REG over the rest; everything over CTRL); top-minus-bottom decile of the full model at the
mid. Train 56 stocks 2022-23, test 56 different stocks 2024; gradient boosting, depth 3, fixed.

Buckets (all bursts, not sampled): at each 30-minute bucket end, predict the next bucket's mid return and the
bucket-end-to-close return (market-excess) from CTRL (bucket return, move since open, bucket index), FLOW (all-trade
flow imbalance, quote OFI) and BURST (per definition, signed burst volume decided inside the bucket / bucket
volume). Cross-sectional rank IC per (day, bucket), averaged within day, Newey-West over days.
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
CTRL = ["since_open", "pre30m", "tod", "spread_dec"]
PATH = ["move_during", "pre60"]
STRUCT = ["log_n", "log_dur", "log_q_adv"]
REG = ["mode_share", "nonround", "size_cv", "size_to_depth", "iat_cv", "iat_med", "phase_R"]
BOOK = ["imb_first", "imb_last", "qofi_pre60", "qofi_during", "tfi_pre60"]
HORIZONS = ["r10", "r60", "r300", "r1800_x", "r_close_x"]


def hgb(tr, cols, y):
    t = tr[tr[y].notna()]; lo, hi = t[y].quantile([0.01, 0.99])
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=200,
                                      early_stopping=False, random_state=1).fit(t[cols].to_numpy(float), t[y].clip(lo, hi).to_numpy())
    return m, lo, hi


def daily_ic(dates, p, y, minn=20):
    t = pd.DataFrame(dict(d=dates, p=p, y=y)).dropna()
    ic, sp = [], []
    for _, g in t.groupby("d"):
        if len(g) < minn:
            continue
        ic.append(g.p.rank().corr(g.y.rank()))
        r = g.p.rank(method="first"); k = max(1, len(g) // 10)
        sp.append(g.y[r > len(g) - k].mean() - g.y[r <= k].mean())
    return pd.Series(ic), PA.nw_t(np.array(ic)), PA.nw_t(np.array(sp))


def add_market(d, tcol, cols_out):
    z = np.load(D / "market_index" / "TEST_mkt.npz"); idx = dict(zip(z["dates"], z["idx"]))
    k0 = np.clip(np.ceil((d[tcol].to_numpy() - 34200.0) / 60.0).astype(int) - 1, 0, 389)
    k30 = np.clip(np.ceil((np.minimum(d[tcol].to_numpy() + 1800, 57600.0) - 34200.0) / 60.0).astype(int) - 1, 0, 389)
    dates = d.date.astype(str).to_numpy()
    mc = np.array([idx[x][389] - idx[x][a] if x in idx else np.nan for x, a in zip(dates, k0)])
    m30 = np.array([idx[x][b] - idx[x][a] if x in idx else np.nan for x, a, b in zip(dates, k0, k30)])
    d["r_close_x"] = d.r_close - d.side * mc * 1e4
    d["r1800_x"] = d.r1800 - d.side * m30 * 1e4
    return d


def bursts_part(train_names):
    d = pd.read_csv(D / "burst_defs_raw2" / "RAW2_bursts.csv.gz", dtype={"date": str})
    nd = pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str},
                     usecols=["permno", "date", "family", "adv20"])
    d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
    d["log_n"] = np.log(d.n); d["log_dur"] = np.log1p(d.dur); d["log_q_adv"] = np.log(d.vol / d.adv20)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    for c in ("r10", "r60", "r300", "r1800", "r_close"):
        d.loc[d[c].abs() > 1000, c] = np.nan
    d = add_market(d, "t_dec", None)
    tr_all = d[d.permno.isin(train_names) & d.date.str[:4].isin(["2022", "2023"])]
    te_all = d[~d.permno.isin(train_names) & (d.date.str[:4] == "2024")]
    print("bursts: train %d on %d stocks (2022-23), test %d on %d stocks (2024)" % (len(tr_all), tr_all.permno.nunique(), len(te_all), te_all.permno.nunique()))
    sets = {"CTRL": CTRL, "PATH": PATH, "REG": REG, "BOOK": BOOK, "ALL-BOOK": CTRL + PATH + STRUCT + REG,
            "ALL-REG": CTRL + PATH + STRUCT + BOOK, "ALL": CTRL + PATH + STRUCT + REG + BOOK}
    res, rows = {}, []
    for defn in sorted(d.defn.unique()):
        tr, te = tr_all[tr_all.defn == defn], te_all[te_all.defn == defn]
        if len(tr) < 3000 or len(te) < 1500:
            continue
        res[defn] = {}
        for y in HORIZONS:
            ics, st = {}, {}
            for k, cols in sets.items():
                m, lo, hi = hgb(tr, cols, y)
                e = te[te[y].notna()]
                s, ic, sp = daily_ic(e.date.to_numpy(), m.predict(e[cols].to_numpy(float)), e[y].clip(lo, hi).to_numpy())
                ics[k] = s; st[k] = dict(ic=ic, spread=sp)
            g_book = PA.nw_t((ics["ALL"] - ics["ALL-BOOK"]).to_numpy())
            g_reg = PA.nw_t((ics["ALL"] - ics["ALL-REG"]).to_numpy())
            g_all = PA.nw_t((ics["ALL"] - ics["CTRL"]).to_numpy())
            res[defn][y] = dict(sets=st, gain_book=g_book, gain_reg=g_reg, gain_all_over_ctrl=g_all)
            rows.append(dict(defn=defn, horizon=y, n_test=int(te[y].notna().sum()),
                             ic_ctrl=st["CTRL"]["ic"]["mean"], t_ctrl=st["CTRL"]["ic"]["t"],
                             ic_book_alone=st["BOOK"]["ic"]["mean"], t_book_alone=st["BOOK"]["ic"]["t"],
                             ic_reg_alone=st["REG"]["ic"]["mean"], t_reg_alone=st["REG"]["ic"]["t"],
                             ic_all=st["ALL"]["ic"]["mean"], t_all=st["ALL"]["ic"]["t"],
                             gain_all=g_all["mean"], t_gain_all=g_all["t"], gain_book=g_book["mean"], t_gain_book=g_book["t"],
                             gain_reg=g_reg["mean"], t_gain_reg=g_reg["t"],
                             d10_d1_all_bps=st["ALL"]["spread"]["mean"], t_d10_d1=st["ALL"]["spread"]["t"]))
        print("  %s done" % defn, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(D / "burst_defs_raw2" / "forecast_by_definition_horizon.csv", index=False)
    pd.set_option("display.width", 250)
    for y in HORIZONS:
        print("\n### horizon %s (test stocks, 2024; IC with NW t in brackets)" % y)
        s = R[R.horizon == y]
        print(s.apply(lambda r: pd.Series(dict(defn=r.defn, CTRL="%+.4f (%.1f)" % (r.ic_ctrl, r.t_ctrl),
                                               BOOK_alone="%+.4f (%.1f)" % (r.ic_book_alone, r.t_book_alone),
                                               REG_alone="%+.4f (%.1f)" % (r.ic_reg_alone, r.t_reg_alone),
                                               ALL="%+.4f (%.1f)" % (r.ic_all, r.t_all),
                                               gain_all_over_CTRL="%+.4f (%.1f)" % (r.gain_all, r.t_gain_all),
                                               gain_BOOK="%+.4f (%.1f)" % (r.gain_book, r.t_gain_book),
                                               gain_REG="%+.4f (%.1f)" % (r.gain_reg, r.t_gain_reg),
                                               D10_D1_bps="%+.2f (%.1f)" % (r.d10_d1_all_bps, r.t_d10_d1))), axis=1).to_string(index=False))
    return res


def buckets_part(train_names):
    b = pd.read_csv(D / "burst_defs_raw2" / "RAW2_buckets.csv.gz", dtype={"date": str})
    z = np.load(D / "market_index" / "TEST_mkt.npz"); idx = dict(zip(z["dates"], z["idx"]))
    b = b.sort_values(["permno", "date", "bucket"])
    with np.errstate(invalid="ignore", divide="ignore"):
        b["ret_b"] = np.log(b.m_end / b.m_start) * 1e4
        b["since_open"] = np.log(b.m_end / b.m_open) * 1e4
        b["to_close"] = np.log(b.m_close / b.m_end) * 1e4
        b["flow_imb"] = b.flow / b.volume
    b["next_ret"] = b.groupby(["permno", "date"]).ret_b.shift(-1)
    kend = (30 * (b.bucket + 1) - 1).clip(upper=389).astype(int).to_numpy()
    dates = b.date.to_numpy()
    mk_open = np.array([idx[x][k] if x in idx else np.nan for x, k in zip(dates, kend)]) * 1e4
    mk_close = np.array([idx[x][389] - idx[x][k] if x in idx else np.nan for x, k in zip(dates, kend)]) * 1e4
    knext = np.minimum(kend + 30, 389)
    mk_next = np.array([idx[x][k2] - idx[x][k] if x in idx else np.nan for x, k, k2 in zip(dates, kend, knext)]) * 1e4
    b["since_open_x"] = b.since_open - mk_open; b["to_close_x"] = b.to_close - mk_close; b["next_ret_x"] = b.next_ret - mk_next
    sv = [c for c in b.columns if c.startswith("sv_")]
    for c in sv:
        b["x_" + c[3:]] = b[c] / b.volume.where(b.volume > 0)
    burst_cols = ["x_" + c[3:] for c in sv]
    b = b.replace([np.inf, -np.inf], np.nan)
    b = b[b.bucket <= 11]                                         # the last bucket has no next bucket before the close
    for c in ("next_ret_x", "to_close_x"):
        b.loc[b[c].abs() > 1000, c] = np.nan
    tr = b[b.permno.isin(train_names) & b.date.str[:4].isin(["2022", "2023"])]
    te = b[~b.permno.isin(train_names) & (b.date.str[:4] == "2024")]
    CT = ["ret_b", "since_open_x", "bucket"]; FL = ["flow_imb", "qofi"]
    sets = {"CTRL": CT, "FLOW": FL, "BURST": burst_cols, "CTRL+FLOW": CT + FL, "CTRL+FLOW+BURST": CT + FL + burst_cols}
    print("\n### 30-minute buckets: train %d bucket-stocks, test %d (2024, different stocks)" % (len(tr), len(te)))
    res = {}
    for y in ("next_ret_x", "to_close_x"):
        ics = {}
        for k, cols in sets.items():
            m, lo, hi = hgb(tr, cols, y)
            e = te[te[y].notna()]
            key = e.date + "_" + e.bucket.astype(str)
            s, ic, sp = daily_ic(key.to_numpy(), m.predict(e[cols].to_numpy(float)), e[y].clip(lo, hi).to_numpy(), minn=20)
            ics[k] = (s, ic, sp)
        g_flow = PA.nw_t((ics["CTRL+FLOW"][0] - ics["CTRL"][0]).to_numpy())
        g_burst = PA.nw_t((ics["CTRL+FLOW+BURST"][0] - ics["CTRL+FLOW"][0]).to_numpy())
        res[y] = {k: dict(ic=v[1], spread=v[2]) for k, v in ics.items()}
        res[y]["gain_flow"] = g_flow; res[y]["gain_burst"] = g_burst
        print("  target %-11s " % y + " | ".join("%s IC %+.4f (t %.1f)" % (k, v[1]["mean"], v[1]["t"]) for k, v in ics.items()))
        print("  %-18s FLOW adds %+.4f (t %.2f); BURST adds %+.4f (t %.2f) over CTRL+FLOW; D10-D1 full %+.2f bps (t %.2f)"
              % ("", g_flow["mean"], g_flow["t"], g_burst["mean"], g_burst["t"], ics["CTRL+FLOW+BURST"][2]["mean"], ics["CTRL+FLOW+BURST"][2]["t"]))
    return res


def main():
    train = set(pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    out = dict(bursts=bursts_part(train), buckets=buckets_part(train))
    (D / "burst_defs_raw2" / "forecast_v2.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
