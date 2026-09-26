#!/usr/bin/env python3
"""burst-trading-eval-v2: trading ideas T1.1 / T1.2 (fade or ride at 10 s / 60 s), T2.1 (5 min), T2.x (30 min) and
T3.1 (to the close), evaluated on the v2 real-time panel as forecasts at the mid, with the spread shown as a cost
overlay for reference.

Per definition and horizon: gradient boosting on every v2 feature (controls, burst path, structure, regularity,
book), trained on the train stocks 2022-23. On the test stocks 2024 each burst is traded in the direction of the
predicted signed move when |prediction| is in the top 20% (threshold = the training 80th percentile of
|prediction|). Direction = + with the burst when the forecast is positive (ride), - against it when negative (fade).
Reported: trades, share that are fades, mean mid P&L per trade (bps), hit rate, t of the daily mean; cost overlay:
full quoted spread for the intraday exits (half in, half out) and half spread + 1 bp for the close.
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p4_analyze as PA
import burst_defs_raw2_model as M

FEATS = M.CTRL + M.PATH + M.STRUCT + M.REG + M.BOOK
HOR = {"r10": "full", "r60": "full", "r300": "full", "r1800_x": "full", "r_close_x": "close"}


def main():
    train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    d = pd.read_csv(M.D / "burst_defs_raw2" / "RAW2_bursts.csv.gz", dtype={"date": str})
    nd = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str},
                     usecols=["permno", "date", "family", "adv20"])
    d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
    d["log_n"] = np.log(d.n); d["log_dur"] = np.log1p(d.dur); d["log_q_adv"] = np.log(d.vol / d.adv20)
    d = d.replace([np.inf, -np.inf], np.nan); d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    for c in ("r10", "r60", "r300", "r1800", "r_close"):
        d.loc[d[c].abs() > 1000, c] = np.nan
    d = M.add_market(d, "t_dec", None)
    tr_all = d[d.permno.isin(train) & d.date.str[:4].isin(["2022", "2023"])]
    te_all = d[~d.permno.isin(train) & (d.date.str[:4] == "2024")]
    rows, buckets = [], []
    for defn in ("run0.1", "run0.25", "run0.5", "run1", "run2", "stream0.5", "stream1", "clip5", "run60"):
        tr, te = tr_all[tr_all.defn == defn], te_all[te_all.defn == defn]
        for y, cost_kind in HOR.items():
            m, lo, hi = M.hgb(tr, FEATS, y)
            thr = np.quantile(np.abs(m.predict(tr[FEATS].to_numpy(float))), 0.8)
            e = te[te[y].notna()].copy()
            p = m.predict(e[FEATS].to_numpy(float))
            sel = np.abs(p) >= thr
            e = e[sel]; p = p[sel]
            pnl = np.sign(p) * e[y].to_numpy()
            cost = e.spread_dec.to_numpy() if cost_kind == "full" else e.spread_dec.to_numpy() / 2 + 1.0
            day = pd.DataFrame(dict(date=e.date.to_numpy(), g=pnl, n=pnl - cost)).groupby("date").mean()
            g, n = PA.nw_t(day.g.to_numpy()), PA.nw_t(day.n.to_numpy())
            for lo_s, hi_s in ((0, 2), (2, 5), (5, 10), (10, 500)):
                b = (e.spread_dec.to_numpy() > lo_s) & (e.spread_dec.to_numpy() <= hi_s)
                if b.sum() >= 200:
                    dd = pd.DataFrame(dict(date=e.date.to_numpy()[b], g=pnl[b], n=(pnl - cost)[b])).groupby("date").mean()
                    buckets.append(dict(defn=defn, horizon=y, spread_bucket="%g-%g bps" % (lo_s, hi_s), trades=int(b.sum()),
                                        mid_pnl_bps=float(pnl[b].mean()), cost_bps=float(cost[b].mean()),
                                        net_bps=float((pnl - cost)[b].mean()), t_net=PA.nw_t(dd.n.to_numpy())["t"]))
            rows.append(dict(defn=defn, horizon=y, trades=int(len(e)), per_day=len(e) / max(len(day), 1),
                             fade_share=float((p < 0).mean()), mid_pnl_bps=float(pnl.mean()), hit=float((pnl > 0).mean()),
                             t_mid=g["t"], cost_bps=float(cost.mean()), net_bps=float(np.mean(pnl - cost)), t_net=n["t"]))
        print(defn, "done", flush=True)
    B = pd.DataFrame(buckets); B.to_csv(M.D / "burst_defs_raw2" / "trading_eval_by_spread.csv", index=False)
    R = pd.DataFrame(rows)
    R.to_csv(M.D / "burst_defs_raw2" / "trading_eval.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print("\nBY SPREAD BUCKET (net = mid P&L - round-trip cost)")
    print(B.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
