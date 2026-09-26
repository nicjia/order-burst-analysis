#!/usr/bin/env python3
"""burst-trading-eval-v3: trading ideas on the v3 panel at the mid (T1.3 decide during the burst, the new
definitions, and T1.19 latency: the same trades entered 1 s late). Top 20% most confident forecasts (training 80th
percentile of |prediction|), traded in the predicted direction; test stocks 2024. Cost overlay: full quoted spread
for intraday exits, half spread + 1 bp for the close."""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p4_analyze as PA
import burst_defs_raw2_model as M
import burst_defs_raw3_model as V

F = V.CTRL + V.BOOK + V.BURST
train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
d = pd.read_csv(M.D / "burst_defs_raw3" / "RAW3_events.csv.gz", dtype={"date": str})
nd = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str}, usecols=["permno", "date", "family", "adv20"])
d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
d["log_n"] = np.log(d.n_used); d["log_dur"] = np.log1p(d.dur); d["log_q_adv"] = np.log(d.vol_used / d.adv20)
d = d.replace([np.inf, -np.inf], np.nan); d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
for c in ("r10", "r60", "r300", "r1800", "r_close", "r10_L1", "r60_L1", "r10_L10", "r60_L10"):
    d.loc[d[c].abs() > 1000, c] = np.nan
d = M.add_market(d, "t_dec", None)
tr_all = d[d.permno.isin(train) & d.date.str[:4].isin(["2022", "2023"])]
te_all = d[~d.permno.isin(train) & (d.date.str[:4] == "2024")]
rows = []
for dn in ("early3", "early5", "run0.01", "run0.1", "levelclear", "cancel", "sweep", "hawkesfix", "hidden", "dom0.5"):
    tr, te = tr_all[tr_all.defn == dn], te_all[te_all.defn == dn]
    for y, outs in (("r10", ["r10", "r10_L1", "r10_L10"]), ("r60", ["r60", "r60_L1", "r60_L10"]), ("r300", ["r300"]),
                    ("r1800_x", ["r1800_x"]), ("r_close_x", ["r_close_x"])):
        m, lo, hi = M.hgb(tr, F, y)
        thr = np.quantile(np.abs(m.predict(tr[F].to_numpy(float))), 0.8)
        p_all = m.predict(te[F].to_numpy(float))
        for o in outs:
            sel = (np.abs(p_all) >= thr) & te[o].notna().to_numpy()
            e = te[sel]; p = p_all[sel]
            pnl = np.sign(p) * e[o].to_numpy()
            cost = e.spread_dec.to_numpy() if o != "r_close_x" else e.spread_dec.to_numpy() / 2 + 1.0
            day = pd.DataFrame(dict(date=e.date.to_numpy(), g=pnl)).groupby("date").g.mean()
            tight = e.spread_dec.to_numpy() <= 5
            rows.append(dict(defn=dn, outcome=o, trades=int(len(e)), mid_pnl_bps=float(pnl.mean()), hit=float((pnl > 0).mean()),
                             t_mid=PA.nw_t(day.to_numpy())["t"], cost_bps=float(cost.mean()),
                             tight_trades=int(tight.sum()), tight_mid_pnl=float(pnl[tight].mean()) if tight.sum() else np.nan,
                             tight_net=float((pnl - cost)[tight].mean()) if tight.sum() else np.nan))
    print(dn, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(M.D / "burst_defs_raw3" / "trading_eval_v3.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(R.round(3).to_string(index=False))
