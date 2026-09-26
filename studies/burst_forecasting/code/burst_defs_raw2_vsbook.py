#!/usr/bin/env python3
"""burst-defs-raw-v2 follow-up: do BURST features add beyond the ORDER-BOOK state (the standard high-frequency
predictors), at each horizon? Sets: CTRL+BOOK (non-burst controls plus queue imbalance, quote OFI, trade-flow
imbalance) vs CTRL+BOOK+PATH vs CTRL+BOOK+PATH+STRUCT+REG. Same split, model and metrics as burst_defs_raw2_model.py.
Hawkes is reported but flagged: its decision time t_e + 1.3 s can precede the moment its cluster is confirmed."""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p4_analyze as PA
import burst_defs_raw2_model as M

train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
d = pd.read_csv(M.D / "burst_defs_raw2" / "RAW2_bursts.csv.gz", dtype={"date": str})
nd = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str}, usecols=["permno", "date", "family", "adv20"])
d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
d["log_n"] = np.log(d.n); d["log_dur"] = np.log1p(d.dur); d["log_q_adv"] = np.log(d.vol / d.adv20)
d = d.replace([np.inf, -np.inf], np.nan); d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
for c in ("r10", "r60", "r300", "r1800", "r_close"):
    d.loc[d[c].abs() > 1000, c] = np.nan
d = M.add_market(d, "t_dec", None)
tr_all = d[d.permno.isin(train) & d.date.str[:4].isin(["2022", "2023"])]
te_all = d[~d.permno.isin(train) & (d.date.str[:4] == "2024")]
sets = {"CTRL+BOOK": M.CTRL + M.BOOK, "+PATH": M.CTRL + M.BOOK + M.PATH, "+PATH+STRUCT+REG": M.CTRL + M.BOOK + M.PATH + M.STRUCT + M.REG}
rows = []
for defn in ("run0.1", "run0.5", "run1", "stream0.5", "clip5", "run60", "hawkes"):
    tr, te = tr_all[tr_all.defn == defn], te_all[te_all.defn == defn]
    for y in ("r10", "r60", "r300", "r1800_x", "r_close_x"):
        ics = {}
        for k, cols in sets.items():
            m, lo, hi = M.hgb(tr, cols, y)
            e = te[te[y].notna()]
            s, ic, sp = M.daily_ic(e.date.to_numpy(), m.predict(e[cols].to_numpy(float)), e[y].clip(lo, hi).to_numpy())
            ics[k] = (s, ic)
        g1 = PA.nw_t((ics["+PATH"][0] - ics["CTRL+BOOK"][0]).to_numpy())
        g2 = PA.nw_t((ics["+PATH+STRUCT+REG"][0] - ics["CTRL+BOOK"][0]).to_numpy())
        rows.append(dict(defn=defn, horizon=y, ic_ctrl_book=ics["CTRL+BOOK"][1]["mean"], t_ctrl_book=ics["CTRL+BOOK"][1]["t"],
                         ic_full=ics["+PATH+STRUCT+REG"][1]["mean"], gain_path=g1["mean"], t_gain_path=g1["t"],
                         gain_burst=g2["mean"], t_gain_burst=g2["t"]))
    print(defn, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(M.D / "burst_defs_raw2" / "burst_vs_book.csv", index=False)
print(R.round(4).to_string(index=False))
