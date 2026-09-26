#!/usr/bin/env python3
"""burst279-grid: the leak-free 279 pipeline across burst families, burst definitions, exits and decision rules.

Families: T (trade bursts, economic packets, run 60 s) and S (submission bursts: qualifying adds at/inside the
touch; extra order-book features exec_dec / cancel_dec = how much of the burst's orders executed or were
cancelled by T_dec). Definitions are sub-populations of each family, with thresholds taken from DEV:
all, n>=5, n>=10, fast (<=5 s), slow (>=60 s), large (size/ADV >= DEV p80), fingerprint (non-round repeated
clip, modal share >= 0.5), program (program score >= DEV p80), linked (>=1 same-side same-size burst in the
previous 30 min), multiday (a repeated-clip program on this side yesterday), hidden (hidden share >= DEV p80),
tight (spread at T_dec <= DEV median).
Exits: close (d_close), next open (d_open), next close (d_cc); cost = half the quoted spread at T_dec + 1 bp.
Rules: FOLLOW / FADE on the DEV top / bottom decile of the classifier's P(continue); REG-FOLLOW / REG-FADE when
the regression's predicted signed move exceeds the trade's own cost.
Pass rule (fixed before running): net daily-P&L t > 2 in VAL AND in TEST, same sign, net > 0.
Train DEV 2017-19 group 0; test VAL 2020-21 and TEST 2022-25 groups 1-2 (disjoint names and years).
Usage: burst279_grid.py FAMILY
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import p4_analyze as PA
import fp_multiday_h1 as H
import burst279_model as B

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "burst_forecasting" / "burst279_v1"
EXITS = {"close": "d_close", "open": "d_open", "cc": "d_cc"}
EXTRA_S = ["exec_dec", "cancel_dec"]


def load(cell, fam, cal, n=None):
    cols = B.COLS + ["d_open", "d_cc"] + (EXTRA_S if fam == "S" else [])
    d = pd.read_csv(B.AGG / cell / ("sample_%s.csv.gz" % cell), usecols=list(dict.fromkeys(cols)), dtype={"date": str})
    d = d[d.family == fam].drop(columns="family")
    if n and len(d) > n:
        d = d.sample(n, random_state=0)
    d["log_n"] = np.log1p(d.n); d["log_dur"] = np.log1p(d.t_e - d.t_b); d["dur"] = d.t_e - d.t_b
    p = B.prior_day_programs(cell, cal)
    d = d.merge(p.rename(columns={"cnt": "prog_same_prev"}), on=["permno", "date", "side"], how="left")
    d = d.merge(p.assign(side=-p.side).rename(columns={"cnt": "prog_opp_prev"}), on=["permno", "date", "side"], how="left")
    d[["prog_same_prev", "prog_opp_prev"]] = d[["prog_same_prev", "prog_opp_prev"]].fillna(0.0)
    if fam == "S":
        d["exec_frac"] = d.exec_dec / d.n.clip(lower=1); d["cancel_frac"] = d.cancel_dec / d.n.clip(lower=1)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["spread_dec"])
    return d[(d.spread_dec > 0) & (d.spread_dec < 500)]


def definitions(d, thr):
    return {"all": np.ones(len(d), bool), "n>=5": (d.n >= 5).to_numpy(), "n>=10": (d.n >= 10).to_numpy(),
            "fast<=5s": (d.dur <= 5).to_numpy(), "slow>=60s": (d.dur >= 60).to_numpy(),
            "large": (d.log_q_adv >= thr["large"]).to_numpy(),
            "fingerprint": ((d.mode_nonround == 1) & (d.mode_share >= 0.5)).to_numpy(),
            "program": (d.program_score >= thr["program"]).to_numpy(), "linked": (d.link_back >= 1).to_numpy(),
            "multiday": (d.prog_same_prev >= 1).to_numpy(), "hidden": (d.hidden_share >= thr["hidden"]).to_numpy(),
            "tight": (d.spread_dec <= thr["tight"]).to_numpy()}


def pnl_stats(df, mask, sign, col):
    t = df[mask & df[col].notna().to_numpy()]
    if len(t) < 200:
        return None
    gross = sign * t[col]; net = gross - t.spread_dec / 2 - B.EXIT_COST_BPS
    dg, dn = gross.groupby(t.date).mean(), net.groupby(t.date).mean()
    g, n = PA.nw_t(dg.to_numpy()), PA.nw_t(dn.to_numpy())
    return dict(trades=int(len(t)), gross=float(gross.mean()), gross_t=g["t"], net=float(dn.mean()), net_t=n["t"],
                sr_net=float(dn.mean() / dn.std() * np.sqrt(252)) if dn.std() > 0 else None)


def main():
    fam = sys.argv[1]
    t0 = time.time(); cal = H.calendar()
    feats = B.BASE + B.NEW + (["exec_frac", "cancel_frac"] if fam == "S" else [])
    tr = load("DEV", fam, cal, 300000)
    thr = dict(large=float(tr.log_q_adv.quantile(0.8)), program=float(tr.program_score.quantile(0.8)),
               hidden=float(tr.hidden_share.quantile(0.8)) if tr.hidden_share.notna().any() else np.inf,
               tight=float(tr.spread_dec.median()))
    models = {}
    for ex, col in EXITS.items():
        t = tr[tr[col].notna()]
        X = t[feats].to_numpy(float); y = t[col].to_numpy(float)
        lo, hi = np.nanquantile(y, [0.01, 0.99]); yw = np.clip(y, lo, hi)
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05, min_samples_leaf=200,
                                             early_stopping=False, random_state=1).fit(X, (y > 0).astype(int))
        reg = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, min_samples_leaf=200,
                                            early_stopping=False, random_state=1).fit(X, yw)
        s = clf.predict_proba(X)[:, 1]
        models[ex] = dict(clf=clf, reg=reg, q10=float(np.quantile(s, 0.1)), q90=float(np.quantile(s, 0.9)))
    print("%s: trained on DEV %d rows, %d features (%.0fs)" % (fam, len(tr), len(feats), time.time() - t0), flush=True)
    rows = []
    for cell in ("VAL", "TEST"):
        te = load(cell, fam, cal, 1500000)
        defs = definitions(te, thr)
        X = te[feats].to_numpy(float)
        cost = (te.spread_dec / 2 + B.EXIT_COST_BPS).to_numpy()
        for ex, col in EXITS.items():
            m = models[ex]
            p = m["clf"].predict_proba(X)[:, 1]; r = m["reg"].predict(X)
            rules = {"FOLLOW top decile": (p >= m["q90"], +1), "FADE bottom decile": (p <= m["q10"], -1),
                     "REG-FOLLOW pred>cost": (r > cost, +1), "REG-FADE pred<-cost": (r < -cost, -1)}
            for dn, dm in defs.items():
                for rn, (rm, sg) in rules.items():
                    st = pnl_stats(te, dm & rm, sg, col)
                    if st:
                        rows.append(dict(family=fam, cell=cell, definition=dn, exit=ex, rule=rn, **st))
        print("  scored %s (%.0fs)" % (cell, time.time() - t0), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / ("grid_%s.csv" % fam), index=False)
    w = res.pivot_table(index=["definition", "exit", "rule"], columns="cell", values=["gross", "net", "net_t", "trades"])
    w.columns = ["%s_%s" % (a, b) for a, b in w.columns]
    w = w.reset_index()
    w["pass"] = (w.net_t_VAL > 2) & (w.net_t_TEST > 2) & (w.net_VAL > 0) & (w.net_TEST > 0)
    w.to_csv(OUT / ("grid_%s_wide.csv" % fam), index=False)
    n = len(w)
    print("\n%s: %d cells (definition x exit x rule). PASS (net t > 2 in both VAL and TEST): %d" % (fam, n, int(w["pass"].sum())))
    top = w.sort_values("net_t_TEST", ascending=False).head(12)
    print(top[["definition", "exit", "rule", "trades_TEST", "gross_VAL", "net_VAL", "net_t_VAL", "gross_TEST", "net_TEST", "net_t_TEST", "pass"]].round(2).to_string(index=False))
    print("\nbest GROSS cells in TEST (before costs):")
    print(w.assign(gt=w.gross_TEST).sort_values("gt", ascending=False).head(6)[["definition", "exit", "rule", "gross_VAL", "gross_TEST", "net_TEST", "net_t_TEST"]].round(2).to_string(index=False))
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
