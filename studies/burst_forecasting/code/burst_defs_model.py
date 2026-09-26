#!/usr/bin/env python3
"""burst-defs-v1 stage 2: the 279 model on four burst definitions with a REAL-TIME decision (first minute mark
after the episode ends + 1 s) and intraday exits.

For each definition (run60, merge5m, merge30m, fpchain): HGB classifier of continuation r_h > 0 trained on DEV,
tested on VAL and TEST; permanence from start at the close ((so_far + r_close) >= 0.5 * so_far, bursts with a
positive move so far) shown for the overlap contrast. Trading: FOLLOW the top / FADE the bottom DEV decile of
P(continue); exits +5, +30, +60 minutes (cost = the full quoted spread at the burst, both crossings) and the
close (half spread + 1 bp). Daily equal-weight P&L, Newey-West(10). Pass: net t > 2 in VAL and TEST, same sign.
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import p4_analyze as PA

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "results" / "burst_forecasting" / "burst_defs_v1"
F = ["log_q_adv", "log_nb", "log_nch", "log_dur", "fp_count", "fp_share", "hidden", "trunc", "pscore", "spread_b",
     "tod", "so_far_bps", "pre30_bps", "open_dec_bps"]
EXITS = {"5m": "r5", "30m": "r30", "60m": "r60", "close": "rclose"}


def load(cell):
    d = pd.read_csv(D / ("%s_defs.csv.gz" % cell), dtype={"date": str})
    nd = pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "agg" / cell / ("nameday_%s.csv.gz" % cell), dtype={"date": str},
                     usecols=["permno", "date", "family", "adv20"])
    d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="inner")
    d["log_q_adv"] = np.log(d.vol / d.adv20); d["log_nb"] = np.log(d.nb); d["log_nch"] = np.log1p(d.nch)
    d["log_dur"] = np.log1p(d.t_e - d.t_b)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d[(d.spread_b > 0) & (d.spread_b < 500)]
    for c in ("r5", "r30", "r60", "rclose", "so_far_bps"):
        d.loc[d[c].abs() > 1000, c] = np.nan
    return d


def stats(t, sign, col, intraday):
    t = t[t[col].notna()]
    if len(t) < 200:
        return None
    gross = sign * t[col]
    cost = t.spread_b if intraday else t.spread_b / 2 + 1.0
    net = gross - cost
    dg, dn = gross.groupby(t.date).mean(), net.groupby(t.date).mean()
    return dict(trades=int(len(t)), gross=float(gross.mean()), gross_t=PA.nw_t(dg.to_numpy())["t"],
                net=float(dn.mean()), net_t=PA.nw_t(dn.to_numpy())["t"])


def main():
    t0 = time.time()
    cells = {c: load(c) for c in ("DEV", "VAL", "TEST") if (D / ("%s_defs.csv.gz" % c)).exists()}
    rows, qual = [], []
    for defn in ("run60", "merge5m", "merge30m", "fpchain"):
        tr = cells["DEV"][cells["DEV"].defn == defn]
        tr = tr.sample(min(300000, len(tr)), random_state=0)
        models = {}
        for ex, col in EXITS.items():
            t = tr[tr[col].notna()]
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05, min_samples_leaf=200,
                                               early_stopping=False, random_state=1).fit(t[F].to_numpy(float), (t[col] > 0).astype(int))
            s = m.predict_proba(t[F].to_numpy(float))[:, 1]
            models[ex] = (m, np.quantile(s, 0.1), np.quantile(s, 0.9))
        pos = tr[tr.so_far_bps > 0].dropna(subset=["rclose"])
        mp = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05, min_samples_leaf=200,
                                            early_stopping=False, random_state=1).fit(
            pos[F].to_numpy(float), ((pos.so_far_bps + pos.rclose) >= 0.5 * pos.so_far_bps).astype(int))
        for cell in ("VAL", "TEST"):
            if cell not in cells:
                continue
            te = cells[cell][cells[cell].defn == defn]
            X = te[F].to_numpy(float)
            pp = te[te.so_far_bps > 0].dropna(subset=["rclose"])
            q = dict(defn=defn, cell=cell, episodes=int(len(te)),
                     auc_perm_from_start=float(roc_auc_score(((pp.so_far_bps + pp.rclose) >= 0.5 * pp.so_far_bps).astype(int),
                                                             mp.predict_proba(pp[F].to_numpy(float))[:, 1])))
            for ex, col in EXITS.items():
                m, q10, q90 = models[ex]
                ok = te[col].notna().to_numpy()
                p = m.predict_proba(X)[:, 1]
                q["auc_cont_" + ex] = float(roc_auc_score((te[col][ok] > 0).astype(int), p[ok]))
                for rule, mask, sg in (("FOLLOW top decile", p >= q90, 1), ("FADE bottom decile", p <= q10, -1)):
                    st = stats(te[mask], sg, col, ex != "close")
                    if st:
                        rows.append(dict(defn=defn, cell=cell, exit=ex, rule=rule, **st))
            qual.append(q)
        print("  %s done (%.0fs)" % (defn, time.time() - t0), flush=True)
    Q = pd.DataFrame(qual)
    print("\nPREDICTION QUALITY (AUC; 0.5 = chance)")
    print(Q.round(4).to_string(index=False))
    R = pd.DataFrame(rows)
    W = R.pivot_table(index=["defn", "exit", "rule"], columns="cell", values=["gross", "net", "net_t", "trades"])
    W.columns = ["%s_%s" % (a, b) for a, b in W.columns]; W = W.reset_index()
    W["pass"] = (W.net_t_VAL > 2) & (W.net_t_TEST > 2) & (W.net_VAL > 0) & (W.net_TEST > 0)
    print("\nTRADING (bps per trade; %d cells; PASS = net t > 2 in both VAL and TEST): %d pass" % (len(W), int(W["pass"].sum())))
    print(W[["defn", "exit", "rule", "trades_TEST", "gross_VAL", "net_VAL", "net_t_VAL", "gross_TEST", "net_TEST", "net_t_TEST", "pass"]].round(2).to_string(index=False))
    Q.to_csv(D / "quality.csv", index=False); W.to_csv(D / "trading.csv", index=False)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
