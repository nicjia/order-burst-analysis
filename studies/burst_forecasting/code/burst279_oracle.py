#!/usr/bin/env python3
"""burst279-oracle: does the burst setup make money if the label were known perfectly (AUC = 1)?

For each label, a perfect oracle trades every burst at T_dec: FOLLOW (in the burst's direction) when the label is
1, FADE when it is 0, exit at the stated horizon, cost = half the quoted spread at T_dec + 1 bp at the auction.
Then noisy oracles: score = label + sigma * N(0,1), sigma set so that the score's AUC hits a target, trading the
top decile (follow) and bottom decile (fade). The break-even AUC is where net P&L crosses zero.

Labels (trade bursts, p4-revisit-v1 per-burst samples):
  PERM_x   phi_x >= 0.5: at least half of the peak impact still standing at x (close / next open / next close),
           measured from the pre-burst mid -- the 279 permanence label;
  CONT_x   d_x > 0: price keeps moving the burst's way AFTER T_dec;
  NET_x    d_x > cost: moves far enough after T_dec to pay for the trade.
Usage: burst279_oracle.py [CELL ...]
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import p4_analyze as PA
import fp_multiday_h1 as H
import burst279_model as B

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "burst_forecasting" / "burst279_v1"
EXITS = {"close": ("d_close", "phi_close"), "open": ("d_open", "phi_open"), "cc": ("d_cc", "phi_cc")}
AUCS = [0.52, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]


def load(cell, n=1500000):
    cols = ["permno", "date", "family", "side", "spread_dec", "d_close", "d_open", "d_cc", "phi_close", "phi_open", "phi_cc"]
    d = pd.read_csv(B.AGG / cell / ("sample_%s.csv.gz" % cell), usecols=cols, dtype={"date": str})
    d = d[d.family == "T"].drop(columns="family")
    if len(d) > n:
        d = d.sample(n, random_state=0)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    d["cost"] = d.spread_dec / 2 + B.EXIT_COST_BPS
    return d


def book(t, sign, col):
    t = t[t[col].notna()]
    if len(t) < 100:
        return None
    gross = sign * t[col]; net = gross - t.cost
    dg, dn = gross.groupby(t.date).mean(), net.groupby(t.date).mean()
    return dict(trades=int(len(t)), share=None, gross=float(gross.mean()), net=float(net.mean()),
                win_rate_net=float((net > 0).mean()), net_t=PA.nw_t(dn.to_numpy())["t"])


def main():
    cells = sys.argv[1:] or ["VAL", "TEST"]
    res = {}
    rng = np.random.default_rng(20260924)
    for cell in cells:
        d = load(cell)
        r = {"perfect": {}, "noisy": {}}
        print("\n=== %s  bursts %d  mean cost %.2f bps (half-spread %.2f + %.1f)" % (cell, len(d), d.cost.mean(), (d.spread_dec / 2).mean(), B.EXIT_COST_BPS))
        print("  PERFECT ORACLE (AUC = 1): trade every burst, follow if label=1, fade if label=0")
        print("  %-12s %-6s %8s %9s %9s %9s %8s %9s %9s %9s" % ("label", "exit", "share=1", "follow g", "follow n", "fade g", "fade n", "both net", "t both", "win%"))
        for ex, (dcol, pcol) in EXITS.items():
            base = d[d[dcol].notna()]
            labels = {"PERM_" + ex: (base[pcol] >= 0.5), "CONT_" + ex: (base[dcol] > 0), "NET_" + ex: (base[dcol] > base.cost)}
            for ln, lab in labels.items():
                fol, fad = book(base[lab], +1, dcol), book(base[~lab], -1, dcol)
                both = base.assign(pnl=np.where(lab, base[dcol], -base[dcol]) - base.cost)
                dn = both.pnl.groupby(both.date).mean()
                row = dict(share=float(lab.mean()), follow=fol, fade=fad, both_net=float(both.pnl.mean()),
                           both_t=PA.nw_t(dn.to_numpy())["t"], win=float((both.pnl > 0).mean()))
                r["perfect"][ln] = row
                print("  %-12s %-6s %8.3f %9.2f %9.2f %9.2f %8.2f %9.2f %9.1f %8.1f%%" % (ln, ex, row["share"], fol["gross"], fol["net"], fad["gross"], fad["net"], row["both_net"], row["both_t"], 100 * row["win"]))
        # noisy oracles on the close exit: how good must a model be to break even?
        base = d[d.d_close.notna()].copy()
        print("\n  NOISY ORACLE, exit at close: top decile follow / bottom decile fade (net bps per trade)")
        print("  %-10s " % "label" + " ".join("AUC%-6.2f" % a for a in AUCS))
        for ln, lab in (("PERM_close", (base.phi_close >= 0.5).to_numpy()), ("CONT_close", (base.d_close > 0).to_numpy())):
            line_f, line_d, cur = [], [], {}
            for a in AUCS:
                sigma = 1e-9 if a >= 1.0 else 1.0 / (np.sqrt(2) * norm.ppf(a))
                s = lab.astype(float) + sigma * rng.standard_normal(len(lab))
                q10, q90 = np.quantile(s, [0.1, 0.9])
                if a >= 1.0:    # ties at the perfect oracle: break them at random
                    s = lab.astype(float) + 1e-6 * rng.standard_normal(len(lab)); q10, q90 = np.quantile(s, [0.1, 0.9])
                fo = book(base[s >= q90], +1, "d_close"); fa = book(base[s <= q10], -1, "d_close")
                real_auc = float(roc_auc_score(lab, s))
                cur[str(a)] = dict(auc=real_auc, follow=fo, fade=fa)
                line_f.append("%+8.2f" % fo["net"]); line_d.append("%+8.2f" % fa["net"])
            r["noisy"][ln] = cur
            print("  %-10s " % (ln + " F") + " ".join(line_f))
            print("  %-10s " % (ln + " D") + " ".join(line_d))
        res[cell] = r
    (OUT / "oracle.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
