#!/usr/bin/env python3
"""
conditional_intraday.py — honest test of the claim "you can't profit intraday."

My prior assertion used the AVERAGE 3-min markout (+2 bps < spread). But a strategy
does not trade the average -- it trades the subset where the signal is predictably
large. This script asks: is the intraday hidden-burst markout PREDICTABLE from
information available AT burst termination (burst count n, imbalance COI, signable
prints n_sig, volume), and does any conditional subset lift the expected markout
ABOVE a realistic round-trip taker cost?

Entry mechanics for a taker: observe the burst at termination, cross the spread to
enter (pay ~half-spread), hold 3 min, cross to exit (pay ~half-spread) => the hurdle
is roughly ONE full spread. For these liquid names the paper uses 4-12 bps; we test
against 5 and 10 bps.

Conditioning variables are all known at burst termination, so a strategy on them has
no look-ahead. Inference is date-clustered (Newey-West on the daily-mean series).
"""
import math, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = 1000.0


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 30: return (np.nan, np.nan)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return m, m / np.sqrt(v / T)


def dstat(sub, col="mk3"):
    """date-clustered mean + t on a name-day column."""
    dm = sub.groupby("date")[col].mean()
    m, t = nw(dm.values)
    return m, t, len(sub)


def main():
    d = pd.read_csv(os.path.join(REPO, "results/research/hidden_xsec_daily.csv"))
    d["date"] = d["date"].astype(int)
    d = d[(d.mk3.abs() <= CAP) & (d.mk15.abs() <= CAP) & (d.mk30.abs() <= CAP)].copy()
    d["absCOI"] = d["COI"].abs()
    d["vol"] = d["buy"] + d["sell"]
    print("cleaned panel: %d name-days, %d names, %d dates" % (len(d), d.ticker.nunique(), d.date.nunique()))
    m, t, n = dstat(d, "mk3")
    print("unconditional 3-min markout: %+.2f bps  (date-clustered t=%.1f)\n" % (m, t))

    print("=" * 82)
    print("CONDITIONAL MARKOUT — does any observable-at-termination subset beat the spread?")
    print("=" * 82)
    print("(hurdle: a round-trip taker pays ~1 spread; test vs 5 and 10 bps)\n")
    for var, lab in [("n", "burst count"), ("absCOI", "|imbalance|"),
                     ("n_sig", "signable prints"), ("vol", "burst volume")]:
        print("  conditioning on %s (deciles):" % lab)
        d["dec"] = pd.qcut(d[var].rank(method="first"), 10, labels=False)
        for q in [9, 8, 7, 0]:                    # top deciles + bottom
            sub = d[d.dec == q]
            m, t, n = dstat(sub, "mk3")
            tag = "  <== > 5bp" if m > 5 else ""
            print("    decile %d  markout %+6.2f bps  (t=%5.1f, n=%d)%s"
                  % (q + 1, m, t, n, tag))
        print()

    print("=" * 82)
    print("MOST EXTREME JOINT CONDITION — top burst-count AND top imbalance")
    print("=" * 82)
    hi = d[(d.n >= d.n.quantile(0.9)) & (d.absCOI >= d.absCOI.quantile(0.9))]
    m, t, n = dstat(hi, "mk3")
    print("  top-10%% n AND top-10%% |COI|:  markout %+.2f bps  (t=%.1f, n=%d)" % (m, t, n))
    for h in ("mk15", "mk30"):
        mm, tt, _ = dstat(hi, h)
        print("    same subset, %s: %+.2f bps (t=%.1f)" % (h, mm, tt))

    print("\n" + "=" * 82)
    print("PREDICTABILITY — regress next-nothing; here regress mk3 on the conditioners")
    print("=" * 82)
    X = d[["n", "absCOI", "n_sig", "vol"]].copy()
    X = (X - X.mean()) / (X.std() + 1e-9)
    X["const"] = 1.0
    y = d["mk3"].values
    Xm = X[["const", "n", "absCOI", "n_sig", "vol"]].values
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    yhat = Xm @ beta
    ss = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print("  OLS mk3 ~ n + |COI| + n_sig + vol :  R^2 = %.4f" % ss)
    print("  coefficients (bps per 1sd):  " +
          "  ".join("%s=%+.2f" % (nm, b) for nm, b in zip(["const", "n", "|COI|", "n_sig", "vol"], beta)))
    print("\n  => if R^2 ~ 0 and no decile clears the spread, the markout is essentially")
    print("     unconditional 2 bps of noise -> no conditional taker trade exists.")
    print("     if some decile clears 5-10 bps with t>3, there IS a conditional intraday edge.")


if __name__ == "__main__":
    main()
