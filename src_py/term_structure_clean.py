#!/usr/bin/env python3
"""
term_structure_clean.py — re-run the 48-name hidden term structure (paper Table 18)
under an explicit outlier rule, and diagnose the to-close placebo cell.

Two problems motivated this:
  (a) the 474-name panel contains name-days with impossible markouts (|mk| up to
      114,627 bps); the same contamination can distort this table.
  (b) five of six published placebo cells sit within +/-0.35 bps of zero while the
      to-close cell is -0.80, and that single cell converts a GROSS decline
      (+2.38 at 2h -> +1.64 at close) into a net "+2.44 persists". If it is
      outlier-driven, the persistence claim is not established.

Rule: drop any name-day whose burst or placebo markout at the horizon exceeds
1000 bps in absolute value (a 10% move attributed to a single burst window).
Inference: equal-weighted daily means, Newey-West (10 lags) on the day series.
"""
import math, os
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
CAP = 1000.0
H = [("3 min", "mk3", "pmk3"), ("15 min", "mk15", "pmk15"), ("30 min", "mk30", "pmk30"),
     ("1 hour", "mk60", "pmk60"), ("2 hour", "mk120", "pmk120"), ("to close", "mkclose", "pmkclose")]


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 20: return (np.nan, np.nan)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return m, m / np.sqrt(v / T)


def main():
    d = pd.read_csv(SP + "/hidden_term_rows.csv")
    d["date"] = d["date"].astype(int)
    print("loaded %d name-days, %d names, %d dates\n" % (len(d), d.ticker.nunique(), d.date.nunique()))

    print("=" * 92)
    print("(A) OUTLIER CENSUS — how extreme does each horizon get?")
    print("=" * 92)
    print("%-9s %12s %12s %10s %10s" % ("horizon", "burst min", "burst max", "|b|>1000", "|p|>1000"))
    for lab, b, p in H:
        print("%-9s %12.1f %12.1f %10d %10d"
              % (lab, d[b].min(), d[b].max(), (d[b].abs() > CAP).sum(), (d[p].abs() > CAP).sum()))

    print("\n" + "=" * 92)
    print("(B) TERM STRUCTURE — as published vs outlier-censored (|markout| <= %d bps)" % CAP)
    print("=" * 92)
    print("%-9s | %-26s | %-26s" % ("", "as published (raw)", "censored"))
    print("%-9s | %7s %8s %7s %6s | %7s %8s %7s %6s"
          % ("horizon", "burst", "placebo", "net", "t", "burst", "placebo", "net", "t"))
    for lab, b, p in H:
        row = []
        for sub in (d, d[(d[b].abs() <= CAP) & (d[p].abs() <= CAP)]):
            gb = sub.groupby("date")[b].mean()
            gp = sub.groupby("date")[p].mean()
            net = (gb - gp).dropna()
            _, t = nw(net.values)
            row += [sub[b].mean(), sub[p].mean(), net.mean(), t]
        print("%-9s | %+7.2f %+8.2f %+7.2f %+6.2f | %+7.2f %+8.2f %+7.2f %+6.2f" % (lab, *row))

    print("\n" + "=" * 92)
    print("(C) THE to-close PLACEBO CELL — what produces -0.80?")
    print("=" * 92)
    pc = d["pmkclose"]
    print("  pmkclose: mean %+.2f  median %+.2f  sd %.1f  min %.1f  max %.1f"
          % (pc.mean(), pc.median(), pc.std(), pc.min(), pc.max()))
    print("  other placebos, median: " + "  ".join("%s=%+.2f" % (l, d[p].median()) for l, _, p in H[:-1]))
    print("\n  worst 8 pmkclose name-days:")
    w = d.reindex(pc.abs().sort_values(ascending=False).index).head(8)
    print(w[["ticker", "date", "n", "mkclose", "pmkclose"]].to_string(index=False))
    for cap in (5000, 2000, 1000, 500):
        sub = d[d.pmkclose.abs() <= cap]
        print("  censor |pmkclose|<=%5d : placebo mean %+6.2f   (drops %d name-days)"
              % (cap, sub.pmkclose.mean(), len(d) - len(sub)))
    print("\n  => if the placebo mean moves toward zero as the cap tightens, the -0.80 is")
    print("     a tail artifact and the 'persists to close' net figure is overstated.")

    print("\n" + "=" * 92)
    print("(D) CENSORED headline for the paper")
    print("=" * 92)
    for lab, b, p in H:
        sub = d[(d[b].abs() <= CAP) & (d[p].abs() <= CAP)]
        gb = sub.groupby("date")[b].mean(); gp = sub.groupby("date")[p].mean()
        net = (gb - gp).dropna(); _, t = nw(net.values)
        print("  %-9s burst %+5.2f  placebo %+5.2f  net %+5.2f  NW t=%+5.2f  (n=%d name-days)"
              % (lab, sub[b].mean(), sub[p].mean(), net.mean(), t, len(sub)))


if __name__ == "__main__":
    main()
