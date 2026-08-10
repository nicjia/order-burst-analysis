#!/usr/bin/env python3
"""
hidden_jump_direction.py — the test that serves BOTH directions.

STRATEGY: the hidden-flow overnight book earns +2.44 Sharpe but 89% of P&L is in ~600
name-days with large overnight moves. That is only tradeable if the SIGN of the jump is
predictable (directional options / stock) rather than just its magnitude (straddles).

IDENTIFICATION (referee alt-#1, the 'fatal one'): if hidden-flow sign on day t predicts
the SIGN of a large overnight move on t+1, hidden traders are ANTICIPATING information,
not just mechanically impacting price -- which is exactly the impact-vs-information
distinction the intraday placebo cannot make. A positive result reframes the causal
language; it does not save it, but it is direct evidence on the question.

We measure, on the 2023-24 hidden panel: conditional on a large next-overnight move,
does sign(hidden COI_t) match sign(overnight_{t+1})? Hit rate vs 50%, by move size.
"""
import math, os
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "measurements", "data")


def to_int(df):
    df = df.copy(); df.index = [int(pd.Timestamp(x).strftime("%Y%m%d")) for x in df.index]; return df


def main():
    d = pd.read_csv(os.path.join(REPO, "results/research/hidden_xsec_daily.csv"))
    d["date"] = d["date"].astype(int)
    d["nf"] = d.buy - d.sell
    FL = d.pivot_table(index="date", columns="ticker", values="nf")
    COI = d.pivot_table(index="date", columns="ticker", values="COI")
    O = to_int(pd.read_parquet(os.path.join(DATA, "opens24.parquet")))
    C = to_int(pd.read_parquet(os.path.join(DATA, "closes24.parquet")))
    dates = sorted(set(FL.index) & set(O.index))
    cols = [c for c in FL.columns if c in O.columns]
    FL = FL.reindex(dates, columns=cols); COI = COI.reindex(dates, columns=cols)
    O = O.reindex(dates, columns=cols); C = C.reindex(dates, columns=cols)
    ON = (O.shift(-1) / C - 1.0)                      # overnight t -> t+1, indexed at t

    sig = np.sign(FL).replace(0, np.nan)
    # stack to long form: (signal at t, overnight move realized t->t+1)
    s = sig.stack(); r = ON.stack()
    df = pd.concat([s, r], axis=1).dropna(); df.columns = ["sig", "on"]
    df["absmove"] = df.on.abs()
    df["hit"] = (np.sign(df.on) == df.sig).astype(int)
    print("hidden panel 2023-24: %d name-day predictions\n" % len(df))

    print("=== DIRECTIONAL HIT RATE conditional on next-overnight move size ===")
    print("  (sign of hidden flow_t vs sign of overnight_{t+1}; 50%% = no skill)\n")
    print("  %-22s %10s %10s %12s" % ("move bucket", "n", "hit rate", "binom z"))
    bins = [(0.0, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 0.10), (0.10, 1.0)]
    for lo, hi in bins:
        m = (df.absmove >= lo) & (df.absmove < hi)
        n = int(m.sum()); h = df.hit[m].mean()
        z = (h - 0.5) / math.sqrt(0.25 / n) if n > 20 else float("nan")
        print("  %5.0f-%3.0f%% overnight   %10d %9.1f%% %+12.1f" % (lo * 100, hi * 100, n, 100 * h, z))

    big = df[df.absmove >= 0.05]
    print("\n  large moves (|overnight|>=5%%): n=%d  hit=%.1f%%" % (len(big), 100 * big.hit.mean()))
    print("  => >55%% with z>3 on large moves = directional, tradeable via options AND")
    print("     evidence hidden flow anticipates news. ~50%% = magnitude-only (straddle),")
    print("     still informative for identification but not a directional signal.")

    # a cleaner economic version: value-weighted directional P&L on big-move days only,
    # net of a round-trip option-ish cost is out of scope; report the raw edge.
    print("\n=== raw directional edge on big-move days (bps/day, follow hidden sign) ===")
    for lo in (0.03, 0.05, 0.10):
        m = df.absmove >= lo
        pnl = (df.sig[m] * df.on[m])
        # date-clustered t
        dd = pd.Series(pnl.values, index=[i[0] for i in pnl.index]).groupby(level=0).mean()
        t = dd.mean() / (dd.std() / math.sqrt(len(dd))) if len(dd) > 20 else float("nan")
        print("  |move|>=%2.0f%%:  mean %+.1f bps  (per-day t=%+.1f, n=%d obs, %d days)"
              % (lo * 100, pnl.mean() * 1e4, t, m.sum(), len(dd)))


if __name__ == "__main__":
    main()
