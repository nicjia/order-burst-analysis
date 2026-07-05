#!/usr/bin/env python3
"""
m7_signed_volume.py — Referee M7 (ii): the PLAIN signed-volume baseline.

Builds a daily signed trade-volume signal with NO geometry gate, NO D_b/kappa
filter, and NO SGD — just aggressor-signed net volume summed over ALL bursts in
the un-gated master files `results/bursts_<T>_baseline_unfiltered.csv`:

    signed_vol[day, tk] = Sum over bursts that day of (BuyVolume - SellVolume)

then runs it through the EXACT same tick-constrained reversal machinery as
`m7_reversal_baseline.py` (imported), so the row is directly comparable.

(Caveat: bursts are still Hawkes-clustered — a truly no-Hawkes total signed
volume needs raw LOBSTER trades, deleted for quota; but the geometry gate and
SGD — the "apparatus" the referee names — are both fully removed here.)
"""
import glob, numpy as np, pandas as pd
import m7_reversal_baseline as m7

MASTER = "results/bursts_*_baseline_unfiltered.csv"


def build_signed_volume(dis, cols):
    """daily net aggressor volume per ticker-day, reindexed to (dis, cols)."""
    panels = []
    for f in sorted(glob.glob(MASTER)):
        tk = f.split('/')[-1].replace('bursts_', '').replace('_baseline_unfiltered.csv', '')
        if tk not in cols: continue
        try:
            d = pd.read_csv(f, usecols=["Date", "BuyVolume", "SellVolume"])
        except Exception:
            continue
        if d.empty: continue
        d["net"] = d["BuyVolume"] - d["SellVolume"]
        g = d.groupby("Date")["net"].sum()
        g.index = pd.to_datetime(g.index).strftime("%Y%m%d").astype(int)
        panels.append(g.rename(tk))
    SV = pd.concat(panels, axis=1)
    return SV.reindex(index=dis, columns=cols)


def main():
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    SV = build_signed_volume(dis, cols)
    n_names = SV.notna().any().sum()
    print(f"[M7-ii] un-gated signed volume built for {n_names} names "
          f"(of {len(cols)} traded), {len(dis)} days\n")

    # sign-convention sanity check: does un-gated signed volume agree with the
    # gated SGD flow_signal on the same name-days? (expect strongly positive)
    both = (SV.notna() & FL.notna())
    xs, ys = SV.where(both), FL.where(both)
    corr = xs.corrwith(ys).mean()
    print(f"sign-check: mean per-name corr(un-gated signed vol, gated flow_signal) = {corr:+.3f}")
    print("  (positive => same sign convention as the paper's flow signal)\n")

    Zsv = m7.zscore(SV)
    Mfull = m7.full_sample_universe(cpx, dis, cols, m7.BOTTOM_K)
    Mwf   = m7.walkforward_universe(cpx, dis, cols, m7.BOTTOM_K)
    oos_days = [d for d in dis if d >= dis[m7.BURN]]

    fs = m7.strat_returns(Zsv, R, Mfull)
    sf, tf, nf = m7.sharpe_t(fs)
    ws = m7.strat_returns(Zsv, R, Mwf).loc[oos_days]
    sw, tw, nw = m7.sharpe_t(ws)

    print("=== M7 (ii): PLAIN un-gated signed-volume reversal "
          "(tick-constrained bottom-100, H=20, net@1bp) ===")
    print(f"{'signal':<24s} {'full_Sharpe':>12s} {'full_t':>8s} {'OOS_Sharpe':>11s} {'OOS_t':>7s}")
    print(f"{'signed_vol_ungated':<24s} {sf:>+12.2f} {tf:>+8.2f} {sw:>+11.2f} {tw:>+7.2f}")

    pd.DataFrame([("signed_vol_ungated", sf, tf, sw, tw)],
                 columns=["signal", "full_Sharpe", "full_t", "OOS_Sharpe", "OOS_t"]
                 ).to_csv("results/research/m7_signed_volume.csv", index=False)
    print("\nsaved: results/research/m7_signed_volume.csv")


if __name__ == "__main__":
    main()
