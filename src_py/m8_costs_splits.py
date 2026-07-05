#!/usr/bin/env python3
"""
m8_costs_splits.py — Referee M8 (iii) spread-based costs + (i) split/adjustment check.

(iii) The paper's cost grid stops at 3 bps flat, but the tick-constrained reversal
lives in low-priced, WIDE-spread names whose effective half-spread at the close is
much larger. Re-cost the reversal with PER-NAME effective half-spreads
(results/research/name_relspread_bps.csv) instead of a flat 1 bp, and report the
net Sharpe (full sample and walk-forward OOS).

(i) Corporate-action check: scan the tick-constrained (bottom-100) reversal universe
for single-day close-to-close moves so large they are almost certainly unadjusted
splits / reverse-splits, which would inject spurious P&L. Report the worst offenders.
"""
import numpy as np, pandas as pd, math
import m7_reversal_baseline as m7

RELSPREAD = "results/research/name_relspread_bps.csv"


def load_relspread(cols):
    s = pd.read_csv(RELSPREAD, header=0)
    s.columns = ["name", "relspread_bps"]
    s["tk"] = s["name"].str.replace("bursts_", "", regex=False)
    return s.set_index("tk")["relspread_bps"].reindex(cols)


def strat_with_cost(Zsig, R, univ_mask, half_spread_bps, flat_bps=None):
    """reversal net return with per-name half-spread cost on turnover (bps series)."""
    z = Zsig.where(univ_mask)
    P = -np.sign(z); P = P.where(np.isfinite(P), 0.0)
    W = P.rolling(m7.H, min_periods=1).mean().shift(1)
    W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan); W = W.div(g, axis=0).fillna(0.0)
    ret = (W * R).sum(axis=1)
    dW = (W - W.shift(1)).abs()
    if flat_bps is not None:
        cost = (flat_bps / 1e4) * dW.sum(axis=1)
    else:
        # per-name effective half-spread (bps) applied to each name's turnover
        hs = (half_spread_bps.reindex(dW.columns) / 1e4).fillna(half_spread_bps.median() / 1e4)
        cost = (dW * hs.values[None, :]).sum(axis=1)
    return ret - cost


def sh_t(s):
    s = s[s != 0].dropna()
    if len(s) < 60: return (np.nan, np.nan)
    return (s.mean()/s.std()*math.sqrt(252), s.mean()/(s.std()/math.sqrt(len(s))))


def main():
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    Zflow = m7.zscore(FL)
    Mfull = m7.full_sample_universe(cpx, dis, cols, m7.BOTTOM_K)
    Mwf = m7.walkforward_universe(cpx, dis, cols, m7.BOTTOM_K)
    oos = [d for d in dis if d >= dis[m7.BURN]]
    rel = load_relspread(cols); half = rel / 2.0

    # tick-constrained subset spread profile
    sub = cpx.mean().nsmallest(m7.BOTTOM_K).index
    print(f"=== M8(iii): per-name spread-based costs, tick-constrained bottom-{m7.BOTTOM_K} reversal ===")
    print(f"  bottom-100 relative spread (bps): median={rel.reindex(sub).median():.1f}, "
          f"mean={rel.reindex(sub).mean():.1f}, p90={rel.reindex(sub).quantile(.9):.1f}")
    print(f"  (vs the paper's flat 1-3 bps cost grid)\n")
    rows = []
    for lab, M, sel in [("full-sample", Mfull, slice(None)), ("walk-fwd OOS", Mwf, oos)]:
        base = strat_with_cost(Zflow, R, M, half, flat_bps=1.0)
        sp = strat_with_cost(Zflow, R, M, half, flat_bps=None)
        if lab == "walk-fwd OOS":
            base, sp = base.loc[sel], sp.loc[sel]
        s1, t1 = sh_t(base); s2, t2 = sh_t(sp)
        rows.append((lab, s1, t1, s2, t2))
    print(f"{'series':<14s} {'Sharpe@1bp':>11s} {'t':>7s} | {'Sharpe@spread':>13s} {'t':>7s}")
    for lab, s1, t1, s2, t2 in rows:
        print(f"{lab:<14s} {s1:>+11.2f} {t1:>+7.2f} | {s2:>+13.2f} {t2:>+7.2f}")

    # (i) split / corporate-action scan on the reversal universe
    print(f"\n=== M8(i): single-day close-to-close moves suggesting unadjusted (reverse-)splits ===")
    clc = close.reindex(dis)[list(sub)].pct_change(fill_method=None)
    ext = []
    for tk in sub:
        x = clc[tk].dropna()
        big = x[(x.abs() > 0.5)]  # >50% one-day move
        for di, v in big.items():
            ext.append((tk, int(di), float(v)))
    ext = sorted(ext, key=lambda r: -abs(r[2]))
    print(f"  name-days with |1-day close-to-close| > 50% in the bottom-100 universe: {len(ext)}")
    for tk, di, v in ext[:12]:
        print(f"    {tk:6s} {di}  {v*100:+.0f}%")
    names_hit = sorted(set(t for t, _, _ in ext))
    print(f"  affected names ({len(names_hit)}): {', '.join(names_hit)}")
    print("  -> these require split/adjustment verification; unadjusted, each injects spurious P&L.")


if __name__ == "__main__":
    main()
