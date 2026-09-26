#!/usr/bin/env python3
"""
borrow_costs.py — referee fix 4b: add stock-loan (borrow) fees to the SHORT leg of
the tick-constrained reversal. Shorting cheap, illiquid names is expensive, so we
re-cost the dollar-neutral book with an annual borrow fee charged on the short
weights (in addition to the 1 bp/turnover trading cost), across a range of fees
that brackets easy-to-borrow to hard-to-borrow names.
"""
import math
import numpy as np, pandas as pd
import m7_reversal_baseline as m7


def strat_returns_borrow(Zsig, R, mask, ann_fee):
    z = Zsig.where(mask)
    P = -np.sign(z); P = P.where(np.isfinite(P), 0.0)
    W = P.rolling(m7.H, min_periods=1).mean().shift(1)
    W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0.0)
    ret = (W * R).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    tc = (m7.COST_BPS / 1e4) * turn
    short_gross = W.clip(upper=0).abs().sum(axis=1)     # sum of |short weights| (~0.5)
    borrow = (ann_fee / 252.0) * short_gross
    return ret - tc - borrow


def sh_t(s):
    s = s[s != 0].dropna()
    return (s.mean() / s.std() * math.sqrt(252), s.mean() / (s.std() / math.sqrt(len(s))))


def main():
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    Z = m7.zscore(FL)
    oos = [d for d in dis if d >= dis[m7.BURN]]
    Mfull = m7.full_sample_universe(cpx, dis, cols, m7.BOTTOM_K)
    Mwf = m7.walkforward_universe(cpx, dis, cols, m7.BOTTOM_K)
    # report the average short gross so the drag is interpretable
    z = Z.where(Mwf); P = (-np.sign(z)).where(lambda x: np.isfinite(x), 0.0)
    W = P.rolling(m7.H, min_periods=1).mean().shift(1); W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan); W = W.div(g, axis=0).fillna(0.0)
    print(f"avg short gross exposure (OOS): {W.clip(upper=0).abs().sum(axis=1).loc[oos].mean():.3f}")
    print("\nborrow fee/yr | full Sharpe | OOS Sharpe (t)")
    for fee in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]:
        f = strat_returns_borrow(Z, R, Mfull, fee); sf, tf = sh_t(f)
        w = strat_returns_borrow(Z, R, Mwf, fee).loc[oos]; sw, tw = sh_t(w)
        print(f"  {fee*100:4.0f}%      |   {sf:+.2f}    |  {sw:+.2f} (t={tw:+.2f})")


if __name__ == "__main__":
    main()
