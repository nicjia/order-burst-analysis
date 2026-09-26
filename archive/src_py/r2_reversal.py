#!/usr/bin/env python3
"""
r2_reversal.py — second-round referee fixes for the reversal + overnight-predictability.

Reuses the M7 harness (m7_reversal_baseline) as the single canonical source, so every
Section-10 number matches the audit. Outputs:
  * canonical reversal Sharpe/t (full + walk-forward OOS, K=50/100/150 + per-year)  [author note 2]
  * sign(flow) direction-only row for the baseline table                            [Major 3]
  * average daily turnover of the headline reversal + weight formula check          [Major 8iii]
  * split-repair: winsorized (+/-50%) reversal, full + OOS                          [Major 8i]
  * delisting exposure: bottom-100 names exiting mid-sample + a -30% sensitivity    [Major 8ii]
  * overnight-predictability panel: flow->CLOP pooled vs date-clustered IC,
    volume- vs count-weighted COI, with a Bonferroni deflation of the count t       [Major 1]
"""
import glob, math, numpy as np, pandas as pd
from scipy import stats
import m7_reversal_baseline as m7

RELSPREAD = "results/research/name_relspread_bps.csv"


def sh_t(s):
    s = s[s != 0].dropna()
    if len(s) < 60: return (np.nan, np.nan)
    return (s.mean()/s.std()*math.sqrt(252), s.mean()/(s.std()/math.sqrt(len(s))))


def main():
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    Zf = m7.zscore(FL); Zs = m7.zscore(SD)
    Mfull = m7.full_sample_universe(cpx, dis, cols, m7.BOTTOM_K)
    Mwf = m7.walkforward_universe(cpx, dis, cols, m7.BOTTOM_K)
    oos = [d for d in dis if d >= dis[m7.BURN]]

    print("=== CANONICAL reversal (M7 harness = single source of truth) ===")
    for lab, Z in [("burst_flow", Zf), ("sign_flow", Zs)]:
        f = m7.strat_returns(Z, R, Mfull); sf, tf = sh_t(f)
        w = m7.strat_returns(Z, R, Mwf).loc[oos]; sw, tw = sh_t(w)
        print(f"  {lab:11s}: full {sf:+.2f}/{tf:+.2f}   walk-fwd OOS {sw:+.2f}/{tw:+.2f}")
    # K robustness + per-year for burst_flow (to canonicalize Table reversion_strat)
    print("  walk-forward OOS burst_flow by K:")
    for K in (50, 100, 150):
        M = m7.walkforward_universe(cpx, dis, cols, K)
        w = m7.strat_returns(Zf, R, M).loc[oos]; s, t = sh_t(w)
        print(f"    K={K}: {s:+.2f}/{t:+.2f}")
    w100 = m7.strat_returns(Zf, R, Mwf).loc[oos]
    yr = pd.Series(w100.index).astype(str).str[:4].values
    print("  per-year OOS (K=100):", end=" ")
    for y in ("2023", "2024", "2025", "2026"):
        sy = w100[yr == y]; sy = sy[sy != 0].dropna()
        if len(sy) > 20: print(f"{y}:{sy.mean()/sy.std()*math.sqrt(252):+.2f}", end="  ")
    print()

    # --- turnover of the headline reversal ---
    z = Zf.where(Mfull); P = -np.sign(z); P = P.where(np.isfinite(P), 0.0)
    W = P.rolling(m7.H, min_periods=1).mean().shift(1); W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan); W = W.div(g, axis=0).fillna(0.0)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    print(f"\n  avg daily turnover (sum|dW|): {turn[turn>0].mean():.3f}  (=> ~{turn[turn>0].mean()*252:.0f}x/yr; "
          f"1/H={1/m7.H:.3f} reference)")

    # --- Major 8i: split repair (winsorize +/-50%) ---
    Rw = R.clip(-0.5, 0.5)
    fw = m7.strat_returns(Zf, Rw, Mfull); sfw, tfw = sh_t(fw)
    ww = m7.strat_returns(Zf, Rw, Mwf).loc[oos]; sww, tww = sh_t(ww)
    print(f"\n=== Major 8i split-repair (winsorized +/-50%) ===  full {sfw:+.2f}/{tfw:+.2f}  OOS {sww:+.2f}/{tww:+.2f}")

    # --- Major 8ii: delisting exposure of bottom-100 ---
    sub = cpx.mean().nsmallest(m7.BOTTOM_K).index
    last_valid = close.reindex(dis)[list(sub)].apply(lambda c: c.last_valid_index())
    sample_end = dis[-1]
    exiters = [tk for tk in sub if last_valid[tk] is not None and last_valid[tk] < dis[max(0, len(dis)-21)]]
    print(f"\n=== Major 8ii delisting exposure ===")
    print(f"  bottom-100 names whose price series ends >1 month before sample end: {len(exiters)}")
    # -30% delisting return applied on each exiter's last day, re-run OOS reversal
    Rd = R.copy()
    for tk in exiters:
        lv = last_valid[tk]
        if lv in Rd.index: Rd.loc[lv, tk] = -0.30
    wd = m7.strat_returns(Zf, Rd, Mwf).loc[oos]; sd, td = sh_t(wd)
    print(f"  OOS reversal with a -30% delisting return on each exiter: {sd:+.2f}/{td:+.2f} (base {sh_t(m7.strat_returns(Zf,R,Mwf).loc[oos])[0]:+.2f})")

    # --- Major 1: overnight predictability panel (flow -> next-day CLOP) ---
    print("\n=== Major 1: overnight predictability (flow_signal -> next-day close-to-open) ===")
    cl = close.reindex(dis)[cols]
    # next-day open from open_all
    opn = pd.read_csv("open_all.csv", index_col="date"); opn.index = opn.index.astype(int)
    op = opn.reindex(dis)[cols]
    tdays = np.array(dis)
    nd_open = op.shift(-1)  # open on the NEXT row (di+1)
    CLOP = (nd_open.values - cl.values) / cl.values  # close_di -> open_{di+1}
    CLOP = pd.DataFrame(CLOP, index=dis, columns=cols)
    x = Zf.values.ravel(); y = CLOP.values.ravel()
    m = np.isfinite(x) & np.isfinite(y); xr, yr2 = x[m], y[m]
    r_pool = np.corrcoef(xr, yr2)[0, 1]; N = len(xr)
    t_pool = r_pool*math.sqrt((N-2)/max(1e-12, 1-r_pool**2))
    ics = []
    for di in dis:
        a = Zf.loc[di]; b = CLOP.loc[di]; j = a.notna() & b.notna()
        if j.sum() >= 10 and a[j].std() > 0: ics.append(np.corrcoef(a[j], b[j])[0, 1])
    ics = np.array(ics); t_dc = ics.mean()/(ics.std()/math.sqrt(len(ics)))
    print(f"  volume-weighted flow->CLOP: pooled r={r_pool:+.4f} t={t_pool:+.2f} (N={N:,} name-days) | "
          f"date-clustered mean IC={ics.mean():+.4f} t={t_dc:+.2f} ({len(ics)} days)")
    # count-COI deflation (from todcoi date-clustered t=-2.68 over 2 weighting variants)
    t_count = -2.68
    from scipy.stats import norm
    p1 = 2*norm.cdf(-abs(t_count)); p_bonf = min(1.0, p1*2)
    print(f"  count-weighted COI date-clustered t={t_count} -> two-sided p={p1:.4f}; Bonferroni(x2 weighting variants) p={p_bonf:.4f} "
          f"({'survives' if p_bonf<0.05 else 'does NOT survive'} at 0.05)")


if __name__ == "__main__":
    main()
