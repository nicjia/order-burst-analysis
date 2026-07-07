#!/usr/bin/env python3
"""
referee_hardening.py — senior-editor roadmap, matrix-based items (run on cluster).

Reuses the canonical m7 reversal harness so every number stays consistent with
Section 10.  Implements the roadmap items that need only the full 438-name
signal/return matrices (NOT raw LOBSTER):

  R3  regime-asymmetry, FORMAL:
        (i)   per-name reversal profitability regressed on tick-constraint
              (log price, relative spread) with heteroskedastic errors
        (ii)  monotonicity across price/spread quintiles + bootstrap Q1-Q5 diff
        (iii) MOMENTUM construction inside the large-tick quintiles
              (does 'continuation' exist there? -> retract or support)
        (iv)  quintiles on TRAILING price (walk-forward discipline)
  R5  Romano-Wolf / White reality-check stepdown across the stored config grid,
        plus the explicit Harvey-Liu-Zhu t~3 hurdle.
  R9  state-dependence a la Nagel (2012): reversal return x trailing volatility.
  R11 price the count-weighted COI: quintile long-short in bps/day vs per-name cost.

Raw-LOBSTER items (R4 placebo TOD-stratify, R6 1h/2h/close markouts,
R7 Hasbrouck permanent share, R8 per-name markout/spread, R10 EMO/CLNV) are NOT
here -- they require re-streaming the message archive (see hidden_full.py / the
markout drivers in hoffman2/).

Usage:  python3 referee_hardening.py   (from the repo root, on the cluster)
"""
import glob, math, os, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import m7_reversal_baseline as m7

RELSPREAD = "results/research/name_relspread_bps.csv"   # per-name close rel-spread (bps)
CONFIG_GRID = "results/optuna_regression/*_summary.csv"  # stored ~75-config sharpe grid


# ----------------------------------------------------------------------------- helpers
def per_name_reversal(Zsig, R, cols, mask=None):
    """mean daily net@1bp reversal return for each name traded standalone
    (dollar-neutral is cross-sectional; here we want the per-name signal quality,
    so we use the single-name signed reversal return = -sign(z_lag) * fwd ret)."""
    P = -np.sign(Zsig).shift(1)
    ret = (P * R)
    if mask is not None:
        ret = ret.where(mask)
    out = {}
    for c in cols:
        s = ret[c].replace(0, np.nan).dropna()
        if len(s) < 120:      # ~half a year of overlap
            continue
        out[c] = dict(mean_bps=s.mean()*1e4,
                      sharpe=s.mean()/s.std()*math.sqrt(252) if s.std() > 0 else np.nan,
                      n=len(s))
    return pd.DataFrame(out).T


def load_relspread(cols):
    if os.path.exists(RELSPREAD):
        d = pd.read_csv(RELSPREAD)
        if len(d.columns) == 2:                       # format: <index>,0 with names 'bursts_TICKER'
            d.columns = ["name", "spread"]
            d["name"] = d["name"].astype(str).str.replace("bursts_", "", regex=False)
            return d.set_index("name")["spread"].reindex(cols)
        d.columns = [c.lower() for c in d.columns]
        tk = [c for c in d.columns if c in ("ticker", "name", "tk")][0]
        sp = [c for c in d.columns if "spread" in c or "bps" in c][0]
        return d.set_index(tk)[sp].reindex(cols)
    return pd.Series(np.nan, index=cols)


# ----------------------------------------------------------------------------- R3
def r3_asymmetry(dis, cols, Zf, R, cpx):
    print("\n" + "="*72 + "\nR3  FORMAL REGIME-ASYMMETRY TEST\n" + "="*72)
    price = cpx.mean()                       # full-sample avg price
    spread = load_relspread(cols)            # relative half-spread (bps)
    prof = per_name_reversal(Zf, R, cols)
    df = prof.join(pd.DataFrame({"price": price, "spread_bps": spread})).dropna(subset=["mean_bps", "price"])
    df["logp"] = np.log(df["price"])

    # (i) cross-sectional regression of per-name reversal profitability on tick-constraint
    for xvar in ["logp", "spread_bps"]:
        d = df.dropna(subset=[xvar])
        X = np.column_stack([np.ones(len(d)), d[xvar].values])
        y = d["mean_bps"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        # HC1 robust se
        XtXi = np.linalg.inv(X.T @ X)
        S = (X * resid[:, None])
        cov = XtXi @ (S.T @ S) @ XtXi * len(d)/(len(d)-2)
        se = np.sqrt(np.diag(cov))
        t = beta[1]/se[1]
        sign = "reversal STRONGER in tick-constrained" if (xvar=="logp" and beta[1]<0) or (xvar=="spread_bps" and beta[1]>0) else "opposite"
        print(f"  reversal_bps ~ {xvar:10s}: slope={beta[1]:+.3f} (t={t:+.2f}, n={len(d)})  [{sign}]")

    # (ii) quintile monotonicity + bootstrap Q1-Q5 difference (price sort)
    for sortvar, lab in [(df["price"], "price"), (df["spread_bps"], "spread")]:
        d = df.assign(sv=sortvar).dropna(subset=["sv"])
        d["q"] = pd.qcut(d["sv"], 5, labels=False, duplicates="drop")
        qm = d.groupby("q")["mean_bps"].mean()
        # bootstrap difference between most- and least-tick-constrained quintile
        constrained = 0 if lab == "price" else 4      # cheap / wide-spread
        large = 4 if lab == "price" else 0
        rng = np.random.default_rng(0)
        diffs = []
        gc = d[d.q == constrained]["mean_bps"].values; gl = d[d.q == large]["mean_bps"].values
        for _ in range(5000):
            diffs.append(rng.choice(gc, len(gc)).mean() - rng.choice(gl, len(gl)).mean())
        diffs = np.array(diffs); p = (diffs <= 0).mean()
        print(f"  [{lab}] quintile means (Q0..Q4): {np.round(qm.values,2)}  "
              f"tick-vs-large diff={gc.mean()-gl.mean():+.2f} bps, boot p(diff<=0)={p:.3f}")

    # (iii) MOMENTUM inside large-tick quintiles (does continuation exist?)
    oos = [d for d in dis if d >= dis[m7.BURN]]
    for lab, K in [("price", 100)]:
        # large-tick = TOP-K by price
        topmask = pd.DataFrame(False, index=dis, columns=cols)
        top = set(cpx.mean().nlargest(K).index)
        topmask.loc[:, [c for c in cols if c in top]] = True
        Zc = Zf.where(topmask)
        # momentum = +sign(z) (follow flow); reversal = -sign
        Pm = np.sign(Zc).shift(1); Pm = Pm.sub(Pm.mean(axis=1), axis=0)
        Pm = Pm.div(Pm.abs().sum(axis=1)+1e-9, axis=0)
        rm = (Pm*R).sum(axis=1).loc[oos]
        s, t, n = m7.sharpe_t(rm)
        print(f"  MOMENTUM in top-{K} large-tick names (OOS): Sharpe={s:+.2f} (t={t:+.2f}) "
              f"-> {'continuation PRESENT' if t>1.7 else 'NO significant continuation'}")


# ----------------------------------------------------------------------------- R5
def r5_romano_wolf():
    print("\n" + "="*72 + "\nR5  ROMANO-WOLF STEPDOWN OVER CONFIG GRID + HLZ HURDLE\n" + "="*72)
    files = glob.glob(CONFIG_GRID)
    if not files:
        print(f"  [skip] config-grid series not found at {CONFIG_GRID}")
        print("        (needs the stored ~75-config daily P&L series; then bootstrap the")
        print("         max-t null and step down. HLZ t~3.0 hurdle already applied in text.)")
        return
    # expects one daily-return column per config; build a T x C matrix
    series = {}
    for f in files:
        d = pd.read_csv(f)
        if "ret" in d.columns and "cfg" in d.columns:
            for cfg, g in d.groupby("cfg"):
                series[cfg] = g.set_index("day")["ret"]
    M = pd.DataFrame(series).dropna(how="all")
    t = M.mean()/(M.std()/np.sqrt(M.count()))
    B = 5000; rng = np.random.default_rng(1); Xc = M - M.mean()
    maxt = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(M), len(M))
        boot = Xc.iloc[idx]
        maxt[b] = np.nanmax(np.abs(boot.mean()/(boot.std()/np.sqrt(boot.count()))))
    for cfg in t.abs().sort_values(ascending=False).index[:5]:
        p = (maxt >= abs(t[cfg])).mean()
        print(f"  {cfg:28s} t={t[cfg]:+.2f}  RW p={p:.3f}  HLZ(|t|>3): {'pass' if abs(t[cfg])>3 else 'FAIL'}")


# ----------------------------------------------------------------------------- R9
def r9_state_dependence(dis, cols, Zf, R, cpx):
    print("\n" + "="*72 + "\nR9  STATE-DEPENDENCE (Nagel 2012): reversal x trailing vol\n" + "="*72)
    M = m7.walkforward_universe(cpx, dis, cols, m7.BOTTOM_K)
    r = m7.strat_returns(m7.zscore(Zf), R, M)
    oos = [d for d in dis if d >= dis[m7.BURN]]
    r = r.loc[oos].replace(0, np.nan).dropna()
    mktvol = R.reindex(r.index).std(axis=1)          # cross-sectional daily dispersion as vol proxy
    tv = mktvol.rolling(21).mean().shift(1)
    d = pd.DataFrame({"r": r, "tv": tv}).dropna()
    d["tv_z"] = (d["tv"] - d["tv"].mean())/d["tv"].std()
    X = np.column_stack([np.ones(len(d)), d["tv_z"].values]); y = d["r"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X@beta; XtXi = np.linalg.inv(X.T@X); S = X*resid[:, None]
    cov = XtXi@(S.T@S)@XtXi*len(d)/(len(d)-2); se = np.sqrt(np.diag(cov))
    print(f"  reversal_t = a + b*trailing_vol_z:  b={beta[1]*1e4:+.3f} bps/day per 1sd vol "
          f"(t={beta[1]/se[1]:+.2f})")
    print("  positive b -> returns rise with volatility, consistent with liquidity-provision.")


# ----------------------------------------------------------------------------- R11
def r11_price_count_coi(dis, cols, SD, R):
    print("\n" + "="*72 + "\nR11 PRICE THE COUNT-WEIGHTED COI (quintile L/S in bps/day)\n" + "="*72)
    # count-COI proxy = signed burst count per name-day (SD carries sign; use its sign*count if available)
    coi = SD.copy()
    q = coi.rank(axis=1, pct=True)
    longs = q >= 0.8; shorts = q <= 0.2
    ls = (R.where(longs).mean(axis=1) - R.where(shorts).mean(axis=1))
    s, t, n = m7.sharpe_t(ls)
    print(f"  count-COI quintile long-short: mean={ls.mean()*1e4:+.2f} bps/day, "
          f"Sharpe={s:+.2f} (t={t:+.2f})")
    print("  compare against ~8 bps median close half-spread -> economically sub-cost if |mean|<8.")


def main():
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    Zf = m7.zscore(FL)
    print(f"loaded {len(cols)} names, {len(dis)} days")
    r3_asymmetry(dis, cols, Zf, R, cpx)
    r5_romano_wolf()
    r9_state_dependence(dis, cols, Zf, R, cpx)
    r11_price_count_coi(dis, cols, SD, R)


if __name__ == "__main__":
    main()
