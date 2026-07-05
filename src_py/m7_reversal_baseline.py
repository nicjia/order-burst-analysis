#!/usr/bin/env python3
"""
m7_reversal_baseline.py — Referee M7.

Does the burst/ML apparatus add anything over the OBVIOUS baseline (plain
short-term reversal in cheap stocks; Jegadeesh 1990, Lehmann 1990, Nagel 2012)?

We hold the deployed positive result's construction FIXED and only swap the
*signal*:
  universe  = bottom-K names by price (tick-constrained), full-sample select
              (= paper headline universe) AND quarterly walk-forward re-select
  strategy  = dollar-neutral, unit-gross, H=20 overlapping hold, close-to-close,
              weight = -sign(z(signal)) (short high-signal), 1 bps/turnover cost,
              no look-ahead (weights shifted one day).

Signals compared on that identical pipeline:
  (A) burst_flow      = SGD daily flow_signal      (the paper's headline signal)
  (B) ret_lag_{1,5,20}= trailing k-day close-to-close return (PLAIN reversal;
                        NO order data, NO Hawkes, NO geometry gate, NO SGD)
  (C) sign_flow       = sign(flow_signal) only     (direction-only / signed-vol proxy)
  (D) sgd_pred        = SGD model prediction `pred` (full ML score)
  (E) flow_orth_ret   = flow_signal residualised cross-sectionally each day on
                        ret_lag_{1,5,20} (does burst flow add *incremental* info
                        beyond lagged returns?)

Reports full-sample and strictly-OOS (2023-2026, 2022 burn-in) Sharpe/t and
net@1bp for each, so the comparison is apples-to-apples with the headline.
"""
import glob, math, numpy as np, pandas as pd

BOTTOM_K = 100        # tick-constrained subset size (paper headline uses 100)
H        = 20         # overlapping holding period
BURN     = 252        # ~2022 burn-in for the walk-forward
REBAL    = 63         # quarterly re-selection of the tick-constrained subset
COST_BPS = 1.0
GLOB     = "results/sgd_backtests_oos/*_reg_clop_b1p0_i0p5_debug_trades.csv"


def load_panels():
    close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int)
    tdays = np.array(sorted(close.index), dtype=np.int64)
    clc = close.reindex(tdays).pct_change(fill_method=None)
    fl, pr, sd = [], [], []
    for f in glob.glob(GLOB):
        tk = f.split('/')[-1].split('_reg_clop')[0]
        if tk not in close.columns: continue
        try: d = pd.read_csv(f, usecols=["day", "flow_signal", "pred", "side"])
        except Exception: continue
        if d.empty: continue
        d["di"] = pd.to_datetime(d["day"]).dt.strftime("%Y%m%d").astype(int); d["tk"] = tk
        # one flag per name-day (aggregate any intraday dupes by sum flow / mean pred)
        g = d.groupby("di").agg(flow_signal=("flow_signal", "sum"),
                                pred=("pred", "mean"), side=("side", "sum"))
        g["tk"] = tk; g = g.reset_index()
        fl.append(g[["di", "tk", "flow_signal"]])
        pr.append(g[["di", "tk", "pred"]])
        sd.append(g[["di", "tk", "side"]].assign(side=np.sign(g["side"])))
    FL = pd.concat(fl).pivot_table(index="di", columns="tk", values="flow_signal")
    PR = pd.concat(pr).pivot_table(index="di", columns="tk", values="pred")
    SD = pd.concat(sd).pivot_table(index="di", columns="tk", values="side")
    dis = sorted(FL.index)
    cols = FL.columns
    FL, PR, SD = FL.reindex(dis), PR.reindex(dis, columns=cols), SD.reindex(dis, columns=cols)
    R   = clc.reindex(index=dis, columns=cols)
    cpx = close.reindex(dis)[cols]
    return dis, cols, FL, PR, SD, R, cpx, close


def zscore(df):
    """cross-sectional z per day, clipped [-4,4]."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def strat_returns(Zsig, R, univ_mask, weight="sign"):
    """dollar-neutral, unit-gross, H-day overlapping reversal on signal Zsig,
    restricted to univ_mask (bool DataFrame). Returns net@1bp daily series."""
    z = Zsig.where(univ_mask)
    P = -np.sign(z) if weight == "sign" else -z
    P = P.where(np.isfinite(P), 0.0)
    W = P.rolling(H, min_periods=1).mean().shift(1)
    W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0.0)
    ret = (W * R).sum(axis=1)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    return ret - (COST_BPS / 1e4) * turn


def sharpe_t(s):
    s = s[s != 0].dropna()
    if len(s) < 60: return (np.nan, np.nan, 0)
    mu, sd = s.mean(), s.std()
    return (mu / sd * math.sqrt(252), mu / (sd / math.sqrt(len(s))), len(s))


def full_sample_universe(cpx, dis, cols, K):
    """bottom-K by FULL-SAMPLE avg price (= paper headline, in-sample selection)."""
    avg = cpx.mean()
    sub = set(avg.nsmallest(K).index)
    M = pd.DataFrame(False, index=dis, columns=cols)
    M.loc[:, [c for c in cols if c in sub]] = True
    return M


def walkforward_universe(cpx, dis, cols, K):
    """bottom-K re-selected quarterly from trailing price only; True only on OOS days."""
    M = pd.DataFrame(False, index=dis, columns=cols)
    for i, ri in enumerate(range(BURN, len(dis), REBAL)):
        idxs = list(range(BURN, len(dis), REBAL))
        d0 = dis[ri]; d1 = dis[idxs[idxs.index(ri) + 1]] if idxs.index(ri) + 1 < len(idxs) else dis[-1] + 1
        trail = cpx.iloc[:ri].mean()
        sub = set(trail.nsmallest(K).index)
        seg = (np.array(dis) >= d0) & (np.array(dis) < d1)
        M.loc[np.array(dis)[seg], [c for c in cols if c in sub]] = True
    return M


def build_signals(dis, cols, FL, PR, SD, close):
    """z-scored signal panels aligned to (dis, cols)."""
    sig = {}
    sig["burst_flow"] = zscore(FL)
    sig["sign_flow"]  = zscore(SD)          # direction-only proxy for plain signed volume
    sig["sgd_pred"]   = zscore(PR)          # full ML score
    # plain lagged returns (NO order data): trailing k-day close-to-close, known as of day di
    lag_z = {}
    for k in (1, 5, 20):
        lr = close.reindex(dis)[cols].pct_change(k, fill_method=None)
        z = zscore(lr)
        sig[f"ret_lag_{k}"] = z
        lag_z[k] = z
    # burst flow residualised on the three lagged-return z's, cross-sectionally each day
    fz = sig["burst_flow"]
    resid = fz.copy() * np.nan
    Xs = [lag_z[k] for k in (1, 5, 20)]
    for di in dis:
        y = fz.loc[di]
        cols_ok = y.dropna().index
        if len(cols_ok) < 10: continue
        Xcols = [x.loc[di] for x in Xs]
        design = pd.concat([pd.Series(1.0, index=y.index)] + Xcols, axis=1)
        mask = y.notna() & design.notna().all(axis=1)
        if mask.sum() < 10: continue
        Xm = design[mask].values; ym = y[mask].values
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        resid.loc[di, y[mask].index] = ym - Xm @ beta
    sig["flow_orth_ret"] = zscore(resid)
    return sig


def main():
    dis, cols, FL, PR, SD, R, cpx, close = load_panels()
    print(f"[M7] {len(cols)} traded names, {len(dis)} days "
          f"({dis[0]}..{dis[-1]}); bottom-K={BOTTOM_K}, H={H}, cost={COST_BPS}bp\n")

    sig = build_signals(dis, cols, FL, PR, SD, close)
    Mfull = full_sample_universe(cpx, dis, cols, BOTTOM_K)
    Mwf   = walkforward_universe(cpx, dis, cols, BOTTOM_K)
    oos_days = [d for d in dis if d >= dis[BURN]]

    order = ["burst_flow", "ret_lag_1", "ret_lag_5", "ret_lag_20",
             "sign_flow", "sgd_pred", "flow_orth_ret"]
    rows = []
    for name in order:
        Z = sig[name]
        fs = strat_returns(Z, R, Mfull)
        sf, tf, nf = sharpe_t(fs)
        ws = strat_returns(Z, R, Mwf).loc[oos_days]
        sw, tw, nw = sharpe_t(ws)
        rows.append((name, sf, tf, sw, tw))
    res = pd.DataFrame(rows, columns=["signal", "full_Sharpe", "full_t", "OOS_Sharpe", "OOS_t"])

    print("=== M7: signal comparison, tick-constrained bottom-100, H=20, dollar-neutral, net@1bp ===")
    print("  full = bottom-100 by full-sample avg price (paper headline, in-sample selection)")
    print("  OOS  = quarterly walk-forward re-selection, 2022 burn-in, 2023-2026\n")
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    res.to_csv("results/research/m7_reversal_baseline.csv", index=False)
    print("\nsaved: results/research/m7_reversal_baseline.csv")

    # --- Direction-only ablation: does the SGD prediction ≈ follow burst side? ---
    agree = []
    for f in glob.glob(GLOB):
        try: d = pd.read_csv(f, usecols=["pred", "side"])
        except Exception: continue
        d = d.dropna()
        if len(d) < 20: continue
        a = (np.sign(d["pred"]) == np.sign(d["side"])).mean()
        r = np.corrcoef(np.sign(d["side"]), d["pred"])[0, 1] if d["pred"].std() > 0 else np.nan
        agree.append((a, r))
    ag = pd.DataFrame(agree, columns=["frac_sign_agree", "corr_side_pred"])
    print("\n=== Direction-only ablation (Sec 5.1: SGD ≈ follow burst side) ===")
    print(f"  names: {len(ag)}")
    print(f"  mean fraction sign(pred)==sign(side): {ag['frac_sign_agree'].mean():.3f} "
          f"(median {ag['frac_sign_agree'].median():.3f})")
    print(f"  mean corr(sign(side), pred):          {ag['corr_side_pred'].mean():.3f}")
    print("  => reversal rows burst_flow vs sign_flow vs sgd_pred above quantify "
          "whether magnitude/ML beats raw direction.")


if __name__ == "__main__":
    main()
