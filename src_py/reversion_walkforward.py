#!/usr/bin/env python3
"""
reversion_walkforward.py — expanding-window OOS test of the tick-constrained reversal.

Addresses the concern that the "100 lowest-price names, H=20" choice may be
in-sample optimized. Here the tick-constrained subset is RE-SELECTED every quarter
using only trailing price data (expanding mean up to the rebalance date), 2022 is
held out as burn-in, and the market-neutral reversal is evaluated strictly on the
subsequent (out-of-sample) days. Reports OOS Sharpe/t for a range of subset sizes K
(so the result does not hinge on K=100) and a per-year breakdown.
"""
import glob, numpy as np, pandas as pd, math

def main():
    close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int)
    tdays = np.array(sorted(close.index), dtype=np.int64)
    clc = close.reindex(tdays).pct_change(fill_method=None)
    rows = []
    for f in glob.glob("results/sgd_backtests_oos/*_reg_clop_b1p0_i0p5_debug_trades.csv"):
        tk = f.split('/')[-1].split('_reg_clop')[0]
        if tk not in close.columns: continue
        try: d = pd.read_csv(f, usecols=["day", "flow_signal"])
        except Exception: continue
        if d.empty: continue
        d["di"] = pd.to_datetime(d["day"]).dt.strftime("%Y%m%d").astype(int); d["tk"] = tk
        rows.append(d[["di", "tk", "flow_signal"]])
    A = pd.concat(rows, ignore_index=True)
    A["z"] = A.groupby("di")["flow_signal"].transform(lambda x: ((x-x.mean())/(x.std()+1e-9)).clip(-4, 4))
    Z = A.pivot_table(index="di", columns="tk", values="z"); dis = sorted(Z.index); Z = Z.reindex(dis)
    R = clc.reindex(index=dis, columns=Z.columns)
    cpx = close.reindex(dis)[Z.columns]                  # daily close of traded names
    P = pd.DataFrame(np.where(np.isfinite(Z.values), -np.sign(Z.values), 0.0), index=Z.index, columns=Z.columns)

    BURN = 252                                            # hold out ~2022
    REBAL = 63                                            # quarterly re-selection
    H = 20
    def walk(K):
        M = pd.DataFrame(0.0, index=Z.index, columns=Z.columns)
        rebal_idx = list(range(BURN, len(dis), REBAL))
        for i, ri in enumerate(rebal_idx):
            d0 = dis[ri]; d1 = dis[rebal_idx[i+1]] if i+1 < len(rebal_idx) else dis[-1]+1
            trail = cpx.iloc[:ri].mean()                  # avg price using only PAST data
            sub = set(trail.nsmallest(K).index)
            seg = (Z.index >= d0) & (Z.index < d1)
            M.loc[seg, [c for c in M.columns if c in sub]] = 1.0
        Pm = P * M
        W = Pm.rolling(H, min_periods=1).mean().shift(1)
        W = W.sub(W.mean(axis=1), axis=0); g = W.abs().sum(axis=1).replace(0, np.nan); W = W.div(g, axis=0).fillna(0)
        ret = (W*R).sum(axis=1) - (1/1e4)*(W-W.shift(1)).abs().sum(axis=1)
        oos = ret.loc[[d for d in dis if d >= dis[BURN]]]
        oos = oos[oos != 0].dropna()
        return oos
    def stat(s):
        if len(s) < 60: return (np.nan, np.nan)
        return (s.mean()/s.std()*math.sqrt(252), s.mean()/(s.std()/math.sqrt(len(s))))

    print("=== EXPANDING-WINDOW OOS (2023-2026; subset re-picked quarterly from trailing price; H=20) ===")
    print(f"{'K (subset size)':16s} {'OOS Sharpe':>11s} {'t':>7s} {'OOS days':>9s}")
    base=None
    for K in [50, 100, 150, 146]:
        s = walk(K); sh, t = stat(s); print(f"{K:<16d} {sh:>+11.2f} {t:>+7.2f} {len(s):>9d}")
        if K == 100: base = s
    print("\n=== Per-year OOS Sharpe (K=100) ===")
    yr = pd.Series(base.index).astype(str).str[:4].values
    for y in ["2023", "2024", "2025", "2026"]:
        sy = base[yr == y]
        if len(sy) > 20: sh, t = stat(sy); print(f"  {y}: Sharpe={sh:+.2f} t={t:+.2f} days={len(sy)}")
    sh, t = stat(base)
    print(f"\nHeadline OOS (K=100): Sharpe={sh:+.2f} t={t:+.2f} over {len(base)} OOS days (2022 burn-in excluded)")

if __name__ == "__main__":
    main()
