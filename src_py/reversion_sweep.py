#!/usr/bin/env python3
"""
reversion_sweep.py — mean-reversion strategy search (Referee R3).

Market-neutral (dollar-neutral) reversal on the SGD daily burst-flow signal, with
two turnover-reducing / conviction levers:
  * multi-day overlapping holding H (turnover ~ 1/H): effective weight = trailing
    H-day mean of the daily reversal weights (past-only), applied to close-to-close
    returns.
  * high-conviction |z| tail selection: trade only names whose daily flow z-score
    exceeds a threshold (the largest overshoots, where reversion should be strongest).

Reports gross and net (1 bps/turnover) annualized Sharpe, t-stat, and a
Deflated-Sharpe z (deflated for the number of configs searched).
No look-ahead: weights formed from signals strictly before the return day.
"""
import glob, numpy as np, pandas as pd

def main():
    close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int)
    tdays = np.array(sorted(close.index), dtype=np.int64)
    # close-to-close daily returns on consecutive trading days
    clc = close.reindex(tdays).pct_change()

    rows = []
    for f in glob.glob("results/sgd_backtests_oos/*_reg_clop_b1p0_i0p5_debug_trades.csv"):
        tk = f.split('/')[-1].split('_reg_clop')[0]
        if tk not in close.columns: continue
        try: d = pd.read_csv(f, usecols=["day", "flow_signal"])
        except Exception: continue
        if d.empty: continue
        d["di"] = pd.to_datetime(d["day"]).dt.strftime("%Y%m%d").astype(int)
        d["tk"] = tk
        rows.append(d[["di", "tk", "flow_signal"]])
    A = pd.concat(rows, ignore_index=True)
    A["z"] = A.groupby("di")["flow_signal"].transform(lambda x: ((x - x.mean())/(x.std()+1e-9)).clip(-4, 4))
    Z = A.pivot_table(index="di", columns="tk", values="z")
    dis = sorted(Z.index); Z = Z.reindex(dis)
    R = clc.reindex(index=dis, columns=Z.columns)   # return realized ON day di (prev close->di close)

    def run(thr, weight, H, cost_bps):
        z = Z.values.copy()
        mask = np.abs(z) >= thr
        if weight == "sign":
            p = -np.sign(z)
        else:  # magnitude
            p = -z
        p = np.where(mask & np.isfinite(z), p, 0.0)
        P = pd.DataFrame(p, index=Z.index, columns=Z.columns)
        # overlapping H-day hold = trailing mean of daily weights, past-only
        W = P.rolling(H, min_periods=1).mean().shift(1)
        # enforce dollar-neutrality + unit gross each day
        W = W.sub(W.mean(axis=1), axis=0)
        gross = W.abs().sum(axis=1).replace(0, np.nan)
        W = W.div(gross, axis=0).fillna(0.0)
        ret = (W * R).sum(axis=1)
        turn = (W - W.shift(1)).abs().sum(axis=1)
        net = ret - (cost_bps/1e4) * turn
        s = net.dropna()
        s = s[s != 0]
        if len(s) < 60: return None
        mu, sd = s.mean(), s.std()
        shp = mu/sd*np.sqrt(252) if sd > 0 else np.nan
        t = mu/(sd/np.sqrt(len(s)))
        return dict(mu=mu*1e4, shp=shp, t=t, days=len(s), turn=turn.mean())

    configs = []
    for weight in ["sign", "magnitude"]:
        for thr in [0.0, 1.0, 1.5, 2.0]:
            for H in [1, 2, 3, 5, 10]:
                configs.append((weight, thr, H))
    ntrials = len(configs)
    results = []
    for (weight, thr, H) in configs:
        g = run(thr, weight, H, 0.0); n = run(thr, weight, H, 1.0)
        if g is None or n is None: continue
        # Deflated-Sharpe z: net t deflated by expected max under ntrials nulls
        emax = np.sqrt(2*np.log(max(ntrials, 2)))
        dsr_z = n["t"] - emax
        results.append((weight, thr, H, g["shp"], n["shp"], n["t"], dsr_z, n["turn"], n["mu"]))
    res = pd.DataFrame(results, columns=["weight","zthr","H","grossSharpe","netSharpe@1bp","net_t","DSR_z","turnover","net_mu_bps"])
    res = res.sort_values("net_t", ascending=False)
    pd.set_option("display.width", 200)
    print(f"REVERSION SWEEP — {ntrials} configs, deflation E[max]={np.sqrt(2*np.log(ntrials)):.2f}")
    print(f"(DSR_z>0 => net t survives multiple-testing deflation; t>2 => nominal sig)\n")
    print(res.head(15).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    res.to_csv("results/research/reversion_sweep.csv", index=False)
    print("\nsaved: results/research/reversion_sweep.csv")

if __name__ == "__main__":
    main()
