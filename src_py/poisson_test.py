#!/usr/bin/env python3
"""
poisson_test.py — Referee B2: is burst clustering an artifact of Poisson arrivals?

For one LOBSTER message file: take visible executions (type 4), aggressor sign =
-Direction, and compare the OBSERVED count of same-side delta-clustered bursts
(gap < 1 s, run length >= 3) to a HOMOGENEOUS POISSON null with intensity matched
to the day's trade rate and i.i.d. signs at the empirical buy fraction. Also report
the Fano factor (index of dispersion of trade counts in 1 s bins; Poisson => 1).

Emits: ticker,date,n_trades,fano,obs_bursts,poisson_mean,poisson_std,z
"""
import argparse, os, re, sys
import numpy as np, pandas as pd

RTH0, RTH1 = 34200.0, 57600.0  # 9:30 .. 16:00


def count_bursts(sign, times, gap=1.0, minrun=3):
    """vectorized: same-side runs with inter-arrival < gap, length >= minrun."""
    n = len(sign)
    if n < minrun: return 0
    brk = (np.diff(sign) != 0) | (np.diff(times) >= gap)
    run_id = np.concatenate([[0], np.cumsum(brk)])
    return int((np.bincount(run_id) >= minrun).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    args = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(args.msg))
    date = int(m.group(1)+m.group(2)+m.group(3)) if m else 0
    try:
        df = pd.read_csv(args.msg, header=None, usecols=[0, 1, 3, 5], names=["t", "ty", "sz", "dr"])
        tr = df[(df.ty == 4) & (df.t >= RTH0) & (df.t <= RTH1)]
        t = tr.t.to_numpy(float); s = (-tr.dr).to_numpy(np.int8)
        N = len(t)
        if N < 100:
            print(f"{args.ticker},{date},{N},nan,nan,nan,nan,nan,nan"); return
        # Fano factor over 1 s bins
        counts, _ = np.histogram(t, bins=np.arange(RTH0, RTH1 + 1.0, 1.0))
        fano = counts.var() / counts.mean() if counts.mean() > 0 else np.nan
        obs = count_bursts(s, t)
        pbuy = (s == 1).mean()
        rng = np.random.default_rng(date + hash(args.ticker) % 10000)
        B = 200
        # homogeneous Poisson null: uniform arrivals, iid signs
        hom = np.empty(B)
        for b in range(B):
            tt = np.sort(rng.uniform(RTH0, RTH1, N))
            ss = np.where(rng.random(N) < pbuy, 1, -1).astype(np.int8)
            hom[b] = count_bursts(ss, tt)
        z_hom = (obs - hom.mean()) / hom.std() if hom.std() > 0 else np.nan
        # INHOMOGENEOUS Poisson null: empirical 60 s intraday intensity profile, iid signs
        edges = np.arange(RTH0, RTH1 + 60.0, 60.0)
        prof = counts if len(counts) == len(edges) - 1 else np.histogram(t, bins=edges)[0]
        inh = np.empty(B)
        for b in range(B):
            nk = rng.poisson(prof)                       # Poisson arrivals per bin at empirical rate
            tt = np.concatenate([rng.uniform(edges[k], edges[k+1], nk[k]) for k in range(len(nk))])
            tt.sort()
            ss = np.where(rng.random(len(tt)) < pbuy, 1, -1).astype(np.int8)
            inh[b] = count_bursts(ss, tt)
        z_inh = (obs - inh.mean()) / inh.std() if inh.std() > 0 else np.nan
        print(f"{args.ticker},{date},{N},{fano:.3f},{obs},{hom.mean():.2f},{z_hom:.3f},{inh.mean():.2f},{z_inh:.3f}")
    except Exception as e:
        print(f"{args.ticker},{date},ERR,{e}", file=sys.stderr)
        print(f"{args.ticker},{date},0,nan,nan,nan,nan,nan,nan")


if __name__ == "__main__":
    main()
