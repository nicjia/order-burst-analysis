#!/usr/bin/env python3
"""
hidden_term.py — term-structure extension of hidden_full.py (referee #3).

Adds the missing intraday horizons the title promises: 1h, 2h, and burst-to-close
markouts for Lee-Ready-signed hidden-execution bursts, and a TIME-OF-DAY-STRATIFIED
placebo (same directions, random times drawn WITHIN each burst's own 30-minute
intraday bucket, so the placebo preserves the U-shaped intraday intensity/markout
profile that a uniform-random placebo ignores).

Output columns:
  ticker,date,n,mk3,mk15,mk30,mk60,mk120,mkclose,
  pmk3,pmk15,pmk30,pmk60,pmk120,pmkclose,n_sig
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NCOL = 15
NA = "{t},{d},0," + ",".join(["nan"] * (NCOL - 3))


def bursts_from(ht, sign, minrun=3, gap=1.0):
    nz = sign != 0
    ht, sign = ht[nz], sign[nz]
    ends, dirs = [], []
    i, n = 0, len(ht)
    while i < n:
        j = i
        while j + 1 < n and sign[j + 1] == sign[i] and (ht[j + 1] - ht[j]) < gap:
            j += 1
        if j - i + 1 >= minrun:
            ends.append(ht[j]); dirs.append(int(sign[i]))
        i = j + 1
    return np.array(ends, float), np.array(dirs)


def mk(bt, bm, t0, dirs, base, dt=None, target_t=None):
    """directional markout (bps) from base mid at t0 to mid at t0+dt (or target_t)."""
    tt = (t0 + dt) if target_t is None else np.full(len(t0), target_t)
    e = BA.mid_at(bt, bm, tt)
    v = dirs * (e - base) / base * 1e4
    v = v[np.isfinite(v)]
    return np.nanmean(v) if len(v) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    date = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        df = pd.read_csv(a.msg, header=None, usecols=[0, 1, 3, 4], names=["t", "ty", "sz", "px"])
        h = df[df.ty == 5]
        if len(h) < 3 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE
        midh = BA.mid_at(bt, bm, ht); ok = np.isfinite(midh)
        ht, hpx, midh = ht[ok], hpx[ok], midh[ok]
        qsign = np.where(hpx > midh, 1, np.where(hpx < midh, -1, 0))
        n_sig = int((qsign != 0).sum())
        ends, dirs = bursts_from(ht, qsign)
        if len(ends) == 0:
            print(NA.format(t=a.ticker, d=date)); return
        e0 = BA.mid_at(bt, bm, ends)
        H = [180., 900., 1800., 3600., 7200.]
        mks = [mk(bt, bm, ends, dirs, e0, dt=dt) for dt in H]
        mkclose = mk(bt, bm, ends, dirs, e0, target_t=RTH1)

        # --- TOD-stratified placebo: random time within each burst's own 30-min bucket ---
        rng = np.random.default_rng(date + hash(a.ticker) % 100000)
        buckets = RTH0 + np.floor(np.clip(ends - RTH0, 0, None) / 1800.0) * 1800.0
        rt = np.clip(buckets + rng.uniform(0, 1800.0, len(ends)), RTH0, RTH1 - 1.0)
        p0 = BA.mid_at(bt, bm, rt)
        pmks = [mk(bt, bm, rt, dirs, p0, dt=dt) for dt in H]
        pmkclose = mk(bt, bm, rt, dirs, p0, target_t=RTH1)

        vals = [len(ends)] + mks + [mkclose] + pmks + [pmkclose] + [n_sig]
        print(f"{a.ticker},{date}," + ",".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in vals))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
