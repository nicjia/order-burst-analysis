#!/usr/bin/env python3
"""
hidden_signrobust.py — stale-mid robustness for the aggressive-hidden footprint (referee
concern 7: a print above a lagging NASDAQ-only mid is mechanically classified a buy just
before the mid ticks up, manufacturing a positive markout). Per name-day, recompute the
3-min aggressive-hidden markout signing against:
  now  : the mid prevailing at the trade (baseline, = the paper's +2.09)
  lag1 : the mid 1s BEFORE the trade (deliberately stale) -- if staleness inflates the
         footprint, this makes it worse
  fwd1 : the mid 1s AFTER the trade (conservative, removes any pre-update leakage)
  far  : keep only prints |px - mid| > 0.5*(ask-bid), i.e. executing outside the quote,
         where the aggressor side is unambiguous regardless of small mid errors
Markouts are always measured from the contemporaneous mid forward; only the SIGN rule
changes. If the footprint survives fwd1 and far, it is not a stale-mid artifact.

Output: ticker,date,n_now,mk3_now,mk3_lag,mk3_fwd,n_far,mk3_far
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

NA = "{t},{d},0,nan,nan,nan,0,nan"


def bursts_from(ht, hsz, sign, minrun=3, gap=1.0):
    nz = sign != 0
    ht2, sign2 = ht[nz], sign[nz]
    ends, dirs = [], []
    n = len(sign2); i = 0
    while i < n:
        j = i
        while j + 1 < n and sign2[j + 1] == sign2[i] and (ht2[j + 1] - ht2[j]) < gap:
            j += 1
        if j - i + 1 >= minrun:
            ends.append(ht2[j]); dirs.append(int(sign2[i]))
        i = j + 1
    return np.array(ends, float), np.array(dirs, int)


def markout(bt, bm, ends, dirs, dt=180.0):
    if len(ends) == 0:
        return np.nan
    b0 = BA.mid_at(bt, bm, ends); e = BA.mid_at(bt, bm, ends + dt)
    mk = dirs * (e - b0) / b0 * 1e4; mk = mk[np.isfinite(mk)]
    return float(np.nanmean(mk)) if len(mk) else np.nan


def mk_for_sign(bt, bm, ht, hsz, s):
    e, d = bursts_from(ht, hsz, s)
    return markout(bt, bm, e, d)


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
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE; hsz = h.sz.to_numpy(np.int64)
        mid_now = BA.mid_at(bt, bm, ht)
        mid_lag = BA.mid_at(bt, bm, ht - 1.0)
        mid_fwd = BA.mid_at(bt, bm, ht + 1.0)
        bid, ask = BA.bbo_at(bt, bb, ba, ht)
        ok = np.isfinite(mid_now) & (mid_now > 0) & np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
        ht, hpx, hsz = ht[ok], hpx[ok], hsz[ok]
        mid_now, mid_lag, mid_fwd = mid_now[ok], mid_lag[ok], mid_fwd[ok]
        bid, ask = bid[ok], ask[ok]
        if len(ht) < 3:
            print(NA.format(t=a.ticker, d=date)); return
        s_now = np.where(hpx > mid_now, 1.0, np.where(hpx < mid_now, -1.0, 0.0))
        s_lag = np.where(hpx > mid_lag, 1.0, np.where(hpx < mid_lag, -1.0, 0.0))
        s_fwd = np.where(hpx > mid_fwd, 1.0, np.where(hpx < mid_fwd, -1.0, 0.0))
        # far-from-mid: strictly outside the quote (unambiguous aggressor)
        half = 0.5 * (ask - bid)
        far = np.abs(hpx - mid_now) > half
        s_far = np.where(far, s_now, 0.0)
        n_now = int((s_now != 0).sum()); n_far = int((s_far != 0).sum())
        mk_now = mk_for_sign(bt, bm, ht, hsz, s_now)
        mk_lag = mk_for_sign(bt, bm, ht, hsz, s_lag)
        mk_fwd = mk_for_sign(bt, bm, ht, hsz, s_fwd)
        mk_far = mk_for_sign(bt, bm, ht, hsz, s_far)
        print(f"{a.ticker},{date},{n_now},{mk_now:.5f},{mk_lag:.5f},{mk_fwd:.5f},{n_far},{mk_far:.5f}")
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
