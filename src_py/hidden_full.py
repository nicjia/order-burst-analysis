#!/usr/bin/env python3
"""
hidden_full.py — process ONE LOBSTER message file into a daily hidden-flow row.

Reconstructs the BBO, signs each hidden (type-5) execution, clusters same-sign hidden
trades into bursts, and emits one line. Second-round additions:
  * PLACEBO markout (Major 2): identical directions at RANDOM within-day times, so the
    burst markout net of placebo removes the day-direction x intraday-drift component.
  * TICK-RULE variant (Major 2): sign the ~49% at-midpoint prints by the tick rule
    (price vs previous different trade price) instead of dropping them, and report the
    3-min markout under this alternative classifier.

Output columns:
  ticker,date,n,mk3,mk15,mk30,buy,sell,n_mid,n_sig,pmk3,pmk15,pmk30,n_tick,mk3_tick
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NA = "{t},{d},0,nan,nan,nan,0,0,{nmid},{nsig},nan,nan,nan,0,nan"


def bursts_from(ht, hsz, sign, minrun=3, gap=1.0):
    """same-sign clustered runs -> (end_times, directions)."""
    nz = sign != 0
    ht, hsz, sign = ht[nz], hsz[nz], sign[nz]
    ends, dirs = [], []
    i = 0; n = len(ht)
    while i < n:
        j = i
        while j+1 < n and sign[j+1] == sign[i] and (ht[j+1]-ht[j]) < gap:
            j += 1
        if j-i+1 >= minrun:
            ends.append(ht[j]); dirs.append(int(sign[i]))
        i = j+1
    return np.array(ends, float), np.array(dirs)


def markout(bt, bm, ends, dirs, dt, base=None):
    b0 = BA.mid_at(bt, bm, ends) if base is None else base
    e = BA.mid_at(bt, bm, ends+dt)
    mk = dirs*(e-b0)/b0*1e4; mk = mk[np.isfinite(mk)]
    return np.nanmean(mk) if len(mk) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    args = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(args.msg))
    date = int(m.group(1)+m.group(2)+m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(args.msg)
        df = pd.read_csv(args.msg, header=None, usecols=[0, 1, 3, 4], names=["t", "ty", "sz", "px"])
        h = df[df.ty == 5]
        if len(h) < 3 or len(bt) < 50:
            print(NA.format(t=args.ticker, d=date, nmid=0, nsig=0)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float)/BA.SCALE; hsz = h.sz.to_numpy(np.int64)
        midh = BA.mid_at(bt, bm, ht); ok = np.isfinite(midh)
        ht, hpx, hsz, midh = ht[ok], hpx[ok], hsz[ok], midh[ok]
        qsign = np.where(hpx > midh, 1, np.where(hpx < midh, -1, 0))  # quote rule
        n_mid = int((qsign == 0).sum()); n_sig = int((qsign != 0).sum())

        # --- quote-rule bursts + markouts ---
        ends, dirs = bursts_from(ht, hsz, qsign)
        if len(ends) == 0:
            print(NA.format(t=args.ticker, d=date, nmid=n_mid, nsig=n_sig)); return
        e0 = BA.mid_at(bt, bm, ends)
        mk3 = markout(bt, bm, ends, dirs, 180.0, e0)
        mk15 = markout(bt, bm, ends, dirs, 900.0, e0)
        mk30 = markout(bt, bm, ends, dirs, 1800.0, e0)
        # signed hidden volume for daily COI
        vall = []; i = 0; n = len(ht); nz = qsign != 0
        htz, hszz, sz = ht[nz], hsz[nz], qsign[nz]
        i = 0; nn = len(htz); vols = []
        while i < nn:
            j = i
            while j+1 < nn and sz[j+1] == sz[i] and (htz[j+1]-htz[j]) < 1.0: j += 1
            if j-i+1 >= 3: vols.append((int(sz[i]), int(hszz[i:j+1].sum())))
            i = j+1
        buy = sum(v for s2, v in vols if s2 == 1); sell = sum(v for s2, v in vols if s2 == -1)

        # --- PLACEBO: same directions, random within-day times ---
        rng = np.random.default_rng(date + hash(args.ticker) % 100000)
        rt = np.sort(rng.uniform(RTH0, RTH1-1800.0, len(ends)))
        p0 = BA.mid_at(bt, bm, rt)
        pmk3 = markout(bt, bm, rt, dirs, 180.0, p0)
        pmk15 = markout(bt, bm, rt, dirs, 900.0, p0)
        pmk30 = markout(bt, bm, rt, dirs, 1800.0, p0)

        # --- TICK-RULE variant: sign at-mid prints by tick rule, recluster ---
        tsign = qsign.copy()
        px = hpx
        last = np.nan; lastsign = 0
        for k in range(len(px)):
            if tsign[k] == 0:
                if np.isfinite(last):
                    if px[k] > last: tsign[k] = 1
                    elif px[k] < last: tsign[k] = -1
                    else: tsign[k] = lastsign
            if np.isfinite(px[k]) and (not np.isfinite(last) or px[k] != last):
                last = px[k]
            if tsign[k] != 0: lastsign = tsign[k]
        te, td = bursts_from(ht, hsz, tsign)
        mk3_tick = markout(bt, bm, te, td, 180.0) if len(te) else np.nan

        print(f"{args.ticker},{date},{len(ends)},{mk3:.5f},{mk15:.5f},{mk30:.5f},{buy},{sell},"
              f"{n_mid},{n_sig},{pmk3:.5f},{pmk15:.5f},{pmk30:.5f},{len(te)},{mk3_tick:.5f}")
    except Exception as e:
        print(f"{args.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=args.ticker, d=date, nmid=0, nsig=0))


if __name__ == "__main__":
    main()
