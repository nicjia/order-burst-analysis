#!/usr/bin/env python3
"""
hidden_full.py — process ONE LOBSTER message file into a daily hidden-flow row.

Reconstructs the BBO, signs each hidden (type-5) execution by Lee-Ready (exec price
vs prevailing mid), clusters same-sign hidden trades into bursts, and emits one line:

    ticker,date,n_bursts,mk3,mk15,mk30,buy_vol,sell_vol,n_mid,n_signed

Signing is proper (not the always-+1 raw Direction field). Markouts are κ=0 (from the
burst-end mid, forward to 3/15/30 min). buy_vol/sell_vol are burst-level signed hidden
volume for daily COI = (buy_vol-sell_vol)/(buy_vol+sell_vol). n_mid / n_signed give the
fraction of hidden prints exactly at the midpoint (referee M12a: where the quote rule is
undefined and the tick rule is noisy).
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
import burst_alt as BA

NA_ROW = "{t},{d},0,nan,nan,nan,0,0,{nmid},{nsig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    args = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(args.msg))
    date = int(m.group(1)+m.group(2)+m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(args.msg)
        df = pd.read_csv(args.msg, header=None, usecols=[0, 1, 3, 4], names=["t", "ty", "sz", "px"])
        h = df[df.ty == 5]
        if len(h) < 3 or len(bt) < 50:
            print(NA_ROW.format(t=args.ticker, d=date, nmid=0, nsig=0)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float)/BA.SCALE; hsz = h.sz.to_numpy(np.int64)
        midh = BA.mid_at(bt, bm, ht); ok = np.isfinite(midh)
        ht, hpx, hsz, midh = ht[ok], hpx[ok], hsz[ok], midh[ok]
        sign = np.where(hpx > midh, 1, np.where(hpx < midh, -1, 0))
        n_mid = int((sign == 0).sum()); n_sig = int((sign != 0).sum())
        nz = sign != 0; ht, hsz, sign = ht[nz], hsz[nz], sign[nz]
        i = 0; n = len(ht); ends = []; dirs = []; vols = []
        while i < n:
            j = i
            while j+1 < n and sign[j+1] == sign[i] and (ht[j+1]-ht[j]) < 1.0:
                j += 1
            if j-i+1 >= 3:
                ends.append(ht[j]); dirs.append(int(sign[i])); vols.append(int(hsz[i:j+1].sum()))
            i = j+1
        if not ends:
            print(NA_ROW.format(t=args.ticker, d=date, nmid=n_mid, nsig=n_sig)); return
        ends = np.array(ends, float); dirs = np.array(dirs); vols = np.array(vols)
        e0 = BA.mid_at(bt, bm, ends)

        def markout(dt):
            e = BA.mid_at(bt, bm, ends+dt)
            mk = dirs*(e-e0)/e0*1e4
            mk = mk[np.isfinite(mk)]
            return np.nanmean(mk) if len(mk) else np.nan
        mk3, mk15, mk30 = markout(180.0), markout(900.0), markout(1800.0)
        buy = int(vols[dirs == 1].sum()); sell = int(vols[dirs == -1].sum())
        print(f"{args.ticker},{date},{len(ends)},{mk3:.5f},{mk15:.5f},{mk30:.5f},{buy},{sell},{n_mid},{n_sig}")
    except Exception as e:
        print(f"{args.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA_ROW.format(t=args.ticker, d=date, nmid=0, nsig=0))


if __name__ == "__main__":
    main()
