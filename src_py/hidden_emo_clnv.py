#!/usr/bin/env python3
"""
hidden_emo_clnv.py — re-sign hidden (type-5) executions under FOUR canonical trade-sign
classifiers and report the 3/15/30-min directional markout under each. This pins the
referee's "3x magnitude uncertainty" (drop-mid +2.0 vs tick +0.6) by adding the two
standard intermediate classifiers.

Classifiers (all on the reconstructed NASDAQ BBO; true SIP NBBO needs external data):
  QD   quote rule, at-mid prints DROPPED     (the paper's headline, +2.0)
  TICK at-mid prints signed by tick rule       (the paper's low end, +0.6)
  EMO  Ellis-Michaely-O'Hara: at-ask=buy, at-bid=sell, inside spread -> tick
  CLNV Chakrabarty et al.: upper 30% of spread=buy, lower 30%=sell, middle 40% -> tick

Output row:
  ticker,date,n_qd,n_mid,n_sig,mk3_qd,mk15_qd,mk30_qd,mk3_tick,mk3_emo,mk3_clnv
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NA = "{t},{d},0,0,0,nan,nan,nan,nan,nan,nan"


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


def markout(bt, bm, ends, dirs, dt):
    if len(ends) == 0:
        return np.nan
    b0 = BA.mid_at(bt, bm, ends)
    e = BA.mid_at(bt, bm, ends + dt)
    mk = dirs * (e - b0) / b0 * 1e4
    mk = mk[np.isfinite(mk)]
    return float(np.nanmean(mk)) if len(mk) else np.nan


def tick_sign(px, seed):
    s = np.zeros(len(px), int); last = np.nan; lastsign = 0
    for k in range(len(px)):
        if seed[k] != 0:                      # keep a firm quote-rule sign if present
            s[k] = seed[k]
        elif np.isfinite(last):
            if px[k] > last: s[k] = 1
            elif px[k] < last: s[k] = -1
            else: s[k] = lastsign
        if np.isfinite(px[k]) and (not np.isfinite(last) or px[k] != last):
            last = px[k]
        if s[k] != 0: lastsign = s[k]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    args = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(args.msg))
    date = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(args.msg)
        df = pd.read_csv(args.msg, header=None, usecols=[0, 1, 3, 4], names=["t", "ty", "sz", "px"])
        h = df[df.ty == 5]
        if len(h) < 3 or len(bt) < 50:
            print(NA.format(t=args.ticker, d=date)); return
        ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE; hsz = h.sz.to_numpy(np.int64)
        mid = BA.mid_at(bt, bm, ht)
        bid, ask = BA.bbo_at(bt, bb, ba, ht)
        ok = np.isfinite(mid) & np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
        ht, hpx, hsz, mid, bid, ask = ht[ok], hpx[ok], hsz[ok], mid[ok], bid[ok], ask[ok]
        if len(ht) < 3:
            print(NA.format(t=args.ticker, d=date)); return
        spr = ask - bid

        # QD quote rule (drop at-mid)
        qd = np.where(hpx > mid, 1, np.where(hpx < mid, -1, 0))
        n_mid = int((qd == 0).sum()); n_sig = int((qd != 0).sum())
        # TICK: at-mid -> tick rule
        tick = tick_sign(hpx, qd)
        # EMO: at/through ask -> buy, at/through bid -> sell, strictly inside -> tick
        emo_seed = np.where(hpx >= ask, 1, np.where(hpx <= bid, -1, 0))
        emo = tick_sign(hpx, emo_seed)
        # CLNV: upper 30% -> buy, lower 30% -> sell, middle 40% -> tick
        clnv_seed = np.where(hpx >= ask - 0.3 * spr, 1, np.where(hpx <= bid + 0.3 * spr, -1, 0))
        clnv = tick_sign(hpx, clnv_seed)

        eqd, dqd = bursts_from(ht, hsz, qd)
        mk3 = markout(bt, bm, eqd, dqd, 180.0)      # AGGRESSIVE (away-from-mid) hidden bursts
        mk15 = markout(bt, bm, eqd, dqd, 900.0)
        mk30 = markout(bt, bm, eqd, dqd, 1800.0)
        et, dt_ = bursts_from(ht, hsz, tick);  mk3_t = markout(bt, bm, et, dt_, 180.0)
        ee, de = bursts_from(ht, hsz, emo);    mk3_e = markout(bt, bm, ee, de, 180.0)
        ec, dc = bursts_from(ht, hsz, clnv);   mk3_c = markout(bt, bm, ec, dc, 180.0)
        # DECOMPOSITION: at-mid prints on their own, tick-signed among themselves.
        # if this is ~0, at-mid prints are uninformed (dilution); if strongly signed,
        # they carry information the drop-mid rule discards.
        am = (qd == 0)
        mk3_mid = mk15_mid = mk30_mid = float("nan"); n_midburst = 0
        if am.sum() >= 3:
            msign = tick_sign(hpx[am], np.zeros(int(am.sum()), int))
            em2, dm2 = bursts_from(ht[am], hsz[am], msign)
            n_midburst = len(em2)
            mk3_mid = markout(bt, bm, em2, dm2, 180.0)
            mk15_mid = markout(bt, bm, em2, dm2, 900.0)
            mk30_mid = markout(bt, bm, em2, dm2, 1800.0)
        print(f"{args.ticker},{date},{len(eqd)},{n_mid},{n_sig},{mk3:.5f},{mk15:.5f},{mk30:.5f},"
              f"{mk3_t:.5f},{mk3_e:.5f},{mk3_c:.5f},{n_midburst},{mk3_mid:.5f},{mk15_mid:.5f},{mk30_mid:.5f}")
    except Exception as e:
        print(f"{args.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=args.ticker, d=date))


if __name__ == "__main__":
    main()
