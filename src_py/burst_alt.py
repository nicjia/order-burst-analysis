#!/usr/bin/env python3
"""
burst_alt.py — alternative burst definitions (NOT Hawkes; C++ data_processor untouched).

Reconstructs the visible book / BBO from a LOBSTER message file (message-only archives)
and emits bursts under one of three definitions, each κ=0 (no D_b / no forward-price
gating), to its own labeled output so nothing overwrites:

  --method ofi     Order-Flow-Imbalance bursts (Cont–Kukanov–Stoikov): contiguous runs
                   of same-sign top-of-book OFI whose windowed magnitude exceeds a
                   per-stock threshold.  -> bursts_<T>_ofi.csv
  --method hidden  Hidden-execution bursts: temporal clusters of same-aggressor-side
                   hidden (type-5) trades.                          -> bursts_<T>_hidden.csv
  --method refill  Book-resilience bursts: aggressive sweeps of the best level whose
                   consumed side FAILS to replenish within a short window (informed =
                   non-refill; uses depth, not forward price).      -> bursts_<T>_refill.csv

Each row: Ticker,Date,StartTime,EndTime,Direction,TradeCount,EndBid,EndAsk,EndPrice,Mid_3m
so the κ=0 3-min markout is  Direction*(Mid_3m - EndMid)/EndMid  (EndMid=(EndBid+EndAsk)/2),
directly comparable to the Hawkes grid output.

Usage:
  python3 src_py/burst_alt.py --method ofi --ticker AAPL --stock-folder <dir> --out <path>
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

SCALE = 10000.0  # LOBSTER integer price -> dollars


def reconstruct(msg_path):
    """Return (bbo_t, bbo_mid, bbo_bid, bbo_ask, trades) for one day.
    trades: structured arrays t, sign(+1 buy aggressor), size, hidden(bool), price."""
    df = pd.read_csv(msg_path, header=None, usecols=[0, 1, 3, 4, 5],
                     names=["t", "ty", "sz", "px", "dr"])
    t = df["t"].to_numpy(float); ty = df["ty"].to_numpy(np.int8)
    sz = df["sz"].to_numpy(np.int64); px = df["px"].to_numpy(np.int64)
    dr = df["dr"].to_numpy(np.int8)
    bid = {}; ask = {}
    best_bid = 0; best_ask = 1 << 62
    bt = []; bmid = []; bb = []; ba = []; bbsz = []; basz = []
    tr_t = []; tr_sign = []; tr_sz = []; tr_hid = []
    ofi = {}                       # int-second -> accumulated Cont-Kukanov-Stoikov OFI
    pbb = pba = pbs = pas = 0      # prev best bid/ask price & size (for OFI increments)
    def rescan_bid():
        return max((k for k, v in bid.items() if v > 0), default=0)
    def rescan_ask():
        return min((k for k, v in ask.items() if v > 0), default=(1 << 62))
    n = len(t)
    for i in range(n):
        typ = ty[i]; p = px[i]; s = sz[i]; d = dr[i]
        if typ == 1:  # submit
            if d == 1:
                bid[p] = bid.get(p, 0) + s
                if p > best_bid: best_bid = p
            else:
                ask[p] = ask.get(p, 0) + s
                if p < best_ask: best_ask = p
        elif typ == 2 or typ == 3:  # cancel / delete (reduce resting)
            book = bid if d == 1 else ask
            if p in book:
                book[p] -= s
                if book[p] <= 0:
                    del book[p]
                    if d == 1 and p >= best_bid: best_bid = rescan_bid()
                    elif d == -1 and p <= best_ask: best_ask = rescan_ask()
        elif typ == 4:  # visible execution: resting limit on side d is hit; aggressor = -d
            book = bid if d == 1 else ask
            if p in book:
                book[p] -= s
                if book[p] <= 0:
                    del book[p]
                    if d == 1 and p >= best_bid: best_bid = rescan_bid()
                    elif d == -1 and p <= best_ask: best_ask = rescan_ask()
            tr_t.append(t[i]); tr_sign.append(-d); tr_sz.append(s); tr_hid.append(False)
        elif typ == 5:  # hidden execution: no visible-book change; aggressor = -d
            tr_t.append(t[i]); tr_sign.append(-d); tr_sz.append(s); tr_hid.append(True)
        else:
            continue
        if 0 < best_bid < best_ask < (1 << 62):
            cbs = bid.get(best_bid, 0); cas = ask.get(best_ask, 0)
            if best_bid != pbb or best_ask != pba or cbs != pbs or cas != pas:
                # Cont-Kukanov-Stoikov OFI increment (buy pressure positive)
                if pbb:
                    db = cbs if best_bid > pbb else (cbs - pbs if best_bid == pbb else -pbs)
                    da = cas if best_ask < pba else (cas - pas if best_ask == pba else -pas)
                    sec = int(t[i]); ofi[sec] = ofi.get(sec, 0.0) + db - da
                m = (best_bid + best_ask) / (2 * SCALE)
                bt.append(t[i]); bmid.append(m); bb.append(best_bid); ba.append(best_ask)
                bbsz.append(cbs); basz.append(cas)
                pbb, pba, pbs, pas = best_bid, best_ask, cbs, cas
    return (np.array(bt), np.array(bmid), np.array(bb, float)/SCALE, np.array(ba, float)/SCALE,
            np.array(bbsz, float), np.array(basz, float), ofi,
            (np.array(tr_t), np.array(tr_sign, np.int8), np.array(tr_sz, np.int64), np.array(tr_hid, bool)))


def mid_at(bbo_t, bbo_mid, q):
    """last mid at or before each query time q (forward-in-time lookup with +180s handled by caller)."""
    idx = np.searchsorted(bbo_t, q, side="right") - 1
    out = np.full(len(q), np.nan)
    ok = idx >= 0
    out[ok] = bbo_mid[idx[ok]]
    return out


def bbo_at(bbo_t, bbo_bid, bbo_ask, q):
    idx = np.searchsorted(bbo_t, q, side="right") - 1
    b = np.full(len(q), np.nan); a = np.full(len(q), np.nan)
    ok = idx >= 0
    b[ok] = bbo_bid[idx[ok]]; a[ok] = bbo_ask[idx[ok]]
    return b, a


def finalize(rows, bbo_t, bbo_mid, bbo_bid, bbo_ask, date, ticker):
    """rows: list of (start,end,direction,tradecount). attach EndBid/EndAsk/Mid_3m."""
    if not rows:
        return pd.DataFrame(columns=["Ticker","Date","StartTime","EndTime","Direction",
                                     "TradeCount","EndBid","EndAsk","EndPrice","Mid_3m"])
    r = np.array(rows, float)
    end = r[:, 1]
    eb, ea = bbo_at(bbo_t, bbo_bid, bbo_ask, end)
    m3 = mid_at(bbo_t, bbo_mid, end + 180.0)
    return pd.DataFrame(dict(Ticker=ticker, Date=date, StartTime=r[:,0], EndTime=end,
                             Direction=r[:,2].astype(int), TradeCount=r[:,3].astype(int),
                             EndBid=eb, EndAsk=ea, EndPrice=(eb+ea)/2, Mid_3m=m3))


def depth_at(bbo_t, bbsz, basz, q):
    idx = np.searchsorted(bbo_t, q, side="right") - 1
    out = np.full(len(q), np.nan); ok = idx >= 0
    out[ok] = bbsz[idx[ok]] + basz[idx[ok]]
    return out


# ── definitions (ctx = bt,bm,bb,ba,bbsz,basz,ofi,trades) ─────────────────────
def bursts_ofi(ctx, win=10.0):
    bt, bm, bb, ba, bbsz, basz, ofi, trades = ctx
    if not ofi: return []
    secs = np.array(sorted(ofi)); val = np.array([ofi[s] for s in secs], float)
    roll = pd.Series(val).rolling(int(win), min_periods=1).sum().to_numpy()  # true CKS OFI, 10s window
    thr = np.nanpercentile(np.abs(roll), 95)
    if not np.isfinite(thr) or thr <= 0: return []
    hot = np.abs(roll) > thr
    rows = []; i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j+1 < len(hot) and hot[j+1]: j += 1
            seg = roll[i:j+1]; d = 1 if seg[np.argmax(np.abs(seg))] > 0 else -1
            rows.append((float(secs[i]), float(secs[j]), d, j-i+1))
            i = j+1
        else: i += 1
    return rows


def bursts_hidden(ctx, gap=1.0, min_ct=3):
    tt, sign, sz, hid = ctx[7]
    ht, hs = tt[hid], sign[hid]
    if len(ht) < min_ct: return []
    rows = []; i = 0; n = len(ht)
    while i < n:
        j = i
        while j+1 < n and hs[j+1] == hs[i] and (ht[j+1]-ht[j]) < gap: j += 1
        if j-i+1 >= min_ct: rows.append((float(ht[i]), float(ht[j]), int(hs[i]), j-i+1))
        i = j+1
    return rows


def bursts_refill(ctx, gap=0.5, min_ct=3, delta=10.0, refill_frac=0.5):
    """Non-refill test uses DEPTH (queue size), not price; markout measured from end+delta so the
    10s observation window cannot overlap the 3-min markout window (κ=0, no forward price)."""
    bt, bm, bb, ba, bbsz, basz, ofi, trades = ctx
    tt, sign, sz, hid = trades
    vt, vs = tt[~hid], sign[~hid]
    if len(vt) < min_ct or len(bt) < 50: return []
    rows = []; i = 0; n = len(vt)
    while i < n:
        j = i
        while j+1 < n and vs[j+1] == vs[i] and (vt[j+1]-vt[j]) < gap: j += 1
        if j-i+1 >= min_ct:
            start = vt[i]; end = vt[j]; d = int(vs[i])
            pre = depth_at(bt, bbsz, basz, np.array([start]))[0]
            post = depth_at(bt, bbsz, basz, np.array([end+delta]))[0]
            if np.isfinite(pre) and np.isfinite(post) and pre > 0 and post < refill_frac*pre:
                rows.append((float(start), float(end+delta), d, j-i+1))  # ref time = end+delta
        i = j+1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ofi","hidden","refill"])
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--stock-folder", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fn = {"ofi":bursts_ofi, "hidden":bursts_hidden, "refill":bursts_refill}[args.method]
    frames = []
    for msg in sorted(glob.glob(os.path.join(args.stock_folder, "*message*.csv"))):
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(msg))
        date = int(m.group(1)+m.group(2)+m.group(3)) if m else 0
        bt, bm, bb, ba, bbsz, basz, ofi, trades = reconstruct(msg)
        rows = fn((bt, bm, bb, ba, bbsz, basz, ofi, trades))
        frames.append(finalize(rows, bt, bm, bb, ba, date, args.ticker))
        print(f"  {args.method} {args.ticker} {date}: {len(rows)} bursts", flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else finalize([],None,None,None,None,0,args.ticker)
    out.to_csv(args.out, index=False)
    print(f"WROTE {args.out}: {len(out)} bursts")


if __name__ == "__main__":
    main()
