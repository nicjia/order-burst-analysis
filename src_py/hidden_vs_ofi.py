#!/usr/bin/env python3
"""
hidden_vs_ofi.py — is the hidden-execution footprint INCREMENTAL to visible order-flow
imbalance? (Referee concern 8.) Per name-day, on a fixed 10s clock, regress the forward
3-min midpoint return on (i) Lee-Ready-signed aggressive-hidden volume and (ii) the
visible Cont-Kukanov-Stoikov OFI, both standardized. We report the hidden coefficient
univariate and jointly (controlling for OFI); if it survives the control, hidden flow
carries information beyond standard visible OFI.

Output per name-day: ticker,date,nbins,b_hid_uni,b_ofi_uni,b_hid_joint,b_ofi_joint
(coefficients in bps of forward 3-min return per 1 SD of the regressor)
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
DELTA = 10.0
FWD = 180.0
NA = "{t},{d},0,nan,nan,nan,nan"


def zc(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-12 else x * 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    date = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        if len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        df = pd.read_csv(a.msg, header=None, usecols=[0, 1, 3, 4], names=["t", "ty", "sz", "px"])
        h = df[df.ty == 5]
        edges = np.arange(RTH0, RTH1 + DELTA, DELTA); nb = len(edges) - 1
        # hidden signed (Lee-Ready, aggressive) volume per bin
        hflow = np.zeros(nb)
        if len(h) >= 3:
            ht = h.t.to_numpy(float); hpx = h.px.to_numpy(float) / BA.SCALE; hsz = h.sz.to_numpy(float)
            mid = BA.mid_at(bt, bm, ht)
            q = np.where(hpx > mid, 1.0, np.where(hpx < mid, -1.0, 0.0))
            ok = (q != 0) & (ht >= RTH0) & (ht < RTH1) & np.isfinite(mid)
            bi = ((ht[ok] - RTH0) / DELTA).astype(int)
            np.add.at(hflow, np.clip(bi, 0, nb - 1), q[ok] * hsz[ok])
        # visible CKS OFI per bin (ofi: sec->accumulated OFI)
        oflow = np.zeros(nb)
        for sec, v in ofi.items():
            if RTH0 <= sec < RTH1:
                oflow[int((sec - RTH0) / DELTA)] += v
        # forward 3-min mid return (bps) from each bin edge
        me = BA.mid_at(bt, bm, edges)
        me = pd.Series(me).ffill().bfill().to_numpy()
        mf = BA.mid_at(bt, bm, edges[:nb] + FWD)
        mf = pd.Series(mf).ffill().bfill().to_numpy()
        r = 1e4 * (mf / me[:nb] - 1.0)
        r = np.clip(r, -500, 500)
        ok = np.isfinite(r) & np.isfinite(hflow) & np.isfinite(oflow)
        if ok.sum() < 200 or np.std(hflow[ok]) < 1e-9 or np.std(oflow[ok]) < 1e-9:
            print(NA.format(t=a.ticker, d=date)); return
        H = zc(hflow[ok]); O = zc(oflow[ok]); Y = r[ok]
        # univariate slopes
        bh_u = np.polyfit(H, Y, 1)[0]
        bo_u = np.polyfit(O, Y, 1)[0]
        # joint OLS Y ~ [1, H, O]
        X = np.column_stack([np.ones(len(Y)), H, O])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        print(f"{a.ticker},{date},{int(ok.sum())},{bh_u:.5f},{bo_u:.5f},{beta[1]:.5f},{beta[2]:.5f}")
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
