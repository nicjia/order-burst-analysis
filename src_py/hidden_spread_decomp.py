#!/usr/bin/env python3
"""
hidden_spread_decomp.py — Huang-Stoll (1996) effective/realized/price-impact spread
decomposition for aggressive hidden executions, per name-day. This validates the
referee's demand (concern 3): show that a standard decomposition yields an
adverse-selection component near the +2 bps footprint, and it fixes the units concern
(14c) by comparing per-trade permanent impact to the effective HALF-spread.

For each aggressive-hidden trade t (quote-rule signed, away from mid), with sign q,
trade price p, quote midpoint m_t, quote 3 min later m_{t+3}, and quoted spread s_t:
  effective half-spread   = q*(p - m_t)/m_t * 1e4          (what the taker pays)
  price impact (adv.sel.) = q*(m_{t+3} - m_t)/m_t * 1e4    (permanent move = the markout)
  realized half-spread    = effective - price impact        (what the provider keeps)
  quoted half-spread      = 0.5*(ask - bid)/m_t * 1e4
Identity: effective = price_impact + realized. A price-impact component near +2 bps,
larger than the realized half-spread, confirms the footprint is the adverse-selection
component of the spread; a negative realized half-spread means hidden liquidity providers
are adversely selected by aggressive hidden takers.

Output: ticker,date,n,eff_hs,quoted_hs,price_impact,realized_hs
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

NA = "{t},{d},0,nan,nan,nan,nan"


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
        m0 = BA.mid_at(bt, bm, ht)                       # mid at the trade
        m3 = BA.mid_at(bt, bm, ht + 180.0)              # mid 3 min later
        bid, ask = BA.bbo_at(bt, bb, ba, ht)
        ok = (np.isfinite(m0) & (m0 > 0) & np.isfinite(m3) & (m3 > 0)
              & np.isfinite(bid) & np.isfinite(ask) & (ask > bid))
        ht, hpx, m0, m3, bid, ask = ht[ok], hpx[ok], m0[ok], m3[ok], bid[ok], ask[ok]
        q = np.where(hpx > m0, 1.0, np.where(hpx < m0, -1.0, 0.0))   # aggressive sign
        agg = q != 0
        if agg.sum() < 3:
            print(NA.format(t=a.ticker, d=date)); return
        hpx, m0, m3, bid, ask, q = hpx[agg], m0[agg], m3[agg], bid[agg], ask[agg], q[agg]
        eff = q * (hpx - m0) / m0 * 1e4
        pi = q * (m3 - m0) / m0 * 1e4
        rlz = eff - pi
        qs = 0.5 * (ask - bid) / m0 * 1e4
        # winsorize per-trade to guard corrupt prints
        def wm(x):
            x = x[np.isfinite(x)]; x = x[np.abs(x) <= 1000]
            return float(np.mean(x)) if len(x) else np.nan
        print(f"{a.ticker},{date},{int(agg.sum())},{wm(eff):.5f},{wm(qs):.5f},{wm(pi):.5f},{wm(rlz):.5f}")
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
