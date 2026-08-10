#!/usr/bin/env python3
"""
buffered_mids.py — extract LOBSTER intraday midpoints at 9:35 and 15:55 (plus 9:30 open
and 16:00 close mids) per name-day, to resolve the overnight anomaly (referee concern 5).
The Sample B overnight Sharpe of +2.44 uses Yahoo official open/close, which may carry
auction-print / stale-data artifacts. Re-measuring the overnight return mid-to-mid with a
5-minute buffer (15:55 -> next 9:35) purges the auction prints: genuine overnight-impounded
information survives the buffer; a print artifact does not.

Output: ticker,date,mid_open,mid_0935,mid_1555,mid_close  (dollars; nan if unavailable)
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

OPEN, T0935, T1555, CLOSE = 34200.0, 34500.0, 57300.0, 57600.0
NA = "{t},{d},nan,nan,nan,nan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    date = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        if len(bt) < 20:
            print(NA.format(t=a.ticker, d=date)); return
        q = BA.mid_at(bt, bm, np.array([OPEN + 60, T0935, T1555, CLOSE - 1]))
        vals = ["%.4f" % v if np.isfinite(v) and v > 0 else "nan" for v in q]
        print(f"{a.ticker},{date}," + ",".join(vals))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
