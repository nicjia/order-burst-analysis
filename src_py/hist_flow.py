#!/usr/bin/env python3
"""
hist_flow.py — lightweight daily-flow extractor for the 2017-2021 historical
out-of-sample probe (NYSE subset). No book reconstruction and no SGD: the paper's
baselines show the deployable content is the SIGN of daily net aggressive flow,
so we compute exactly that (plus burst count for the volatility probe) directly
from the LOBSTER message stream.

Per message file -> one row:
  ticker,date,netflow,n_bursts,buy,sell
where netflow = sum over RTH type-4 executions of (aggressor_sign * size),
aggressor_sign = -Direction (LOBSTER convention), and n_bursts = same-sign
sub-second-clustered runs (gap<1s, length>=3), the paper's burst definition.
"""
import argparse, os, re, sys
import numpy as np, pandas as pd

RTH0, RTH1 = 34200.0, 57600.0


def count_bursts(times, sign, gap=1.0, minrun=3):
    n = len(sign)
    if n < minrun:
        return 0
    brk = (np.diff(sign) != 0) | (np.diff(times) >= gap)
    rid = np.concatenate([[0], np.cumsum(brk)])
    return int((np.bincount(rid) >= minrun).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    date = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        df = pd.read_csv(a.msg, header=None, usecols=[0, 1, 3, 5], names=["t", "ty", "sz", "dr"])
        tr = df[(df.ty == 4) & (df.t >= RTH0) & (df.t <= RTH1)]
        if len(tr) < 10:
            print(f"{a.ticker},{date},0,0,0,0"); return
        t = tr.t.to_numpy(float); s = (-tr.dr).to_numpy(np.int8); sz = tr.sz.to_numpy(np.int64)
        netflow = int((s.astype(np.int64) * sz).sum())
        buy = int(sz[s > 0].sum()); sell = int(sz[s < 0].sum())
        nb = count_bursts(t, s)
        print(f"{a.ticker},{date},{netflow},{nb},{buy},{sell}")
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(f"{a.ticker},{date},0,0,0,0")


if __name__ == "__main__":
    main()
