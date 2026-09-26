#!/usr/bin/env python3
"""ETF burst extraction (cluster, per name-day) for the ETF -> constituent lead test (backlog T1.10 / B11.3 / G4).
Writes every run burst (gap 0.5 s, >= 3 packets) of an ETF: t_b, t_e, side, children, volume, and the ETF's mid
move over the burst. Usage: etf_bursts_extract.py --msg FILE --ticker TK --out OUT.csv.gz [--helper p4_bbo]"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import p4_extract as X
import p4_packets as PK
import fingerprint_packets as FP
import fingerprint_stats as FS
import burst_defs_raw2 as B2

ap = argparse.ArgumentParser()
ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--helper", default=None)
a = ap.parse_args()
date = "".join(re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(a.msg).name).groups())
msg = X.read_messages(a.msg)
context, _, _, _ = X.bbo_context(a.msg, msg, a.helper)
mid = X.MidPath(context[0], context[1], context[2], context[3])
day = FP.packet_arrays(PK.fast_packets(msg, context))
r = (day["time"] >= X.RTH0) & (day["time"] < X.RTH1)
t, sign, vol = day["time"][r], day["sign"][r].astype(int), day["volume"][r]
ids, _ = FS.burst_ids(t, sign, 0.5, "run")
b, m = B2.make_bursts(np.where(sign != 0, ids, -1), t, sign, vol)
out = pd.DataFrame()
if b is not None:
    m0, m1 = mid.at(b["t_b"]), mid.at(b["t_e"] + 0.01)
    out = pd.DataFrame(dict(date=date, etf=a.ticker, t_b=b["t_b"], t_e=b["t_e"], side=b["side"], n=b["n"], vol=b["vol"],
                            move_bps=b["side"] * (m1 - m0) / m0 * 1e4))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.out, index=False)
print(len(out))
