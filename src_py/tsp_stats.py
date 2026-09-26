#!/usr/bin/env python3
"""program-evidence-v1 module J: per-day tape descriptors for the multi-year and Tick Size Pilot panels.

For each cached day of one name: signed packets, untruncated share, mean pre-trade half-spread, and
for run/60 bursts (>= 3 packets) the 3-minute markout measured from the first packet after the
burst (so the conditioning and measurement windows are disjoint) with the half-spread at that
packet. Also the program-burst volume share under the stage-3b score (transported, descriptive).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import program_bursts as PB

RTH1 = 57600.0
HORIZON = 180.0


def day_stats(day, model):
    t = day["time"].astype(float); sign = day["sign"].astype(int); mid = day["mid"].astype(float)
    spread = day["spread_bps"].astype(float); vol = day["volume"].astype(float)
    sg = sign != 0
    row = dict(n_signed=int(sg.sum()), untruncated_share=float(day["untruncated"][sg].mean()) if sg.any() else np.nan,
               half_spread_bps=float(np.nanmean(spread[sg]) / 2) if sg.any() else np.nan)
    member, table = PB.run_bursts(day)
    if not table:
        row.update(n_bursts=0, mk3_mean_bps=np.nan, mk3_n=0, half_spread_end_bps=np.nan, program_volume_share=np.nan)
        return row
    end = table["end"]; side = table["side"]
    nxt = np.searchsorted(t, end, side="right")
    ok = (nxt < len(t)) & (end + HORIZON < RTH1)
    later = np.searchsorted(t, end + HORIZON, side="left")
    ok &= later < len(t)
    m0 = np.where(ok, mid[np.minimum(nxt, len(t) - 1)], np.nan)
    m1 = np.where(ok, mid[np.minimum(later, len(t) - 1)], np.nan)
    mk3 = side * (m1 - m0) / m0 * 1e4
    hs = np.where(ok, spread[np.minimum(nxt, len(t) - 1)] / 2, np.nan)
    good = np.isfinite(mk3) & np.isfinite(hs) & (np.abs(mk3) < 1000)
    s = PB.score(table, model)
    prog = np.isfinite(s) & (s >= model["program_threshold_q80"])
    inb = member >= 0
    pvol = float(vol[inb][prog[member[inb]]].sum())
    row.update(n_bursts=int(len(end)), mk3_mean_bps=float(mk3[good].mean()) if good.any() else np.nan, mk3_n=int(good.sum()),
               half_spread_end_bps=float(hs[good].mean()) if good.any() else np.nan,
               program_volume_share=pvol / float(vol[sg].sum()) if sg.any() and vol[sg].sum() > 0 else np.nan)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = PB.load_model(args.model)
    rows = []
    for path in sorted(glob.glob(str(Path(args.packets) / "*.npz"))):
        with np.load(path) as z:
            if "time" not in z.files or len(z["time"]) == 0:
                continue
            day = {k: z[k] for k in z.files}
        r = day_stats(day, model)
        r.update(ticker=args.ticker, date=Path(path).stem)
        rows.append(r)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows))


if __name__ == "__main__":
    main()
