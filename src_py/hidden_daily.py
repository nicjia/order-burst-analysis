#!/usr/bin/env python3
"""
hidden_daily.py — lightweight hidden-execution burst extractor (no book reconstruction).

For daily hidden-flow COI we only need the type-5 (hidden execution) stream: cluster
same-aggressor-side hidden trades into bursts and record per-burst date/direction/volume.
Aggressor sign = -Direction (LOBSTER Direction = side of the executed resting order).

Output: bursts_<TICKER>_hidden.csv with columns Ticker,Date,Direction,Volume,TradeCount
(one row per hidden burst), directly consumable by a COI aggregation.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd


def day_bursts(msg_path, gap=1.0, min_ct=3):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(msg_path))
    date = int(m.group(1)+m.group(2)+m.group(3)) if m else 0
    df = pd.read_csv(msg_path, header=None, usecols=[0, 1, 3, 5], names=["t", "ty", "sz", "dr"])
    h = df[df["ty"] == 5]
    if len(h) < min_ct:
        return []
    t = h["t"].to_numpy(float); s = (-h["dr"]).to_numpy(np.int8); sz = h["sz"].to_numpy(np.int64)
    rows = []; i = 0; n = len(t)
    while i < n:
        j = i
        while j+1 < n and s[j+1] == s[i] and (t[j+1]-t[j]) < gap:
            j += 1
        if j-i+1 >= min_ct:
            rows.append((date, int(s[i]), int(sz[i:j+1].sum()), j-i+1))
        i = j+1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-folder", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for msg in sorted(glob.glob(os.path.join(args.stock_folder, "*message*.csv"))):
        rows.extend(day_bursts(msg))
    df = pd.DataFrame(rows, columns=["Date", "Direction", "Volume", "TradeCount"])
    df.insert(0, "Ticker", args.ticker)
    df.to_csv(args.out, index=False)
    print(f"WROTE {args.out}: {len(df)} hidden bursts over {df['Date'].nunique() if len(df) else 0} days")


if __name__ == "__main__":
    main()
