#!/usr/bin/env python3
"""Cross-name spread scaling for economic hidden-execution packets."""
import argparse
import glob
import json

import numpy as np
import pandas as pd


CATEGORIES = ("known", "outside", "away_quote", "all_conventional")


def load(pattern):
    frames = []
    for path in sorted(glob.glob(pattern)):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("no packet-spread outputs")
    return pd.concat(frames, ignore_index=True)


def scaling(frame, category):
    ncol = "n_" + category; mk = "mk3_" + category; hs = "halfspread_" + category
    use = frame[(frame[ncol] > 0) & np.isfinite(frame[mk]) & np.isfinite(frame[hs])]
    by_name = use.groupby("ticker").agg(
        n_days=("date", "nunique"), markout=(mk, "mean"), halfspread=(hs, "mean")
    )
    by_name = by_name[by_name.n_days >= 20]
    if len(by_name) < 5:
        return {"n_names": int(len(by_name))}
    x = by_name.halfspread.to_numpy(float); y = by_name.markout.to_numpy(float)
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ratio = y / x
    try:
        quintile = pd.qcut(x, 5, labels=False, duplicates="drop")
        ratios = [float(np.mean(ratio[quintile == q])) for q in sorted(set(quintile))]
    except ValueError:
        ratios = []
    return {
        "n_names": int(len(by_name)), "intercept": float(beta[0]),
        "slope": float(beta[1]), "correlation": float(np.corrcoef(x, y)[0, 1]),
        "mean_markout": float(y.mean()), "mean_halfspread": float(x.mean()),
        "mean_ratio": float(ratio.mean()),
        "n_markout_gt_2x_halfspread": int((y > 2.0 * x).sum()),
        "ratio_by_spread_quintile": ratios,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = load(args.input)
    periods = {
        "replication_2023_2024": data[data.date.astype(int) // 10000 <= 2024],
        "confirmation_2025": data[data.date.astype(int) // 10000 == 2025],
    }
    result = {"experiment": "hidden-packet-spread-scaling-v1", "periods": {}}
    for period, frame in periods.items():
        result["periods"][period] = {
            category: scaling(frame, category) for category in CATEGORIES
        }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
