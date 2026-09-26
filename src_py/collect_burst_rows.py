#!/usr/bin/env python3
"""Concatenate per-name burst rows into one compact file of usable rows (pairs > 0) plus coverage counts."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEEP = ["ticker", "date", "n_packets", "n_opposite", "n_unsigned", "dm_pairs", "dm_repeats", "dm_expected",
        "duration", "intensity", "iat_cv", "iat_median", "truncated_share", "hidden_share", "spread_bps",
        "log_exec_depth", "imbalance", "tod", "trailing_activity", "size_roundlot_share", "size_median"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    frames = []; coverage = {}
    for path in sorted(glob.glob(str(Path(args.dir) / "*.csv.gz"))):
        try:
            f = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        tk = Path(path).name.split(".")[0]
        coverage[tk] = dict(bursts=int(len(f)))
        if f.empty:
            continue
        u = f[(f.dm_pairs > 0) & np.isfinite(f.dm_expected)][KEEP]
        coverage[tk].update(usable=int(len(u)), packets_in_bursts=int(f.n_packets.sum()))
        frames.append(u)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False, float_format="%.6g", compression="gzip")
    Path(args.out).with_suffix(".coverage.json").write_text(json.dumps(coverage, indent=1))
    print(len(out), "usable rows from", len(coverage), "names")


if __name__ == "__main__":
    main()
