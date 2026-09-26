#!/usr/bin/env python3
"""Daily-unit Newey-West aggregation for corrected packet-level Hurst placebos."""
import argparse
import glob

import numpy as np
import pandas as pd


def nw(x, lags=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 20:
        return np.nan, np.nan, n
    mean = x.mean(); residual = x - mean
    variance = residual.dot(residual) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * residual[lag:].dot(residual[:-lag]) / n
    se = np.sqrt(max(variance, 0.0) / n)
    return mean, mean / se if se else np.nan, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True); args = ap.parse_args()
    frames = []
    for path in sorted(glob.glob(args.input)):
        try:
            frame = pd.read_csv(path)
            if "dH" in frame:
                frames.append(frame)
        except Exception:
            pass
    if not frames:
        raise FileNotFoundError("no valid bq2 outputs match %s" % args.input)
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for definition, group in data.groupby("defn"):
        daily = group.groupby("date").mean(numeric_only=True).sort_index()
        mean, t_stat, n_days = nw(daily["dH"])
        rows.append({
            "defn": definition, "n_name_days": int(group.dH.notna().sum()),
            "n_days": int(n_days), "mean_h_packet": float(group.h_packet.mean()),
            "mean_h_real": float(group.h_real.mean()),
            "mean_h_placebo": float(group.h_placebo.mean()),
            "mean_dH": float(mean), "nw_t_dH": float(t_stat),
            "share_randomization_p_le_005": float((group.p_value <= 0.05).mean()),
            "mean_bursts": float(group.n_bursts.mean()),
        })
    out = pd.DataFrame(rows).sort_values("mean_dH", ascending=False)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

