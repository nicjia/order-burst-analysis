#!/usr/bin/env python3
"""Aggregate compact 2024 name-day summaries with the project's inference convention."""
import argparse
import glob
import json

import numpy as np
import pandas as pd


def nw(values, lags=10):
    x = np.asarray(values, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 20:
        return {"mean": np.nan, "t": np.nan, "n_days": int(n)}
    mean = x.mean(); residual = x - mean
    variance = residual.dot(residual) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * residual[lag:].dot(residual[:-lag]) / n
    se = np.sqrt(max(variance, 0.0) / n)
    return {"mean": float(mean), "t": float(mean / se) if se else np.nan,
            "n_days": int(n)}


def daily_stat(frame, column, eligibility=None):
    use = frame
    if eligibility is not None:
        use = use[use[eligibility] > 0]
    daily = use.groupby("date")[column].mean().sort_index()
    return nw(daily.to_numpy(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = sorted(glob.glob(args.input))
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("no nonempty OOS files match %s" % args.input)
    data = pd.concat(frames, ignore_index=True)
    result = {
        "n_name_days": int(len(data)), "n_names": int(data.ticker.nunique()),
        "date_min": int(data.date.min()), "date_max": int(data.date.max()), "cohorts": {},
    }
    for cohort, frame in [("seen_names", data[data.name_holdout == 0]),
                          ("heldout_names", data[data.name_holdout == 1])]:
        values = {"n_name_days": int(len(frame)), "n_names": int(frame.ticker.nunique())}
        for prefix in ("price_free_reg", "price_free_cls", "post_end_reg", "post_end_cls"):
            values[prefix] = {
                "permanent": daily_stat(frame, prefix + "_permanent", prefix + "_n"),
                "persistent_rate": daily_stat(frame, prefix + "_persistent_rate", prefix + "_n"),
            }
            for minutes in (5, 15, 30):
                values[prefix]["pnl_%dm" % minutes] = daily_stat(
                    frame, prefix + "_pnl_%dm" % minutes, prefix + "_trades_%dm" % minutes
                )
        values["parent_reconstruction"] = {
            "campaign_join_rate": daily_stat(frame, "campaign_join_rate"),
        }
        for horizon in ("60s", "300s"):
            residual = "sim_high_flow_residual_%s" % horizon
            raw = "sim_high_future_flow_%s" % horizon
            if residual in frame:
                values["parent_reconstruction"]["future_flow_residual_%s" % horizon] = (
                    daily_stat(frame, residual, "sim_high_n")
                )
                values["parent_reconstruction"]["future_flow_raw_%s" % horizon] = (
                    daily_stat(frame, raw, "sim_high_n")
                )
        # Preserve the first-run key names when aggregating its older schema.
        if "sim_high_flow_residual_300s" not in frame:
            values["parent_reconstruction"]["future_flow_residual"] = daily_stat(
                frame, "sim_high_flow_residual", "sim_high_n"
            )
            values["parent_reconstruction"]["future_flow_raw"] = daily_stat(
                frame, "sim_high_future_flow", "sim_high_n"
            )
        values["post_completion_reversal"] = {}
        for minutes in (5, 15, 30):
            values["post_completion_reversal"]["pnl_%dm" % minutes] = daily_stat(
                frame, "reversal_pnl_%dm" % minutes, "reversal_trades_%dm" % minutes
            )
        result["cohorts"][cohort] = values
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
