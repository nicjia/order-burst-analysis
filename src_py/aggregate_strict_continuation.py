#!/usr/bin/env python3
"""Day-unit Newey-West aggregation for the frozen 2025 continuation test."""
import argparse
import glob
import json

import numpy as np
import pandas as pd

import strict_continuation_common as SC
import two_avenue_evaluate as EV


def daily_stat(frame, column, eligible="n_fragments"):
    use = frame[frame[eligible] > 0]
    daily = use.groupby("date")[column].mean().sort_index()
    return EV.nw_mean(daily.to_numpy(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    frames = []
    for path in sorted(glob.glob(args.input)):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("no nonempty files match %s" % args.input)
    data = pd.concat(frames, ignore_index=True)
    result = {
        "experiment": "strict-continuation-v1",
        "n_name_days": int(len(data)), "n_names": int(data.ticker.nunique()),
        "date_min": int(data.date.min()), "date_max": int(data.date.max()),
        "cohorts": {},
    }
    for label, frame in (("seen_names", data[data.name_holdout == 0]),
                         ("heldout_names", data[data.name_holdout == 1])):
        values = {
            "n_name_days": int(len(frame)), "n_names": int(frame.ticker.nunique()),
            "selection_rate": daily_stat(frame, "selection_rate"), "targets": {},
        }
        for target in sorted(SC.TARGETS):
            base_mse = daily_stat(frame, target + "_base_mse")
            aug_mse = daily_stat(frame, target + "_aug_mse")
            delta = daily_stat(frame, target + "_delta_mse")
            values["targets"][target] = {
                "base_mse": base_mse, "aug_mse": aug_mse, "delta_mse": delta,
                "incremental_mse_fraction": (
                    delta["mean"] / base_mse["mean"] if base_mse["mean"] else np.nan
                ),
                "selected_target": daily_stat(frame, target + "_selected_target",
                                              "n_selected"),
                "selected_base_residual": daily_stat(
                    frame, target + "_selected_base_residual", "n_selected"
                ),
                "selected_aug_residual": daily_stat(
                    frame, target + "_selected_aug_residual", "n_selected"
                ),
            }
        result["cohorts"][label] = values
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
