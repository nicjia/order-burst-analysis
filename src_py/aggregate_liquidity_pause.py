#!/usr/bin/env python3
"""Name-day/Newey-West(10) aggregation for liquidity-pause-v1."""
import argparse
import glob
import json

import pandas as pd

import two_avenue_evaluate as EV


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
        raise FileNotFoundError("no nonempty liquidity-pause OOS files")
    return pd.concat(frames, ignore_index=True)


def stat(frame, column, eligible="n_risk"):
    daily = frame[frame[eligible] > 0].groupby("date")[column].mean().sort_index()
    return EV.nw_mean(daily.to_numpy(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = load(args.input)
    with open(args.frozen) as handle:
        frozen = json.load(handle)
    result = {
        "experiment": "liquidity-pause-v1", "n_names": int(data.ticker.nunique()),
        "n_name_days": int(len(data)), "training_mechanism_signs":
            frozen["training_mechanism_signs"], "cohorts": {},
    }
    for label, frame in (("seen_names", data[data.name_holdout == 0]),
                         ("heldout_names", data[data.name_holdout == 1])):
        values = {"n_names": int(frame.ticker.nunique()),
                  "n_name_days": int(len(frame)), "models": {}}
        for model in ("spread", "depth", "joint"):
            values["models"][model] = {
                "delta_logloss": stat(frame, model + "_delta_logloss"),
                "selected_delta_logloss": stat(
                    frame, model + "_selected_delta_logloss", "n_selected_risk"
                ),
            }
        result["cohorts"][label] = values
    signs = result["training_mechanism_signs"]
    conditions = [signs["selected_x_spread_change"] < 0,
                  signs["selected_x_contra_depth_change"] > 0]
    for cohort in result["cohorts"].values():
        joint = cohort["models"]["joint"]["delta_logloss"]
        conditions.append(joint["mean"] > 0 and joint["t"] > 2)
        conditions.append(cohort["models"]["spread"]["delta_logloss"]["mean"] > 0)
        conditions.append(cohort["models"]["depth"]["delta_logloss"]["mean"] > 0)
    result["frozen_gate_conditions"] = conditions
    result["frozen_gate_pass"] = bool(all(conditions))
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
