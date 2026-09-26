#!/usr/bin/env python3
"""Fit and freeze the four predeclared 2023 liquidity-pause hazard models."""
import argparse
import glob
import json

import numpy as np
import pandas as pd

import liquidity_pause_common as LP
import metaorder_models as MM
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
        raise FileNotFoundError("no nonempty liquidity-pause training files")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = load(args.input)
    data = data[(data.date.astype(int) // 10000 == 2023)].copy()
    data = data[~data.ticker.map(EV.stable_holdout)].copy()
    y = data.event.to_numpy(float)
    if not np.isfinite(y).all() or not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("invalid hazard labels")
    result = {
        "experiment": "liquidity-pause-v1", "ridge": 10.0,
        "n_risk_rows": int(len(data)), "n_events": int(y.sum()),
        "n_names": int(data.ticker.nunique()), "models": {},
    }
    for label, features in LP.MODEL_FEATURES.items():
        model = MM.RidgeLogit(ridge=10.0, max_iter=60).fit(LP.matrix(data, features), y)
        result["models"][label] = model.to_dict(features)
    joint = result["models"]["joint"]
    result["training_mechanism_signs"] = {
        "selected_x_spread_change": float(
            joint["coef"][1 + joint["features"].index("selected_x_spread_change")]
        ),
        "selected_x_contra_depth_change": float(
            joint["coef"][1 + joint["features"].index("selected_x_contra_depth_change")]
        ),
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in (
        "experiment", "n_risk_rows", "n_events", "n_names",
        "training_mechanism_signs")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
