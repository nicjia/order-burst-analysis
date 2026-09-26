#!/usr/bin/env python3
"""Fit and freeze nested continuation models from 2023 non-holdout names only."""
import argparse
import glob
import json

import numpy as np
import pandas as pd

import strict_continuation_common as SC
import two_avenue_evaluate as EV


def load_training(pattern):
    frames = []
    for path in sorted(glob.glob(pattern)):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("no usable training files match %s" % pattern)
    data = pd.concat(frames, ignore_index=True)
    data = data[(data.date >= 20230101) & (data.date <= 20231231)].copy()
    data["name_holdout"] = data.ticker.map(EV.stable_holdout)
    return data[~data.name_holdout].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train = load_training(args.input)
    result = {
        "experiment": "strict-continuation-v1",
        "training_window": [20230101, 20231231],
        "final_test_window": [20250101, 20251231],
        "exploratory_year_excluded_from_final_test": 2024,
        "name_holdout_rule": "sha256(ticker) first32 mod 5 == 0",
        "n_fragments": int(len(train)), "n_names": int(train.ticker.nunique()),
        "score_top_decile": float(np.nanquantile(train.fragment_score, 0.90)),
        "control_features": SC.CONTROL_FEATURES,
        "augmented_features": SC.AUGMENTED_FEATURES,
        "targets": {},
    }
    x_base = EV._matrix(train, SC.CONTROL_FEATURES)
    x_aug = EV._matrix(train, SC.AUGMENTED_FEATURES)
    for target in sorted(SC.TARGETS):
        y = SC.target_values(train, target)
        base = EV.Ridge(10.0).fit(x_base, y)
        augmented = EV.Ridge(10.0).fit(x_aug, y)
        result["targets"][target] = {
            "source": SC.TARGETS[target][0], "transform": SC.TARGETS[target][1],
            "base": base.to_dict(SC.CONTROL_FEATURES),
            "augmented": augmented.to_dict(SC.AUGMENTED_FEATURES),
        }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "targets"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
