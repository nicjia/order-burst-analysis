#!/usr/bin/env python3
"""Fit and freeze Avenue-1 and flow-baseline models from 2023 sampled fragments only."""
import argparse
import glob
import json

import numpy as np
import pandas as pd

import metaorder_models as MM
import two_avenue_evaluate as EV


def load_training(pattern):
    frames = []
    for path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("no usable training files match %s" % pattern)
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_numeric(data["date"], errors="coerce")
    data = data[(data.date >= 20230101) & (data.date <= 20231231)].copy()
    data["name_holdout"] = data["ticker"].map(EV.stable_holdout)
    return data[~data.name_holdout].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train = load_training(args.input)
    result = {
        "training_window": [20230101, 20231231],
        "name_holdout_rule": "sha256(ticker) first32 mod 5 == 0",
        "n_fragments": int(len(train)), "n_names": int(train.ticker.nunique()),
        "models": {},
    }
    for label, features in [("price_free", EV.BASE_FEATURES),
                            ("post_end", EV.POST_END_FEATURES)]:
        X = EV._matrix(train, features)
        target = train["permanent_proxy"].to_numpy(float)
        ridge = EV.Ridge(10.0).fit(X, target)
        prediction = ridge.predict(X)
        persistent = train["persistent_positive"].to_numpy(float)
        classifier = MM.RidgeLogit(ridge=10.0).fit(X, persistent)
        probability = classifier.predict_proba(X)
        result["models"][label] = {
            "ridge": ridge.to_dict(features),
            "ridge_top_decile": float(np.nanquantile(prediction, 0.90)),
            "classifier": classifier.to_dict(features),
            "classifier_top_decile": float(np.nanquantile(probability, 0.90)),
        }
    result["flow_baselines"] = {}
    for horizon in ("60s", "300s"):
        flow = EV.Ridge(10.0).fit(
            EV._matrix(train, EV.FLOW_BASELINE),
            train["future_count_imbalance_%s" % horizon].to_numpy(float),
        )
        result["flow_baselines"][horizon] = flow.to_dict(EV.FLOW_BASELINE)
    # Backward-compatible alias for readers of the first frozen schema.
    result["flow_baseline"] = result["flow_baselines"]["300s"]
    result["fragment_score_top_decile"] = float(np.nanquantile(train.fragment_score, 0.90))
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
