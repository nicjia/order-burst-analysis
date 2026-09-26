#!/usr/bin/env python3
"""Apply frozen nested models to one untouched 2025 ticker-day."""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import strict_continuation_common as SC
import strict_continuation_extract as EX
import two_avenue_evaluate as EV


def _mean(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def summarize(frame, ticker, date, frozen):
    threshold = float(frozen["score_top_decile"])
    selected = frame.fragment_score.to_numpy(float) >= threshold
    result = {
        "ticker": ticker, "date": int(date),
        "name_holdout": int(EV.stable_holdout(ticker)),
        "n_fragments": int(len(frame)), "n_selected": int(selected.sum()),
        "selection_rate": float(selected.mean()),
    }
    for target, spec in sorted(frozen["targets"].items()):
        y = SC.target_values(frame, target)
        base = EV.Ridge.from_dict(spec["base"])
        aug = EV.Ridge.from_dict(spec["augmented"])
        base_pred = base.predict(EV._matrix(frame, spec["base"]["features"]))
        aug_pred = aug.predict(EV._matrix(frame, spec["augmented"]["features"]))
        ok = np.isfinite(y) & np.isfinite(base_pred) & np.isfinite(aug_pred)
        result[target + "_base_mse"] = _mean((y[ok] - base_pred[ok]) ** 2)
        result[target + "_aug_mse"] = _mean((y[ok] - aug_pred[ok]) ** 2)
        result[target + "_delta_mse"] = _mean(
            (y[ok] - base_pred[ok]) ** 2 - (y[ok] - aug_pred[ok]) ** 2
        )
        high = ok & selected
        result[target + "_selected_target"] = _mean(y[high])
        result[target + "_selected_base_residual"] = _mean(y[high] - base_pred[high])
        result[target + "_selected_aug_residual"] = _mean(y[high] - aug_pred[high])
    return pd.DataFrame([result])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--simulation-model", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()
    with open(args.frozen) as handle:
        frozen = json.load(handle)
    frame = EX.extract(args.msg, args.ticker, args.simulation_model)
    if frame.empty:
        return
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(args.msg))
    date = int("".join(match.groups())) if match else 0
    summarize(frame, args.ticker, date, frozen).to_csv(
        sys.stdout, index=False, header=args.header
    )


if __name__ == "__main__":
    main()
