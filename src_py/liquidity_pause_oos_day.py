#!/usr/bin/env python3
"""Apply frozen liquidity-pause hazards to one untouched 2025 ticker-day."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import liquidity_pause_common as LP
import liquidity_pause_extract as EX
import metaorder_models as MM
import two_avenue_evaluate as EV


def _logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def summarize(frame, ticker, date, frozen):
    y = frame.event.to_numpy(float)
    losses = {}
    for label, spec in frozen["models"].items():
        model = MM.RidgeLogit.from_dict(spec)
        losses[label] = _logloss(y, model.predict_proba(LP.matrix(frame, spec["features"])))
    selected = frame.selected.to_numpy(float) > 0
    result = {
        "ticker": ticker, "date": int(date),
        "name_holdout": int(EV.stable_holdout(ticker)),
        "n_risk": int(len(frame)), "n_events": int(y.sum()),
        "n_selected_risk": int(selected.sum()),
        "n_selected_events": int(y[selected].sum()),
        "base_logloss": float(losses["base"].mean()),
    }
    for label in ("spread", "depth", "joint"):
        result[label + "_logloss"] = float(losses[label].mean())
        result[label + "_delta_logloss"] = float(
            (losses["base"] - losses[label]).mean()
        )
        result[label + "_selected_delta_logloss"] = (
            float((losses["base"][selected] - losses[label][selected]).mean())
            if selected.any() else np.nan
        )
    return pd.DataFrame([result])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--simulation-model", required=True)
    ap.add_argument("--strict-frozen", required=True)
    ap.add_argument("--pause-frozen", required=True)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()
    frame = EX.extract(args.msg, args.ticker, args.simulation_model, args.strict_frozen)
    if frame.empty:
        return
    with open(args.pause_frozen) as handle:
        frozen = json.load(handle)
    date = int(frame.date.iloc[0])
    summarize(frame, args.ticker, date, frozen).to_csv(
        sys.stdout, index=False, header=args.header
    )


if __name__ == "__main__":
    main()
