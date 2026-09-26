#!/usr/bin/env python3
"""Schema/finiteness gate for compact liquidity-pause OOS output."""
import argparse

import numpy as np
import pandas as pd


def validate(path, expected_year=None):
    frame = pd.read_csv(path)
    required = {"ticker", "date", "name_holdout", "n_risk", "n_events",
                "n_selected_risk", "n_selected_events", "base_logloss"}
    for model in ("spread", "depth", "joint"):
        required.update({model + "_logloss", model + "_delta_logloss",
                         model + "_selected_delta_logloss"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing compact liquidity-pause columns: %s" % missing)
    numeric = frame[sorted(required - {"ticker"})].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite compact liquidity-pause values")
    if expected_year is not None and not (
            frame.date.astype(int) // 10000 == int(expected_year)).all():
        raise ValueError("compact liquidity-pause year mismatch")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--expected-year", type=int)
    args = ap.parse_args()
    frame = validate(args.input, args.expected_year)
    print("validated %d rows and %d columns" % frame.shape)


if __name__ == "__main__":
    main()
