#!/usr/bin/env python3
"""Schema and finiteness gate for strict-continuation compact output."""
import argparse

import numpy as np
import pandas as pd

import strict_continuation_common as SC


def validate(path, expected_year=None):
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("empty strict-continuation output")
    required = {"ticker", "date", "name_holdout", "n_fragments", "n_selected",
                "selection_rate"}
    for target in SC.TARGETS:
        required.update({
            target + "_base_mse", target + "_aug_mse", target + "_delta_mse",
            target + "_selected_target", target + "_selected_base_residual",
            target + "_selected_aug_residual",
        })
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing strict-continuation columns: %s" % missing)
    numeric = frame[sorted(required - {"ticker"})].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite strict-continuation pilot values")
    if expected_year is not None:
        years = frame.date.astype(int) // 10000
        if not (years == int(expected_year)).all():
            raise ValueError("output year does not match frozen test year")
    if not ((frame.selection_rate >= 0.0) & (frame.selection_rate <= 1.0)).all():
        raise ValueError("invalid selection rate")
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
