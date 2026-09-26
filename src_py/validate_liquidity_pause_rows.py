#!/usr/bin/env python3
"""Schema/finiteness gate for liquidity-pause risk-set rows."""
import argparse

import numpy as np
import pandas as pd

import liquidity_pause_common as LP


def validate(path, expected_year=None):
    frame = pd.read_csv(path)
    required = {"ticker", "date", "fragment_id", "sign", "interval_id",
                "risk_time", "event"} | set(LP.JOINT_FEATURES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing liquidity-pause columns: %s" % missing)
    numeric = frame[sorted(required - {"ticker"})].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite liquidity-pause risk values")
    if not set(frame.event.unique()).issubset({0.0, 1.0}):
        raise ValueError("non-binary liquidity-pause event")
    if not frame.interval_id.between(0, len(LP.INTERVALS) - 1).all():
        raise ValueError("invalid liquidity-pause interval")
    if expected_year is not None and not (
            frame.date.astype(int) // 10000 == int(expected_year)).all():
        raise ValueError("liquidity-pause year mismatch")
    keys = ["ticker", "date", "fragment_id"]
    if (frame.groupby(keys).event.sum() > 1).any():
        raise ValueError("multiple events in one risk episode")
    if frame.duplicated(keys + ["interval_id"]).any():
        raise ValueError("duplicate liquidity-pause risk interval")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--expected-year", type=int)
    args = ap.parse_args()
    frame = validate(args.input, args.expected_year)
    print("validated %d risk rows and %d columns" % frame.shape)


if __name__ == "__main__":
    main()
