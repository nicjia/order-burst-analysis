#!/usr/bin/env python3
"""Schema and conditional-finiteness gate for hidden-packet-bound rows."""
import argparse

import numpy as np
import pandas as pd

import hidden_packet_bounds as HP


BASE = {
    "ticker", "date", "n_hidden_packets", "n_hidden_messages", "hidden_volume",
    "n_mixed_packets", "n_outside_packets", "n_unsigned_packets",
    "n_unsigned_away_packets", "n_midpoint_packets",
    "n_unresolved_convention_packets", "mixed_hidden_volume",
    "outside_hidden_volume", "unsigned_hidden_volume", "midpoint_hidden_volume",
}


def validate(path, expected_year=None):
    frame = pd.read_csv(path)
    required = set(BASE)
    pairs = []
    for seconds in HP.HORIZONS:
        horizon = "%dm" % (seconds // 60)
        for name in ("mixed", "outside", "known", "away_quote", "mid_tick",
                     "all_conventional"):
            count = "n_%s_%s" % (name, horizon)
            value = "mk_%s_%s" % (name, horizon)
            required.update({count, value}); pairs.append((count, value))
        for population in ("all", "away"):
            count = "n_bound_%s_%s" % (population, horizon)
            lo = "bound_%s_lo_%s" % (population, horizon)
            hi = "bound_%s_hi_%s" % (population, horizon)
            required.update({count, lo, hi}); pairs.extend([(count, lo), (count, hi)])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing hidden-packet-bound columns: %s" % missing)
    core = sorted(BASE - {"ticker"})
    if not np.isfinite(frame[core].to_numpy(float)).all():
        raise ValueError("non-finite hidden-packet coverage values")
    for count, value in pairs:
        eligible = frame[count] > 0
        if not np.isfinite(frame.loc[eligible, value].to_numpy(float)).all():
            raise ValueError("non-finite eligible values for %s" % value)
    for seconds in HP.HORIZONS:
        horizon = "%dm" % (seconds // 60)
        for population in ("all", "away"):
            lo = frame["bound_%s_lo_%s" % (population, horizon)]
            hi = frame["bound_%s_hi_%s" % (population, horizon)]
            eligible = frame["n_bound_%s_%s" % (population, horizon)] > 0
            if not (lo[eligible] <= hi[eligible]).all():
                raise ValueError("reversed hidden-packet bounds")
    if expected_year is not None and not (
            frame.date.astype(int) // 10000 == int(expected_year)).all():
        raise ValueError("hidden-packet-bound year mismatch")
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
