#!/usr/bin/env python3
"""Day-unit aggregation for hidden-packet identification bounds."""
import argparse
import glob
import json

import pandas as pd

import two_avenue_evaluate as EV


HORIZONS = ("3m", "15m", "30m")


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
        raise FileNotFoundError("no hidden-packet outputs")
    return pd.concat(frames, ignore_index=True)


def stat(frame, column, eligible):
    daily = frame[frame[eligible] > 0].groupby("date")[column].mean().sort_index()
    return EV.nw_mean(daily.to_numpy(float))


def summarize_period(frame):
    values = {
        "n_names": int(frame.ticker.nunique()), "n_name_days": int(len(frame)),
        "coverage": {}, "horizons": {},
    }
    for name, numerator, denominator in (
        ("mixed_packet_share", "n_mixed_packets", "n_hidden_packets"),
        ("outside_packet_share", "n_outside_packets", "n_hidden_packets"),
        ("unsigned_packet_share", "n_unsigned_packets", "n_hidden_packets"),
        ("midpoint_packet_share", "n_midpoint_packets", "n_hidden_packets"),
        ("mixed_volume_share", "mixed_hidden_volume", "hidden_volume"),
        ("unsigned_volume_share", "unsigned_hidden_volume", "hidden_volume"),
    ):
        ratio = frame[numerator] / frame[denominator].where(frame[denominator] > 0)
        temp = frame.assign(_ratio=ratio)
        values["coverage"][name] = stat(temp, "_ratio", denominator)
    for horizon in HORIZONS:
        h = {}
        for name in ("mixed", "outside", "known", "away_quote", "mid_tick",
                     "all_conventional"):
            h[name] = stat(frame, "mk_%s_%s" % (name, horizon),
                           "n_%s_%s" % (name, horizon))
        for bound in ("all_lo", "all_hi", "away_lo", "away_hi"):
            population = "all" if bound.startswith("all") else "away"
            h["bound_" + bound] = stat(
                frame, "bound_%s_%s" % (bound, horizon),
                "n_bound_%s_%s" % (population, horizon),
            )
        values["horizons"][horizon] = h
    return values


PERIODS = ("replication_2023_2024", "confirmation_2025")


def split_periods(data):
    """Both frozen periods, or an error: a gate must never be judged on a partial panel.

    Added 2026-09-13. The original version silently dropped an empty period and evaluated the
    gates over whichever period remained. Output for a complete panel is unchanged.
    """
    year = data.date.astype(int) // 10000
    periods = {"replication_2023_2024": data[year <= 2024], "confirmation_2025": data[year == 2025]}
    empty = [label for label, frame in periods.items() if frame.empty]
    if empty:
        raise ValueError("refusing to evaluate gates; empty period(s): " + ", ".join(empty))
    return periods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = load(args.input)
    periods = split_periods(data)
    result = {"experiment": "hidden-packet-bounds-v1", "periods": {
        label: summarize_period(frame) for label, frame in periods.items()
    }}
    # Note: the frozen bifurcation gate checks signs only, not significance.
    minimal = []; bifurcation = []
    for period in result["periods"].values():
        h3 = period["horizons"]["3m"]
        minimal.append(h3["bound_all_lo"]["mean"] > 0
                       and h3["bound_all_lo"]["t"] > 2)
        bifurcation.extend([h3["away_quote"]["mean"] > 0,
                            h3["mid_tick"]["mean"] < 0])
    result["minimal_positive_identification"] = bool(minimal and all(minimal))
    result["conventional_bifurcation_replicates"] = bool(
        bifurcation and all(bifurcation)
    )
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
