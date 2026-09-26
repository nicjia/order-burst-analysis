#!/usr/bin/env python3
"""Packet-level burst validation with geometry-exact, multi-draw Hurst placebos.

For each real definition, placebo draws keep the exact burst boundaries in packet index
space.  Signs are circularly rotated within 30-minute bins, so every draw matches time of
day, coverage, packet counts, durations, and inter-burst gaps while breaking the alignment
between the detector's boundaries and signed order flow.  Rotations preserve the sign
sequence's own within-bin memory, making this a stricter null than shuffling signs.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import execution_packets as EP


def hurst(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 400 or np.std(x) < 1e-12:
        return np.nan
    block_sizes = [m for m in (2, 4, 8, 16, 32, 64, 128) if n // m >= 20]
    if len(block_sizes) < 4:
        return np.nan
    variances = []
    for size in block_sizes:
        count = n // size
        variances.append(np.var(x[:count * size].reshape(count, size).mean(axis=1)))
    variances = np.asarray(variances)
    ok = variances > 0
    if ok.sum() < 4:
        return np.nan
    slope = np.polyfit(np.log(np.asarray(block_sizes)[ok]), np.log(variances[ok]), 1)[0]
    return float(slope / 2.0 + 1.0)


def runs(times, signs, gap, minrun):
    blocks = []; i = 0
    while i < len(times):
        j = i
        while (j + 1 < len(times) and signs[j + 1] == signs[i] and
               times[j + 1] - times[j] < gap):
            j += 1
        if j - i + 1 >= minrun:
            blocks.append((i, j))
        i = j + 1
    return blocks


def counter(times, signs, beta, threshold, minrun=3):
    blocks = []; i = 0
    while i < len(times):
        j = i; intensity = 1.0
        while j + 1 < len(times):
            gap = times[j + 1] - times[j]
            if intensity * np.exp(-beta * gap) < threshold or signs[j + 1] != signs[i]:
                break
            intensity = intensity * np.exp(-beta * gap) + 1.0
            j += 1
        if j - i + 1 >= minrun:
            blocks.append((i, j))
        i = j + 1
    return blocks


def block_signs(blocks, signs):
    return np.asarray([np.sign(signs[a:b + 1].sum()) for a, b in blocks], float)


def rotate_within_tod(times, signs, rng, bin_seconds=1800.0):
    rotated = np.asarray(signs, float).copy()
    bins = ((np.asarray(times) - EP.RTH0) // bin_seconds).astype(int)
    for value in np.unique(bins):
        idx = np.flatnonzero(bins == value)
        if len(idx) < 2:
            continue
        shift = int(rng.integers(1, len(idx)))
        rotated[idx] = np.roll(rotated[idx], shift)
    return rotated


def emit(name, blocks, times, signs, h_packet, rng, ticker, date, draws):
    if len(blocks) < 400:
        return
    sizes = np.asarray([b - a + 1 for a, b in blocks], float)
    h_real = hurst(block_signs(blocks, signs))
    placebo = []
    for _ in range(draws):
        rotated = rotate_within_tod(times, signs, rng)
        placebo.append(hurst(block_signs(blocks, rotated)))
    placebo = np.asarray(placebo, float)
    placebo = placebo[np.isfinite(placebo)]
    h_mean = float(placebo.mean()) if len(placebo) else np.nan
    h_sd = float(placebo.std(ddof=1)) if len(placebo) > 1 else np.nan
    d_h = h_mean - h_real if np.isfinite(h_mean) and np.isfinite(h_real) else np.nan
    p_value = ((1.0 + np.sum(placebo <= h_real)) / (len(placebo) + 1.0)
               if len(placebo) and np.isfinite(h_real) else np.nan)
    duration = np.mean([times[b] - times[a] for a, b in blocks])
    values = [h_packet, h_real, h_mean, h_sd, d_h, p_value]
    formatted = [("%.6f" % x) if np.isfinite(x) else "nan" for x in values]
    print("%s,%d,%s,%d,%.3f,%.6f,%s" % (
        ticker, date, name, len(blocks), sizes.mean(), duration, ",".join(formatted)
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--draws", type=int, default=50)
    args = ap.parse_args()
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(args.msg))
    date = int("".join(match.groups())) if match else 0
    try:
        _context, packets = EP.reconstruct_packets(args.msg)
        packets = EP.signed_packets(packets)
        if len(packets) < 3000:
            return
        times = packets["time"].to_numpy(float)
        signs = packets["sign"].to_numpy(float)
        sizes = packets["volume"].to_numpy(float)
        h_packet = hurst(signs)
        rng = np.random.default_rng(EP.deterministic_seed(args.ticker, date, "bq2"))
        for gap in (0.25, 0.5, 1.0, 2.0, 5.0):
            emit("sil_%g" % gap, runs(times, signs, gap, 3), times, signs,
                 h_packet, rng, args.ticker, date, args.draws)
        for beta in (0.5, 1.0, 2.0, 5.0):
            for threshold in (0.3, 0.5, 0.8):
                emit("hwk_%g_%g" % (beta, threshold),
                     counter(times, signs, beta, threshold), times, signs,
                     h_packet, rng, args.ticker, date, args.draws)
        for minimum in (2, 3, 5, 10):
            emit("run_%d" % minimum, runs(times, signs, 1.0, minimum), times, signs,
                 h_packet, rng, args.ticker, date, args.draws)
        cumulative = np.cumsum(sizes); total = cumulative[-1]
        for fraction in (0.0005, 0.001, 0.002, 0.005):
            bucket = (cumulative / (total * fraction)).astype(int)
            changes = np.flatnonzero(np.diff(bucket)) + 1
            blocks = list(zip(np.r_[0, changes], np.r_[changes - 1, len(times) - 1]))
            blocks = [(a, b) for a, b in blocks if b >= a + 2]
            emit("vol_%g" % fraction, blocks, times, signs,
                 h_packet, rng, args.ticker, date, args.draws)
    except Exception as error:
        print("%s,%d,ERR,%s" % (args.ticker, date, error), file=sys.stderr)


if __name__ == "__main__":
    main()

