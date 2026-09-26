#!/usr/bin/env python3
"""Sign-blind packet-cluster validation with geometry-exact Hurst placebos.

Version 2 formed most candidate blocks from same-side runs.  Comparing the signs of those
blocks with rotated signs is circular because sign helped choose the real boundaries.  This
version predeclares three timing-only definitions.  A boundary is determined exclusively by
the empirical quantile of inter-packet gaps; changing every packet sign leaves every block
unchanged.

For each fixed geometry, signs are circularly rotated within 30-minute bins.  The statistic
is H(placebo) - H(real).  It can show that timing clusters compress sign persistence beyond a
geometry-matched null, but it cannot identify institutional parents in anonymous data.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import execution_packets as EP
from burst_quality_v2 import block_signs, emit, hurst


# Predeclared before looking at v3 outcomes.  Adaptive gap quantiles give the same intended
# cluster density in liquid and illiquid names without using sign, price, or future outcomes.
GAP_QUANTILES = (0.30, 0.50, 0.70)
MIN_PACKETS = 3


def timing_blocks(times, gap_quantile, min_packets=MIN_PACKETS):
    """Return contiguous blocks separated only by unusually long inter-packet gaps."""
    times = np.asarray(times, float)
    if len(times) < min_packets:
        return [], np.nan
    gaps = np.diff(times)
    finite = gaps[np.isfinite(gaps) & (gaps >= 0.0)]
    if not len(finite):
        return [], np.nan
    cutoff = float(np.quantile(finite, float(gap_quantile)))
    # ``<=`` is intentional: simultaneous economic packets may remain after conservative
    # signing, and timestamp ties are timing information rather than sign information.
    changes = np.flatnonzero(gaps > cutoff) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, len(times) - 1]
    blocks = [(int(a), int(b)) for a, b in zip(starts, ends)
              if int(b) - int(a) + 1 >= int(min_packets)]
    return blocks, cutoff


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
        if len(packets) < 3000:
            return
        times = packets["time"].to_numpy(float)
        signs = packets["sign"].to_numpy(float)
        h_packet = hurst(signs)
        rng = np.random.default_rng(EP.deterministic_seed(args.ticker, date, "bq3"))
        for quantile in GAP_QUANTILES:
            blocks, _cutoff = timing_blocks(times, quantile, MIN_PACKETS)
            emit("time_q%02d_m%d" % (round(100 * quantile), MIN_PACKETS), blocks,
                 times, signs, h_packet, rng, args.ticker, date, args.draws)
    except Exception as error:
        print("%s,%d,ERR,%s" % (args.ticker, date, error), file=sys.stderr)


if __name__ == "__main__":
    main()
