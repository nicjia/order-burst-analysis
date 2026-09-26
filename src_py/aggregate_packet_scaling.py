#!/usr/bin/env python3
"""Cross-name aggregation of the packet-level spread-scaling law.

Reports, for each packet definition and each measurement base, the regression of the
name's mean markout on its mean half-spread, the cross-name correlation, the ratio
distribution by spread quintile, and the count of names whose markout exceeds a round
trip.  The last is the only number with a trading interpretation.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

GROUP = sys.argv[1] if len(sys.argv) > 1 else "results/packet_scaling"
VARIANTS = ("all", "blk5", "blk10", "run3", "clust3")
BASES = ("mk3_t0", "mk3_t1", "mk30_t1")
MIN_DAYS = 40
MIN_N = 3


def load(group):
    frames = []
    for path in sorted(glob.glob(os.path.join(group, "out", "*.csv"))):
        if os.path.getsize(path) == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise SystemExit("no usable rows under %s/out" % group)
    return pd.concat(frames, ignore_index=True)


def main():
    panel = load(GROUP)
    panel = panel[np.isfinite(panel.halfsp_day) & (panel.halfsp_day > 0)]
    report = {
        "group": GROUP,
        "name_days": int(len(panel)),
        "names": int(panel.ticker.nunique()),
        "dates": int(panel.date.nunique()),
        "date_min": int(panel.date.min()),
        "date_max": int(panel.date.max()),
        "message_to_packet_ratio": float(
            panel.n_messages_in_signed.sum() / max(panel.n_signed_packets.sum(), 1)),
        "definitions": {},
    }
    for variant in VARIANTS:
        ncol = "%s_n" % variant
        if ncol not in panel.columns:
            continue
        block = {}
        for base in BASES:
            col = "%s_%s" % (variant, base)
            if col not in panel.columns:
                continue
            use = panel[np.isfinite(panel[col]) & (panel[ncol] >= MIN_N)]
            if use.empty:
                continue
            byname = use.groupby("ticker").agg(
                mk=(col, "mean"), hs=("halfsp_day", "mean"), days=(col, "size"))
            byname = byname[byname.days >= MIN_DAYS]
            if len(byname) < 10:
                continue
            mk = byname.mk.to_numpy(float)
            hs = byname.hs.to_numpy(float)
            slope, intercept = np.polyfit(hs, mk, 1)
            ratio = mk / hs
            quintile = pd.qcut(hs, 5, labels=False, duplicates="drop")
            block[base] = {
                "names": int(len(byname)),
                "panel_mean_markout_bps": float(np.mean(mk)),
                "mean_half_spread_bps": float(np.mean(hs)),
                "slope_on_half_spread": float(slope),
                "intercept_bps": float(intercept),
                "cross_name_correlation": float(np.corrcoef(hs, mk)[0, 1]),
                "ratio_median": float(np.median(ratio)),
                "ratio_iqr": [float(np.percentile(ratio, 25)),
                              float(np.percentile(ratio, 75))],
                "ratio_by_spread_quintile": [
                    float(np.mean(ratio[quintile == q]))
                    for q in range(int(np.nanmax(quintile)) + 1)
                ],
                "names_clearing_round_trip": int(np.sum(mk > 2.0 * hs)),
                "names_clearing_half_spread": int(np.sum(mk > hs)),
            }
        if block:
            report["definitions"][variant] = block
    path = os.path.join(GROUP, "summary.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nwrote %s" % path, file=sys.stderr)


if __name__ == "__main__":
    main()
