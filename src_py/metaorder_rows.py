#!/usr/bin/env python3
"""Metaorder-v1 stage 1 (M2, M4): one row per burst for rules run60 and stream5, one ticker.

Columns: identifiers, whole-burst features, first-three-packet prefix features, backward context,
within-burst identical-size evidence (dm_pairs, dm_repeats, dm_expected; same definition as
fingerprint-v1 burst rows) and forward link evidence with other bursts over (end, end + 1,800 s]
for same-side (link_same_*) and opposite-side (link_opp_*) packets. Licensed-data derivative.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fingerprint_stats as FS
import metaorder_features as MF

RULES = (("run", 60.0, "run60"), ("stream", 5.0, "stream5"))
DROP = ("first", "last")


def rows_for_day(day, date, ticker, dbin, rates):
    out = []
    for rule, gap, label in RULES:
        member, tab = MF.burst_table(day, rule, gap)
        if not tab:
            continue
        pairs, reps, exp = MF.within_evidence(day, member, tab, dbin, rates)
        link = MF.link_evidence(day, member, tab, dbin, rates)
        df = pd.DataFrame({k: v for k, v in tab.items() if k not in DROP})
        df.insert(0, "rule", label); df.insert(0, "date", date); df.insert(0, "ticker", ticker)
        df["dm_pairs"] = pairs; df["dm_repeats"] = reps; df["dm_expected"] = exp
        for rel, lab in ((0, "same"), (1, "opp")):
            df["link_%s_pairs" % lab] = link[:, rel, 0]
            df["link_%s_matches" % lab] = link[:, rel, 1]
            df["link_%s_expected" % lab] = link[:, rel, 2]
        out.append(df)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = [tuple(x.split()) for x in Path(args.pairs).read_text().splitlines() if x.strip()]
    days = FS.load_days(args.packets, [d for p in pairs for d in p])
    frames = []
    for a, b in pairs:
        if a not in days or b not in days:
            continue
        pool = np.concatenate([days[x]["exec_depth"][days[x]["untruncated"].astype(bool) & (days[x]["sign"] != 0)] for x in (a, b)])
        pool = pool[np.isfinite(pool)]
        thr = np.quantile(pool, FS.DEPTH_QUANTILES) if len(pool) else np.array([np.inf])
        rates = MF.depth_rates((days[a], days[b]), thr)
        for d in (a, b):
            frames += rows_for_day(days[d], d, args.ticker, FS.depth_bins(days[d]["exec_depth"], thr), rates)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tmp = out.with_name(out.name + ".part")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "rows": int(len(df))}))


if __name__ == "__main__":
    main()
