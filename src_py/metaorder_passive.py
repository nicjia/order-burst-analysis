#!/usr/bin/env python3
"""Metaorder-v1 M5 driver: real-time burst triggers and queue-aware passive orders for one ticker-day.

Reconstructs economic packets from the LOBSTER message file, forms bursts with the adopted rule, scores each
burst at its third own-side packet with the real-time model, posts a 100-share order at the touch on the side
the burst trades against, and adds placebo postings at uniform times (10:00-15:30, random side), as many as
program triggers. Writes one row per trigger with fill outcome and provider-signed markouts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import execution_packets as EP
import fingerprint_packets as FP
import metaorder_features as MF
import queue_sim as QS

RULES = {"run60": ("run", 60.0), "stream5": ("stream", 5.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--rule", required=True, choices=list(RULES))
    ap.add_argument("--model", required=True, help="real-time model JSON (M4)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = json.loads(Path(args.model).read_text())
    _ctx, packets = EP.reconstruct_packets(args.msg)
    day = FP.packet_arrays(packets)
    rule, gap = RULES[args.rule]
    member, tab = MF.burst_table(day, rule, gap)
    rows = []
    if tab:
        s = MF.score(model, tab)
        label = np.where(s >= model["threshold_q80"], "program", np.where(s <= model["threshold_q20"], "bottom", "middle"))
        ok = np.isfinite(s)
        rows.append(pd.DataFrame(dict(tau=tab["t3"][ok], side=-tab["side"][ok], label=label[ok], score=s[ok],
                                      burst_side=tab["side"][ok])))
        n_prog = int((label[ok] == "program").sum())
    else:
        n_prog = 0
    seed = int.from_bytes(hashlib.sha256(("metaorder-v1|M5|%s|%s" % (args.ticker, args.date)).encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    if n_prog:
        rows.append(pd.DataFrame(dict(tau=np.sort(rng.uniform(36000, 55800, n_prog)), side=rng.choice([-1, 1], n_prog),
                                      label="placebo", score=np.nan, burst_side=0)))
    trig = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["tau", "side", "label", "score", "burst_side"])
    trig = trig[(trig.tau >= 34260) & (trig.tau <= 57240)].reset_index(drop=True)
    msgs = QS.read_messages(args.msg)
    res, bbo = QS.simulate(msgs, trig)
    res = QS.markouts(res, bbo)
    res.insert(0, "date", args.date); res.insert(0, "ticker", args.ticker)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    res.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "date": args.date, "triggers": int(len(res)), "filled": int(res.filled.sum())}))


if __name__ == "__main__":
    main()
