#!/usr/bin/env python3
"""P4 revisit v1, Q0a/Q0c summary: legacy-detector anomalies under the three trade streams.

Input: the Q0 jsonl rows (one row per name-day and stream, src_py/p4_q0_legacy.py), concatenated.
Output: JSON with, per stream,
  sell_share_count / sell_share_volume  sells among directional legacy bursts, pooled
  net_short_namedays                    share of name-days with sell burst volume > buy burst volume
  hit_end / hit_start                   1-minute directional hit rate (zero moves count as misses), from the
                                        end mid and from the start mid (the legacy Perm_t1m base)
  hit_end_nonzero                       same, zero moves excluded
  corr_flow_ret_pooled                  corr over name-days of (buy - sell burst volume) / executed volume with
                                        the NASDAQ open-to-close mid return
  corr_flow_ret_within_name             mean over names of that correlation across the name's dates (>= 10)
  mk3_all / mk3_k0.5 / mk3_k1.085       mean 3-minute markout (bps) of directional bursts, all and legacy-D_b-gated
and the share of regular-hours type-5 messages with Direction = +1.
"""
import argparse
import json

import numpy as np
import pandas as pd


def summarize(df):
    out = {}
    for stream, g in df.groupby("stream"):
        with np.errstate(invalid="ignore", divide="ignore"):
            flow = (g.buy_vol - g.sell_vol) / g.exec_volume
            ret = g.mid_close / g.mid_open - 1
        ok = np.isfinite(flow) & np.isfinite(ret)
        within = []
        for _, h in g[ok].assign(flow=flow[ok], ret=ret[ok]).groupby("ticker"):
            if len(h) >= 10 and h.flow.std() > 0 and h.ret.std() > 0:
                within.append(np.corrcoef(h.flow, h.ret)[0, 1])
        r = dict(namedays=int(len(g)), directional_bursts=int(g.n_dir.sum()),
                 sell_share_count=float(g.n_sell.sum() / max(g.n_dir.sum(), 1)),
                 sell_share_volume=float(g.sell_vol.sum() / max((g.buy_vol + g.sell_vol).sum(), 1)),
                 net_short_namedays=float(((g.sell_vol > g.buy_vol) & (g.n_dir > 0)).sum() / max((g.n_dir > 0).sum(), 1)),
                 hit_end=float(g.pos_end.sum() / max(g.hit_end.sum(), 1)),
                 hit_start=float(g.pos_start.sum() / max(g.hit_start.sum(), 1)),
                 hit_end_nonzero=float(g.pos_end.sum() / max(g.nz_end.sum(), 1)),
                 corr_flow_ret_pooled=float(np.corrcoef(flow[ok], ret[ok])[0, 1]),
                 corr_flow_ret_within_name=float(np.mean(within)) if within else None, names_within=len(within))
        if "mk3_n" in g:
            r["mk3_all"] = float(g.mk3_sum.sum() / max(g.mk3_n.sum(), 1))
            for k in ("0.5", "1.085"):
                r["mk3_k" + k] = float(g["mk3_sum_k" + k].sum() / max(g["mk3_n_k" + k].sum(), 1))
                r["mk3_share_gated_k" + k] = float(g["mk3_n_k" + k].sum() / max(g.mk3_n.sum(), 1))
        out[stream] = r
    first = df.drop_duplicates(["ticker", "date"])
    out["type5_direction_plus_share"] = float(first.hidden_dir_plus.sum() / max(first.hidden_msgs.sum(), 1))
    out["hidden_share_of_executed_volume"] = float(first.hidden_volume.sum() / first.exec_volume.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_json(args.rows, lines=True, dtype={"date": str})
    res = summarize(df)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
