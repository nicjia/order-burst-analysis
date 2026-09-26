#!/usr/bin/env python3
"""Post-hoc module F2 (after the F1 read): markouts by program score within truncation-share strata.

Same markout as evidence_markouts.py (untruncated packets, horizons 1/10/60/300 s, next packet
within 2h), with an extra stratum: the truncation share of the packet's run/60 burst in
[0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1]. Tests whether the program-minus-bottom contrast
survives among bursts that sweep the book equally often.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS
import program_bursts as PB

HORIZONS = (1, 10, 60, 300)
T_EDGES = np.array([0, 0.25, 0.5, 0.75, 1.0000001])
N_GROUPS, N_DECILES = 3, 10                              # 0 bottom, 1 middle, 2 program (bursts only)


def day_markouts(day, model):
    t = day["time"].astype(float); sign = day["sign"].astype(int); mid = day["mid"].astype(float)
    spread = day["spread_bps"].astype(float)
    member, table = PB.run_bursts(day)
    out = np.zeros((3, N_GROUPS, len(T_EDGES) - 1, N_DECILES, len(HORIZONS)))
    if not table:
        return out
    s = PB.score(table, model)
    g = np.where(s >= model["program_threshold_q80"], 2, np.where(s <= model["program_threshold_q20"], 0, 1))
    tb = np.clip(np.searchsorted(T_EDGES, table["truncated_share"], side="right") - 1, 0, len(T_EDGES) - 2)
    sel = day["untruncated"].astype(bool) & (sign != 0) & np.isfinite(mid) & np.isfinite(spread) & (member >= 0)
    idx = np.flatnonzero(sel)
    idx = idx[np.isfinite(s[member[idx]])]
    if len(idx) < N_DECILES:
        return out
    q = np.quantile(spread[idx], np.linspace(0, 1, N_DECILES + 1)[1:-1])
    dec = np.searchsorted(q, spread[idx], side="right")
    grp = g[member[idx]]; tbin = tb[member[idx]]
    for hi, h in enumerate(HORIZONS):
        pos = np.searchsorted(t, t[idx] + h, side="left")
        ok = pos < len(t)
        limit = t[idx] + (h + 1.0 if h == 1 else 2.0 * h)
        ok[ok] &= t[pos[ok]] < limit[ok]
        m_next = np.full(len(idx), np.nan); m_next[ok] = mid[pos[ok]]
        pi = spread[idx] / 2.0 - sign[idx] * (m_next - mid[idx]) / mid[idx] * 1e4
        good = np.isfinite(pi) & (np.abs(pi) < 1000)
        key = (grp[good], tbin[good], dec[good], np.full(good.sum(), hi))
        np.add.at(out[0], key, pi[good]); np.add.at(out[1], key, pi[good] ** 2); np.add.at(out[2], key, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = PB.load_model(args.model)
    pairs = [tuple(x.split()) for x in Path(args.pairs).read_text().splitlines() if x.strip()]
    days = FS.load_days(args.packets, [d for p in pairs for d in p])
    dates = sorted(days)
    res = np.zeros((len(dates), 3, N_GROUPS, len(T_EDGES) - 1, N_DECILES, len(HORIZONS)))
    for i, d in enumerate(dates):
        res[i] = day_markouts(days[d], model)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, ticker=np.array(args.ticker), dates=np.array(dates), markouts=res, horizons=np.array(HORIZONS),
                        trunc_edges=T_EDGES)
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "days": len(dates)}))


if __name__ == "__main__":
    main()
