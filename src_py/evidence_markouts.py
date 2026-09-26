#!/usr/bin/env python3
"""program-evidence-v1 module F: liquidity-provider markouts by burst program score (one ticker).

Untruncated packets fill at the touch, so the provider's markout at horizon h is exact up to the
mid measurement: pi_h = half-spread - side * (mid_{t+h} - mid_t) / mid_t (bps). mid_{t+h} is the
pre-trade mid of the first packet at or after t + h, used only if that packet arrives before
t + 2h (t + h + 1 s for h = 1 s); otherwise the observation is dropped for that horizon.

Groups: 0 bottom-quintile bursts, 1 middle, 2 program bursts (top quintile), 3 unscored bursts,
4 packets outside any run/60 burst. Sums, squared sums and counts by group, name-day spread decile
and horizon. Burst scores use whole-burst features: descriptive toxicity, not a real-time signal.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS
import program_bursts as PB

HORIZONS = (1, 10, 60, 300)
N_GROUPS, N_DECILES = 5, 10


def day_markouts(day, model):
    t = day["time"].astype(float); sign = day["sign"].astype(int); mid = day["mid"].astype(float)
    spread = day["spread_bps"].astype(float)
    member, table = PB.run_bursts(day)
    group = np.full(len(t), 4)
    if table:
        s = PB.score(table, model)
        g = np.where(~np.isfinite(s), 3, np.where(s >= model["program_threshold_q80"], 2,
                                                   np.where(s <= model["program_threshold_q20"], 0, 1)))
        inb = member >= 0
        group[inb] = g[member[inb]]
    sel = day["untruncated"].astype(bool) & (sign != 0) & np.isfinite(mid) & np.isfinite(spread)
    idx = np.flatnonzero(sel)
    out = np.zeros((3, N_GROUPS, N_DECILES, len(HORIZONS)))    # sum, sumsq, count
    if len(idx) < N_DECILES:
        return out, np.zeros(N_GROUPS)
    q = np.quantile(spread[idx], np.linspace(0, 1, N_DECILES + 1)[1:-1])
    dec = np.searchsorted(q, spread[idx], side="right")
    for hi, h in enumerate(HORIZONS):
        pos = np.searchsorted(t, t[idx] + h, side="left")
        ok = pos < len(t)
        limit = t[idx] + (h + 1.0 if h == 1 else 2.0 * h)
        ok[ok] &= t[pos[ok]] < limit[ok]
        m_next = np.full(len(idx), np.nan); m_next[ok] = mid[pos[ok]]
        pi = spread[idx] / 2.0 - sign[idx] * (m_next - mid[idx]) / mid[idx] * 1e4
        good = np.isfinite(pi) & (np.abs(pi) < 1000)
        gg, dd, pp = group[idx][good], dec[good], pi[good]
        np.add.at(out[0], (gg, dd, hi), pp)
        np.add.at(out[1], (gg, dd, hi), pp ** 2)
        np.add.at(out[2], (gg, dd, hi), 1.0)
    vol = np.bincount(group[sign != 0], weights=day["volume"][sign != 0], minlength=N_GROUPS)
    return out, vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = PB.load_model(args.model)
    date_pairs = [tuple(x.split()) for x in Path(args.pairs).read_text().splitlines() if x.strip()]
    days = FS.load_days(args.packets, [d for p in date_pairs for d in p])
    dates = sorted(days)
    res = np.zeros((len(dates), 3, N_GROUPS, N_DECILES, len(HORIZONS)))
    vol = np.zeros((len(dates), N_GROUPS))
    for i, d in enumerate(dates):
        res[i], vol[i] = day_markouts(days[d], model)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, ticker=np.array(args.ticker), dates=np.array(dates), markouts=res, group_volume=vol,
                        horizons=np.array(HORIZONS))
    tmp.rename(out)
    print(json.dumps({"ticker": args.ticker, "days": len(dates)}))


if __name__ == "__main__":
    main()
