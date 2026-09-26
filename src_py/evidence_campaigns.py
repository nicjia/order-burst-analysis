#!/usr/bin/env python3
"""program-evidence-v1 module C: price paths of fingerprint-linked campaigns and matched placebos.

Campaign: same-side rare untruncated non-round packets sharing one size, consecutive gaps <= 60 s,
>= 4 packets. The size's same-side base rate on the name's other sampled days (add-one smoothed)
must give a chance link probability lambda * 60 s < 0.05. Placebo: a window of the same duration on
the same name-day and side, starting at a random same-side packet and overlapping no campaign.

Mids are pre-trade mids of the first packet at or after a time. Nothing is fitted here; rows are
written for local analysis (licensed-data derivative: keep under results/, never commit).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fingerprint_stats as FS

LINK_GAP, MIN_CHILDREN, MAX_LINK_PROB = 60.0, 4, 0.05
HORIZONS = (60, 300, 900, 1800, 3600)
FLOW_WINDOW = 300.0
RTH1 = FS.RTH0 + 23400.0


def mid_at(t, mid, tau, strictly_after=False):
    pos = np.searchsorted(t, tau, side="right" if strictly_after else "left")
    out = np.full(np.shape(tau), np.nan)
    ok = pos < len(t)
    out[ok] = mid[pos[ok]]
    return out


def daily_vol_bps(t, mid):
    grid = FS.RTH0 + 300.0 * np.arange(79)
    m = mid_at(t, mid, grid)
    r = np.diff(np.log(m))
    r = r[np.isfinite(r)]
    return float(np.std(r) * np.sqrt(78) * 1e4) if len(r) > 10 else np.nan


def side_base_rates(days, exclude):
    """{side: {size: count}} of u_nonround packets on days not in `exclude`, and the day count."""
    counts = {1: {}, -1: {}}; n_days = 0
    for date, day in days.items():
        if date in exclude:
            continue
        n_days += 1
        size, masks = FS.size_class_masks(day, np.array([], np.int64))
        u = masks["u_nonround"]
        for s in (1, -1):
            vals, c = np.unique(size[u & (day["sign"] == s)], return_counts=True)
            d = counts[s]
            for v, k in zip(vals.tolist(), c.tolist()):
                d[v] = d.get(v, 0) + k
    return counts, n_days


def chains_same_side(t, size, gap=LINK_GAP):
    """Index lists of identical-size chains (input already restricted to one side)."""
    if len(t) == 0:
        return []
    order = np.lexsort((t, size))
    ts, ss = t[order], size[order]
    cut = np.r_[True, (ss[1:] != ss[:-1]) | (np.diff(ts) > gap)]
    starts = np.flatnonzero(cut); ends = np.r_[starts[1:], len(ts)]
    return [order[a:b] for a, b in zip(starts, ends) if b - a >= MIN_CHILDREN]


def window_flows(t, sign, vol, side, t_end, members=None):
    """Same-side (excluding members) and opposite-side volume and counts before and after t_end."""
    lo = np.searchsorted(t, t_end - FLOW_WINDOW, side="left"); mid_i = np.searchsorted(t, t_end, side="left")
    hi = np.searchsorted(t, t_end + FLOW_WINDOW, side="right"); after0 = np.searchsorted(t, t_end, side="right")
    out = {}
    for lab, a, b in (("before", lo, mid_i), ("after", after0, hi)):
        idx = np.arange(a, b)
        if members is not None and len(idx):
            idx = idx[~np.isin(idx, members)]
        s = sign[idx]; v = vol[idx]
        out["same_%s_n" % lab] = int((s == side).sum()); out["same_%s_v" % lab] = float(v[s == side].sum())
        out["opp_%s_v" % lab] = float(v[s == -side].sum())
    return out


def day_rows(day, date, ticker, rates, n_other, seed):
    t = day["time"].astype(float); sign = day["sign"].astype(int); vol = day["volume"].astype(float)
    mid = day["mid"].astype(float)
    size, masks = FS.size_class_masks(day, np.array([], np.int64))
    u = masks["u_nonround"]
    sigma = daily_vol_bps(t, mid)
    v_day = float(vol[sign != 0].sum())
    rows, summary = [], {}
    windows = {1: [], -1: []}
    for side in (1, -1):
        sel = np.flatnonzero(u & (sign == side))
        base = rates[side]
        lam = np.array([(base.get(int(s), 0) + 1) / (max(n_other, 1) * 23400.0) for s in size[sel]])
        rare_share = np.array([base.get(int(s), 0) for s in size[sel]]) / max(sum(base.values()), 1)
        eligible = (lam * LINK_GAP < MAX_LINK_PROB) & (rare_share <= FS.RARE_MAX_SHARE)
        cand = sel[eligible]
        chains = chains_same_side(t[cand], size[cand])
        # chance campaigns implied by base rates, summed over sizes seen on other days
        lam_all = np.array([(c + 1) / (max(n_other, 1) * 23400.0) for s, c in base.items()
                            if (c + 1) / (max(n_other, 1) * 23400.0) * LINK_GAP < MAX_LINK_PROB
                            and c / max(sum(base.values()), 1) <= FS.RARE_MAX_SHARE])
        p = 1 - np.exp(-lam_all * LINK_GAP)
        summary["expected_chance_campaigns_side%+d" % side] = float(np.sum(lam_all * 23400.0 * (1 - p) * p ** (MIN_CHILDREN - 1)))
        summary["campaigns_side%+d" % side] = len(chains)
        for ch in chains:
            idx = cand[ch]; idx = idx[np.argsort(t[idx], kind="stable")]
            t0, t1 = t[idx[0]], t[idx[-1]]
            windows[side].append((t0, t1))
            w0, w1 = np.searchsorted(t, t0, "left"), np.searchsorted(t, t1, "right")
            seg_s, seg_v = sign[w0:w1], vol[w0:w1]
            row = dict(ticker=ticker, date=date, kind="campaign", side=side, t0=t0, t1=t1, n=len(idx),
                       size=int(size[idx[0]]), q=float(vol[idx].sum()), v_same=float(seg_v[seg_s == side].sum()),
                       v_opp=float(seg_v[seg_s == -side].sum()), v_day=v_day, sigma_day_bps=sigma,
                       spread_bps=float(day["spread_bps"][idx[0]]), lambda60=float(lam[eligible][ch].max() * LINK_GAP),
                       m_pre=float(mid[idx[0]]), m_post=float(mid_at(t, mid, np.array([t1]), strictly_after=True)[0]),
                       m_before300=float(mid_at(t, mid, np.array([t0 - 300.0]))[0]) if t0 - 300.0 >= FS.RTH0 else np.nan)
            for h in HORIZONS:
                row["m_%d" % h] = float(mid_at(t, mid, np.array([t1 + h]))[0]) if t1 + h < RTH1 else np.nan
            row.update(window_flows(t, sign, vol, side, t1, members=idx))
            rows.append(row)
    rng = np.random.default_rng(seed)
    for row in list(rows):
        side, dur = row["side"], row["t1"] - row["t0"]
        pool = t[(sign == side) & (t + dur <= RTH1)]
        if not len(pool):
            continue
        for _ in range(50):
            s0 = float(pool[rng.integers(0, len(pool))]); s1 = s0 + dur
            if all(s1 < a or s0 > b for a, b in windows[side]):
                break
        else:
            continue
        w0, w1 = np.searchsorted(t, s0, "left"), np.searchsorted(t, s1, "right")
        seg_s, seg_v = sign[w0:w1], vol[w0:w1]
        pr = dict(ticker=ticker, date=date, kind="placebo", side=side, t0=s0, t1=s1, n=int((seg_s == side).sum()),
                  size=-1, q=float(seg_v[seg_s == side].sum()), v_same=float(seg_v[seg_s == side].sum()),
                  v_opp=float(seg_v[seg_s == -side].sum()), v_day=v_day, sigma_day_bps=sigma,
                  spread_bps=float(day["spread_bps"][w0]) if w0 < len(t) else np.nan, lambda60=np.nan,
                  m_pre=float(mid[w0]) if w0 < len(t) else np.nan,
                  m_post=float(mid_at(t, mid, np.array([s1]), strictly_after=True)[0]),
                  m_before300=float(mid_at(t, mid, np.array([s0 - 300.0]))[0]) if s0 - 300.0 >= FS.RTH0 else np.nan)
        for h in HORIZONS:
            pr["m_%d" % h] = float(mid_at(t, mid, np.array([s1 + h]))[0]) if s1 + h < RTH1 else np.nan
        pr.update(window_flows(t, sign, vol, side, s1))
        rows.append(pr)
    summary.update(ticker=ticker, date=date, v_day=v_day, sigma_day_bps=sigma)
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    date_pairs = [tuple(x.split()) for x in Path(args.pairs).read_text().splitlines() if x.strip()]
    days = FS.load_days(args.packets, [d for p in date_pairs for d in p])
    pair_of = {d: p for p in date_pairs for d in p}
    rows, summaries = [], []
    for date in sorted(days):
        rates, n_other = side_base_rates(days, set(pair_of.get(date, (date,))))
        if n_other == 0:
            continue
        seed = int.from_bytes(hashlib.sha256(("program-evidence-v1|C|%s|%s" % (args.ticker, date)).encode()).digest()[:4], "little")
        r, s = day_rows(days[date], date, args.ticker, rates, n_other, seed)
        rows += r; summaries.append(s)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".summary.json").write_text(json.dumps(summaries) + "\n")
    print(json.dumps({"ticker": args.ticker, "rows": len(rows)}))


if __name__ == "__main__":
    main()
