#!/usr/bin/env python3
"""Prospective burst landmarks and matched-state prediction labels, exploratory v1.

All predictors stop at decision_time. Completion is recognized by a breaking packet or
one second of silence, never at an retrospectively known final execution. Every action
has a one-second latency. Quotes describe infinitesimal market-order timing, not fills
for a finite strategy. No type-5 Direction is used.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import burst_alt as BA
import execution_packets as EP
import fragment_reconstruction as FR
import metaorder_models as MM


WINDOWS = (1, 5, 60, 300)
STAGES = ("third", "sixth", "completion")
BASE = ["spread_bps", "log_depth", "book_imbalance", "tod_sin", "tod_cos"] + [
    "%s_%ds" % (name, w) for w in WINDOWS
    for name in ("count", "count_imbalance", "volume", "volume_imbalance",
                 "ambiguous_count", "return_bps")]
REGIME = ["regime_alignment", "regime_confidence"]
BURST = ["volume", "n_packets", "duration", "cv_packet_volume", "mean_gap",
         "cv_gap", "hidden_share", "spread_start", "depth_start",
         "depth_imbalance_start", "intensity"]
TARGETS = ("flow_60s", "flow_300s", "return_60s", "return_300s", "wait_cost_60s")


def landmarks(packets, gap=1.0):
    t = packets.time.to_numpy(float); s = packets.sign.to_numpy(int)
    if not len(t):
        return []
    edges = np.r_[0, np.flatnonzero((np.diff(t) >= gap) | (s[1:] != s[:-1])
                                  | (s[1:] == 0) | (s[:-1] == 0)) + 1, len(t)]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        if s[a] == 0 or b - a < 3:
            continue
        for count, stage in ((3, "third"), (6, "sixth")):
            if b - a >= count:
                j = int(a + count - 1)
                rows.append((int(a), j, float(t[j]), stage))
        # If tape ends before timeout, no completion was observable on this tape.
        when = min(t[b] if b < len(t) else np.inf, t[b - 1] + gap)
        if when <= t[-1]:
            rows.append((int(a), int(b - 1), float(when), "completion"))
    return rows


def regime_filter(packets):
    """Fixed two-state binomial HMM on completed one-second count bins.

    p(buy|state)=.35/.65, symmetric switch probability .01 per second. This is a
    transparent online regime benchmark, NOT the published score-driven BOCPD model.
    """
    n = int(EP.RTH1 - EP.RTH0)
    bins = np.floor(packets.time.to_numpy(float) - EP.RTH0).astype(int)
    signs = packets.sign.to_numpy(int)
    good = (bins >= 0) & (bins < n)
    buys = np.bincount(bins[good & (signs > 0)], minlength=n)
    sells = np.bincount(bins[good & (signs < 0)], minlength=n)
    posterior = np.empty(n + 1); posterior[0] = 0.5
    logratio = np.log(0.65 / 0.35)
    for i in range(n):
        prior = 0.01 + 0.98 * posterior[i]
        logodds = np.log(prior / (1 - prior)) + (buys[i] - sells[i]) * logratio
        posterior[i + 1] = 1 / (1 + np.exp(-np.clip(logodds, -35, 35)))
    return posterior


def extract_packets(packets, context, ticker, date, model, sample_modulus=32):
    p = packets.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    candidates = landmarks(p)
    chosen = []
    for stage in STAGES:
        group = [x for x in candidates if x[3] == stage
                 and EP.RTH0 + 300 <= x[2] < EP.RTH1 - 301]
        # A public fixed hash permits selection at the decision itself; a full-day
        # fixed-size random sample would require knowing future candidate counts.
        if sample_modulus > 1:
            group = [x for x in group if EP.deterministic_seed(
                ticker, date, stage, x[1], "bi-v1") % sample_modulus == 0]
        chosen.extend(group)
    if not chosen:
        return pd.DataFrame()
    rows = []
    for a, b, decision, stage in chosen:
        row = FR._summarize_fragment(p, a, b, len(rows))
        row.update(decision_time=decision, stage=stage)
        rows.append(row)
    out = pd.DataFrame(rows)
    out = MM.score_fragments(out, model)
    d = out.decision_time.to_numpy(float); s = out.sign.to_numpy(float)
    t = p.time.to_numpy(float); ps = p.sign.to_numpy(float); v = p.volume.to_numpy(float)
    bt, bm, bb, ba, bbs, bas, _ofi, _trades = context
    mid = BA.mid_at(bt, bm, d)
    bid, ask = BA.bbo_at(bt, bb, ba, d)
    bsz, asz = BA.bbo_at(bt, bbs, bas, d)
    out["spread_bps"] = (ask - bid) / mid * 1e4
    out["log_depth"] = np.log1p(bsz + asz)
    out["book_imbalance"] = s * (bsz - asz) / np.maximum(bsz + asz, 1)
    angle = 2 * np.pi * (d - EP.RTH0) / (EP.RTH1 - EP.RTH0)
    out["tod_sin"] = np.sin(angle); out["tod_cos"] = np.cos(angle)
    arrays = {"count": (ps != 0).astype(float), "count_imbalance": ps,
              "volume": v * (ps != 0), "volume_imbalance": v * ps,
              "ambiguous_count": (ps == 0).astype(float)}
    cums = {key: np.r_[0, np.cumsum(values)] for key, values in arrays.items()}
    right = np.searchsorted(t, d, side="right")
    for w in WINDOWS:
        left = np.searchsorted(t, d - w, side="right")
        for key, sums in cums.items():
            values = sums[right] - sums[left]
            if "imbalance" in key:
                values *= s
            out["%s_%ds" % (key, w)] = values
        earlier = BA.mid_at(bt, bm, d - w)
        out["return_bps_%ds" % w] = s * (mid - earlier) / earlier * 1e4
    posterior = regime_filter(p)
    idx = np.floor(d - EP.RTH0).astype(int).clip(0, len(posterior) - 1)
    prob = posterior[idx]  # completed seconds only; no current-second future prints
    out["regime_alignment"] = s * (2 * prob - 1)
    out["regime_confidence"] = np.abs(2 * prob - 1)
    reference = d + 1.0
    refmid = BA.mid_at(bt, bm, reference)
    refbid, refask = BA.bbo_at(bt, bb, ba, reference)
    refbs, refas = BA.bbo_at(bt, bbs, bas, reference)
    out["reference_time"] = reference
    out["reference_mid"] = refmid
    out["reference_touch"] = np.where(s > 0, refask, refbid)
    out["reference_depth"] = np.where(s > 0, refas, refbs)
    left = np.searchsorted(t, reference, side="right")
    for h in (60, 300):
        future_time = reference + h
        right = np.searchsorted(t, future_time, side="right")
        out["flow_%ds" % h] = s * (cums["count_imbalance"][right] - cums["count_imbalance"][left])
        future_mid = BA.mid_at(bt, bm, future_time)
        out["return_%ds" % h] = s * (future_mid - refmid) / refmid * 1e4
        if h == 60:
            fb, fa = BA.bbo_at(bt, bb, ba, future_time)
            fbs, fas = BA.bbo_at(bt, bbs, bas, future_time)
            out["wait_depth"] = np.where(s > 0, fas, fbs)
            out["wait_cost_60s"] = s * (np.where(s > 0, fa, fb)
                                        - out.reference_touch.to_numpy()) / refmid * 1e4
    out.insert(0, "ticker", ticker); out.insert(1, "date", int(date))
    out["row_id"] = ["%s:%d:%s:%d" % (ticker, date, r.stage, r.packet_last)
                     for r in out.itertuples()]
    required = BASE + REGIME + BURST + ["fragment_score"] + list(TARGETS)
    out["valid"] = np.isfinite(out[required].to_numpy(float)).all(axis=1) & (refmid > 0)
    out["extreme_price"] = (np.abs(out[["return_60s", "return_300s", "wait_cost_60s"]]) > 1000).any(axis=1)
    return out.sort_values(["decision_time", "stage"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sample-modulus", type=int, default=32)
    args = ap.parse_args()
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(args.msg).name)
    if not match:
        raise ValueError("message filename must contain YYYY-MM-DD")
    date = int("".join(match.groups()))
    spec = json.loads(Path(args.model).read_text())
    context, packets = EP.reconstruct_packets(args.msg)
    frame = extract_packets(packets, context, args.ticker, date,
                            MM.RidgeLogit.from_dict(spec["fragment_model"]), args.sample_modulus)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(json.dumps({"ticker": args.ticker, "date": date, "packets": len(packets),
                      "rows": len(frame), "valid": int(frame.valid.sum()) if len(frame) else 0}))


if __name__ == "__main__":
    main()
