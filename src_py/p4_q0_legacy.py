#!/usr/bin/env python3
"""P4 revisit v1, Q0a: does the legacy detector's hidden-execution signing produce the old anomalies?

P4_REVISIT_DESIGN.md section 6, Q0. For one ticker-day, rebuild the legacy C++ bursts
(src_cpp/burst.cpp at HEAD) on three trade streams and summarize direction, flow and hit rates:
  legacy     every type-4 and type-5 message; buy iff Direction == -1, so each hidden print is a sell
  no_hidden  type-4 messages only, buy iff Direction == -1
  packets    economic packets with a native or outside-quote sign (execution_packets)
Legacy rule: Hawkes termination with beta = 1 and trigger 0.3 (the burst ends when the intensity,
decayed over the gap since the previous trade, is below the trigger; otherwise intensity decays
and adds 1); direction +1 (-1) iff the buy (sell) count share is >= 0.763 and the opposite volume
is <= 0.28 x the majority volume, else 0; volume filter >= 0.00197 x volume. The legacy filter used a
trailing 14-day ADV; this audit uses the same day's regular-hours executed volume for all streams.
Regular hours only. Mids from the p4_extract book (strictly before the query time).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import p4_extract as PX
import p4_packets as PK

RTH0, RTH1 = PX.RTH0, PX.RTH1
BETA, TRIGGER = 1.0, 0.3
DIR_THRESH, VOL_RATIO, VOL_FRAC = 0.763, 0.28, 0.00197


def hawkes_ids(t):
    ids = np.zeros(len(t), np.int64)
    if len(t) == 0:
        return ids
    lam = 1.0; k = 0
    gaps = np.diff(t)
    for i in range(1, len(t)):
        decayed = lam * np.exp(-BETA * gaps[i - 1])
        if decayed < TRIGGER:
            k += 1; lam = 1.0
        else:
            lam = decayed + 1.0
        ids[i] = k
    return ids


def legacy_bursts(t, buy, size, threshold):
    ids = hawkes_ids(t)
    nb = int(ids[-1]) + 1 if len(ids) else 0
    if nb == 0:
        return {}
    nbuy = np.bincount(ids, weights=buy.astype(float), minlength=nb)
    ntot = np.bincount(ids, minlength=nb).astype(float)
    vbuy = np.bincount(ids, weights=np.where(buy, size, 0.0), minlength=nb)
    vsell = np.bincount(ids, weights=np.where(buy, 0.0, size), minlength=nb)
    first = np.searchsorted(ids, np.arange(nb), side="left")
    last = np.searchsorted(ids, np.arange(nb), side="right") - 1
    br, sr = nbuy / ntot, 1 - nbuy / ntot
    direction = np.zeros(nb, int)
    direction[(br >= DIR_THRESH) & (vbuy > 0) & (vsell <= VOL_RATIO * vbuy)] = 1
    direction[(sr >= DIR_THRESH) & (vsell > 0) & (vbuy <= VOL_RATIO * vsell)] = -1
    vol = vbuy + vsell
    keep = vol >= threshold
    return dict(start=t[first][keep], end=t[last][keep], direction=direction[keep], vol=vol[keep])


LEGACY_KAPPAS = (0.5, 1.085)   # C++ default and the universal Optuna median (main.tex, fixed-parameter provenance)


def summarize(b, mid):
    out = dict(n_bursts=0, n_dir=0, n_buy=0, n_sell=0, buy_vol=0.0, sell_vol=0.0,
               hit_end=0, hit_start=0, nz_end=0, nz_start=0, pos_end=0, pos_start=0, mk3_n=0, mk3_sum=0.0)
    for k in LEGACY_KAPPAS:
        out["mk3_n_k%s" % k] = 0; out["mk3_sum_k%s" % k] = 0.0
    if not b:
        return out
    d = b["direction"]
    out["n_bursts"] = int(len(d))
    m = d != 0
    out["n_dir"] = int(m.sum()); out["n_buy"] = int((d > 0).sum()); out["n_sell"] = int((d < 0).sum())
    out["buy_vol"] = float(b["vol"][d > 0].sum()); out["sell_vol"] = float(b["vol"][d < 0].sum())
    if m.any():
        s = d[m]; te = b["end"][m]; ts = b["start"][m]
        fwd = mid.at(te + 60.0)
        mv_end = s * (fwd - mid.at(np.nextafter(te, np.inf)))
        mv_start = s * (fwd - mid.at(ts))
        for tag, mv in (("end", mv_end), ("start", mv_start)):
            ok = np.isfinite(mv)
            out["hit_" + tag] = int(ok.sum())
            out["pos_" + tag] = int((mv[ok] > 0).sum())
            out["nz_" + tag] = int((mv[ok] != 0).sum())
        # Q0c: legacy D_b = 1/4 sum over 1, 3, 5, 10 min after the end of Q * dir * (Mid - StartPrice), the C++ gate,
        # against the 3-minute markout from the end mid (bps)
        m_start = mid.at(ts)
        db = np.mean([b["vol"][m] * s * (mid.at(te + h) - m_start) for h in (60.0, 180.0, 300.0, 600.0)], axis=0)
        m_end = mid.at(np.nextafter(te, np.inf))
        mk3 = s * (mid.at(te + 180.0) - m_end) / m_end * 1e4
        ok = np.isfinite(mk3)
        out["mk3_n"] = int(ok.sum()); out["mk3_sum"] = float(mk3[ok].sum())
        for k in LEGACY_KAPPAS:
            g = ok & np.isfinite(db) & (db >= k)
            out["mk3_n_k%s" % k] = int(g.sum()); out["mk3_sum_k%s" % k] = float(mk3[g].sum())
    return out


def run(msg_path, ticker, date, helper=None):
    msg = PX.read_messages(msg_path)
    context, _bid, _ask, engine = PX.bbo_context(msg_path, msg, helper)
    mid = PX.MidPath(context[0], context[1], context[2], context[3])
    t = msg["t"].to_numpy(float); ty = msg["ty"].to_numpy(int)
    sz = msg["sz"].to_numpy(float); dr = msg["dr"].to_numpy(int)
    rth = (t >= RTH0) & (t < RTH1)
    ex = rth & ((ty == 4) | (ty == 5))
    threshold = VOL_FRAC * sz[ex].sum()
    rows = []
    streams = {
        "legacy": (t[ex], dr[ex] == -1, sz[ex]),
        "no_hidden": (t[rth & (ty == 4)], dr[rth & (ty == 4)] == -1, sz[rth & (ty == 4)]),
    }
    packets = PK.fast_packets(msg, context)
    if len(packets):
        p = packets.sort_values(["time", "packet_id"], kind="stable")
        p = p[p["sign"] != 0]
        streams["packets"] = (p["time"].to_numpy(float), p["sign"].to_numpy(int) > 0, p["volume"].to_numpy(float))
    first = np.searchsorted(mid.t, RTH0, side="left")
    day = dict(ticker=ticker, date=date, engine=engine, exec_volume=float(sz[ex].sum()),
               hidden_volume=float(sz[rth & (ty == 5)].sum()),
               hidden_dir_plus=int((dr[rth & (ty == 5)] == 1).sum()), hidden_msgs=int((rth & (ty == 5)).sum()),
               mid_open=float(mid.m[first]) if first < len(mid.t) else float("nan"),
               mid_close=float(mid.at(np.array([RTH1]))[0]))
    for name, (tt, buy, size) in streams.items():
        r = dict(day, stream=name)
        r.update(summarize(legacy_bursts(tt, buy, size, threshold), mid))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--helper", default=None)
    args = ap.parse_args()
    date = "".join(re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(args.msg).name).groups())
    rows = run(args.msg, args.ticker, date, args.helper)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
