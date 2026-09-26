#!/usr/bin/env python3
"""P4 revisit v1, stage 1: trade and submission bursts with P4 impact measures for one ticker-day.

Design: P4_REVISIT_DESIGN.md (frozen 2026-09-15), sections 3-4. Licensed-data derivative: the npz
stays on the cluster and is never committed.

Trade bursts (T): economic packets (execution_packets, type-5 Direction never used), run rule with
60 s gaps and >= 3 packets (program_bursts.run_bursts). Submission bursts (S): qualifying adds, same
run rule. A qualifying add is a type-1 message at or better than its own-side best quote prevailing
strictly before it, not the add half of an ITCH replace, not the remainder of a marketable order and
not fleeting (fully deleted within 1 s with no execution before the delete).

Per burst (side s, first event t_b, last event t_e), with m(t) the mid prevailing strictly before t:
  m_ref = m(t_b); peak_raw = max over the mid path in [t_b, t_e + 10 s) of s * (m - m_ref);
  d_h = s * (m(t_b + h) - m_ref) for h in 60, 180, 300, 600 s (P4 eq. 3.2, from initiation);
  t_dec = max(t_b + 600, t_e + 10); m_dec = m(t_dec).
Each real burst has a pseudo-burst with the same side and duration, starting uniformly at random in
[9:30, 16:00 - duration - 600 s], measured on the same mid path. Filters, thresholds, prices from
CRSP and all outcomes are computed downstream.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

import burst_alt as BA
import execution_packets as EP
import fingerprint_packets as FP
import fingerprint_stats as FS
import p4_packets as PK
import program_bursts as PB

RTH0, RTH1 = 34200.0, 57600.0
HORIZONS = (60.0, 180.0, 300.0, 600.0)
PEAK_TAIL = 10.0
DECISION_LAG = 600.0
FLEETING = 1.0
GAP = 60.0
MIN_EVENTS = 3
CLOCKS = {"1530": 55800.0, "1550": 57000.0, "close": RTH1}
SALT = "p4-revisit-v1"


# ---------------------------------------------------------------------------------------------
# Messages and the best quote

def read_messages(path):
    return pd.read_csv(path, header=None, usecols=[0, 1, 2, 3, 4, 5],
                       names=["t", "ty", "oid", "sz", "px", "dr"])


def bbo_context(path, messages, helper=None):
    """Return (context, bid ticks, ask ticks, engine); context matches burst_alt.reconstruct."""
    if helper and Path(helper).exists():
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bbo.bin"
            subprocess.run([str(helper), str(path), str(out)], check=True)
            raw = np.fromfile(out, dtype="<i8")
        n = int(raw[0])
        rec = raw[1:1 + 5 * n].reshape(n, 5)
        t = messages["t"].to_numpy(float)
        bid_i, ask_i = rec[:, 1].copy(), rec[:, 2].copy()
        context = (t[rec[:, 0]], (bid_i + ask_i).astype(float) / (2 * BA.SCALE),
                   bid_i.astype(float) / BA.SCALE, ask_i.astype(float) / BA.SCALE,
                   rec[:, 3].astype(float), rec[:, 4].astype(float), {}, None)
        return context, bid_i, ask_i, "cpp"
    context = BA.reconstruct(path)
    bid_i = np.rint(context[2] * BA.SCALE).astype(np.int64)
    ask_i = np.rint(context[3] * BA.SCALE).astype(np.int64)
    return context, bid_i, ask_i, "python"


class MidPath:
    """Mid, bid and ask prevailing strictly before any query time."""

    def __init__(self, bt, bm, bb, ba):
        self.t, self.m, self.b, self.a = bt, bm, bb, ba

    def index(self, q):
        return np.searchsorted(self.t, np.asarray(q, float), side="left") - 1

    def at(self, q, which="m"):
        q = np.asarray(q, float)
        arr = getattr(self, which)
        out = np.full(q.shape, np.nan)
        ok = np.isfinite(q)
        idx = np.full(q.shape, -1)
        idx[ok] = self.index(q[ok])
        good = idx >= 0
        out[good] = arr[idx[good]]
        return out

    def spread_bps(self, q):
        b, a, m = self.at(q, "b"), self.at(q, "a"), self.at(q, "m")
        with np.errstate(invalid="ignore", divide="ignore"):
            return (a - b) / m * 1e4

    def signed_extreme(self, side, t0, t1, mref):
        """max over the mid prevailing at t0 and every update in [t0, t1) of side * (m - mref)."""
        out = np.full(len(t0), np.nan)
        ok = np.isfinite(t0) & np.isfinite(t1) & np.isfinite(mref)
        i0 = np.full(len(t0), -1)
        i1 = np.zeros(len(t0), dtype=np.int64)
        i0[ok] = self.index(t0[ok])
        i1[ok] = np.searchsorted(self.t, t1[ok], side="left")
        m = self.m
        for k in np.flatnonzero(ok):
            a = max(i0[k], 0); b = i1[k]
            if b <= a:
                continue
            seg = m[a:b]
            out[k] = (seg.max() - mref[k]) if side[k] > 0 else (mref[k] - seg.min())
        return out


# ---------------------------------------------------------------------------------------------
# Events

def modal_size(burst_of, size, n_bursts):
    """Per burst: most frequent size (ties -> smallest) and its count, over member events."""
    mode = np.full(n_bursts, np.nan); count = np.zeros(n_bursts)
    ok = burst_of >= 0
    if not ok.any():
        return mode, count
    b = burst_of[ok].astype(np.int64); s = np.rint(size[ok]).astype(np.int64)
    key, cnt = np.unique(np.stack([b, s], 1), axis=0, return_counts=True)
    order = np.lexsort((key[:, 1], -cnt, key[:, 0]))       # by burst, count desc, size asc
    key, cnt = key[order], cnt[order]
    first = np.r_[True, key[1:, 0] != key[:-1, 0]]
    mode[key[first, 0]] = key[first, 1]; count[key[first, 0]] = cnt[first]
    return mode, count


def trade_bursts(packets, model):
    day = FP.packet_arrays(packets) if len(packets) else None
    if day is None or not len(day["time"]):
        return None, day
    member, table = PB.run_bursts(day, gap=GAP, min_packets=MIN_EVENTS)
    if not table:
        return None, day
    nb = len(table["start"])
    unt = day["untruncated"].astype(bool)
    mode, count = modal_size(np.where(unt, member, -1), day["volume"], nb)
    out = dict(t_b=table["start"], t_e=table["end"], side=table["side"].astype(np.int8),
               n=table["n_packets"], vol=table["volume"], mode_size=mode, mode_count=count,
               truncated_share=table["truncated_share"], hidden_share=table["hidden_share"],
               program_score=PB.score(table, model) if model is not None else np.full(nb, np.nan))
    return out, day


def qualifying_adds(msg, bid_i, ask_i, mid):
    """Boolean masks over the regular-hours adds, plus the add arrays."""
    t = msg["t"].to_numpy(float); ty = msg["ty"].to_numpy(int); oid = msg["oid"].to_numpy(np.int64)
    sz = msg["sz"].to_numpy(np.int64); px = msg["px"].to_numpy(np.int64); dr = msg["dr"].to_numpy(int)
    ia = np.flatnonzero((ty == 1) & (t >= RTH0) & (t < RTH1))
    ta, oa, sa, pa, da = t[ia], oid[ia], sz[ia], px[ia], dr[ia]
    if len(bid_i):
        j = mid.index(ta)
        valid = j >= 0
        jj = np.maximum(j, 0)
        touch = np.where(da > 0, bid_i[jj], ask_i[jj])
        at_or_better = valid & np.where(da > 0, pa >= touch, pa <= touch)
    else:
        at_or_better = np.zeros(len(ta), bool)
    deletes = pd.MultiIndex.from_arrays([t[ty == 3], dr[ty == 3]])
    replace = pd.MultiIndex.from_arrays([ta, da]).isin(deletes)
    remainder = np.isin(ta, np.unique(t[(ty == 4) | (ty == 5)]))
    first_del = pd.Series(t[ty == 3], index=oid[ty == 3]).groupby(level=0).min()
    first_exec = pd.Series(t[ty == 4], index=oid[ty == 4]).groupby(level=0).min()
    del_t = first_del.reindex(oa).to_numpy(float)
    exe_t = first_exec.reindex(oa).to_numpy(float)
    with np.errstate(invalid="ignore"):
        fleeting = np.isfinite(del_t) & (del_t - ta < FLEETING) & ~(np.isfinite(exe_t) & (exe_t <= del_t))
    qual = at_or_better & ~replace & ~remainder & ~fleeting
    adds = dict(t=ta, oid=oa, size=sa, side=da)
    masks = dict(all=np.ones(len(ta), bool), at_or_better=at_or_better, replace=replace,
                 remainder=remainder, fleeting=fleeting, qualifying=qual)
    return adds, masks


def submission_bursts(msg, adds, qual):
    tq = adds["t"][qual]; sq = adds["side"][qual]; zq = adds["size"][qual]; oq = adds["oid"][qual]
    if len(tq) == 0:
        return None
    ids, _ = FS.burst_ids(tq, sq, GAP, "run")
    counts = np.bincount(ids)
    first = np.r_[0, np.cumsum(counts)[:-1]]
    keep = np.flatnonzero(counts >= MIN_EVENTS)
    if not len(keep):
        return None
    remap = np.full(len(counts), -1); remap[keep] = np.arange(len(keep))
    member = remap[ids]
    f = first[keep]; last = f + counts[keep] - 1
    nb = len(keep)
    mode, mcount = modal_size(member, zq.astype(float), nb)
    vol = np.bincount(member[member >= 0], weights=zq[member >= 0].astype(float), minlength=nb)
    out = dict(t_b=tq[f], t_e=tq[last], side=sq[f].astype(np.int8), n=counts[keep].astype(float),
               vol=vol, mode_size=mode, mode_count=mcount)
    out["_member"] = member; out["_oid"] = oq
    return out


def order_outcomes_by_decision(msg, bursts):
    """Executed and cancelled shares of each submission burst's own orders by its decision time."""
    member, oq = bursts.pop("_member"), bursts.pop("_oid")
    nb = len(bursts["t_b"])
    inb = member >= 0
    pos = pd.Series(member[inb], index=oq[inb])
    pos = pos[~pos.index.duplicated()]
    t = msg["t"].to_numpy(float); ty = msg["ty"].to_numpy(int)
    oid = msg["oid"].to_numpy(np.int64); sz = msg["sz"].to_numpy(float)
    res = {}
    for name, mask in (("exec_dec", ty == 4), ("cancel_dec", (ty == 2) | (ty == 3))):
        k = pos.reindex(oid[mask]).to_numpy(float)
        ok = np.isfinite(k)
        b = k[ok].astype(np.int64)
        within = t[mask][ok] <= bursts["t_dec"][b]
        res[name] = np.bincount(b[within], weights=sz[mask][ok][within], minlength=nb)
    bursts.update(res)


# ---------------------------------------------------------------------------------------------
# Measures

def seeded_uniform(n, *key):
    seed = int(hashlib.sha256("|".join((SALT,) + tuple(str(k) for k in key)).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).random(n)


def measure(bursts, mid, key):
    side = bursts["side"].astype(float)
    tb, te = bursts["t_b"], bursts["t_e"]
    mref = mid.at(tb)
    bursts["m_ref"] = mref
    bursts["peak_raw"] = mid.signed_extreme(side, tb, te + PEAK_TAIL, mref)
    for h in HORIZONS:
        bursts["d%d" % h] = side * (mid.at(tb + h) - mref)
    bursts["dmean"] = np.mean([bursts["d%d" % h] for h in HORIZONS], axis=0)
    bursts["t_dec"] = np.maximum(tb + DECISION_LAG, te + PEAK_TAIL)
    bursts["m_dec"] = mid.at(bursts["t_dec"])
    bursts["spread_b"] = mid.spread_bps(tb)
    bursts["spread_dec"] = mid.spread_bps(bursts["t_dec"])
    pre = tb - 1800.0
    bursts["m_pre30"] = np.where(pre >= RTH0, mid.at(pre), np.nan)
    dur = te - tb
    hi = RTH1 - dur - DECISION_LAG
    u = RTH0 + seeded_uniform(len(tb), *key) * (hi - RTH0)
    u = np.where(hi > RTH0, u, np.nan)
    ps_ref = mid.at(u)
    bursts["ps_t"] = u
    bursts["ps_mref"] = ps_ref
    bursts["ps_peak_raw"] = mid.signed_extreme(side, u, u + dur + PEAK_TAIL, ps_ref)
    bursts["ps_dmean"] = np.mean([side * (mid.at(u + h) - ps_ref) for h in HORIZONS], axis=0)
    bursts["ps_t_dec"] = np.maximum(u + DECISION_LAG, u + dur + PEAK_TAIL)
    bursts["ps_m_dec"] = mid.at(bursts["ps_t_dec"])
    return bursts


def day_summary(day, mid, masks):
    out = {}
    if day is not None and len(day["time"]):
        t = day["time"]; s = day["sign"]; v = day["volume"]
        for name, clock in CLOCKS.items():
            before = t < clock
            out["buy_" + name] = float(v[before & (s > 0)].sum())
            out["sell_" + name] = float(v[before & (s < 0)].sum())
            out["unsigned_" + name] = float(v[before & (s == 0)].sum())
        out["hidden_volume"] = float((v * day["hidden_share"]).sum())
        out["n_packets"] = int(len(t))
    for name, clock in CLOCKS.items():
        out["mid_" + name] = float(mid.at(np.array([clock]))[0])
        out["spread_" + name] = float(mid.spread_bps(np.array([clock]))[0])
    first = np.searchsorted(mid.t, RTH0, side="left")
    out["mid_open"] = float(mid.m[first]) if first < len(mid.t) else float("nan")
    for k, m in masks.items():
        out["adds_" + k] = int(m.sum())
    return out


# Derivable downstream: t_dec = max(t_b + 600, t_e + 10), the same for pseudo-bursts; dmean keeps
# the P4 average and d60/d600 its profile.
NOT_SAVED = {"d180", "d300", "t_dec", "ps_t_dec"}


def to_arrays(prefix, bursts):
    arrays = {}
    if not bursts:
        return arrays
    for k, v in bursts.items():
        if k in NOT_SAVED:
            continue
        v = np.asarray(v)
        if k == "side":
            v = v.astype(np.int8)
        else:
            v = v.astype(np.float32)      # times to ~7 ms, prices to ~1e-5 relative
        arrays[prefix + k] = v
    return arrays


def extract(msg_path, ticker, date, helper=None, model=None, canonical_packets=False):
    clock = {}
    t0 = _time.time()
    msg = read_messages(msg_path)
    clock["read"] = _time.time() - t0
    t0 = _time.time()
    context, bid_i, ask_i, engine = bbo_context(msg_path, msg, helper)
    mid = MidPath(context[0], context[1], context[2], context[3])
    clock["bbo"] = _time.time() - t0
    t0 = _time.time()
    if canonical_packets:
        _ctx, packets = EP.reconstruct_packets(msg_path, context=context)
    else:
        packets = PK.fast_packets(msg, context)
    T, day = trade_bursts(packets, model)
    clock["trade"] = _time.time() - t0
    t0 = _time.time()
    adds, masks = qualifying_adds(msg, bid_i, ask_i, mid)
    S = submission_bursts(msg, adds, masks["qualifying"])
    clock["submission"] = _time.time() - t0
    t0 = _time.time()
    if T is not None:
        measure(T, mid, (ticker, date, "T"))
    if S is not None:
        measure(S, mid, (ticker, date, "S"))
        order_outcomes_by_decision(msg, S)
    clock["measure"] = _time.time() - t0
    grid_t = RTH0 + 60.0 * np.arange(1, 391)
    arrays = dict(grid_mid=mid.at(grid_t).astype(np.float32))
    arrays.update(to_arrays("T_", T)); arrays.update(to_arrays("S_", S))
    summary = day_summary(day, mid, masks)
    status = dict(ticker=ticker, date=date, engine=engine, packets="canonical" if canonical_packets else "fast", n_messages=int(len(msg)), n_bbo=int(len(mid.t)),
                  n_T=int(len(T["t_b"])) if T else 0, n_S=int(len(S["t_b"])) if S else 0,
                  seconds={k: round(v, 3) for k, v in clock.items()}, day=summary)
    return arrays, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--helper", default=None, help="path to the compiled p4_bbo binary")
    ap.add_argument("--model", default=str(PB.MODEL_PATH))
    ap.add_argument("--canonical-packets", action="store_true")
    args = ap.parse_args()
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(args.msg).name)
    if not match:
        raise ValueError("message filename must contain YYYY-MM-DD")
    date = "".join(match.groups())
    model = PB.load_model(args.model) if args.model and Path(args.model).exists() else None
    arrays, status = extract(args.msg, args.ticker, date, args.helper, model, args.canonical_packets)
    status["program_model"] = model is not None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, day_json=np.array(json.dumps(status["day"])), **arrays)
    tmp.rename(out)
    print(json.dumps(status))


if __name__ == "__main__":
    main()
