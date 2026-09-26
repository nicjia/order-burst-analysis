#!/usr/bin/env python3
"""Vectorized economic execution packets, identical to execution_packets.reconstruct_packets.

The canonical module groups execution rows timestamp by timestamp in Python, which takes tens of
seconds on a busy name-day. This builds the same packets with array operations. Equality with the
canonical output (every column exact except vwap, which is summed in a different order) is tested
in tests/test_p4_extract.py on synthetic tapes and checked on real name-days before use.

Rules reproduced (execution_packets docstring): regular-hours type-4 and type-5 rows grouped by exact
timestamp; a unique visible native sign (-Direction) signs every row at that timestamp; with both
native signs, visible rows split by sign (visible rows with an invalid direction are dropped) and
hidden rows are signed only outside the pre-timestamp quote; with no native sign, visible rows form
an invalid-direction packet of sign 0 and hidden rows are signed outside the quote, else sign 0.
Packets are ordered by (time, first message row, sign).
"""
import numpy as np
import pandas as pd

import burst_alt as BA
import execution_packets as EP

SOURCES = {0: "native+timestamp", 1: "native-conflict-split", 2: "invalid-native-direction"}


def fast_packets(msg, context, rth_only=True):
    bt, _bm, bb, ba, bbsz, basz = context[:6]
    t = msg["t"].to_numpy(float); ty = msg["ty"].to_numpy(np.int64)
    sz = msg["sz"].to_numpy(np.int64); px = msg["px"].to_numpy(np.int64); dr = msg["dr"].to_numpy(np.int64)
    row = np.arange(len(msg), dtype=np.int64)
    m = (ty == 4) | (ty == 5)
    if rth_only:
        m &= (t >= EP.RTH0) & (t < EP.RTH1)
    if not m.any():
        return EP._empty_packets()
    t, ty, sz, px, dr, row = t[m], ty[m], sz[m].astype(float), px[m], dr[m], row[m]
    price = px.astype(float) / BA.SCALE
    times, g = np.unique(t, return_inverse=True)
    pre_q = np.nextafter(times, -np.inf)
    pre_bid, pre_ask = BA.bbo_at(bt, bb, ba, pre_q)
    pre_bsz, pre_asz = BA.bbo_at(bt, bbsz, basz, pre_q)

    vis = ty == 4
    native = np.where(vis, -dr, 0)
    n_groups = len(times)
    has_pos = np.bincount(g, weights=(vis & (native == 1)).astype(float), minlength=n_groups) > 0
    has_neg = np.bincount(g, weights=(vis & (native == -1)).astype(float), minlength=n_groups) > 0
    n_native = (has_pos.astype(np.int64) + has_neg.astype(np.int64))[g]

    pb, pa = pre_bid[g], pre_ask[g]
    finite = np.isfinite(price) & np.isfinite(pb) & np.isfinite(pa)
    with np.errstate(invalid="ignore"):
        inferred = np.where(finite & (price > pa + 1e-12), 1, np.where(finite & (price < pb - 1e-12), -1, 0))

    cat = np.full(len(t), -1, np.int64); sign = np.zeros(len(t), np.int64)
    a = n_native == 1
    cat[a] = 0; sign[a] = np.where(has_pos[g[a]], 1, -1)
    b = (n_native == 2) & vis & (native != 0)
    cat[b] = 1; sign[b] = native[b]
    c = (n_native == 0) & vis
    cat[c] = 2
    h = (n_native != 1) & ~vis
    cat[h] = 3; sign[h] = inferred[h]
    keep = cat >= 0
    t, g, cat, sign, sz, price, vis, row = t[keep], g[keep], cat[keep], sign[keep], sz[keep], price[keep], vis[keep], row[keep]

    key = (g.astype(np.int64) * 4 + cat) * 3 + (sign + 1)
    order = np.lexsort((row, key))
    key, g, cat, sign, sz, price, vis, row = (x[order] for x in (key, g, cat, sign, sz, price, vis, row))
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    pid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(key)]))
    n = len(starts)
    volume = np.bincount(pid, weights=sz, minlength=n)
    vis_volume = np.bincount(pid, weights=np.where(vis, sz, 0.0), minlength=n)
    n_msg = np.bincount(pid, minlength=n).astype(np.int64)
    n_vis = np.bincount(pid, weights=vis.astype(float), minlength=n).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(volume > 0, np.bincount(pid, weights=sz * price, minlength=n) / volume, np.nan)
    gp = g[starts]; cp = cat[starts]; sp = sign[starts]
    source = np.array([SOURCES[int(k)] if k < 3 else ("outside-prequote" if s != 0 else "ambiguous-hidden")
                       for k, s in zip(cp, sp)], dtype=object)
    frame = pd.DataFrame({
        "packet_id": np.zeros(n, np.int64), "time": times[gp], "sign": sp.astype(np.int64),
        "sign_source": source, "volume": volume, "vwap": vwap,
        "min_price": np.minimum.reduceat(price, starts), "max_price": np.maximum.reduceat(price, starts),
        "n_messages": n_msg, "n_visible": n_vis, "n_hidden": n_msg - n_vis,
        "visible_volume": vis_volume, "hidden_volume": volume - vis_volume,
        "pre_bid": pre_bid[gp], "pre_ask": pre_ask[gp], "pre_bid_size": pre_bsz[gp], "pre_ask_size": pre_asz[gp],
        "spread": pre_ask[gp] - pre_bid[gp], "depth": pre_bsz[gp] + pre_asz[gp],
        "row_first": np.minimum.reduceat(row, starts), "row_last": np.maximum.reduceat(row, starts),
    }, columns=EP.PACKET_COLUMNS)
    frame = frame.sort_values(["time", "row_first", "sign"], kind="stable").reset_index(drop=True)
    frame["packet_id"] = np.arange(n, dtype=np.int64)
    return frame
