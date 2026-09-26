#!/usr/bin/env python3
"""Metaorder-v1 M5: queue-aware simulation of small passive orders posted at trigger times.

Book: LOBSTER messages (1 add, 2 partial cancel, 3 delete, 4 visible execution, 5 hidden execution,
6 cross, 7 halt) with order-level tracking. A virtual order of V shares posted at time tau at price P
on side S (+1 bid, -1 ask) joins the back of the displayed queue after all messages at tau.

Fill rule (price-time priority; displayed orders only):
  - executions or cancellations of orders that were ahead reduce the queue ahead;
  - a visible execution at (P, S) of an order that was NOT ahead means everything ahead is gone and
    the virtual order was hit first: it fills at that message's time;
  - a visible execution on side S at a price beyond P (the aggressor walked past P) fills it;
  - hidden executions never fill it (their side is unknown in LOBSTER).
Cancel rule: 60 s after posting, or as soon as a better price appears on the order's own side, or
when its level empties through cancellations only (the order would be alone at the touch; exiting is
conservative).
Markouts are provider-signed: side_sign * (mid_{fill+h} - P) / mid_fill * 1e4 with side_sign = +1 for a
bid (bought at P) and -1 for an ask (sold at P); mid_{fill+h} is the last mid at or before fill + h.
"""
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd

RTH0, RTH1 = 34200.0, 57600.0
SCALE = 10000.0
MAX_WAIT = 60.0
HORIZONS = (1, 10, 60, 300)


def read_messages(path):
    df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3, 4, 5], names=["t", "ty", "oid", "sz", "px", "dr"])
    return df


def simulate(messages, triggers, size=100, max_wait=MAX_WAIT):
    """messages: DataFrame t, ty, oid, sz, px (int ticks), dr. triggers: DataFrame with columns tau, side (+1 bid,
    -1 ask), label. Returns (per-trigger results DataFrame, bbo arrays (times, bid, ask) in ticks)."""
    t = messages.t.to_numpy(float); ty = messages.ty.to_numpy(int); oid = messages.oid.to_numpy(np.int64)
    sz = messages.sz.to_numpy(np.int64); px = messages.px.to_numpy(np.int64); dr = messages.dr.to_numpy(int)
    trig = triggers.sort_values("tau", kind="stable").reset_index(drop=True)
    tau = trig.tau.to_numpy(float); tside = trig.side.to_numpy(int)
    n_trig = len(trig)
    res = dict(posted=np.zeros(n_trig, bool), price=np.full(n_trig, -1, np.int64), ahead0=np.zeros(n_trig, np.int64),
               filled=np.zeros(n_trig, bool), fill_time=np.full(n_trig, np.nan), end_time=np.full(n_trig, np.nan),
               exit_reason=np.array([""] * n_trig, dtype=object), spread_ticks=np.full(n_trig, -1, np.int64))
    orders = {}                                            # oid -> [side, price, remaining]
    levels = {1: defaultdict(int), -1: defaultdict(int)}   # price -> displayed size
    level_ids = {1: defaultdict(set), -1: defaultdict(set)}
    best = {1: 0, -1: 1 << 62}
    active = {1: defaultdict(list), -1: defaultdict(list)}  # price -> list of trigger idx
    ahead_ids = {}; ahead_rem = {}
    expiry = []                                              # heap of (time, idx)
    bbo_t, bbo_b, bbo_a = [], [], []

    def rescan(side):
        lv = levels[side]
        if side == 1:
            best[1] = max((p for p, v in lv.items() if v > 0), default=0)
        else:
            best[-1] = min((p for p, v in lv.items() if v > 0), default=1 << 62)

    def close(i, when, reason, filled=False):
        side, price = tside[i], res["price"][i]
        lst = active[side].get(price)
        if lst is not None and i in lst:
            lst.remove(i)
            if not lst:
                del active[side][price]
        res["end_time"][i] = when; res["exit_reason"][i] = reason
        if filled:
            res["filled"][i] = True; res["fill_time"][i] = when
        ahead_ids.pop(i, None); ahead_rem.pop(i, None)

    n = len(t); j = 0; k = 0
    while j < n or k < n_trig:
        next_t = t[j] if j < n else np.inf
        # post triggers strictly after all messages at their timestamp
        if k < n_trig and tau[k] < next_t:
            i = k; k += 1
            side = tside[i]
            if not (0 < best[1] < best[-1] < (1 << 62)):
                res["exit_reason"][i] = "no_quote"; continue
            price = best[1] if side == 1 else best[-1]
            res["posted"][i] = True; res["price"][i] = price; res["spread_ticks"][i] = best[-1] - best[1]
            ids = set(level_ids[side][price])
            ahead_ids[i] = ids
            ahead_rem[i] = int(sum(orders[o][2] for o in ids))
            res["ahead0"][i] = ahead_rem[i]
            active[side][price].append(i)
            heapq.heappush(expiry, (tau[i] + max_wait, i))
            continue
        tj = t[j]
        while expiry and expiry[0][0] <= tj:
            when, i = heapq.heappop(expiry)
            if res["posted"][i] and np.isnan(res["end_time"][i]):
                close(i, when, "timeout")
        typ = ty[j]; o = oid[j]; s = sz[j]; p = px[j]; d = dr[j]
        prev_best = (best[1], best[-1])
        if typ == 1:
            orders[o] = [d, p, s]; levels[d][p] += s; level_ids[d][p].add(o)
            if (d == 1 and p > best[1]) or (d == -1 and p < best[-1]):
                best[d] = p
        elif typ in (2, 3, 4):
            rec = orders.get(o)
            if rec is not None:
                side, price, rem = rec
                red = rem if typ == 3 else min(s, rem)
                if typ == 4:
                    # trade-through: an execution beyond an active order's price fills it (price priority)
                    through = [q for q in active[side] if (side == 1 and q > price) or (side == -1 and q < price)]
                    for q in through:
                        for i in list(active[side][q]):
                            close(i, tj, "filled_through", filled=True)
                if typ == 4 and price in active[side]:
                    for i in list(active[side][price]):
                        if o in ahead_ids[i]:
                            ahead_rem[i] -= red
                        else:
                            close(i, tj, "filled", filled=True)
                elif typ in (2, 3) and price in active[side]:
                    for i in active[side][price]:
                        if o in ahead_ids[i]:
                            ahead_rem[i] -= red
                rec[2] -= red; levels[side][price] -= red
                if rec[2] <= 0:
                    del orders[o]; level_ids[side][price].discard(o)
                if levels[side][price] <= 0:
                    del levels[side][price]; level_ids[side].pop(price, None)
                    if price in active[side] and typ in (2, 3):
                        for i in list(active[side][price]):
                            close(i, tj, "level_emptied")
                    if (side == 1 and price >= best[1]) or (side == -1 and price <= best[-1]):
                        rescan(side)
        # a better price on an active order's own side: cancel orders behind the new touch
        for side in (1, -1):
            if best[side] != prev_best[0 if side == 1 else 1]:
                worse = [p for p in active[side] if (side == 1 and p < best[side]) or (side == -1 and p > best[side])]
                for p in worse:
                    for i in list(active[side][p]):
                        close(i, tj, "improved_away")
        if 0 < best[1] < best[-1] < (1 << 62):
            if not bbo_t or (bbo_b[-1], bbo_a[-1]) != (best[1], best[-1]):
                bbo_t.append(tj); bbo_b.append(best[1]); bbo_a.append(best[-1])
        j += 1
    while expiry:
        when, i = heapq.heappop(expiry)
        if res["posted"][i] and np.isnan(res["end_time"][i]):
            close(i, when, "timeout")
    out = trig.copy()
    for key, val in res.items():
        out[key] = val
    return out, (np.array(bbo_t), np.array(bbo_b, float), np.array(bbo_a, float))


def markouts(results, bbo):
    bt, bb, ba = bbo
    mid = (bb + ba) / 2
    out = results.copy()
    for h in HORIZONS:
        val = np.full(len(out), np.nan)
        f = out.filled.to_numpy(bool)
        if f.any() and len(bt):
            ft = out.fill_time.to_numpy(float)[f]
            i0 = np.searchsorted(bt, ft, side="right") - 1
            ih = np.searchsorted(bt, ft + h, side="right") - 1
            ok = (i0 >= 0) & (ih >= 0) & (ft + h <= RTH1)
            m0 = np.where(ok, mid[np.maximum(i0, 0)], np.nan); mh = np.where(ok, mid[np.maximum(ih, 0)], np.nan)
            price = out.price.to_numpy(float)[f]
            sgn = out.side.to_numpy(int)[f]
            val[f] = sgn * (mh - price) / m0 * 1e4
        out["markout_%ds_bps" % h] = val
    return out
