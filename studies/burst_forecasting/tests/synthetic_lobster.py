#!/usr/bin/env python3
"""A synthetic LOBSTER message day for testing the burst extractors without licensed data.

A small price-time-priority matching engine emits LOBSTER rows (time, type, order id, size, price x 10000, direction):
passive adds (1), partial cancels (2), deletions (3), executions of resting visible orders (4, direction of the resting
order), and occasional hidden executions (5: order id 0, direction +1, as in the real files). Marketable orders arrive
alone or in planted bursts of 3-8 same-side orders 10-400 ms apart. Every order is created inside the file, so a book
rebuilt from these messages must equal the quote path exactly. Nothing here is estimated from market data.
"""
import numpy as np

TICK = 100


def synth_day(path, seed=0, rate=6.0, t0=32400.0, t1=57660.0):
    rng = np.random.default_rng(seed)
    book = {1: {}, -1: {}}                       # side -> price -> list of [oid, size] (FIFO)
    oid = [0]
    rows = []

    def emit(t, ty, o, sz, px, dr):
        rows.append((t, ty, o, sz, px, dr))

    def best(side):
        lv = book[side]
        if not lv:
            return None
        return max(lv) if side == 1 else min(lv)

    def add(t, side, px, sz):
        oid[0] += 1
        book[side].setdefault(px, []).append([oid[0], sz])
        emit(t, 1, oid[0], sz, px, side)

    def seed_levels(t, mid_i):
        for k in range(1, 11):
            for _ in range(rng.integers(1, 4)):
                add(t, 1, mid_i - k * TICK, int(rng.integers(1, 6)) * 100)
                add(t, -1, mid_i + k * TICK, int(rng.integers(1, 6)) * 100)

    def market(t, side, qty):
        """An aggressive order of `side` (+1 buy) walks the opposite book; one row per resting order it touches."""
        opp = -side
        while qty > 0 and book[opp]:
            px = best(opp)
            q = book[opp][px]
            o, s = q[0]
            take = min(s, qty)
            emit(t, 4, o, take, px, opp)
            qty -= take
            if take == s:
                q.pop(0)
                if not q:
                    del book[opp][px]
            else:
                q[0][1] = s - take
        if qty > 0 and rng.random() < 0.3:        # an unseen hidden remainder
            emit(t, 5, 0, int(qty), best(opp) or 0, 1)

    seed_levels(t0, 500000)                       # $50.00
    t = t0 + 1.0
    while t < t1:
        t += rng.exponential(1.0 / rate)
        bb, ba = best(1), best(-1)
        if bb is None or ba is None or len(book[1]) < 4 or len(book[-1]) < 4:
            ref = bb if bb is not None else (ba - TICK if ba is not None else 500000 - TICK)
            seed_levels(t, ref + TICK if bb is not None else ref)
            continue
        u = rng.random()
        if u < 0.47:                               # passive add, never crossing
            side = 1 if rng.random() < 0.5 else -1
            k = int(rng.geometric(0.45)) - 1
            px = (min(bb + TICK, ba - TICK) - k * TICK) if side == 1 else (max(ba - TICK, bb + TICK) + k * TICK)
            if side == 1 and px >= ba or side == -1 and px <= bb:
                px = bb if side == 1 else ba
            add(t, side, px, int(rng.integers(1, 6)) * 100)
        elif u < 0.82:                             # cancel a random resting order
            side = 1 if rng.random() < 0.5 else -1
            lv = book[side]
            px = list(lv)[rng.integers(len(lv))]
            q = lv[px]; i = int(rng.integers(len(q))); o, s = q[i]
            if s >= 200 and rng.random() < 0.3:
                cut = int(rng.integers(1, s // 100)) * 100
                q[i][1] = s - cut
                emit(t, 2, o, cut, px, side)
            else:
                q.pop(i)
                if not q:
                    del lv[px]
                emit(t, 3, o, s, px, side)
        elif u < 0.93:                             # a single marketable order
            market(t, 1 if rng.random() < 0.5 else -1, int(rng.integers(1, 8)) * 100)
        elif u < 0.95:                             # hidden-heavy orders: a small visible fill signs the packet, most volume hidden
            side = 1 if rng.random() < 0.5 else -1
            for j in range(int(rng.integers(1, 4))):
                if j:
                    t += rng.uniform(0.05, 0.6)
                px = best(-side)
                if px is None:
                    break
                market(t, side, 100)
                emit(t, 5, 0, int(rng.integers(2, 6)) * 100, px, 1)
        elif u < 0.97:                             # a planted cancellation burst at one touch
            side = 1 if rng.random() < 0.5 else -1
            for j in range(int(rng.integers(5, 9))):
                t += rng.uniform(0.01, 0.3)
                px = best(side)
                if px is None or len(book[side]) < 3:
                    break
                o, sz = book[side][px].pop(0)
                if not book[side][px]:
                    del book[side][px]
                emit(t, 3, o, sz, px, side)
        else:                                      # a planted burst (half with 1-9 ms gaps)
            side = 1 if rng.random() < 0.5 else -1
            clip = int(rng.integers(1, 4)) * 100 + (37 if rng.random() < 0.5 else 0)
            fast = rng.random() < 0.5
            for j in range(int(rng.integers(3, 9))):
                if j:
                    t += rng.uniform(0.001, 0.009) if fast else rng.uniform(0.01, 0.4)
                market(t, side, clip)
    with open(path, "w") as fh:
        for r in rows:
            fh.write("%.9f,%d,%d,%d,%d,%d\n" % r)
    return len(rows)


if __name__ == "__main__":
    import sys
    print(synth_day(sys.argv[1], seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0))
