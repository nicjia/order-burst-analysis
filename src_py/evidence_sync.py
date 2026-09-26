#!/usr/bin/env python3
"""program-evidence-v1 module G: cross-name synchrony of untruncated non-round packets (one date).

For packets i, j from different names, counts ordered pairs with t_j - t_i - o in [-delta, delta)
for offsets o = 0 and +-0.731, +-2.371, +-7.129 s (non-integer, so whole-second clocks do not
enter the baseline) and delta = 0.1, 1, 10 ms. Pooled counts by side relation and by the program
class of each packet (program = run/60 burst with score >= 2024 q80); name-pair counts at 1 ms by the
first packet's program class (so program-versus-other contrasts can be bootstrapped over names).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS
import program_bursts as PB

OFFSETS = np.array([0.0, 0.731, -0.731, 2.371, -2.371, 7.129, -7.129])
DELTAS = np.array([1e-4, 1e-3, 1e-2])
CHUNK = 2_000_000


def name_packets(day, model):
    t = day["time"].astype(float); sign = day["sign"].astype(int)
    size, masks = FS.size_class_masks(day, np.array([], np.int64))
    member, table = PB.run_bursts(day)
    prog = np.zeros(len(t), bool)
    if table:
        s = PB.score(table, model)
        good = np.isfinite(s) & (s >= model["program_threshold_q80"])
        inb = member >= 0
        prog[inb] = good[member[inb]]
    u = masks["u_nonround"]
    return t[u], sign[u], prog[u]


def sync_counts(T, NAME, SIGN, PROG, n_names):
    order = np.argsort(T, kind="stable")
    T, NAME, SIGN, PROG = T[order], NAME[order], SIGN[order], PROG[order]
    pooled = np.zeros((len(DELTAS), len(OFFSETS), 2, 2, 2), np.int64)      # [delta, offset, rel, prog_i, prog_j]
    pair = np.zeros((len(OFFSETS), 2, 2, n_names, n_names), np.int64)       # delta = 1 ms, [offset, rel, prog_i, i, j]
    n = len(T)
    for di, delta in enumerate(DELTAS):
        for oi, o in enumerate(OFFSETS):
            lo = np.searchsorted(T, T + o - delta, side="left")
            hi = np.searchsorted(T, T + o + delta, side="left")
            cnt = hi - lo
            for a in range(0, n, CHUNK):
                b = min(n, a + CHUNK)
                c = cnt[a:b]; tot = int(c.sum())
                if tot == 0:
                    continue
                ii = np.repeat(np.arange(a, b), c)
                offs = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
                jj = np.repeat(lo[a:b], c) + offs
                keep = NAME[ii] != NAME[jj]
                ii, jj = ii[keep], jj[keep]
                rel = (SIGN[ii] != SIGN[jj]).astype(np.int64)
                flat = (rel * 2 + PROG[ii].astype(np.int64)) * 2 + PROG[jj].astype(np.int64)
                pooled[di, oi] += np.bincount(flat, minlength=8).reshape(2, 2, 2)
                if abs(delta - 1e-3) < 1e-12:
                    flat2 = ((rel * 2 + PROG[ii].astype(np.int64)) * n_names + NAME[ii]) * n_names + NAME[jj]
                    pair[oi] += np.bincount(flat2, minlength=4 * n_names * n_names).reshape(2, 2, n_names, n_names)
    return pooled, pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="fingerprint-v1 group directory")
    ap.add_argument("--date", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = PB.load_model(args.model)
    names = [x for x in (Path(args.group) / "universe.txt").read_text().split() if x]
    T, NAME, SIGN, PROG, used = [], [], [], [], []
    for tk in names:
        path = Path(args.group) / "packets" / tk / (args.date + ".npz")
        if not path.is_file():
            continue
        with np.load(path) as z:
            if "time" not in z.files or len(z["time"]) == 0:
                continue
            day = {k: z[k] for k in z.files}
        t, s, p = name_packets(day, model)
        T.append(t); SIGN.append(s); PROG.append(p); NAME.append(np.full(len(t), len(used))); used.append(tk)
    T = np.concatenate(T); NAME = np.concatenate(NAME); SIGN = np.concatenate(SIGN); PROG = np.concatenate(PROG)
    pooled, pair = sync_counts(T, NAME, SIGN, PROG, len(used))
    n_by_name = np.bincount(NAME, minlength=len(used)); prog_by_name = np.bincount(NAME, weights=PROG, minlength=len(used))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, date=np.array(args.date), names=np.array(used), offsets=OFFSETS, deltas=DELTAS,
                        pooled=pooled, pair_1ms=pair, n_packets=n_by_name, n_program=prog_by_name)
    tmp.rename(out)
    print(json.dumps({"date": args.date, "names": len(used), "packets": int(len(T))}))


if __name__ == "__main__":
    main()
