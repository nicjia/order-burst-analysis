#!/usr/bin/env python3
"""Metaorder-v1 M3: inject synthetic parents into real cached packet days and measure detection (one ticker).

Generator (METAORDER_DESIGN.md, with the participation cap recorded in its amendments): six parents per
name-day; side +-1; start U[10:00, 15:00]; duration {600, 1800, 3600} s; interval {5, 15, 30, 60} s;
timing timer-locked (whole-second anchor, |N(0, 2 ms)|) or jittered (N(0, 0.1 interval)); clip a fixed
non-round size in [101, 999] or that clip x U(0.8, 1.2); children capped below the prevailing same-side
executed depth; child count capped so a parent is at most 1% of the day's signed packet volume.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fingerprint_stats as FS
import metaorder_features as MF

RULES = {"run60": ("run", 60.0), "stream5": ("stream", 5.0)}
FIELDS = ("time", "sign", "volume", "untruncated", "mid", "spread_bps", "exec_depth", "imbalance", "hidden_share", "n_messages")


def nonround(x):
    x = np.maximum(np.rint(x).astype(np.int64), 1)
    return np.where(x % 100 == 0, x + 1, x)


def make_parents(day, rng, n_parents=6):
    t = day["time"].astype(float); sign = day["sign"].astype(int)
    vday = float(day["volume"][sign != 0].sum())
    parents, kids = [], []
    for p in range(n_parents):
        side = int(rng.choice([-1, 1])); start = rng.uniform(36000, 54000)
        dur = float(rng.choice([600, 1800, 3600])); dt = float(rng.choice([5, 15, 30, 60]))
        locked = bool(rng.random() < 0.5); fixed = bool(rng.random() < 0.5)
        clip = int(nonround(rng.integers(101, 1000)))
        n = int(dur // dt)
        n = min(n, int(0.01 * vday // clip))
        if n < 4:
            parents.append(dict(parent=p, side=side, start=start, duration=dur, interval=dt, locked=locked, fixed=fixed,
                                clip=clip, n_children=0))
            continue
        k = np.arange(n)
        if locked:
            tt = np.floor(start) + k * dt + np.abs(rng.normal(0, 0.002, n))
        else:
            tt = start + k * dt + rng.normal(0, 0.1 * dt, n)
        sizes = np.full(n, clip) if fixed else nonround(clip * rng.uniform(0.8, 1.2, n))
        tt = np.clip(tt, FS.RTH0 + 1, FS.RTH0 + 23399)
        # prevailing quote state: nearest preceding real packet (any side); depth from the nearest same-side packet
        pos = np.clip(np.searchsorted(t, tt, "right") - 1, 0, len(t) - 1)
        same_idx = np.flatnonzero(sign == side)
        if not len(same_idx):
            continue
        sp = np.clip(np.searchsorted(t[same_idx], tt, "right") - 1, 0, len(same_idx) - 1)
        depth = day["exec_depth"][same_idx[sp]].astype(float)
        cap = np.where(np.isfinite(depth), depth - 1, sizes)
        sizes = np.minimum(sizes, np.maximum(cap, 1)).astype(np.int64)
        ok = sizes >= 1
        kids.append(dict(time=tt[ok], sign=np.full(ok.sum(), side, np.int8), volume=sizes[ok].astype(float),
                         untruncated=np.ones(ok.sum(), bool), mid=day["mid"][pos[ok]], spread_bps=day["spread_bps"][pos[ok]],
                         exec_depth=depth[ok], imbalance=day["imbalance"][pos[ok]], hidden_share=np.zeros(ok.sum()),
                         n_messages=np.ones(ok.sum(), np.int32), parent=np.full(ok.sum(), p)))
        parents.append(dict(parent=p, side=side, start=start, duration=dur, interval=dt, locked=locked, fixed=fixed,
                            clip=clip, n_children=int(ok.sum())))
    return parents, kids


def inject(day, kids):
    out = {f: np.asarray(day[f]) for f in FIELDS}
    parent = np.full(len(out["time"]), -1)
    for kd in kids:
        for f in FIELDS:
            out[f] = np.r_[out[f], kd[f]]
        parent = np.r_[parent, kd["parent"]]
    order = np.argsort(out["time"], kind="stable")
    return {f: v[order] for f, v in out.items()}, parent[order]


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def day_eval(day, rates, thr, rule, models, rng, date, ticker):
    parents, kids = make_parents(day, rng)
    inj, parent = inject(day, kids)
    r, g = RULES[rule]
    member, tab = MF.burst_table(inj, r, g)
    prow = []
    if not tab:
        return pd.DataFrame(parents).assign(date=date, ticker=ticker), dict()
    dbin = FS.depth_bins(inj["exec_depth"], thr)
    link = MF.link_evidence(inj, member, tab, dbin, rates)
    K = len(tab["side"])
    inb = member >= 0
    counts = np.zeros((K, 6))
    for pid in range(6):
        c = np.bincount(member[inb & (parent == pid)], minlength=K)
        counts[:, pid] = c
    tot = np.bincount(member[inb], minlength=K).astype(float)
    major = counts.argmax(1)
    maj_share = counts[np.arange(K), major] / np.maximum(tot, 1)
    injected_majority = maj_share >= 0.5
    scores = {name: MF.score(m, tab) for name, m in models.items()}
    size = np.rint(inj["volume"]).astype(np.int64)
    for p in parents:
        pid = p["parent"]
        mine = parent == pid
        p = dict(p)
        p.update(date=date, ticker=ticker)
        if p["n_children"] == 0:
            prow.append(p); continue
        in_b = mine & inb
        p["children_in_bursts"] = int(in_b.sum())
        bursts = np.unique(member[in_b])
        p["bursts_with_children"] = int(len(bursts))
        p["mean_purity"] = float(np.mean(counts[bursts, pid] / np.maximum(tot[bursts], 1))) if len(bursts) else np.nan
        own_b = bursts[(major[bursts] == pid) & injected_majority[bursts]]
        p["majority_bursts"] = int(len(own_b))
        linked = 0
        for kb in own_b:
            e = tab["end"][kb]
            sizes_k = np.unique(size[(member == kb) & mine])
            later = mine & inb & (member != kb) & (inj["time"] > e) & (inj["time"] <= e + MF.LINK_WINDOW)
            if np.isin(size[later], sizes_k).any():
                linked += 1
        p["majority_bursts_linked_same_parent"] = int(linked)
        for name, s in scores.items():
            p["mean_score_%s" % name] = float(np.nanmean(s[own_b])) if len(own_b) else np.nan
        prow.append(p)
    burst_summary = dict(bursts=int(K), injected_majority=int(injected_majority.sum()),
                         link_matches_injected=float(link[injected_majority, 0, 1].sum()),
                         link_expected_injected=float(np.nansum(link[injected_majority, 0, 2])),
                         link_matches_background=float(link[~injected_majority, 0, 1].sum()),
                         link_expected_background=float(np.nansum(link[~injected_majority, 0, 2])))
    for name, s in scores.items():
        burst_summary["auc_%s" % name] = auc(s, injected_majority)
        burst_summary["_scores_%s" % name] = s.tolist()
    burst_summary["_injected_majority"] = injected_majority.astype(int).tolist()
    return pd.DataFrame(prow), burst_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--rule", required=True, choices=list(RULES))
    ap.add_argument("--models", nargs="+", required=True, help="model JSON files (program, link)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = {}
    for path in args.models:
        m = json.loads(Path(path).read_text()); models[m["kind"]] = m
    pairs = [tuple(x.split()) for x in Path(args.pairs).read_text().splitlines() if x.strip()]
    days = FS.load_days(args.packets, [d for p in pairs for d in p])
    frames, summaries = [], []
    for a, b in pairs:
        if a not in days or b not in days:
            continue
        pool = np.concatenate([days[x]["exec_depth"][days[x]["untruncated"].astype(bool) & (days[x]["sign"] != 0)] for x in (a, b)])
        pool = pool[np.isfinite(pool)]
        thr = np.quantile(pool, FS.DEPTH_QUANTILES) if len(pool) else np.array([np.inf])
        rates = MF.depth_rates((days[a], days[b]), thr)
        seed = int.from_bytes(hashlib.sha256(("metaorder-v1|M3|%s|%s" % (args.ticker, a)).encode()).digest()[:4], "little")
        f, s = day_eval(days[a], rates, thr, args.rule, models, np.random.default_rng(seed), a, args.ticker)
        frames.append(f); s.update(date=a, ticker=args.ticker); summaries.append(s)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    out.with_name(out.stem + ".bursts.json").write_text(json.dumps(summaries) + "\n")
    print(json.dumps({"ticker": args.ticker, "days": len(summaries)}))


if __name__ == "__main__":
    main()
