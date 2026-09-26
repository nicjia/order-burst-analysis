#!/usr/bin/env python3
"""burst-defs-v1 stage 1 (cluster): alternative burst definitions from the p4-revisit-v1 trade bursts, decided in
real time (first minute mark after the episode ends + 1 s), with intraday exits read off the minute mid grid.

Definitions: run60 (each original burst); merge5m / merge30m (consecutive same-side bursts in time order joined when
the gap from one's end to the next's start is <= 300 / 1800 s; an opposite-side burst breaks the chain);
fpchain (bursts with the same side AND the same repeated non-round modal clip joined within 1800 s, whatever
happens in between -- the metaorder-linkage definition). Episodes are sampled at random (seeded), up to CAP per
definition per name-day, so the sample never depends on outcomes or on later bursts. Early-close days excluded.
Usage: burst_defs_extract.py OUTDIR CELLDIR
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import sys, os, zlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

RTH0, CAP = 34200.0, 4
EARLY = set()
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import p4_aggregate as AG
    EARLY = set(AG.EARLY_CLOSE)
except Exception:
    pass


def chains(tb, te, side, key, gap, by_key):
    order = np.argsort(tb, kind="stable")
    ep = np.full(len(tb), -1); nxt = 0
    last = {}
    prev_side, prev_end, cur = None, -1e9, -1
    for i in order:
        if by_key:
            k = (side[i], key[i])
            if key[i] < 0:
                continue
            if k in last and tb[i] - last[k][1] <= gap:
                ep[i] = last[k][0]; last[k] = (last[k][0], max(last[k][1], te[i]))
            else:
                ep[i] = nxt; last[k] = (nxt, te[i]); nxt += 1
        else:
            if side[i] == prev_side and tb[i] - prev_end <= gap:
                ep[i] = cur; prev_end = max(prev_end, te[i])
            else:
                cur = nxt; nxt += 1; ep[i] = cur; prev_side = side[i]; prev_end = te[i]
    return ep


def one_permno(pdir):
    rows = []
    permno = int(pdir.name)
    for f in sorted(pdir.glob("*.npz")):
        date = f.stem
        if date in EARLY:
            continue
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "T_t_b" not in z.files or "grid_mid" not in z.files:
            continue
        g = z["grid_mid"].astype(float)
        tb, te = z["T_t_b"].astype(float), z["T_t_e"].astype(float)
        if len(tb) == 0 or not np.isfinite(g).sum() > 300:
            continue
        side = z["T_side"].astype(int); n = z["T_n"].astype(float); vol = z["T_vol"].astype(float)
        ms, mc = z["T_mode_size"].astype(float), z["T_mode_count"].astype(float)
        fp = (mc >= 2) & np.isfinite(ms) & (np.nan_to_num(ms) % 100 != 0)
        key = np.where(fp, np.nan_to_num(ms), -1).astype(np.int64)
        hid, trc = z["T_hidden_share"].astype(float), z["T_truncated_share"].astype(float)
        ps, spb = z["T_program_score"].astype(float), z["T_spread_b"].astype(float)
        defs = {"run60": np.arange(len(tb)), "merge5m": chains(tb, te, side, key, 300, False),
                "merge30m": chains(tb, te, side, key, 1800, False), "fpchain": chains(tb, te, side, key, 1800, True)}
        rng = np.random.default_rng(zlib.crc32(("%d|%s" % (permno, date)).encode()))
        for dn, ep in defs.items():
            ids = np.unique(ep[ep >= 0])
            if not len(ids):
                continue
            pick = rng.choice(ids, size=min(CAP, len(ids)), replace=False)
            for e in pick:
                m = ep == e
                s = side[m][0]; b0 = tb[m].min(); e1 = te[m].max()
                kdec = int((e1 + 1 - RTH0) // 60)            # grid index of first minute mark after t_e + 1 s
                kpre = int((b0 - RTH0) // 60) - 1            # last minute mark at or before t_b
                if kdec > 388 or kpre < 0:
                    continue
                md = g[kdec]
                if not np.isfinite(md) or md <= 0:
                    continue
                def mv(k):
                    return s * (g[k] - md) / md * 1e4 if 0 <= k <= 389 and np.isfinite(g[k]) else np.nan
                pre30 = s * (g[kpre] - g[max(kpre - 30, 0)]) / g[max(kpre - 30, 0)] * 1e4 if kpre >= 1 else np.nan
                rows.append((dn, permno, date, s, b0, e1, int(m.sum()), n[m].sum(), vol[m].sum(),
                             float(np.nanmax(mc[m])) if np.isfinite(mc[m]).any() else 0.0, float(fp[m].mean()),
                             float(np.nanmean(hid[m])), float(np.nanmean(trc[m])), float(np.nanmax(ps[m])) if np.isfinite(ps[m]).any() else np.nan,
                             float(spb[m][0]), (b0 - RTH0) / 23400.0, kdec,
                             s * (md - g[kpre]) / g[kpre] * 1e4, pre30, s * (md - g[0]) / g[0] * 1e4,
                             mv(kdec + 5), mv(kdec + 30), mv(kdec + 60), mv(389)))
    return rows


COLS = ["defn", "permno", "date", "side", "t_b", "t_e", "nb", "nch", "vol", "fp_count", "fp_share", "hidden",
        "trunc", "pscore", "spread_b", "tod", "kdec", "so_far_bps", "pre30_bps", "open_dec_bps", "r5", "r30", "r60", "rclose"]

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cd = Path(sys.argv[2]); cell = cd.name
    dirs = sorted(p for p in cd.iterdir() if p.is_dir() and p.name.isdigit())
    allrows = []
    with Pool(int(os.environ.get("NSLOTS", "8"))) as pool:
        for r in pool.imap_unordered(one_permno, dirs, chunksize=2):
            allrows.extend(r)
    pd.DataFrame(allrows, columns=COLS).to_csv(out / ("%s_defs.csv.gz" % cell), index=False)
    print(cell, "episodes", len(allrows), flush=True)
