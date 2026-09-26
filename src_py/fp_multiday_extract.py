#!/usr/bin/env python3
"""fingerprint-multiday-v1, stage 1 (cluster): compact per-name-day fingerprint tables from p4-revisit-v1 npz files.

For every trade burst whose modal untruncated child size repeats inside the burst (mode_count >= 2), aggregate by
(permno, date, side, mode_size): number of such bursts, their executed volume and child count. Also one row per
name-day with totals over all trade bursts. No prices or outcomes are read; this is flow structure only.
Usage: fp_multiday_extract.py OUTDIR CELL_DIR [CELL_DIR ...]   (writes OUTDIR/<cell>_fp.csv.gz, <cell>_nd.csv.gz)
"""
import sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd


def one_permno(pdir):
    rows, nd = [], []
    permno = int(pdir.name)
    for f in sorted(pdir.glob("*.npz")):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "T_side" not in z.files:
            continue
        side = z["T_side"].astype(np.int64); vol = z["T_vol"].astype(float); n = z["T_n"].astype(float)
        ms = z["T_mode_size"].astype(float); mc = z["T_mode_count"].astype(float)
        date = f.stem
        nd.append((permno, date, len(side), float(vol.sum()), float((side * vol).sum())))
        k = (mc >= 2) & np.isfinite(ms) & (ms > 0)
        if not k.any():
            continue
        d = pd.DataFrame(dict(side=side[k], size=ms[k].astype(np.int64), vol=vol[k], n=n[k]))
        g = d.groupby(["side", "size"]).agg(nb=("vol", "size"), vol=("vol", "sum"), nch=("n", "sum")).reset_index()
        g.insert(0, "date", date); g.insert(0, "permno", permno)
        rows.append(g)
    fp = pd.concat(rows, ignore_index=True) if rows else None
    return fp, nd


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for cell_dir in sys.argv[2:]:
        cell = Path(cell_dir).name
        dirs = sorted(p for p in Path(cell_dir).iterdir() if p.is_dir() and p.name.isdigit())
        fps, nds = [], []
        with Pool(int(os.environ.get("NSLOTS", "8"))) as pool:
            for fp, nd in pool.imap_unordered(one_permno, dirs, chunksize=2):
                if fp is not None:
                    fps.append(fp)
                nds.extend(nd)
        pd.concat(fps, ignore_index=True).to_csv(out / ("%s_fp.csv.gz" % cell), index=False)
        pd.DataFrame(nds, columns=["permno", "date", "n_bursts", "vol", "signed_vol"]).to_csv(
            out / ("%s_nd.csv.gz" % cell), index=False)
        print(cell, "names", len(dirs), "name-days", len(nds), flush=True)


if __name__ == "__main__":
    main()
