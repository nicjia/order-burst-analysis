#!/usr/bin/env python3
"""Equal-weight intraday market index per date from the p4-revisit-v1 minute mid grids (for market-excess returns).
idx[d, k] = cumulative mean across names of log(mid[k] / mid[k-1]), k = 0..389 (grid marks 9:31 .. 16:00),
idx[d, 0] = 0. Output: OUTDIR/<cell>_mkt.npz with dates and the 390-column matrix. Usage: OUTDIR CELLDIR"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import sys, os
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def one(files):
    rows = []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "grid_mid" in z.files:
            g = z["grid_mid"].astype(float)
            if len(g) == 390 and np.isfinite(g).sum() > 300:
                rows.append(g)
    if len(rows) < 30:
        return None
    M = np.vstack(rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.diff(np.log(M), axis=1)
    R[~np.isfinite(R) | (np.abs(R) > 0.05)] = np.nan
    m = np.nanmean(R, axis=0)
    return np.r_[0.0, np.nancumsum(m)], len(rows)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cd = Path(sys.argv[2]); per = defaultdict(list)
    for p in cd.iterdir():
        if p.is_dir() and p.name.isdigit():
            for f in p.glob("*.npz"):
                per[f.stem].append(f)
    dates = sorted(per)
    with Pool(int(os.environ.get("NSLOTS", "8"))) as pool:
        res = pool.map(one, [per[d] for d in dates], chunksize=1)
    keep = [(d, r) for d, r in zip(dates, res) if r is not None]
    np.savez_compressed(out / ("%s_mkt.npz" % cd.name), dates=np.array([d for d, _ in keep]),
                        idx=np.vstack([r[0] for _, r in keep]), n=np.array([r[1] for _, r in keep]))
    print(cd.name, "dates", len(keep), flush=True)
