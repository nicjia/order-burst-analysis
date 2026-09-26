#!/usr/bin/env python3
"""Random-time benchmark for the intraday reversal forecast. Per name-day (p4-revisit-v1 minute grids), 5 random
minute marks between 10:00 and 15:30 (seeded): x = log mid move open(9:31) -> t, y = log mid move t -> 16:00,
and the same for the equal-weight market index, so both raw and market-excess versions can be computed.
Usage: OUTDIR CELLDIR MKT.npz"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import sys, os, zlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

MKT = {}


def one(pdir):
    rows = []; permno = int(pdir.name)
    for f in sorted(pdir.glob("*.npz")):
        d = f.stem
        if d not in MKT:
            continue
        try:
            g = np.load(f, allow_pickle=True)["grid_mid"].astype(float)
        except Exception:
            continue
        if len(g) != 390 or not np.isfinite(g[[0, 389]]).all():
            continue
        rng = np.random.default_rng(zlib.crc32(("%d|%s" % (permno, d)).encode()))
        m = MKT[d]
        for k in rng.integers(29, 359, size=5):
            if np.isfinite(g[k]) and g[k] > 0:
                rows.append((permno, d, int(k), np.log(g[k] / g[0]) * 1e4, np.log(g[389] / g[k]) * 1e4,
                             (m[k] - m[0]) * 1e4, (m[389] - m[k]) * 1e4))
    return rows


if __name__ == "__main__":
    out, cd, mk = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    z = np.load(mk); MKT.update(dict(zip(z["dates"], z["idx"])))
    dirs = sorted(p for p in cd.iterdir() if p.is_dir() and p.name.isdigit())
    with Pool(int(os.environ.get("NSLOTS", "8"))) as pool:
        rows = [r for rs in pool.imap_unordered(one, dirs, chunksize=4) for r in rs]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["permno", "date", "k", "x_bps", "y_bps", "mx_bps", "my_bps"]).to_csv(out / ("%s_random.csv.gz" % cd.name), index=False)
    print(cd.name, len(rows))
