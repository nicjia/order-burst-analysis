#!/usr/bin/env python3
"""fingerprint-multiday-v1 amendment A1 diagnostic: link purity by repetition count (flow structure only)."""
import sys
import numpy as np
import pandas as pd
import fp_multiday_h1 as H


def keyed(cell, cal):
    fp = pd.read_csv(H.T / ("%s_fp.csv.gz" % cell), dtype={"date": str})
    fp = fp[fp["size"] % 100 != 0].copy()
    fp["day"] = fp.date.map(cal).astype(int)
    w = fp.pivot_table(index=["permno", "size", "day"], columns="side", values=["nb", "vol"], aggfunc="sum", fill_value=0)
    w.columns = ["%s_%s" % (a, "B" if b == 1 else "S") for a, b in w.columns]
    w = w.reset_index()
    prev = w.rename(columns={c: c + "_p" for c in ("nb_B", "nb_S", "vol_B", "vol_S")}); prev["day"] = prev.day + 1
    return w.merge(prev, on=["permno", "size", "day"], how="left").fillna(0)


def purity(w, m):
    """same: size one-sided today (side X) and yesterday on X only; mirror: yesterday on the other side only."""
    out = {}
    for side, o in (("B", "S"), ("S", "B")):
        today = (w["nb_" + side] >= m) & (w["nb_" + o] == 0)
        same = today & (w["nb_%s_p" % side] >= m) & (w["nb_%s_p" % o] == 0)
        mirror = today & (w["nb_%s_p" % o] >= m) & (w["nb_%s_p" % side] == 0)
        out[side] = (w["vol_" + side][same].sum(), w["vol_" + side][mirror].sum(), int(same.sum()))
    vs = out["B"][0] + out["S"][0]; vm = out["B"][1] + out["S"][1]
    return vs, vm, out["B"][2] + out["S"][2]


if __name__ == "__main__":
    cell = sys.argv[1]; cal = H.calendar(); w = keyed(cell, cal)
    tot = (w.vol_B + w.vol_S).sum()
    for m in (1, 2, 3, 5, 10, 20):
        vs, vm, n = purity(w, m)
        print("min bursts per day %2d  same-link vol share %.4f  mirror %.4f  purity %.3f  links %d" % (m, vs / tot, vm / tot, 1 - vm / vs if vs else np.nan, n))
