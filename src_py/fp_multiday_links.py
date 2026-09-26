#!/usr/bin/env python3
"""fingerprint-multiday-v1, amendment A1 link labels (fixed on DEV, before TEST H2).

For name i, day t, key (side, size), primary sizes (not a multiple of 100):
  rare(k)      : the size appeared (either side) on at most k of the trading days t-21 .. t-2;
  same link    : (side, size) present on t-1 and t, and the size absent on the opposite side on both t-1 and t;
  mirror link  : (opposite side, size) present on t-1, size absent on `side` at t-1 and on the opposite side at t
                 (the same size switching side overnight -- symmetric coincidence, the placebo).
purity(k) = 1 - mirror volume / same volume.
"""
import numpy as np
import pandas as pd
import fp_multiday_h1 as H


def labels(cell, cal, k_rare):
    fp = pd.read_csv(H.T / ("%s_fp.csv.gz" % cell), dtype={"date": str})
    fp = fp[fp["size"] % 100 != 0].copy()
    fp["day"] = fp.date.map(cal).astype(int)
    pres = fp.groupby(["permno", "size", "day"]).side.agg(lambda s: (1 if (s == 1).any() else 0) + (2 if (s == -1).any() else 0)).rename("mask")
    pres = pres.reset_index()
    # rarity: number of distinct days in [t-21, t-2] on which the size appeared
    pres = pres.sort_values(["permno", "size", "day"])
    rare_cnt = np.zeros(len(pres), dtype=np.int64)
    for (_, _), idx in pres.groupby(["permno", "size"]).indices.items():
        d = pres.day.to_numpy()[idx]
        rare_cnt[idx] = np.searchsorted(d, d - 1, side="left") - np.searchsorted(d, d - 21, side="left")
    pres["prior_days"] = rare_cnt
    m = pres.set_index(["permno", "size", "day"])["mask"]
    fp = fp.merge(pres[["permno", "size", "day", "prior_days", "mask"]], on=["permno", "size", "day"])
    prev = m.rename("mask_prev").reset_index(); prev["day"] = prev.day + 1
    fp = fp.merge(prev, on=["permno", "size", "day"], how="left")
    fp["mask_prev"] = fp.mask_prev.fillna(0).astype(int)
    own = np.where(fp.side == 1, 1, 2); opp = 3 - own
    now_only_own = (fp["mask"] & opp) == 0
    prev_own = (fp.mask_prev & own) > 0; prev_opp = (fp.mask_prev & opp) > 0
    rare = fp.prior_days <= k_rare
    fp["same_link"] = rare & now_only_own & prev_own & ~prev_opp
    fp["mirror_link"] = rare & now_only_own & prev_opp & ~prev_own
    return fp


if __name__ == "__main__":
    import sys
    cell = sys.argv[1]; cal = H.calendar()
    for k in (0, 1, 2, 5, 20):
        f = labels(cell, cal, k)
        vs, vm = f.vol[f.same_link].sum(), f.vol[f.mirror_link].sum()
        print("k_rare %2d  same-link volume share %.4f  mirror share %.4f  purity %.3f  same-link rows %d"
              % (k, vs / f.vol.sum(), vm / f.vol.sum(), 1 - vm / vs if vs > 0 else np.nan, int(f.same_link.sum())))
