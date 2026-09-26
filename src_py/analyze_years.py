#!/usr/bin/env python3
"""program-evidence-v1 module J: fingerprint and burst descriptors across years; Tick Size Pilot DiD.

J1 (descriptive): per year, name-median C1 ratio (u_nonround, depth-matched cross-day null, 0.5-2 s
and 2-10 s), mean per-name J of run/60 (min 3), untruncated share, program-burst volume share.
Balanced version: names with >= 16 usable days in every year.
J2 (exploratory): pilot groups vs control, April-September 2016 vs November 2016-June 2017; name-level
changes in untruncated share, fingerprint ratio, run/60 J, half-spread at burst ends, 3-minute burst
markout, and the instrumented slope d(markout)/d(half-spread).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate_fingerprint as AF
import fingerprint_stats as FS

ROOT = Path(__file__).resolve().parents[1]
PE = ROOT / "results" / "program_evidence_v1"
FP = ROOT / "results" / "fingerprint_v1"
BOOT = 1000
UN = FS.CLASSES.index("u_nonround")


def name_measures(path):
    z = AF.load_name(path)
    if z is None:
        return None
    k = "cross_day_depth_matched"
    wm, we = z["within_matches_" + k][UN, 0], z["within_expected_" + k][UN, 0]     # [F]
    fine = AF.FINE
    r1 = [(fine[:-1] >= 0.5) & (fine[1:] <= 2), (fine[:-1] >= 2) & (fine[1:] <= 10)]
    out = dict(ticker=z["ticker"], days=z["days"],
               ratio_05_2=float(wm[r1[0]].sum() / we[r1[0]].sum()) if we[r1[0]].sum() > 0 else np.nan,
               ratio_2_10=float(wm[r1[1]].sum() / we[r1[1]].sum()) if we[r1[1]].sum() > 0 else np.nan,
               expected_2_10=float(we[r1[1]].sum()))
    # run/60, min 3, u_nonround J over lags < 300 s (burst bins)
    ri, gi, mi = FS.RULES.index("run"), FS.GAPS.index(60.0), FS.MINSIZES.index(3)
    bci = FS.BURST_CLASSES.index("u_nonround")
    lagm = AF.BURST[:-1] < AF.RECALL_LAG
    bm = z["burst_matches_" + k][ri, gi, mi, bci][lagm]; be = z["burst_expected_" + k][ri, gi, mi, bci][lagm]
    bp = z["burst_pairs_" + k][ri, gi, mi, bci][lagm]
    am = z["all_matches_b_" + k][bci][lagm]; ae = z["all_expected_b_" + k][bci][lagm]; ap = z["all_pairs_b_" + k][bci][lagm]
    x_in = np.clip(bm - be, 0, None).sum(); x_all = np.clip(am - ae, 0, None).sum()
    out["j_run60"] = float(x_in / x_all - bp.sum() / ap.sum()) if x_all > 0 and ap.sum() > 0 else np.nan
    out["expected_all"] = float(ae.sum()); out["excess_all"] = float(x_all)
    return out


def tape(group_dirs):
    frames = []
    for d in group_dirs:
        for p in glob.glob(str(Path(d) / "tape" / "*.csv")):
            try:
                f = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                continue
            if len(f):
                frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def boot_median(v, rng):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return None
    b = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(BOOT)]
    return dict(median=float(np.median(v)), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], names=int(len(v)))


def j1(rng):
    years = {
        "2013": [PE / "years_2013"], "2016": [PE / "years_2016"], "2019": [PE / "years_2019"],
        "2021": [PE / "years_2021", FP / "confirm_2021"], "2024": [PE / "years_2024", FP / "explore_2024"],
    }
    per_year = {}
    for y, dirs in years.items():
        rows = []
        for d in dirs:
            for p in sorted(glob.glob(str(d / "stats_v2" / "*.npz"))):
                m = name_measures(p)
                if m is not None:
                    rows.append(m)
        s = pd.DataFrame(rows)
        t = tape(dirs)
        if len(t):
            tt = t.groupby("ticker").agg(untruncated_share=("untruncated_share", "mean"),
                                         program_volume_share=("program_volume_share", "mean"),
                                         half_spread_bps=("half_spread_bps", "mean"), tape_days=("date", "size"))
            s = s.merge(tt, left_on="ticker", right_index=True, how="left")
        per_year[y] = s
    common = None
    for y, s in per_year.items():
        ok = set(s[(s.days >= 16)].ticker)
        common = ok if common is None else common & ok
    out = {"balanced_names": len(common)}
    for y, s in per_year.items():
        for label, sub in (("all", s[s.days >= 16]), ("balanced", s[s.ticker.isin(common)])):
            e = sub[sub.expected_2_10 >= 20]
            out.setdefault(y, {})[label] = dict(
                names=int(len(sub)), ratio_05_2=boot_median(e.ratio_05_2, rng), ratio_2_10=boot_median(e.ratio_2_10, rng),
                j_run60_mean=float(sub[(sub.expected_all >= 20) & (sub.excess_all >= 10)].j_run60.mean()),
                untruncated_share_median=float(sub.untruncated_share.median()) if "untruncated_share" in sub else None,
                program_volume_share_median=float(sub.program_volume_share.median()) if "program_volume_share" in sub else None,
                half_spread_bps_median=float(sub.half_spread_bps.median()) if "half_spread_bps" in sub else None)
    return out


def j2(rng):
    groups = pd.read_csv(PE / "tsp_names_groups.csv").drop_duplicates("sym_root").set_index("sym_root").grp.astype(str)
    periods = {}
    for per in ("tsp_pre", "tsp_during"):
        rows = [m for m in (name_measures(p) for p in sorted(glob.glob(str(PE / per / "stats_v2" / "*.npz")))) if m is not None]
        s = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
        t = tape([PE / per])
        if len(t):
            w = t[t.mk3_n > 0]
            agg = t.groupby("ticker").agg(untruncated_share=("untruncated_share", "mean"), half_spread_bps=("half_spread_bps", "mean"),
                                          n_signed=("n_signed", "mean"), tape_days=("date", "size"))
            agg2 = w.groupby("ticker").apply(lambda g: pd.Series(dict(
                mk3_bps=np.average(g.mk3_mean_bps, weights=g.mk3_n), hs_end_bps=np.average(g.half_spread_end_bps, weights=g.mk3_n),
                bursts=g.mk3_n.sum())))
            s = agg.join(agg2, how="left").join(s, how="left")
        periods[per] = s
    pre, dur = periods["tsp_pre"], periods["tsp_during"]
    names = sorted(set(pre.index) & set(dur.index))
    cols = ["untruncated_share", "half_spread_bps", "hs_end_bps", "mk3_bps", "ratio_2_10", "j_run60"]
    d = pd.DataFrame({c: dur.loc[names, c] - pre.loc[names, c] for c in cols if c in pre and c in dur})
    d["grp"] = groups.reindex(names).to_numpy()
    d = d[d.grp.notna()]
    ok = (pre.loc[d.index, "tape_days"] >= 8) & (dur.loc[d.index, "tape_days"] >= 8)
    d = d[ok.to_numpy()]
    out = dict(names=int(len(d)), by_group=d.grp.value_counts().to_dict())
    treat = d[d.grp.isin(["1", "2", "3"])]; ctrl = d[d.grp == "C"]
    for c in cols:
        if c not in d:
            continue
        a, b = treat[c].dropna(), ctrl[c].dropna()
        boots = [a.sample(len(a), replace=True, random_state=int(rng.integers(1 << 30))).mean()
                 - b.sample(len(b), replace=True, random_state=int(rng.integers(1 << 30))).mean() for _ in range(BOOT)]
        out["did_" + c] = dict(treated_change=float(a.mean()), control_change=float(b.mean()), did=float(a.mean() - b.mean()),
                               ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], n_t=int(len(a)), n_c=int(len(b)))
        for g in ("1", "2", "3"):
            out["did_%s_group%s" % (c, g)] = float(d[d.grp == g][c].mean() - b.mean())
    if "mk3_bps" in d and "hs_end_bps" in d:
        def iv(tr, co):
            num = tr.mk3_bps.mean() - co.mk3_bps.mean(); den = tr.hs_end_bps.mean() - co.hs_end_bps.mean()
            return num / den if den != 0 else np.nan
        tr, co = treat.dropna(subset=["mk3_bps", "hs_end_bps"]), ctrl.dropna(subset=["mk3_bps", "hs_end_bps"])
        vals = [iv(tr.sample(len(tr), replace=True, random_state=int(rng.integers(1 << 30))),
                   co.sample(len(co), replace=True, random_state=int(rng.integers(1 << 30)))) for _ in range(BOOT)]
        out["iv_slope_markout_on_half_spread"] = dict(value=float(iv(tr, co)), ci95=[float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))])
        pooled = pd.concat([pre[["mk3_bps", "hs_end_bps"]].assign(period="pre"), dur[["mk3_bps", "hs_end_bps"]].assign(period="during")]).dropna()
        out["cross_section_slope_markout_on_half_spread"] = float(np.polyfit(pooled.hs_end_bps, pooled.mk3_bps, 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="J1,J2")
    args = ap.parse_args()
    rng = np.random.default_rng(20260914)
    res = {}
    if "J1" in args.parts:
        res["J1"] = j1(rng)
    if "J2" in args.parts:
        res["J2"] = j2(rng)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
