#!/usr/bin/env python3
"""Aggregate program-evidence-v1 modules A, B and I across names (one period), with gates.

Equal weight per name: medians of per-name ratios with a name bootstrap, or pooled counts with a
name bootstrap where the design says pooled. See PROGRAM_EVIDENCE_DESIGN.md.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import evidence_formulas as EF
import evidence_stats as ES
import fingerprint_stats as FS

BOOT = 1000
DELTAS = (1, 2, 5, 10, 20, 50)
B_RANGES = ((2, 10), (10, 60), (60, 600), (600, 3600))
B2_RANGES = ((600, 1800), (1800, 3600))
MIN_EXPECTED = 20.0


def load(pattern):
    names = []
    for path in sorted(glob.glob(pattern)):
        with np.load(path) as z:
            if len(z["dates"]) == 0:
                continue
            names.append({k: z[k] for k in z.files})
    return names


def boot_median(values, rng):
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return dict(median=None, ci95=None, names=0)
    b = np.median(v[rng.integers(0, len(v), (BOOT, len(v)))], axis=1)
    return dict(median=float(np.median(v)), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                names=int(len(v)), share_above_1=float(np.mean(v > 1)))


def boot_pooled(per_name, stat, rng):
    """per_name: list of arrays to be summed; stat(sum) -> scalar."""
    arr = np.stack(per_name)
    point = stat(arr.sum(0))
    idx = rng.integers(0, len(arr), (BOOT, len(arr)))
    b = np.array([stat(arr[i].sum(0)) for i in idx])
    b = b[np.isfinite(b)]
    return dict(value=float(point), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else None,
                names=len(arr))


def module_a(names, rng):
    out = {}
    for delta in DELTAS:
        r = [EF.phase_ratios(z, delta) for z in names]
        out["delta_%dms" % delta] = dict(
            a1=boot_median([x["a1"] for x in r if x["window_pairs"] >= 1000], rng),
            a2=boot_median([x["a2"] for x in r if x["identical_window_pairs"] >= 500], rng),
            a1_all_signed=boot_median([x["a1_signed"] for x in r if x["window_pairs"] >= 1000], rng),
            a1_opposite_side=boot_median([x["a1_opposite"] for x in r if x["window_pairs"] >= 1000], rng),
            crossday_identical_vs_all=boot_median([x["a1_crossday_identical"] for x in r if x["window_pairs"] >= 1000], rng),
        )
    # A0: absolute timestamp phase, pooled
    frac = np.sum([z["a0_frac"].sum(0) for z in names], axis=0)      # [class, 1000]
    tot = frac.sum(1, keepdims=True)
    share = frac / tot
    out["absolute_phase"] = dict(
        classes=["u_nonround", "all_signed"],
        share_within_10ms_of_whole_second=[float(share[c, :10].sum() + share[c, 990:].sum()) for c in range(2)],
        uniform_share=0.02,
        top_ms_bins=[[int(i) for i in np.argsort(share[c])[::-1][:10]] for c in range(2)],
        top_ms_share_over_uniform=[[float(share[c, i] * 1000) for i in np.argsort(share[c])[::-1][:10]] for c in range(2)],
    )
    # A3: phase J and size J over the 33 definitions (min 3)
    rules, gaps = FS.RULES, FS.GAPS
    pj, sj = [], []
    for z in names:
        p = EF.burst_phase_j(z, 10); s = EF.burst_size_j(z)
        pj.append(np.where((p["expected_locked"] >= MIN_EXPECTED) & (p["excess_locked"] >= 10), p["j"], np.nan))
        sj.append(np.where((s["expected"] >= MIN_EXPECTED) & (s["excess"] >= 10), s["j"], np.nan))
    pj = np.array(pj); sj = np.array(sj)                            # [names, rule, gap]
    mean_p = np.nanmean(pj, 0); mean_s = np.nanmean(sj, 0)
    rho = spearmanr(mean_p.ravel(), mean_s.ravel()).correlation
    boots = []
    for _ in range(BOOT):
        i = rng.integers(0, len(names), len(names))
        with np.errstate(all="ignore"):
            boots.append(spearmanr(np.nanmean(pj[i], 0).ravel(), np.nanmean(sj[i], 0).ravel()).correlation)
    boots = np.array(boots)
    defs = [dict(rule=r, gap_s=g, phase_j=float(mean_p[ri, gi]), size_j=float(mean_s[ri, gi]))
            for ri, r in enumerate(rules) for gi, g in enumerate(gaps)]
    order_p = sorted(defs, key=lambda d: -d["phase_j"]); order_s = sorted(defs, key=lambda d: -d["size_j"])
    out["definitions"] = dict(
        spearman_phase_vs_size_j=float(rho),
        spearman_ci95=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
        names_phase=int(np.isfinite(pj).any((1, 2)).sum()), names_size=int(np.isfinite(sj).any((1, 2)).sum()),
        top5_phase=order_p[:5], top5_size=order_s[:5],
        working_definition_rank_phase=1 + [(d["rule"], d["gap_s"]) for d in order_p].index(("run", 60.0)),
        working_definition_rank_size=1 + [(d["rule"], d["gap_s"]) for d in order_s].index(("run", 60.0)),
        all=defs)
    return out


def module_b(names, rng):
    out = {"lag_edges": ES.B_EDGES.tolist()}
    for ci, cls in enumerate(ES.B_CLASSES):
        per = [EF.side_observed_expected(z, ci, matched=True) for z in names]
        curves = {}
        for rel, lab in ((0, "same_side"), (1, "opposite_side")):
            curves[lab] = []
            for b in range(len(ES.B_EDGES) - 1):
                vals = [o[rel, b] / e[rel, b] for o, e in per if np.isfinite(e[rel, b]) and e[rel, b] >= MIN_EXPECTED]
                curves[lab].append(boot_median(vals, rng))
        ranges = {}
        for lo, hi in B_RANGES + B2_RANGES:
            diffs, same = [], []
            for o, e in per:
                r = EF.side_ratio(o, e, lo, hi)
                m = EF.lag_range_mask(lo, hi)
                es, eo = np.nansum(e[0, m]), np.nansum(e[1, m])
                if es >= MIN_EXPECTED:
                    same.append(r[0])
                    if eo >= MIN_EXPECTED:
                        diffs.append(r[0] - r[1])
            ranges["%g-%g" % (lo, hi)] = dict(same_minus_opposite=boot_median(diffs, rng), same_side_ratio=boot_median(same, rng))
        # one-sidedness index per lag bin, pooled excess
        stacked = [np.stack([o, np.nan_to_num(e)]) for o, e in per]        # [2(obs,exp), rel, bin]

        def os_index(s, b):
            x_same = s[0, 0, b] - s[1, 0, b]; x_opp = s[0, 1, b] - s[1, 1, b]
            return x_same / (x_same + x_opp) if (x_same + x_opp) > 0 else np.nan
        onesided = [boot_pooled(stacked, lambda s, b=b: os_index(s, b), rng) for b in range(len(ES.B_EDGES) - 1)]
        out[cls] = dict(ratio_curves=curves, ranges=ranges, one_sidedness_index=onesided)
    # unmatched sensitivity, u_nonround ranges only
    per_u = [EF.side_observed_expected(z, 0, matched=False) for z in names]
    out["u_nonround_unmatched_ranges"] = {}
    for lo, hi in B_RANGES + B2_RANGES:
        vals = [EF.side_ratio(o, e, lo, hi) for o, e in per_u]
        out["u_nonround_unmatched_ranges"]["%g-%g" % (lo, hi)] = dict(
            same_side=boot_median([v[0] for v in vals], rng), opposite_side=boot_median([v[1] for v in vals], rng))
    # B3 multi-day difference in differences (pooled with bootstrap), u_nonround matched
    parts = []
    for z in names:
        _, d = EF.multiday_did(z)
        parts.append(np.stack([d["adj_match"], d["adj_pairs"], d["dist_match"], d["dist_pairs"]]))

    def did(s):
        am, ap, dm, dp = s
        if np.any(np.array([ap, dp, am, dm]) <= 0):
            return np.nan
        return (am[0] / ap[0] / (dm[0] / dp[0])) / (am[1] / ap[1] / (dm[1] / dp[1]))
    out["b3_multiday_did"] = boot_pooled(parts, did, rng)
    tot = np.stack(parts).sum(0)
    out["b3_rates"] = dict(adjacent_same=float(tot[0, 0] / tot[1, 0]), adjacent_opposite=float(tot[0, 1] / tot[1, 1]),
                           distant_same=float(tot[2, 0] / tot[3, 0]), distant_opposite=float(tot[2, 1] / tot[3, 1]))
    # chains: real vs size-permuted
    real = np.sum([z["b_chain_real"].sum(0) for z in names], axis=0)
    perm = np.sum([z["b_chain_perm"].sum(0) for z in names], axis=0)
    os_mid = (ES.CHAIN_OS_EDGES[:-1] + ES.CHAIN_OS_EDGES[1:]) / 2

    def chain_summary(h):
        by_n = h.sum((1, 2))
        ge5 = h[2:].sum(2)                                               # n >= 5, [n, os]
        n5 = ge5.sum()
        return dict(chains_by_n={"%d-%d" % (ES.CHAIN_N_EDGES[i], ES.CHAIN_N_EDGES[i + 1] - 1): int(by_n[i]) for i in range(len(by_n))},
                    chains_n_ge5=int(n5),
                    share_n_ge5_one_sided_ge_0p9=float(ge5[:, -1].sum() / n5) if n5 else None,
                    mean_one_sidedness_n_ge5=float((ge5.sum(0) * os_mid).sum() / n5) if n5 else None,
                    duration_n_ge5={"%g-%g" % (ES.CHAIN_DUR_EDGES[i], ES.CHAIN_DUR_EDGES[i + 1]): int(h[2:, :, i].sum())
                                    for i in range(len(ES.CHAIN_DUR_EDGES) - 1)})
    out["rare_size_chains"] = dict(real=chain_summary(real), size_permuted_within_30min=chain_summary(perm))
    return out


def module_i(names, rng):
    per = [z["i_counts"].sum(0).astype(float) for z in names]          # [side, near/far, e, dm, ds]
    out = {"e_edges": ES.E_EDGES.tolist()}
    tot = np.sum(per, axis=0)
    d = EF.dollar_d(tot)                                                 # [side, near/far, e]
    out["D"] = {side: {dist: [float(x) for x in d[si, di]] for di, dist in enumerate(("near", "far"))}
                for si, side in enumerate(("buy", "sell"))}
    out["pairs"] = {side: {dist: [int(tot[si, di, e, :2].sum()) for e in range(len(ES.E_EDGES) - 1)]
                           for di, dist in enumerate(("near", "far"))} for si, side in enumerate(("buy", "sell"))}
    for si, side in enumerate(("buy", "sell")):
        out["I1_%s" % side] = boot_pooled(per, lambda s, si=si: float(EF.dollar_d(s)[si, 0, 2] - EF.dollar_d(s)[si, 1, 2]), rng)
        out["near_minus_far_by_e_%s" % side] = [
            boot_pooled(per, lambda s, si=si, e=e: float(EF.dollar_d(s)[si, 0, e] - EF.dollar_d(s)[si, 1, e]), rng)
            for e in range(len(ES.E_EDGES) - 1)]
    return out


def gates(res):
    a = res["A"]["delta_10ms"]
    lower = lambda x: x["ci95"][0] if x and x.get("ci95") else -np.inf
    g = dict(
        A1=bool(lower(a["a1"]) > 1), A2=bool(lower(a["a2"]) > 1),
        A3=bool(res["A"]["definitions"]["spearman_phase_vs_size_j"] > 0.7),
        B1=bool(all(lower(res["B"]["u_nonround"]["ranges"]["%g-%g" % r]["same_minus_opposite"]) > 0 for r in B_RANGES)),
        B2=bool(all(lower(res["B"]["u_nonround"]["ranges"]["%g-%g" % r]["same_side_ratio"]) > 1 for r in B2_RANGES)),
        B3=bool(lower(res["B"]["b3_multiday_did"]) > 1),
        I1=bool(lower(res["I"]["I1_buy"]) > 0 and lower(res["I"]["I1_sell"]) > 0),
    )
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="glob of per-name npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260914)
    names = load(args.stats)
    res = dict(label=args.label, names=len(names), name_days=int(sum(len(z["dates"]) for z in names)))
    res["A"] = module_a(names, rng)
    res["B"] = module_b(names, rng)
    res["I"] = module_i(names, rng)
    res["gates"] = gates(res)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res["gates"], indent=1))


if __name__ == "__main__":
    main()
