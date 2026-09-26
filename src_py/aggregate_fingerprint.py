#!/usr/bin/env python3
"""Aggregate fingerprint-v1 statistics for one group (exploration or confirmation).

Excess same-size matches = within-day matches - within-day pairs x cross-day null rate, with
the null computed separately for every date pair, name and size class (a lag bin with no
cross-day pairs borrows that name's pooled null; with none at all its counts are dropped). Only
days whose paired day also has statistics are used. Uncertainty: bootstrap over names.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS

FINE = FS.FINE_EDGES
BURST = FS.BURST_EDGES
FINE_TO_BURST = np.searchsorted(BURST, FINE[:-1], side="right") - 1
RECALL_LAG = 300.0
MIN_EXPECTED = 20.0   # a name enters equal-weight statistics with >= 20 expected chance matches
BOOT = int(__import__("os").environ.get("FP_BOOT", "1000"))


def to_burst_bins(x):
    out = np.zeros(x.shape[:-1] + (len(BURST) - 1,), x.dtype)
    for f, b in enumerate(FINE_TO_BURST):
        out[..., b] += x[..., f]
    return out


NULLS = ("cross_day_depth_matched", "cross_day", "within_day_long_lag")
PRIMARY_NULL = "cross_day_depth_matched"
LONG_LAG = 900.0


def _rate(matches, pairs, axis_pool):
    """matches/pairs with empty cells borrowing the pooled rate along axis_pool (NaN if none)."""
    q = np.divide(matches, pairs, out=np.full(pairs.shape, np.nan), where=pairs > 0)
    pp, pm = pairs.sum(axis_pool), matches.sum(axis_pool)
    pooled = np.divide(pm, pp, out=np.full(pp.shape, np.nan), where=pp > 0)
    return np.where(np.isnan(q), np.expand_dims(pooled, axis_pool), q)


def load_name(path):
    """Per-name sums of pair counts and expected chance matches under three nulls.

    cross_day_depth_matched (primary): pairs, within-day and cross-day alike, must share an
    executed-side depth quartile (thresholds pooled over the date pair); cross-day rate pooled over
    lags in [0, 1800s). Removes depth-censoring and intraday seasonality; cannot absorb a same-day
    program. Its pair population is same-side pairs in the same depth quartile.
    cross_day: the same without depth matching (sensitivity; exposed to censoring).
    within_day_long_lag: same day's rate at 900-1800s lags (lower bound; absorbs long programs).
    """
    z = np.load(path, allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    pairs = [tuple(str(p).split("_")) for p in z["pairs"]]
    pair_index = {d: i for i, p in enumerate(pairs) for d in p}
    use = [i for i, d in enumerate(dates) if d in pair_index]
    if not use:
        return None
    pidx = np.array([pair_index[dates[i]] for i in use])
    bc = [FS.CLASSES.index(c) for c in FS.BURST_CLASSES]
    long_bins = FINE[:-1] >= LONG_LAG
    wp = z["within_pairs"][use].astype(float); wm = z["within_matches"][use].astype(float)             # [D, C, 2, F]
    wp_dm = np.zeros_like(wp); wm_dm = np.zeros_like(wm)
    wp_dm[:, :, 0, :] = z["within_pairs_dm"][use]; wm_dm[:, :, 0, :] = z["within_matches_dm"][use]
    sources = {
        "cross_day_depth_matched": (wp_dm, wm_dm, z["burst_pairs_dm"][use].astype(float), z["burst_matches_dm"][use].astype(float),
                                    _rate(z["cross_matches_dm"].sum(-1).astype(float), z["cross_pairs_dm"].sum(-1).astype(float), 0)[pidx]),
        "cross_day": (wp, wm, z["burst_pairs"][use].astype(float), z["burst_matches"][use].astype(float),
                      _rate(z["cross_matches"].sum(-1).astype(float), z["cross_pairs"].sum(-1).astype(float), 0)[pidx]),
        "within_day_long_lag": (wp, wm, z["burst_pairs"][use].astype(float), z["burst_matches"][use].astype(float),
                                _rate(wm[:, :, 0, long_bins].sum(-1), wp[:, :, 0, long_bins].sum(-1), 0)),
    }
    out = {"ticker": str(z["ticker"]), "days": len(use),
           "n_class": z["n_class"][use].sum(0), "n_signed": int(z["n_signed"][use].sum()),
           "n_bursts3": z["n_bursts3"][use].sum(0), "packets_in_bursts3": z["packets_in_bursts3"][use].sum(0),
           "recur_count": z["recur_count"][use].sum(0), "recur_sum": z["recur_sum"][use].sum(0),
           "recur_hist": z["recur_hist"][use].sum(0)}
    for name, (WPd, WMd, bp, bm, qk) in sources.items():
        ok = np.isfinite(qk); qz = np.nan_to_num(qk)                                                   # [D, C]
        okw = ok[:, :, None, None]
        out["within_pairs_" + name] = np.where(okw, WPd, 0).sum(0)
        out["within_matches_" + name] = np.where(okw, WMd, 0).sum(0)
        out["within_expected_" + name] = np.where(okw, WPd * qz[:, :, None, None], 0).sum(0)
        okb = ok[:, bc]; qb = qz[:, bc]
        sel = okb[:, None, None, None, :, None]
        out["burst_pairs_" + name] = np.where(sel, bp, 0).sum(0)
        out["burst_matches_" + name] = np.where(sel, bm, 0).sum(0)
        out["burst_expected_" + name] = np.where(sel, bp * qb[:, None, None, None, :, None], 0).sum(0)
        wpb = to_burst_bins(WPd[:, bc, 0, :]); wmb = to_burst_bins(WMd[:, bc, 0, :])
        out["all_pairs_b_" + name] = np.where(okb[:, :, None], wpb, 0).sum(0)
        out["all_matches_b_" + name] = np.where(okb[:, :, None], wmb, 0).sum(0)
        out["all_expected_b_" + name] = np.where(okb[:, :, None], wpb * qb[:, :, None], 0).sum(0)
    return out


def boot_ci(stat_fn, n, rng):
    vals = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        v = stat_fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < BOOT // 2:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default="stats_v2", help="subdirectory of per-name statistics")
    args = ap.parse_args()
    rng = np.random.default_rng(20260913)
    names = [x for x in (load_name(p) for p in sorted(glob.glob(str(Path(args.group) / args.stats / "*.npz"))))
             if x is not None]
    if not names:
        raise ValueError("no usable name statistics")
    N = len(names)
    stack = lambda key: np.stack([x[key] for x in names])
    result = {"group": args.group, "names": N, "name_days": int(sum(x["days"] for x in names)),
              "signed_packets": int(sum(x["n_signed"] for x in names)),
              "class_packets": dict(zip(FS.CLASSES, stack("n_class").sum(0).tolist())),
              "nulls": {"cross_day_depth_matched": "primary: adjacent-day same-clock same-side same-depth-quartile rate, pooled over lags",
                        "cross_day": "adjacent-day same-clock same-side rate, pooled over lags (exposed to depth censoring)",
                        "within_day_long_lag": "same-day same-side rate at 900-1800s lags (lower bound)"}}
    n_signed = np.array([x["n_signed"] for x in names], float)
    nb3, pin3 = stack("n_bursts3"), stack("packets_in_bursts3")                           # [N,R,G]
    L = int(np.searchsorted(BURST, RECALL_LAG, side="right") - 1)
    curves = {}; table = []
    for null in NULLS:
        WP, WM, WE = (stack(k + "_" + null) for k in ("within_pairs", "within_matches", "within_expected"))
        # 1. Excess ratio curves
        for ci, c in enumerate(FS.CLASSES):
            for rel, label in ((0, "same_side"), (1, "opposite_side")):
                rows = []
                for f in range(len(FINE) - 1):
                    m, e, p = WM[:, ci, rel, f], WE[:, ci, rel, f], WP[:, ci, rel, f]
                    ratio_fn = lambda idx: m[idx].sum() / e[idx].sum() if e[idx].sum() > 0 else np.nan
                    elig = e >= MIN_EXPECTED
                    name_ratio = np.divide(m, e, out=np.full(N, np.nan), where=elig)
                    med_fn = lambda idx: np.nanmedian(name_ratio[idx]) if np.isfinite(name_ratio[idx]).sum() >= 5 else np.nan
                    rows.append(dict(lag_lo=float(FINE[f]), lag_hi=float(FINE[f + 1]), pairs=float(p.sum()),
                                     matches=float(m.sum()), expected=float(e.sum()),
                                     names_eligible=int(elig.sum()),
                                     name_median_ratio=float(med_fn(np.arange(N))), name_median_ratio_ci95=boot_ci(med_fn, N, rng),
                                     names_ratio_above_1=float(np.mean(name_ratio[elig] > 1)) if elig.any() else None,
                                     ratio=float(ratio_fn(np.arange(N))), ratio_ci95=boot_ci(ratio_fn, N, rng),
                                     excess_per_1000_pairs=float(1000 * (m.sum() - e.sum()) / p.sum()) if p.sum() else None))
                curves["%s/%s/%s" % (null, c, label)] = rows
        # 2. Burst-definition validation
        BP, BM, BE = (stack(k + "_" + null) for k in ("burst_pairs", "burst_matches", "burst_expected"))
        AP, AM, AE = (stack(k + "_" + null) for k in ("all_pairs_b", "all_matches_b", "all_expected_b"))
        for bci, c in enumerate(FS.BURST_CLASSES):
            total_excess = (AM[:, bci, :L] - AE[:, bci, :L]).sum(1)
            for ri, rule in enumerate(FS.RULES):
                for gi, gap in enumerate(FS.GAPS):
                    for mi, minsize in enumerate(FS.MINSIZES):
                        m = BM[:, ri, gi, mi, bci]; e = BE[:, ri, gi, mi, bci]; p = BP[:, ri, gi, mi, bci]
                        win_excess = (m[:, :L] - e[:, :L]).sum(1)
                        recall = lambda idx: win_excess[idx].sum() / total_excess[idx].sum() if total_excess[idx].sum() > 0 else np.nan

                        def roc(idx, bci=bci, m=m, e=e, p=p):
                            """(TPR, FPR) over lags < 300s. Positive mass: excess same-size matches,
                            clipped at zero per lag bin. Negative mass: all same-side pairs (same-origin
                            pairs are a small share of them). Scale-free: needs no P(same size | same origin)."""
                            x_all = np.clip((AM[idx, bci, :L] - AE[idx, bci, :L]).sum(0), 0, None)
                            x_win = np.minimum(np.clip((m[idx, :L] - e[idx, :L]).sum(0), 0, None), x_all)
                            p_all = AP[idx, bci, :L].sum()
                            tpr = x_win.sum() / x_all.sum() if x_all.sum() > 0 else np.nan
                            fpr = p[idx, :L].sum() / p_all if p_all > 0 else np.nan
                            return tpr, fpr

                        youden = lambda idx: (lambda r: r[0] - r[1])(roc(idx))

                        def name_roc(bci=bci, m=m, e=e, p=p):
                            """Per-name TPR, FPR over lags < 300s; eligible names have >= MIN_EXPECTED
                            chance matches and >= 10 excess matches in total."""
                            x_all = np.clip(AM[:, bci, :L] - AE[:, bci, :L], 0, None)            # [N, L]
                            x_win = np.minimum(np.clip(m[:, :L] - e[:, :L], 0, None), x_all)
                            p_all = AP[:, bci, :L].sum(1)
                            elig = (AE[:, bci, :L].sum(1) >= MIN_EXPECTED) & (x_all.sum(1) >= 10) & (p_all > 0)
                            tpr = np.divide(x_win.sum(1), x_all.sum(1), out=np.full(N, np.nan), where=elig)
                            fpr = np.divide(p[:, :L].sum(1), p_all, out=np.full(N, np.nan), where=elig)
                            return tpr, fpr
                        ntpr, nfpr = name_roc()
                        nj = ntpr - nfpr
                        name_j = lambda idx: np.nanmean(nj[idx]) if np.isfinite(nj[idx]).sum() >= 5 else np.nan

                        def enrich(idx, bci=bci, m=m, e=e, p=p):
                            """Within-burst excess over the excess random same-lag pairs would carry."""
                            x_all = (AM[idx, bci, :L] - AE[idx, bci, :L]).sum(0)
                            p_all = AP[idx, bci, :L].sum(0)
                            rate = np.divide(x_all, p_all, out=np.zeros(L), where=p_all > 0)
                            expected = (p[idx, :L].sum(0) * rate).sum()
                            observed = (m[idx, :L] - e[idx, :L]).sum()
                            return observed / expected if expected > 0 else np.nan
                        dens = lambda idx: 1000 * (m[idx].sum() - e[idx].sum()) / p[idx].sum() if p[idx].sum() > 0 else np.nan
                        by_lag = [float(1000 * (m[:, b].sum() - e[:, b].sum()) / p[:, b].sum())
                                  if p[:, b].sum() > 0 else None for b in range(L)]
                        table.append(dict(null=null, size_class=c, rule=rule, gap_s=gap, min_packets=minsize,
                                          within_pairs=float(p.sum()), within_matches=float(m.sum()),
                                          within_expected=float(e.sum()),
                                          recall_300s=float(recall(np.arange(N))), recall_ci95=boot_ci(recall, N, rng),
                                          enrichment_300s=float(enrich(np.arange(N))), enrichment_ci95=boot_ci(enrich, N, rng),
                                          tpr_300s=float(roc(np.arange(N))[0]), fpr_300s=float(roc(np.arange(N))[1]),
                                          youden_j=float(youden(np.arange(N))), youden_j_ci95=boot_ci(youden, N, rng),
                                          names_eligible=int(np.isfinite(nj).sum()),
                                          name_mean_tpr=float(np.nanmean(ntpr)) if np.isfinite(ntpr).any() else None,
                                          name_mean_fpr=float(np.nanmean(nfpr)) if np.isfinite(nfpr).any() else None,
                                          name_mean_j=float(name_j(np.arange(N))), name_mean_j_ci95=boot_ci(name_j, N, rng),
                                          excess_per_1000_within_pairs=float(dens(np.arange(N))),
                                          density_ci95=boot_ci(dens, N, rng),
                                          excess_per_1000_pairs_by_lag=by_lag,
                                          chance_share_of_within_matches=float(e.sum() / m.sum()) if m.sum() else None,
                                          bursts3=float(nb3[:, ri, gi].sum()),
                                          share_signed_packets_in_bursts3=float(pin3[:, ri, gi].sum() / n_signed.sum())))
    result["excess_curves"] = curves
    # E1 / C1: combined lag ranges, per class and null. Primary statistic: median of per-name ratios.
    e1 = {}
    ranges = {"0.5-2s": (0.5, 2.0), "2-10s": (2.0, 10.0)}
    for null in NULLS:
        WM, WE = stack("within_matches_" + null), stack("within_expected_" + null)
        for ci, c in enumerate(FS.CLASSES):
            for label, (lo, hi) in ranges.items():
                sel = (FINE[:-1] >= lo) & (FINE[1:] <= hi)
                m = WM[:, ci, 0, sel].sum(1); e = WE[:, ci, 0, sel].sum(1)
                elig = e >= MIN_EXPECTED
                per = np.divide(m, e, out=np.full(N, np.nan), where=elig)
                med = lambda idx: np.nanmedian(per[idx]) if np.isfinite(per[idx]).sum() >= 5 else np.nan
                pooled = lambda idx: m[idx].sum() / e[idx].sum() if e[idx].sum() > 0 else np.nan
                e1["%s/%s/%s" % (null, c, label)] = dict(
                    names_eligible=int(elig.sum()), name_median_ratio=float(med(np.arange(N))),
                    name_median_ratio_ci95=boot_ci(med, N, rng),
                    names_ratio_above_1=float(np.mean(per[elig] > 1)) if elig.any() else None,
                    pooled_ratio=float(pooled(np.arange(N))), pooled_ratio_ci95=boot_ci(pooled, N, rng))
    result["e1_existence"] = e1
    result["burst_lag_edges"] = BURST[:L + 1].tolist()
    result["burst_definitions"] = table
    # Pre-declared selection (BURST_FINGERPRINT_DESIGN.md, revised before exploration output was read):
    # u_nonround, min 3 packets, depth-matched cross-day null, maximize the equal-name mean of
    # per-name Youden J = TPR - FPR (pooled J reported alongside).
    selection = {}; auc = {}
    for null in NULLS:
        cand = [r for r in table if r["null"] == null and r["size_class"] == "u_nonround" and r["min_packets"] == 3
                and r["name_mean_j"] is not None and np.isfinite(r["name_mean_j"])]
        ranked = sorted(cand, key=lambda r: -r["name_mean_j"])
        selection[null] = [dict(rank=i + 1, rule=r["rule"], gap_s=r["gap_s"], name_mean_j=r["name_mean_j"],
                                name_mean_j_ci95=r["name_mean_j_ci95"], name_mean_tpr=r["name_mean_tpr"],
                                name_mean_fpr=r["name_mean_fpr"], pooled_youden_j=r["youden_j"],
                                enrichment_300s=r["enrichment_300s"], names_eligible=r["names_eligible"])
                           for i, r in enumerate(ranked)]
        for rule in FS.RULES:
            pts = sorted([(r["name_mean_fpr"], r["name_mean_tpr"]) for r in cand if r["rule"] == rule])
            xs = np.r_[0.0, [x for x, _ in pts], 1.0]; ys = np.r_[0.0, [y for _, y in pts], 1.0]
            auc["%s/%s" % (null, rule)] = float(np.trapz(ys, xs)) if hasattr(np, "trapz") else float(np.trapezoid(ys, xs))
    result["selection"] = selection
    result["roc_auc_by_rule"] = auc
    result["selected_definition"] = selection[PRIMARY_NULL][0] if selection[PRIMARY_NULL] else None

    # 3. State similarity of consecutive same-size recurrences versus lag-matched controls
    RC, RS = stack("recur_count"), stack("recur_sum")                                     # [N,2,2,5], [N,2,2,5,3]
    sim = []
    for rel, label in ((0, "same_side"), (1, "opposite_side")):
        for b in range(len(FS.RECUR_EDGES) - 1):
            for f, feat in enumerate(FS.FEATURES):
                mean = lambda idx, w: RS[idx, rel, w, b, f].sum() / RC[idx, rel, w, b].sum() if RC[idx, rel, w, b].sum() else np.nan
                ratio = lambda idx: mean(idx, 0) / mean(idx, 1)
                okn = (RC[:, rel, 0, b] >= 30) & (RC[:, rel, 1, b] >= 30)
                with np.errstate(invalid="ignore", divide="ignore"):
                    per = (RS[:, rel, 0, b, f] / RC[:, rel, 0, b]) / (RS[:, rel, 1, b, f] / RC[:, rel, 1, b])
                per = np.where(okn, per, np.nan)
                name_med = lambda idx: np.nanmedian(per[idx]) if np.isfinite(per[idx]).sum() >= 5 else np.nan
                sim.append(dict(relation=label, lag_lo=float(FS.RECUR_EDGES[b]), lag_hi=float(FS.RECUR_EDGES[b + 1]),
                                feature=feat, matched_n=int(RC[:, rel, 0, b].sum()), control_n=int(RC[:, rel, 1, b].sum()),
                                matched_mean_absdiff=float(mean(np.arange(N), 0)),
                                control_mean_absdiff=float(mean(np.arange(N), 1)),
                                ratio=float(ratio(np.arange(N))), ratio_ci95=boot_ci(ratio, N, rng),
                                names_eligible=int(np.isfinite(per).sum()),
                                name_median_ratio=float(name_med(np.arange(N))), name_median_ratio_ci95=boot_ci(name_med, N, rng)))
    result["state_similarity"] = sim

    # 4. Clock periodicity of consecutive same-size recurrences
    H = stack("recur_hist").sum(0)                                                         # [2, bins]
    per = []
    for rel, label in ((0, "same_side"), (1, "opposite_side")):
        h = H[rel].astype(float)
        for k in (1, 2, 3, 5, 10, 15, 20, 30, 60, 90):
            c = int(round(k / FS.HIST_STEP))
            peak = h[c - 1:c + 1].sum() / 2
            neigh = np.r_[h[c - 10:c - 2], h[c + 2:c + 10]].mean()
            half = int(round((k + 0.5) / FS.HIST_STEP))
            ctrl = h[half - 1:half + 1].sum() / 2
            per.append(dict(relation=label, lag_s=k, peak_per_bin=float(peak), neighbour_per_bin=float(neigh),
                            peak_ratio=float(peak / neigh) if neigh else None,
                            half_second_control_ratio=float(ctrl / neigh) if neigh else None))
    result["periodicity"] = per

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    print(json.dumps({"names": N, "name_days": result["name_days"]}, indent=1))


if __name__ == "__main__":
    main()
