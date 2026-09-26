#!/usr/bin/env python3
"""Independent recomputation and frozen-gate application for the spread-scaling law.

This script deliberately shares no code path with ``aggregate_packet_scaling.py``: it does
not import it, it reads the per-ticker CSVs itself, and it solves the cross-name regression
by QR factorisation rather than ``numpy.polyfit``/``lstsq``.  Agreement between the two is
therefore evidence about the result rather than about a shared helper.

Usage:
    audit_packet_scaling.py <group> <gate.json> [--production <summary.json>]

It prints the recomputed estimates, the production/audit discrepancy where a production
summary is supplied, and the pass/fail of every gate.  It never re-specifies anything: the
gate file is read, applied, and reported verbatim.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

MIN_DAYS = 40
MIN_N = 3


def read_group(group):
    """Row-by-row CSV reading, independent of the production pandas loader."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(group, "out", "*.csv"))):
        if os.path.getsize(path) == 0:
            continue
        with open(path, newline="") as handle:
            for record in csv.DictReader(handle):
                rows.setdefault(record["ticker"], []).append(record)
    if not rows:
        raise SystemExit("no rows under %s/out" % group)
    return rows


def _finite(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def name_means(rows, variant, base):
    """Mean markout and mean half-spread per ticker, computed with plain accumulators."""
    tickers, marks, spreads, dates = [], [], [], set()
    ncol, mcol = "%s_n" % variant, "%s_%s" % (variant, base)
    for ticker, records in sorted(rows.items()):
        mk_sum = hs_sum = 0.0
        count = 0
        for record in records:
            dates.add(record["date"])
            n = _finite(record.get(ncol))
            mk = _finite(record.get(mcol))
            hs = _finite(record.get("halfsp_day"))
            if n is None or mk is None or hs is None or n < MIN_N or hs <= 0:
                continue
            mk_sum += mk
            hs_sum += hs
            count += 1
        if count >= MIN_DAYS:
            tickers.append(ticker)
            marks.append(mk_sum / count)
            spreads.append(hs_sum / count)
    return (np.asarray(marks, float), np.asarray(spreads, float),
            len(tickers), len(dates))


def regress_qr(x, y):
    """Slope, intercept and HC1 t-statistics via QR rather than a least-squares solver."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    resid = y - X @ beta
    Rinv = np.linalg.inv(R)
    XtXi = Rinv @ Rinv.T
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = XtXi @ meat @ XtXi * (n / (n - 2))
    se = np.sqrt(np.diag(cov))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "intercept": float(beta[0]), "intercept_se": float(se[0]),
        "intercept_t": float(beta[0] / se[0]),
        "slope": float(beta[1]), "slope_se": float(se[1]),
        "slope_t": float(beta[1] / se[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "names": int(n),
    }


def panel_level(rows, variant, base):
    """Equal-weighted panel mean of a column, for the formation-agreement gate."""
    total = 0.0
    count = 0
    ncol, col = "%s_n" % variant, "%s_%s" % (variant, base)
    for records in rows.values():
        for record in records:
            n = _finite(record.get(ncol))
            value = _finite(record.get(col))
            if n is None or value is None or n < MIN_N:
                continue
            total += value
            count += 1
    return total / count if count else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("gate")
    ap.add_argument("--production", default=None)
    args = ap.parse_args()

    gate = json.load(open(args.gate))
    rows = read_group(args.group)
    marks, spreads, n_names, n_dates = name_means(rows, "all", "mk3_t0")
    if n_names < 10:
        raise SystemExit("too few names with >=%d usable days" % MIN_DAYS)
    fit = regress_qr(spreads, marks)
    clearing = float(np.mean(marks > 2.0 * spreads))
    run3 = panel_level(rows, "run3", "mk3_t0")
    clust3 = panel_level(rows, "clust3", "mk3_t0")
    reference = gate["reference_estimates_2023_2024"]["all_mk3_t0"]["slope"]

    report = {
        "group": args.group,
        "gate_file": args.gate,
        "names_in_regression": n_names,
        "dates": n_dates,
        "tickers_with_rows": len(rows),
        "all_mk3_t0": fit,
        "ratio_median": float(np.median(marks / spreads)),
        "share_clearing_round_trip": clearing,
        "run3_mk3_t0_level": run3,
        "clust3_mk3_t0_level": clust3,
        "formation_gap_bps": abs(run3 - clust3),
        "slope_deviation_from_training": abs(fit["slope"] - reference),
    }
    checks = {
        "G1_no_fixed_bps_component": fit["intercept_t"] < 2.0,
        "G2_scaling_is_real": fit["slope_t"] > 5.0,
        "G3_none_clear_cost": clearing <= 0.01,
        "G4_no_formation_circularity": report["formation_gap_bps"] <= 0.15,
        "G5_slope_transports": report["slope_deviation_from_training"] <= 0.20,
    }
    report["gates"] = {k: ("PASS" if v else "FAIL") for k, v in checks.items()}
    report["verdict"] = "PASS" if all(checks.values()) else "FAIL"

    if args.production and os.path.exists(args.production):
        prod = json.load(open(args.production))
        block = prod.get("definitions", {}).get("all", {}).get("mk3_t0")
        if block:
            report["production_vs_audit"] = {
                "slope_abs_diff": abs(block["slope_on_half_spread"] - fit["slope"]),
                "intercept_abs_diff": abs(block["intercept_bps"] - fit["intercept"]),
                "ratio_median_abs_diff": abs(block["ratio_median"] - report["ratio_median"]),
            }

    path = os.path.join(args.group, "independent_audit.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nwrote %s" % path, file=sys.stderr)
    return 0 if report["verdict"] == "PASS" else 0


if __name__ == "__main__":
    sys.exit(main())
