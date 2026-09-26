#!/usr/bin/env python3
"""Metaorder-v1 gate M1: combined size and phase J for stream5 against run60, one year.

Size J: fingerprint-v1 per-name Youden J over lags < 300 s (depth-matched cross-day null, u_nonround,
min 3, excess clipped at zero per lag bin; names with >= 20 expected and >= 10 excess matches).
Phase J: program-evidence-v1 A3 per name (names with >= 20 expected and >= 10 excess locked pairs).
D = combined J(stream5) - combined J(run60), names eligible for both statistics in both rules,
name-paired bootstrap.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import aggregate_fingerprint as AF
import evidence_formulas as EF
import fingerprint_stats as FS

BOOT = 1000
CANDIDATES = (("run", 60.0), ("stream", 5.0))
K = "cross_day_depth_matched"


def size_j(z, rule, gap):
    ri, gi, mi = FS.RULES.index(rule), FS.GAPS.index(gap), FS.MINSIZES.index(3)
    bci = FS.BURST_CLASSES.index("u_nonround")
    lagm = AF.BURST[:-1] < AF.RECALL_LAG
    bm = z["burst_matches_" + K][ri, gi, mi, bci][lagm]; be = z["burst_expected_" + K][ri, gi, mi, bci][lagm]
    bp = z["burst_pairs_" + K][ri, gi, mi, bci][lagm]
    am = z["all_matches_b_" + K][bci][lagm]; ae = z["all_expected_b_" + K][bci][lagm]; ap = z["all_pairs_b_" + K][bci][lagm]
    x_in = np.clip(bm - be, 0, None).sum(); x_all = np.clip(am - ae, 0, None).sum()
    ok = ae.sum() >= 20 and x_all >= 10 and ap.sum() > 0
    return float(x_in / x_all - bp.sum() / ap.sum()) if ok else np.nan


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--size-stats", required=True, help="glob of fingerprint_stats (code_v2) npz")
    ap_.add_argument("--phase-stats", required=True, help="glob of evidence_stats npz (module A)")
    ap_.add_argument("--label", required=True)
    ap_.add_argument("--out", required=True)
    args = ap_.parse_args()
    rng = np.random.default_rng(20260914)
    size = {}
    for p in sorted(glob.glob(args.size_stats)):
        z = AF.load_name(p)
        if z is not None:
            size[z["ticker"]] = [size_j(z, r, g) for r, g in CANDIDATES]
    phase = {}
    for p in sorted(glob.glob(args.phase_stats)):
        with np.load(p) as zz:
            if not len(zz["dates"]):
                continue
            z = {k: zz[k] for k in zz.files}
        pj = EF.burst_phase_j(z, 10)
        ok = (pj["expected_locked"] >= 20) & (pj["excess_locked"] >= 10)
        phase[str(z["ticker"])] = [float(pj["j"][FS.RULES.index(r), FS.GAPS.index(g)]) if ok else np.nan for r, g in CANDIDATES]
    names = sorted(set(size) & set(phase))
    S = np.array([size[n] for n in names]); P = np.array([phase[n] for n in names])
    ok = np.isfinite(S).all(1) & np.isfinite(P).all(1)
    S, P = S[ok], P[ok]
    comb = (S + P) / 2
    d = comb[:, 1] - comb[:, 0]
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    decision = "adopt stream5" if lo > 0 else ("keep run60" if hi < 0 else "tie: keep run60")
    res = dict(label=args.label, names=int(ok.sum()), names_size=len(size), names_phase=len(phase),
               mean_size_j=dict(run60=float(S[:, 0].mean()), stream5=float(S[:, 1].mean())),
               mean_phase_j=dict(run60=float(P[:, 0].mean()), stream5=float(P[:, 1].mean())),
               mean_combined_j=dict(run60=float(comb[:, 0].mean()), stream5=float(comb[:, 1].mean())),
               D=float(d.mean()), D_ci95=[lo, hi], decision=decision,
               D_size_only=float((S[:, 1] - S[:, 0]).mean()), D_phase_only=float((P[:, 1] - P[:, 0]).mean()))
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
