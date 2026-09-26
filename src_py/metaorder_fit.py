#!/usr/bin/env python3
"""Metaorder-v1 M2 and M4 fits (and the rule's program score for M3), exploration 2024 -> confirmation 2021.

Each model maximizes the fingerprint-v1 binomial likelihood p = q + (1 - q) sigmoid(w.x) with ridge 1 and equal
weight per name, where (pairs, matches, expected) are:
  program  : within-burst dm_pairs, dm_repeats, dm_expected, whole-burst features;
  link (M2): forward same-side link pairs/matches/expected, whole-burst + backward context features;
  realtime (M4): within-burst evidence, first-three-packet prefix + backward context features.
Evaluation on 2021: score deciles, excess per 1,000 pairs, name bootstrap; for link also opposite-side excess
in the same deciles (gate L2).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

import metaorder_features as MF

PENALTY, BOOT = 1.0, 1000
TARGETS = {"program": ("dm_pairs", "dm_repeats", "dm_expected"), "realtime": ("dm_pairs", "dm_repeats", "dm_expected"),
           "link": ("link_same_pairs", "link_same_matches", "link_same_expected")}


def load(pattern, rule):
    frames = []
    for p in sorted(glob.glob(pattern)):
        try:
            f = pd.read_csv(p)
        except (pd.errors.EmptyDataError, EOFError):
            continue
        if len(f):
            frames.append(f[f.rule == rule])
    return pd.concat(frames, ignore_index=True)


def design(frame, kind):
    d = MF.derived(frame)
    return np.column_stack([d[f] for f in MF.MODEL_FEATURES[kind]])


def usable(frame, kind):
    pc, mc, ec = TARGETS[kind]
    X = design(frame, kind)
    ok = (frame[pc] > 0) & np.isfinite(frame[ec]) & np.isfinite(X).all(axis=1)
    return frame[ok.to_numpy()].reset_index(drop=True)


def fit(frame, kind):
    pc, mc, ec = TARGETS[kind]
    X = design(frame, kind)
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-12] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    n = frame[pc].to_numpy(float); r = frame[mc].to_numpy(float)
    q = np.clip(frame[ec].to_numpy(float) / n, 1e-9, 1 - 1e-9)
    w = 1.0 / frame.groupby("ticker").ticker.transform("count").to_numpy(); w *= len(w) / w.sum()

    def nll(beta):
        eta = np.clip(Z @ beta, -30, 30); sig = 1 / (1 + np.exp(-eta))
        p = np.clip(q + (1 - q) * sig, 1e-12, 1 - 1e-12)
        ll = w * (r * np.log(p) + (n - r) * np.log(1 - p))
        g = w * (r / p - (n - r) / (1 - p)) * (1 - q) * sig * (1 - sig)
        return -ll.sum() + 0.5 * PENALTY * (beta[1:] ** 2).sum(), -(Z.T @ g) + PENALTY * np.r_[0, beta[1:]]
    b0 = np.zeros(Z.shape[1]); b0[0] = -6.0
    res = minimize(nll, b0, jac=True, method="L-BFGS-B")
    model = dict(kind=kind, features=MF.MODEL_FEATURES[kind], mu=mu.tolist(), sd=sd.tolist(), beta=res.x.tolist(),
                 converged=bool(res.success))
    s = MF.score(model, frame)
    model["threshold_q80"] = float(np.quantile(s, 0.8)); model["threshold_q20"] = float(np.quantile(s, 0.2))
    return model


def deciles(frame, s, cols_list, rng):
    d = frame.assign(score=s)
    d["decile"] = pd.qcut(d.score.rank(method="first"), 10, labels=False)
    names = d.ticker.unique(); pos = {t: i for i, t in enumerate(names)}
    out = {}
    cubes = []
    for lab, (pc, mc, ec) in cols_list:
        agg = d.groupby(["ticker", "decile"])[[mc, ec, pc]].sum()
        cube = np.zeros((len(names), 10, 3))
        for (tk, dec), row in agg.iterrows():
            cube[pos[tk], int(dec)] = row.to_numpy(float)
        cubes.append(cube)
        tot = cube.sum(0)
        out[lab] = dict(excess_per_1000_pairs=(1000 * (tot[:, 0] - tot[:, 1]) / np.maximum(tot[:, 2], 1)).tolist(),
                        ratio=(tot[:, 0] / np.maximum(tot[:, 1], 1e-12)).tolist())
        out[lab]["spearman"] = float(spearmanr(np.arange(10), out[lab]["excess_per_1000_pairs"]).correlation)

    def tmb(cube, idx):
        c = cube[idx].sum(0)
        return 1000 * ((c[9, 0] - c[9, 1]) / max(c[9, 2], 1) - (c[0, 0] - c[0, 1]) / max(c[0, 2], 1))
    boots = rng.integers(0, len(names), (BOOT, len(names)))
    for i, (lab, _) in enumerate(cols_list):
        pt = tmb(cubes[i], np.arange(len(names))); b = np.array([tmb(cubes[i], x) for x in boots])
        out[lab]["top_minus_bottom"] = float(pt)
        out[lab]["ci95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    if len(cubes) == 2:
        pt = tmb(cubes[0], np.arange(len(names))) - tmb(cubes[1], np.arange(len(names)))
        b = np.array([tmb(cubes[0], x) - tmb(cubes[1], x) for x in boots])
        out["directional_difference"] = dict(value=float(pt), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    out["names"] = int(len(names))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--rule", required=True, choices=["run60", "stream5"])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--kinds", default="program,link,realtime")
    ap.add_argument("--evaluate", action="store_true", help="score the confirmation rows (read once)")
    args = ap.parse_args()
    rng = np.random.default_rng(20260914)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_all = load(args.train, args.rule)
    test_all = load(args.test, args.rule) if args.evaluate else None
    summary = dict(rule=args.rule, train_rows=int(len(train_all)))
    for kind in args.kinds.split(","):
        tr = usable(train_all, kind)
        model = fit(tr, kind)
        (out / ("model_%s_%s.json" % (kind, args.rule))).write_text(json.dumps(model, indent=1) + "\n")
        pc, mc, ec = TARGETS[kind]
        cols = [("target", (pc, mc, ec))]
        if kind == "link":
            cols.append(("opposite", ("link_opp_pairs", "link_opp_matches", "link_opp_expected")))
        res = dict(train_usable=int(len(tr)), converged=model["converged"],
                   coefficients=dict(zip(["intercept"] + model["features"], model["beta"])),
                   in_sample=deciles(tr if kind != "link" else tr[(tr.link_opp_pairs > 0) & np.isfinite(tr.link_opp_expected)],
                                     MF.score(model, tr if kind != "link" else tr[(tr.link_opp_pairs > 0) & np.isfinite(tr.link_opp_expected)]),
                                     cols, rng))
        if test_all is not None:
            te = usable(test_all, kind)
            if kind == "link":
                te = te[(te.link_opp_pairs > 0) & np.isfinite(te.link_opp_expected)].reset_index(drop=True)
            res["test_usable"] = int(len(te))
            res["out_of_sample"] = deciles(te, MF.score(model, te), cols, rng)
        summary[kind] = res
        print(kind, json.dumps({k: v for k, v in res.items() if k in ("train_usable", "test_usable", "converged")}))
    (out / ("fit_%s%s.json" % (args.rule, "_evaluated" if args.evaluate else ""))).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
