#!/usr/bin/env python3
"""Stage 3 of fingerprint-v1: a program-likeness score for bursts, validated on real data.

Each burst k (fixed definition) has n_k same-side, same-depth-quartile untruncated non-round pairs,
R_k identical-size pairs among them, and a chance expectation E_k = sum over quartiles of pairs x
cross-day rate. A score s(x_k) built only from features that do not use child sizes is fit by
maximizing the binomial likelihood of R_k with success probability q_k + (1 - q_k) * sigmoid(w.x_k),
q_k = E_k / n_k (pairs within a burst are not independent, so this is an estimating equation;
uncertainty comes from a bootstrap over names).

Validation is out of sample in names and calendar: fit on the 2024 exploration names, evaluate once
on the 2021 confirmation names. A useful score concentrates excess identical-size repeats in its
top decile. Excess is a lower bound on same-origin structure, so decile lift, not calibrated
probability, is the claim.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATURES = ["log_packets", "log_duration", "log_intensity", "iat_cv", "log_iat_median", "truncated_share",
            "hidden_share", "log_spread", "log_exec_depth", "imbalance", "tod", "tod2", "log_activity",
            "opposite_share", "unsigned_share"]
PENALTY = 1.0
BOOT = 1000


def load(pattern):
    frames = []
    for path in sorted(glob.glob(pattern)):
        try:
            f = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(f):
            frames.append(f)
    if not frames:
        raise FileNotFoundError(pattern)
    d = pd.concat(frames, ignore_index=True)
    d["log_packets"] = np.log1p(d.n_packets); d["log_duration"] = np.log1p(d.duration)
    d["log_intensity"] = np.log(d.intensity.clip(lower=1e-6)); d["log_iat_median"] = np.log1p(d.iat_median)
    d["log_spread"] = np.log(d.spread_bps.clip(lower=1e-3)); d["tod2"] = d.tod ** 2
    d["log_activity"] = np.log1p(d.trailing_activity)
    d["opposite_share"] = d.n_opposite / (d.n_packets + d.n_opposite + d.n_unsigned)
    d["unsigned_share"] = d.n_unsigned / (d.n_packets + d.n_opposite + d.n_unsigned)
    return d


def usable(d):
    ok = (d.dm_pairs > 0) & np.isfinite(d.dm_expected) & np.isfinite(d[FEATURES]).all(axis=1)
    return d[ok].reset_index(drop=True)


def fit(train):
    X = train[FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-12] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    n = train.dm_pairs.to_numpy(float); r = train.dm_repeats.to_numpy(float)
    q = np.clip(train.dm_expected.to_numpy(float) / n, 1e-9, 1 - 1e-9)
    w_names = 1.0 / train.groupby("ticker").ticker.transform("count").to_numpy()  # equal weight per name
    w_names *= len(w_names) / w_names.sum()

    def nll(beta):
        eta = np.clip(Z @ beta, -30, 30); sig = 1 / (1 + np.exp(-eta))
        p = np.clip(q + (1 - q) * sig, 1e-12, 1 - 1e-12)
        ll = w_names * (r * np.log(p) + (n - r) * np.log(1 - p))
        dp = (1 - q) * sig * (1 - sig)
        g = w_names * (r / p - (n - r) / (1 - p)) * dp
        pen = PENALTY * np.r_[0, beta[1:]]
        return -ll.sum() + 0.5 * PENALTY * (beta[1:] ** 2).sum(), -(Z.T @ g) + pen
    beta0 = np.zeros(Z.shape[1]); beta0[0] = -6.0
    res = minimize(nll, beta0, jac=True, method="L-BFGS-B")
    return dict(mu=mu, sd=sd, beta=res.x, converged=bool(res.success))


def score(model, d):
    Z = np.column_stack([np.ones(len(d)), (d[FEATURES].to_numpy(float) - model["mu"]) / model["sd"]])
    return Z @ model["beta"]


def decile_table(d, s, rng):
    d = d.assign(score=s)
    d["decile"] = pd.qcut(d.score.rank(method="first"), 10, labels=False)
    rows = []
    for dec, g in d.groupby("decile"):
        rows.append(dict(decile=int(dec), bursts=len(g), pairs=float(g.dm_pairs.sum()),
                         repeats=float(g.dm_repeats.sum()), expected=float(g.dm_expected.sum()),
                         ratio=float(g.dm_repeats.sum() / g.dm_expected.sum()) if g.dm_expected.sum() else None,
                         excess_per_1000_pairs=float(1000 * (g.dm_repeats.sum() - g.dm_expected.sum()) / g.dm_pairs.sum())))
    table = pd.DataFrame(rows)
    # Bootstrap over names using per-name decile totals (identical to resampling rows by name).
    agg = d.groupby(["ticker", "decile"])[["dm_repeats", "dm_expected", "dm_pairs"]].sum()
    names = d.ticker.unique()
    cube = np.zeros((len(names), 10, 3))
    pos = {tk: i for i, tk in enumerate(names)}
    for (tk, dec), row in agg.iterrows():
        cube[pos[tk], int(dec)] = row.to_numpy(float)

    def top_minus_bottom(idx):
        c = cube[idx].sum(0)
        if c[9, 2] == 0 or c[0, 2] == 0:
            return np.nan
        return 1000 * ((c[9, 0] - c[9, 1]) / c[9, 2] - (c[0, 0] - c[0, 1]) / c[0, 2])
    point = top_minus_bottom(np.arange(len(names)))
    boots = [top_minus_bottom(rng.integers(0, len(names), len(names))) for _ in range(BOOT)]
    boots = [b for b in boots if np.isfinite(b)]
    rho = pd.Series(table.excess_per_1000_pairs.to_numpy()).corr(pd.Series(np.arange(10)), method="spearman")
    return table, dict(top_minus_bottom_excess_per_1000_pairs=float(point),
                       ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                       spearman_decile_excess=float(rho), names=int(len(names)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="glob of exploration burst-row CSVs")
    ap.add_argument("--test", required=True, help="glob of confirmation burst-row CSVs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260913)
    train_all, test_all = load(args.train), load(args.test)
    train, test = usable(train_all), usable(test_all)
    model = fit(train)
    in_table, in_stat = decile_table(train, score(model, train), rng)
    out_table, out_stat = decile_table(test, score(model, test), rng)
    coef = dict(zip(["intercept"] + FEATURES, model["beta"].tolist()))
    result = dict(scope="fit on exploration (2024) names; evaluated once on confirmation (2021) names",
                  train_bursts=len(train_all), train_usable=len(train), test_bursts=len(test_all), test_usable=len(test),
                  converged=model["converged"], standardized_coefficients=coef,
                  in_sample=dict(deciles=in_table.to_dict(orient="records"), **in_stat),
                  out_of_sample=dict(deciles=out_table.to_dict(orient="records"), **out_stat))
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("in_sample", "out_of_sample")}, indent=1))
    print(out_table.round(3).to_string(index=False)); print(out_stat)


if __name__ == "__main__":
    main()
