#!/usr/bin/env python3
"""Frozen temporal/name holdout evaluation for both packet-fragment avenues.

Avenue 1 fits a fixed ridge forecast of the future price-discovery proxy using only
formation-time variables.  Avenue 2 tests whether the simulation-trained fragment score
predicts future same-side packet flow after residualizing a timing/liquidity baseline.  Only
after scores are frozen does the script report executable bid/ask P&L.
"""
import argparse
import glob
import hashlib
import json

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "volume", "n_packets", "duration", "cv_packet_volume", "mean_gap", "cv_gap",
    "hidden_share", "spread_start", "depth_start", "depth_imbalance_start", "intensity",
    "tod_sin", "tod_cos",
]
POST_END_FEATURES = BASE_FEATURES + ["impact_to_end", "end_halfspread_bps"]
FLOW_BASELINE = BASE_FEATURES + [
    "prior_total_count_60s", "prior_count_imbalance_60s",
    "prior_total_volume_60s", "prior_volume_imbalance_60s",
    "prior_total_count_300s", "prior_count_imbalance_300s",
    "prior_total_volume_300s", "prior_volume_imbalance_300s",
]


def stable_holdout(ticker, modulus=5):
    value = int.from_bytes(hashlib.sha256(str(ticker).encode()).digest()[:4], "little")
    return value % modulus == 0


def _matrix(frame, columns):
    x = frame[columns].astype(float).to_numpy()
    for j, name in enumerate(columns):
        if (name in ("volume", "n_packets", "duration", "depth_start", "intensity") or
                name.startswith("prior_total_count_") or
                name.startswith("prior_total_volume_")):
            x[:, j] = np.log1p(np.maximum(x[:, j], 0.0))
        elif (name.startswith("prior_count_imbalance_") or
              name.startswith("prior_volume_imbalance_")):
            x[:, j] = np.sign(x[:, j]) * np.log1p(np.abs(x[:, j]))
    return x


class Ridge(object):
    def __init__(self, penalty=10.0):
        self.penalty = float(penalty)
        self.mean = None; self.scale = None; self.coef = None

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[ok]; y = y[ok]
        self.mean = X.mean(axis=0); self.scale = X.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        z = (X - self.mean) / self.scale
        z = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(z.shape[1]) * self.penalty; penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(z.T.dot(z) + penalty, z.T.dot(y))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        z = (X - self.mean) / self.scale
        return np.column_stack([np.ones(len(z)), z]).dot(self.coef)

    def to_dict(self, features):
        return {"penalty": self.penalty, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                "features": list(features)}

    @classmethod
    def from_dict(cls, values):
        model = cls(values.get("penalty", 10.0))
        model.mean = np.asarray(values["mean"], float)
        model.scale = np.asarray(values["scale"], float)
        model.coef = np.asarray(values["coef"], float)
        return model


def nw_mean(values, lags=10):
    x = np.asarray(values, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 20:
        return {"mean": np.nan, "t": np.nan, "n_days": int(n)}
    mean = x.mean(); residual = x - mean
    variance = residual.dot(residual) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * residual[lag:].dot(residual[:-lag]) / n
    standard_error = np.sqrt(max(variance, 0.0) / n)
    return {"mean": float(mean), "t": float(mean / standard_error) if standard_error else np.nan,
            "n_days": int(n)}


def daily_inference(frame, column):
    # Equal weight within name-day, then across names, matching the project convention.
    name_day = frame.groupby(["date", "ticker"])[column].mean()
    daily = name_day.groupby("date").mean().sort_index()
    return nw_mean(daily.to_numpy(float))


def evaluate_cut(frame, score, threshold):
    selected = frame[np.isfinite(score) & (score >= threshold)].copy()
    result = {"n_fragments": int(len(frame)), "n_selected": int(len(selected))}
    for column in ["permanent_proxy", "markout_5m", "markout_15m", "markout_30m",
                   "mean_executable", "executable_5m", "executable_15m", "executable_30m"]:
        if column in selected:
            result[column] = daily_inference(selected, column)
    return result


def load_panel(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("no files match %s" % pattern)
    frames = [pd.read_csv(path) for path in files if path]
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_numeric(panel["date"], errors="coerce")
    panel = panel[panel["date"].notna()].copy()
    panel["date"] = panel["date"].astype(int)
    panel["name_holdout"] = panel["ticker"].map(stable_holdout)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob for per-ticker CSVs")
    ap.add_argument("--train-end", type=int, default=20231231)
    ap.add_argument("--test-start", type=int, default=20240101)
    ap.add_argument("--out")
    args = ap.parse_args()
    panel = load_panel(args.input)
    train = panel[(panel.date <= args.train_end) & (~panel.name_holdout)].copy()
    test_seen = panel[(panel.date >= args.test_start) & (~panel.name_holdout)].copy()
    test_names = panel[(panel.date >= args.test_start) & panel.name_holdout].copy()
    result = {
        "design": {
            "train_end": args.train_end, "test_start": args.test_start,
            "train_names": int(train.ticker.nunique()),
            "temporal_test_names": int(test_seen.ticker.nunique()),
            "double_holdout_names": int(test_names.ticker.nunique()),
        }
    }

    # Two predeclared Avenue-1 specifications.  The second may use within-fragment price
    # response because the proposed decision occurs after fragment termination.
    avenue1 = {}
    for label, features in [("price_free", BASE_FEATURES), ("post_end", POST_END_FEATURES)]:
        forbidden = [x for x in features if x.startswith("markout_") or
                     x.startswith("executable_") or x.startswith("future_")]
        if forbidden:
            raise ValueError("future label leaked into formation features: %s" % forbidden)
        model = Ridge(10.0).fit(_matrix(train, features),
                                train["permanent_proxy"].to_numpy(float))
        train_score = model.predict(_matrix(train, features))
        threshold = float(np.nanquantile(train_score, 0.90))
        avenue1[label] = {
            "threshold_from_training": threshold,
            "temporal_holdout": evaluate_cut(
                test_seen, model.predict(_matrix(test_seen, features)), threshold),
            "temporal_and_name_holdout": evaluate_cut(
                test_names, model.predict(_matrix(test_names, features)), threshold),
        }
    result["avenue1_informed_flow"] = avenue1

    # Avenue 2: does a simulation-trained recovery score forecast order continuation beyond
    # the observable state that mechanically predicts activity?
    flow_model = Ridge(10.0).fit(
        _matrix(train, FLOW_BASELINE), train["future_count_imbalance_300s"].to_numpy(float)
    )
    score_threshold = float(np.nanquantile(train["fragment_score"], 0.90))
    avenue2 = {"score_threshold_from_training": score_threshold}
    for label, sample in [("temporal_holdout", test_seen),
                          ("temporal_and_name_holdout", test_names)]:
        sample = sample.copy()
        baseline = flow_model.predict(_matrix(sample, FLOW_BASELINE))
        sample["flow_residual_300s"] = sample["future_count_imbalance_300s"] - baseline
        selected = sample[sample["fragment_score"] >= score_threshold]
        avenue2[label] = {
            "all_flow_residual": daily_inference(sample, "flow_residual_300s"),
            "high_score_flow_residual": daily_inference(selected, "flow_residual_300s"),
            "high_score_future_count_imbalance": daily_inference(
                selected, "future_count_imbalance_300s"),
            "n_fragments": int(len(sample)), "n_selected": int(len(selected)),
        }
    result["avenue2_parent_reconstruction"] = avenue2
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
