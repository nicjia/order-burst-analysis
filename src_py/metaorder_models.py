#!/usr/bin/env python3
"""Simulation-calibrated fragment scoring and pause-aware campaign stitching.

Real anonymous data have no parent-order answer key.  Models in this file are therefore fit
only where ``parent_id`` is known (simulation or future identified data), then frozen before
application to LOBSTER.  Their output is a probability, not an asserted institutional ID.
"""
import numpy as np
import pandas as pd


FRAGMENT_FEATURES = [
    "log_volume", "log_packets", "log_duration", "cv_packet_volume", "mean_gap",
    "cv_gap", "hidden_share", "spread_start", "log_depth", "depth_imbalance_start",
    "log_intensity", "tod_sin", "tod_cos",
]

JOIN_FEATURES = [
    "log_gap", "log_volume_ratio", "duration_ratio", "spread_change", "depth_change",
    "imbalance_change", "prior_fragment_score", "next_fragment_score",
]


class RidgeLogit(object):
    """Small dependency-free standardized logistic model with ridge regularization."""

    def __init__(self, ridge=1.0, max_iter=100, tol=1e-8):
        self.ridge = float(ridge)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mean_ = None
        self.scale_ = None
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[ok]; y = y[ok]
        if len(y) < 20 or len(np.unique(y)) < 2:
            raise ValueError("logit fit requires >=20 observations and both labels")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        Z = (X - self.mean_) / self.scale_
        Z = np.column_stack([np.ones(len(Z)), Z])
        beta = np.zeros(Z.shape[1], float)
        penalty = np.eye(Z.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        for _ in range(self.max_iter):
            eta = np.clip(Z.dot(beta), -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-eta))
            w = np.maximum(p * (1.0 - p), 1e-6)
            grad = Z.T.dot(y - p) - penalty.dot(beta)
            hess = (Z.T * w).dot(Z) + penalty
            step = np.linalg.solve(hess, grad)
            beta_new = beta + step
            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                break
            beta = beta_new
        self.coef_ = beta
        return self

    def predict_proba(self, X):
        if self.coef_ is None:
            raise ValueError("model is not fit")
        X = np.asarray(X, float)
        Z = (X - self.mean_) / self.scale_
        Z = np.column_stack([np.ones(len(Z)), Z])
        eta = np.clip(Z.dot(self.coef_), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-eta))

    def to_dict(self, features=None):
        if self.coef_ is None:
            raise ValueError("model is not fit")
        result = {
            "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
            "coef": self.coef_.tolist(),
        }
        if features is not None:
            result["features"] = list(features)
        return result

    @classmethod
    def from_dict(cls, values):
        model = cls()
        model.mean_ = np.asarray(values["mean"], float)
        model.scale_ = np.asarray(values["scale"], float)
        model.coef_ = np.asarray(values["coef"], float)
        return model


def fragment_feature_frame(fragments):
    f = fragments.copy()
    out = pd.DataFrame(index=f.index)
    out["log_volume"] = np.log1p(f["volume"].astype(float))
    out["log_packets"] = np.log1p(f["n_packets"].astype(float))
    out["log_duration"] = np.log1p(f["duration"].astype(float))
    out["cv_packet_volume"] = f["cv_packet_volume"].astype(float)
    out["mean_gap"] = f["mean_gap"].astype(float)
    out["cv_gap"] = f["cv_gap"].astype(float)
    out["hidden_share"] = f["hidden_share"].astype(float)
    out["spread_start"] = f["spread_start"].astype(float)
    out["log_depth"] = np.log1p(f["depth_start"].astype(float))
    out["depth_imbalance_start"] = f["depth_imbalance_start"].astype(float)
    out["log_intensity"] = np.log1p(f["intensity"].astype(float))
    out["tod_sin"] = f["tod_sin"].astype(float)
    out["tod_cos"] = f["tod_cos"].astype(float)
    return out[FRAGMENT_FEATURES]


def add_simulated_fragment_labels(fragments, packets, purity_cutoff=0.8):
    """Attach dominant-parent purity and recovery labels using simulated truth."""
    out = fragments.copy()
    purity = []; dominant = []; true_fragment = []
    for row in out.itertuples(index=False):
        ids = packets.iloc[int(row.packet_first):int(row.packet_last) + 1]["parent_id"]
        ids = ids.to_numpy(int)
        valid = ids[ids >= 0]
        if len(valid) == 0:
            dominant.append(-1); purity.append(0.0); true_fragment.append(0.0)
            continue
        values, counts = np.unique(valid, return_counts=True)
        k = int(np.argmax(counts))
        dom = int(values[k]); pur = float(counts[k] / len(ids))
        dominant.append(dom); purity.append(pur)
        true_fragment.append(float(pur >= purity_cutoff))
    out["dominant_parent"] = dominant
    out["parent_purity"] = purity
    out["true_fragment"] = true_fragment
    return out


def fit_fragment_model(simulated_fragments, ridge=1.0):
    X = fragment_feature_frame(simulated_fragments).to_numpy(float)
    y = simulated_fragments["true_fragment"].to_numpy(float)
    return RidgeLogit(ridge=ridge).fit(X, y)


def score_fragments(fragments, model):
    out = fragments.copy()
    out["fragment_score"] = model.predict_proba(
        fragment_feature_frame(out).to_numpy(float)
    )
    return out


def join_feature_frame(fragments):
    """One row for each consecutive pair in each sign-specific fragment stream."""
    rows = []
    for sign, group in fragments.groupby("sign", sort=False):
        group = group.sort_values("start_time")
        idx = group.index.to_numpy()
        for left, right in zip(idx[:-1], idx[1:]):
            a = fragments.loc[left]; b = fragments.loc[right]
            gap = max(0.0, float(b.start_time - a.end_time))
            rows.append({
                "left_index": int(left), "right_index": int(right), "sign": int(sign),
                "gap": gap, "log_gap": np.log1p(gap),
                "log_volume_ratio": abs(np.log((float(b.volume) + 1.0) /
                                               (float(a.volume) + 1.0))),
                "duration_ratio": abs(np.log((float(b.duration) + 1.0) /
                                              (float(a.duration) + 1.0))),
                "spread_change": abs(float(b.spread_start) - float(a.spread_start)),
                "depth_change": abs(np.log((float(b.depth_start) + 1.0) /
                                           (float(a.depth_start) + 1.0))),
                "imbalance_change": abs(float(b.depth_imbalance_start) -
                                        float(a.depth_imbalance_start)),
                "prior_fragment_score": float(a.get("fragment_score", 0.5)),
                "next_fragment_score": float(b.get("fragment_score", 0.5)),
            })
    return pd.DataFrame(rows)


def fit_join_model(simulated_fragments, ridge=1.0, max_gap=1800.0):
    pairs = join_feature_frame(simulated_fragments)
    if pairs.empty:
        raise ValueError("no fragment pairs available")
    pairs = pairs[pairs["gap"] <= max_gap].copy()
    left_parent = simulated_fragments.loc[pairs["left_index"], "dominant_parent"].to_numpy(int)
    right_parent = simulated_fragments.loc[pairs["right_index"], "dominant_parent"].to_numpy(int)
    y = ((left_parent >= 0) & (left_parent == right_parent)).astype(float)
    model = RidgeLogit(ridge=ridge).fit(pairs[JOIN_FEATURES].to_numpy(float), y)
    return model


def stitch_campaigns(fragments, join_model, threshold=0.5, max_gap=1800.0):
    """Join fragments within sign streams while allowing liquidity-sensitive pauses."""
    out = fragments.copy().reset_index(drop=True)
    pairs = join_feature_frame(out)
    if not pairs.empty:
        pairs["join_probability"] = join_model.predict_proba(
            pairs[JOIN_FEATURES].to_numpy(float)
        )
    campaign_id = np.full(len(out), -1, int)
    join_probability = np.full(len(out), np.nan, float)
    next_id = 0
    pair_lookup = {}
    for row in pairs.itertuples(index=False):
        pair_lookup[(int(row.left_index), int(row.right_index))] = (
            float(row.join_probability), float(row.gap)
        )
    for sign, group in out.groupby("sign", sort=False):
        idx = group.sort_values("start_time").index.to_numpy(int)
        previous = None
        for current in idx:
            if previous is None:
                campaign_id[current] = next_id; next_id += 1
            else:
                prob, gap = pair_lookup[(int(previous), int(current))]
                join_probability[current] = prob
                if prob >= threshold and gap <= max_gap:
                    campaign_id[current] = campaign_id[previous]
                else:
                    campaign_id[current] = next_id; next_id += 1
            previous = current
    out["campaign_id"] = campaign_id
    out["join_probability"] = join_probability
    return out


def recovery_metrics(labeled_fragments, packets):
    """Ground-truth fragment metrics; never callable on anonymous real data as proof."""
    if labeled_fragments.empty:
        return {"n_fragments": 0, "mean_purity": np.nan, "true_fragment_rate": np.nan,
                "parent_coverage": np.nan}
    parent_packets = packets[packets["parent_id"] >= 0]
    recovered = set()
    for row in labeled_fragments.itertuples(index=False):
        if int(row.dominant_parent) >= 0 and float(row.true_fragment) > 0:
            recovered.add(int(row.dominant_parent))
    all_parents = set(int(x) for x in parent_packets["parent_id"].unique())
    return {
        "n_fragments": int(len(labeled_fragments)),
        "mean_purity": float(labeled_fragments["parent_purity"].mean()),
        "true_fragment_rate": float(labeled_fragments["true_fragment"].mean()),
        "parent_coverage": float(len(recovered) / len(all_parents)) if all_parents else np.nan,
    }
