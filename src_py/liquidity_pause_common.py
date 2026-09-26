#!/usr/bin/env python3
"""Frozen specification and risk-set construction for liquidity-pause-v1."""
import numpy as np
import pandas as pd

import burst_alt as BA
import strict_continuation_common as SC


INTERVALS = ((1.0, 2.0), (2.0, 5.0), (5.0, 10.0),
             (10.0, 30.0), (30.0, 60.0), (60.0, 300.0))
INTERVAL_DUMMIES = ("interval_2_5", "interval_5_10", "interval_10_30",
                    "interval_30_60", "interval_60_300")
RISK_FEATURES = [
    "fragment_score", "selected",
] + list(INTERVAL_DUMMIES) + [
    "end_spread_bps", "end_log_contra_depth", "end_log_same_depth",
    "risk_spread_change", "risk_contra_depth_change", "risk_same_depth_change",
    "risk_prior_total_count_1s", "risk_prior_count_imbalance_1s",
    "risk_prior_total_volume_1s", "risk_prior_volume_imbalance_1s",
]
BASE_FEATURES = list(SC.CONTROL_FEATURES) + RISK_FEATURES
SPREAD_FEATURES = BASE_FEATURES + ["selected_x_spread_change"]
DEPTH_FEATURES = BASE_FEATURES + ["selected_x_contra_depth_change"]
JOINT_FEATURES = BASE_FEATURES + [
    "selected_x_spread_change", "selected_x_contra_depth_change",
    "selected_x_same_depth_change",
]
MODEL_FEATURES = {
    "base": BASE_FEATURES,
    "spread": SPREAD_FEATURES,
    "depth": DEPTH_FEATURES,
    "joint": JOINT_FEATURES,
}


def matrix(frame, columns):
    """Apply only predeclared monotone transforms before model standardization."""
    x = frame[columns].astype(float).to_numpy()
    for j, name in enumerate(columns):
        if (name in ("volume", "n_packets", "duration", "depth_start", "intensity")
                or name.startswith("prior_total_count_")
                or name.startswith("prior_total_volume_")
                or name.startswith("risk_prior_total_count_")
                or name.startswith("risk_prior_total_volume_")):
            x[:, j] = np.log1p(np.maximum(x[:, j], 0.0))
        elif (name.startswith("prior_count_imbalance_")
              or name.startswith("prior_volume_imbalance_")
              or name.startswith("risk_prior_count_imbalance_")
              or name.startswith("risk_prior_volume_imbalance_")):
            x[:, j] = np.sign(x[:, j]) * np.log1p(np.abs(x[:, j]))
    return x


def _quote(context, query):
    bt, bm, bb, ba, bbsz, basz, _ofi, _trades = context
    q = np.asarray(query, float)
    bid, ask = BA.bbo_at(bt, bb, ba, q)
    bsz, asz = BA.bbo_at(bt, bbsz, basz, q)
    mid = BA.mid_at(bt, bm, q)
    return bid, ask, bsz, asz, mid


def _nonoverlap_fragments(fragments, horizon=300.0):
    """Chronological fixed-window selection, separately by sign and outcome-free."""
    keep = []
    for sign, group in fragments.groupby("sign", sort=True):
        window_end = -np.inf
        for idx, row in group.sort_values(["end_time", "fragment_id"], kind="stable").iterrows():
            end = float(row.end_time)
            if end >= window_end:
                keep.append(idx)
                window_end = end + float(horizon)
    return fragments.loc[sorted(keep)].sort_values(
        ["end_time", "fragment_id"], kind="stable"
    )


def build_risk_rows(fragments, packets, context, score_threshold):
    """Create discrete-time survival rows using only interval-start information."""
    if fragments.empty or packets.empty:
        return pd.DataFrame()
    fragments = fragments[(fragments.end_time + INTERVALS[-1][1]) < 57600.0].copy()
    fragments = _nonoverlap_fragments(fragments)
    if fragments.empty:
        return pd.DataFrame()
    p = packets.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    pt = p.time.to_numpy(float)
    ps = p.sign.to_numpy(int)
    pv = p.volume.to_numpy(float)
    rows = []
    for fragment in fragments.itertuples(index=False):
        end = float(fragment.end_time); sign = int(fragment.sign)
        end_q = np.nextafter(end, np.inf)
        end_bid, end_ask, end_bsz, end_asz, end_mid = _quote(context, [end_q])
        end_spread = (end_ask[0] - end_bid[0]) / end_mid[0] * 1e4
        end_contra = end_asz[0] if sign > 0 else end_bsz[0]
        end_same = end_bsz[0] if sign > 0 else end_asz[0]
        if not (np.isfinite(end_spread) and end_spread > 0
                and np.isfinite(end_contra) and end_contra >= 0
                and np.isfinite(end_same) and end_same >= 0):
            continue
        first = np.searchsorted(pt, end + 1.0, side="right")
        candidates = np.flatnonzero(ps[first:] == sign)
        event_time = pt[first + candidates[0]] if len(candidates) else np.inf
        selected = float(fragment.fragment_score >= float(score_threshold))
        base = {name: getattr(fragment, name) for name in SC.CONTROL_FEATURES}
        base.update({
            "fragment_id": int(fragment.fragment_id), "sign": sign,
            "fragment_score": float(fragment.fragment_score), "selected": selected,
            "end_spread_bps": float(end_spread),
            "end_log_contra_depth": float(np.log1p(end_contra)),
            "end_log_same_depth": float(np.log1p(end_same)),
        })
        for interval_id, (left, right) in enumerate(INTERVALS):
            if event_time <= end + left:
                break
            risk_time = end + left
            query = np.nextafter(risk_time, -np.inf)
            bid, ask, bsz, asz, mid = _quote(context, [query])
            spread = (ask[0] - bid[0]) / mid[0] * 1e4
            contra = asz[0] if sign > 0 else bsz[0]
            same = bsz[0] if sign > 0 else asz[0]
            if not (np.isfinite(spread) and spread > 0 and np.isfinite(contra)
                    and contra >= 0 and np.isfinite(same) and same >= 0):
                break
            lo = np.searchsorted(pt, risk_time - 1.0, side="left")
            hi = np.searchsorted(pt, risk_time, side="left")
            prior_sign = ps[lo:hi]; prior_volume = pv[lo:hi]
            same_prior = prior_sign == sign; opposite_prior = prior_sign == -sign
            row = dict(base)
            for dummy_id, dummy in enumerate(INTERVAL_DUMMIES, start=1):
                row[dummy] = float(interval_id == dummy_id)
            row.update({
                "interval_id": int(interval_id), "risk_time": float(risk_time),
                "event": float(event_time <= end + right),
                "risk_spread_change": float(np.log(spread / end_spread)),
                "risk_contra_depth_change": float(np.log1p(contra) - np.log1p(end_contra)),
                "risk_same_depth_change": float(np.log1p(same) - np.log1p(end_same)),
                "risk_prior_total_count_1s": int(same_prior.sum() + opposite_prior.sum()),
                "risk_prior_count_imbalance_1s": int(same_prior.sum() - opposite_prior.sum()),
                "risk_prior_total_volume_1s": float(
                    prior_volume[same_prior].sum() + prior_volume[opposite_prior].sum()
                ),
                "risk_prior_volume_imbalance_1s": float(
                    prior_volume[same_prior].sum() - prior_volume[opposite_prior].sum()
                ),
            })
            row["selected_x_spread_change"] = selected * row["risk_spread_change"]
            row["selected_x_contra_depth_change"] = (
                selected * row["risk_contra_depth_change"]
            )
            row["selected_x_same_depth_change"] = selected * row["risk_same_depth_change"]
            rows.append(row)
            if row["event"] > 0:
                break
    return pd.DataFrame(rows)


def validate_specification():
    forbidden = [name for name in BASE_FEATURES if name.startswith("future_")]
    if forbidden or set(SC.CONTROL_FEATURES) - set(BASE_FEATURES):
        raise ValueError("invalid liquidity-pause controls: %s" % forbidden)
    if JOINT_FEATURES[-3:] != ["selected_x_spread_change",
                              "selected_x_contra_depth_change",
                              "selected_x_same_depth_change"]:
        raise ValueError("invalid liquidity-pause interactions")


validate_specification()
