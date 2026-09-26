#!/usr/bin/env python3
"""Session-scoped join candidates. Legacy join functions remain unchanged for provenance."""
import numpy as np
import pandas as pd

import metaorder_models as legacy


def session_columns(frame):
    return [c for c in ("simulation_day", "ticker", "date") if c in frame.columns]


def join_feature_frame(fragments):
    if not fragments.index.is_unique:
        raise ValueError("join input requires unique row indices")
    keys = session_columns(fragments) + ["sign"]
    rows = []
    for _session, group in fragments.groupby(keys, sort=False, dropna=False):
        # Legacy feature extraction is valid for a single instrument/session/sign.
        pairs = legacy.join_feature_frame(group)
        if not pairs.empty:
            rows.append(pairs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pair_labels(fragments, pairs):
    left = fragments.loc[pairs.left_index]
    right = fragments.loc[pairs.right_index]
    for column in session_columns(fragments):
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise ValueError("candidate pair crosses a session")
    a = left.dominant_parent.to_numpy(int); b = right.dominant_parent.to_numpy(int)
    return ((a >= 0) & (a == b)).astype(float)


def fit_join_model(fragments, ridge=1.0, max_gap=1800.0):
    pairs = join_feature_frame(fragments)
    if pairs.empty:
        raise ValueError("no within-session join candidates")
    pairs = pairs[pairs.gap <= max_gap].copy()
    return legacy.RidgeLogit(ridge=ridge).fit(pairs[legacy.JOIN_FEATURES].to_numpy(float),
                                             pair_labels(fragments, pairs))


def stitch_campaigns(fragments, join_model, threshold=0.5, max_gap=1800.0):
    """Never share campaign IDs across simulation days, dates, or instruments."""
    out = fragments.copy().reset_index(drop=True)
    out["campaign_id"] = -1
    out["join_probability"] = np.nan
    keys = session_columns(out)
    groups = out.groupby(keys, sort=False, dropna=False) if keys else [(None, out)]
    offset = 0
    for _session, group in groups:
        stitched = legacy.stitch_campaigns(group, join_model, threshold, max_gap)
        out.loc[group.index, "campaign_id"] = stitched.campaign_id.to_numpy() + offset
        out.loc[group.index, "join_probability"] = stitched.join_probability.to_numpy()
        if len(stitched):
            offset += int(stitched.campaign_id.max()) + 1
    return out
