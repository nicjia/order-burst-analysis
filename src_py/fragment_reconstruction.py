#!/usr/bin/env python3
"""Price-free fragment formation and outcome attachment for the two research avenues.

Formation uses only information available by the fragment end.  Future prices are attached
as *labels* in a separate function and are never allowed to decide whether a fragment exists.
"""
import numpy as np
import pandas as pd


FRAGMENT_COLUMNS = [
    "fragment_id", "packet_first", "packet_last", "start_time", "end_time", "sign",
    "n_packets", "n_messages", "volume", "duration", "mean_packet_volume",
    "cv_packet_volume", "mean_gap", "cv_gap", "max_gap", "hidden_share",
    "ambiguous_hidden_share", "spread_start", "depth_start", "depth_imbalance_start",
    "intensity", "tod_sin", "tod_cos",
]


def _cv(x):
    x = np.asarray(x, float)
    if len(x) < 2 or not np.isfinite(x).all() or abs(x.mean()) < 1e-12:
        return 0.0
    return float(x.std(ddof=1) / abs(x.mean()))


def _summarize_fragment(packets, i, j, fragment_id):
    g = packets.iloc[i:j + 1]
    times = g["time"].to_numpy(float)
    volumes = g["volume"].to_numpy(float)
    gaps = np.diff(times)
    start = float(times[0]); end = float(times[-1])
    hidden_volume = float(g["hidden_volume"].sum())
    total_volume = float(volumes.sum())
    bsz = float(g["pre_bid_size"].iloc[0]); asz = float(g["pre_ask_size"].iloc[0])
    depth = bsz + asz
    seconds = start - 34200.0
    angle = 2.0 * np.pi * seconds / 23400.0
    return {
        "fragment_id": int(fragment_id),
        "packet_first": int(i),
        "packet_last": int(j),
        "start_time": start,
        "end_time": end,
        "sign": int(g["sign"].iloc[0]),
        "n_packets": int(len(g)),
        "n_messages": int(g["n_messages"].sum()),
        "volume": total_volume,
        "duration": max(0.0, end - start),
        "mean_packet_volume": float(volumes.mean()),
        "cv_packet_volume": _cv(volumes),
        "mean_gap": float(gaps.mean()) if len(gaps) else 0.0,
        "cv_gap": _cv(gaps),
        "max_gap": float(gaps.max()) if len(gaps) else 0.0,
        "hidden_share": hidden_volume / total_volume if total_volume > 0 else 0.0,
        "ambiguous_hidden_share": 0.0,
        "spread_start": float(g["spread"].iloc[0]),
        "depth_start": depth,
        "depth_imbalance_start": (bsz - asz) / depth if depth > 0 else 0.0,
        "intensity": len(g) / max(end - start, 1e-3),
        "tod_sin": float(np.sin(angle)),
        "tod_cos": float(np.cos(angle)),
    }


def form_fragments(packets, gap=1.0, min_packets=3):
    """Form same-side local execution episodes from economic packets.

    A sign-0 packet ends a directional run.  This is conservative: unidentified hidden flow
    is retained in the packet tape but cannot be smuggled into a buy or sell fragment.
    """
    if packets.empty:
        return pd.DataFrame(columns=FRAGMENT_COLUMNS)
    p = packets.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    rows = []
    i = 0
    while i < len(p):
        sign = int(p.at[i, "sign"])
        if sign == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(p):
            next_sign = int(p.at[j + 1, "sign"])
            next_gap = float(p.at[j + 1, "time"] - p.at[j, "time"])
            if next_sign != sign or next_gap >= gap:
                break
            j += 1
        if j - i + 1 >= min_packets:
            rows.append(_summarize_fragment(p, i, j, len(rows)))
        i = j + 1
    return pd.DataFrame(rows, columns=FRAGMENT_COLUMNS)


def attach_price_outcomes(fragments, context, buffer_seconds=1.0,
                          horizons=(300.0, 900.0, 1800.0)):
    """Attach non-overlapping price-discovery labels and executable P&L.

    These columns are outcomes, not formation inputs.  The reference price is one second
    after fragment termination.  ``persistent_positive`` requires the signed move to be
    positive at every requested horizon.  This is a price-discovery proxy, not proof of
    private information and not literal infinite-horizon permanence.
    """
    import burst_alt as BA

    out = fragments.copy()
    if out.empty:
        return out
    bt, bm, bb, ba, _bbsz, _basz, _ofi, _trades = context
    end = out["end_time"].to_numpy(float)
    sign = out["sign"].to_numpy(float)
    ref_time = end + float(buffer_seconds)
    ref_mid = BA.mid_at(bt, bm, ref_time)
    entry_bid, entry_ask = BA.bbo_at(bt, bb, ba, ref_time)
    out["reference_time"] = ref_time
    out["reference_mid"] = ref_mid
    markout_cols = []
    executable_cols = []
    for horizon in horizons:
        label = "%dm" % int(round(horizon / 60.0))
        mid_h = BA.mid_at(bt, bm, end + horizon)
        exit_bid, exit_ask = BA.bbo_at(bt, bb, ba, end + horizon)
        with np.errstate(invalid="ignore", divide="ignore"):
            markout = sign * (mid_h - ref_mid) / ref_mid * 1e4
            entry = np.where(sign > 0, entry_ask, entry_bid)
            exit_price = np.where(sign > 0, exit_bid, exit_ask)
            executable = sign * (exit_price - entry) / ref_mid * 1e4
        col_m = "markout_%s" % label
        col_x = "executable_%s" % label
        out[col_m] = markout
        out[col_x] = executable
        markout_cols.append(col_m)
        executable_cols.append(col_x)
    values = out[markout_cols].to_numpy(float)
    out["permanent_proxy"] = np.nanmedian(values, axis=1)
    out["persistent_positive"] = np.where(
        np.isfinite(values).all(axis=1), (values.min(axis=1) > 0).astype(float), np.nan
    )
    out["mean_executable"] = np.nanmean(out[executable_cols].to_numpy(float), axis=1)
    return out


def attach_formation_price_features(fragments, context):
    """Attach price response observed by fragment end (valid only for post-end decisions)."""
    import burst_alt as BA

    out = fragments.copy()
    if out.empty:
        return out
    bt, bm, bb, ba, _bbsz, _basz, _ofi, _trades = context
    start = out["start_time"].to_numpy(float)
    end = out["end_time"].to_numpy(float)
    sign = out["sign"].to_numpy(float)
    m0 = BA.mid_at(bt, bm, np.nextafter(start, -np.inf))
    m1 = BA.mid_at(bt, bm, end)
    end_bid, end_ask = BA.bbo_at(bt, bb, ba, end)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["impact_to_end"] = sign * (m1 - m0) / m0 * 1e4
        out["end_halfspread_bps"] = (end_ask - end_bid) / (2.0 * m1) * 1e4
    return out


def attach_future_flow_outcomes(fragments, packets, horizons=(60.0, 300.0)):
    """Attach future packet-flow labels, separate from price and from formation."""
    out = fragments.copy()
    if out.empty:
        return out
    p = packets.sort_values("time").reset_index(drop=True)
    times = p["time"].to_numpy(float)
    signs = p["sign"].to_numpy(int)
    volumes = p["volume"].to_numpy(float)
    for horizon in horizons:
        label = "%ds" % int(horizon)
        same_count = []; opposite_count = []; same_volume = []; opposite_volume = []
        for row in out.itertuples(index=False):
            left = np.searchsorted(times, float(row.end_time), side="right")
            right = np.searchsorted(times, float(row.end_time) + horizon, side="right")
            s = signs[left:right]; v = volumes[left:right]
            same = s == int(row.sign); opposite = s == -int(row.sign)
            same_count.append(int(same.sum())); opposite_count.append(int(opposite.sum()))
            same_volume.append(float(v[same].sum())); opposite_volume.append(float(v[opposite].sum()))
        out["future_same_count_%s" % label] = same_count
        out["future_opposite_count_%s" % label] = opposite_count
        out["future_same_volume_%s" % label] = same_volume
        out["future_opposite_volume_%s" % label] = opposite_volume
        out["future_count_imbalance_%s" % label] = (
            out["future_same_count_%s" % label] - out["future_opposite_count_%s" % label]
        )
        out["future_volume_imbalance_%s" % label] = (
            out["future_same_volume_%s" % label] - out["future_opposite_volume_%s" % label]
        )
    return out


def attach_prior_flow_state(fragments, packets, horizons=(60.0, 300.0)):
    """Attach signed-flow state observable strictly before each fragment starts.

    This is a formation control, not an outcome.  Packets sharing the fragment's first
    timestamp are excluded because their economic ordering is not observable.
    """
    out = fragments.copy()
    if out.empty:
        return out
    p = packets.sort_values("time").reset_index(drop=True)
    times = p["time"].to_numpy(float)
    signs = p["sign"].to_numpy(int)
    volumes = p["volume"].to_numpy(float)
    for horizon in horizons:
        label = "%ds" % int(horizon)
        same_count = []; opposite_count = []; same_volume = []; opposite_volume = []
        for row in out.itertuples(index=False):
            start = float(row.start_time)
            left = np.searchsorted(times, start - horizon, side="left")
            right = np.searchsorted(times, start, side="left")
            s = signs[left:right]; v = volumes[left:right]
            same = s == int(row.sign); opposite = s == -int(row.sign)
            same_count.append(int(same.sum())); opposite_count.append(int(opposite.sum()))
            same_volume.append(float(v[same].sum())); opposite_volume.append(float(v[opposite].sum()))
        out["prior_same_count_%s" % label] = same_count
        out["prior_opposite_count_%s" % label] = opposite_count
        out["prior_total_count_%s" % label] = np.asarray(same_count) + np.asarray(opposite_count)
        out["prior_count_imbalance_%s" % label] = (
            np.asarray(same_count) - np.asarray(opposite_count)
        )
        out["prior_same_volume_%s" % label] = same_volume
        out["prior_opposite_volume_%s" % label] = opposite_volume
        out["prior_total_volume_%s" % label] = np.asarray(same_volume) + np.asarray(opposite_volume)
        out["prior_volume_imbalance_%s" % label] = (
            np.asarray(same_volume) - np.asarray(opposite_volume)
        )
    return out
