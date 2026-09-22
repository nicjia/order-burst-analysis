#!/usr/bin/env python3
"""Canonical reconstruction of economic aggressive execution packets.

LOBSTER rows are execution *messages*, not necessarily economic child orders.  One
incoming marketable order can match several resting orders and therefore create several
type-4/type-5 rows at the same timestamp.  This module consolidates those rows before any
burst detector sees them.

Signing rules are deliberately conservative:

* type 4: ``Direction`` is the resting side, so aggressor sign is ``-Direction``;
* type 5: ``Direction`` is ignored;
* a type-5 row inherits the unique type-4 aggressor sign at the same timestamp;
* a hidden-only timestamp is signed only when its execution price lies outside the
  displayed quote immediately before the timestamp;
* everything else is sign 0 (ambiguous), never silently assigned to a side.

The historical ``burst_alt.reconstruct`` API is left untouched because its checksum is
pinned to existing result panels.  New research code must use this module.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0

PACKET_COLUMNS = [
    "packet_id", "time", "sign", "sign_source", "volume", "vwap",
    "min_price", "max_price", "n_messages", "n_visible", "n_hidden",
    "visible_volume", "hidden_volume", "pre_bid", "pre_ask", "pre_bid_size",
    "pre_ask_size", "spread", "depth", "row_first", "row_last",
]


def _empty_packets():
    return pd.DataFrame(columns=PACKET_COLUMNS)


def _outside_quote_sign(price, bid, ask, tol=1e-12):
    """Return aggressor sign only when price is unambiguous from the pre-event quote."""
    if not (np.isfinite(price) and np.isfinite(bid) and np.isfinite(ask)):
        return 0
    if price > ask + tol:
        return 1
    if price < bid - tol:
        return -1
    return 0


def _packet_row(group, sign, source, pre_bid, pre_ask, pre_bsz, pre_asz,
                packet_number):
    size = group["sz"].to_numpy(float)
    price = group["price"].to_numpy(float)
    volume = float(size.sum())
    vwap = float(np.dot(size, price) / volume) if volume > 0 else np.nan
    visible = group["ty"].to_numpy(int) == 4
    return {
        "packet_id": int(packet_number),
        "time": float(group["t"].iloc[0]),
        "sign": int(sign),
        "sign_source": source,
        "volume": volume,
        "vwap": vwap,
        "min_price": float(np.nanmin(price)),
        "max_price": float(np.nanmax(price)),
        "n_messages": int(len(group)),
        "n_visible": int(visible.sum()),
        "n_hidden": int((~visible).sum()),
        "visible_volume": float(size[visible].sum()),
        "hidden_volume": float(size[~visible].sum()),
        "pre_bid": float(pre_bid),
        "pre_ask": float(pre_ask),
        "pre_bid_size": float(pre_bsz),
        "pre_ask_size": float(pre_asz),
        "spread": float(pre_ask - pre_bid),
        "depth": float(pre_bsz + pre_asz),
        "row_first": int(group["row"].min()),
        "row_last": int(group["row"].max()),
    }


def reconstruct_packets(msg_path, rth_only=True, context=None):
    """Return ``(context, packets)`` for one LOBSTER message file.

    ``context`` is the tuple returned by :func:`burst_alt.reconstruct`.  Passing an existing
    context avoids repeating book reconstruction.  Exact timestamps define packet candidates;
    conflicting native signs at one timestamp are kept as separate packets.
    """
    if context is None:
        context = BA.reconstruct(msg_path)
    bt, _bm, bb, ba, bbsz, basz, _ofi, _legacy_trades = context

    df = pd.read_csv(
        msg_path, header=None, usecols=[0, 1, 2, 3, 4, 5],
        names=["t", "ty", "order_id", "sz", "px", "dr"],
    )
    df["row"] = np.arange(len(df), dtype=np.int64)
    df = df[df["ty"].isin([4, 5])].copy()
    if rth_only:
        df = df[(df["t"] >= RTH0) & (df["t"] < RTH1)]
    if df.empty:
        return context, _empty_packets()
    df["price"] = df["px"].to_numpy(float) / BA.SCALE
    df["native_sign"] = np.where(df["ty"].to_numpy(int) == 4,
                                  -df["dr"].to_numpy(int), 0)

    times = df["t"].drop_duplicates().to_numpy(float)
    pre_q = np.nextafter(times, -np.inf)
    pre_bid, pre_ask = BA.bbo_at(bt, bb, ba, pre_q)
    pre_bsz, pre_asz = BA.bbo_at(bt, bbsz, basz, pre_q)
    quote = {
        float(t): (pre_bid[i], pre_ask[i], pre_bsz[i], pre_asz[i])
        for i, t in enumerate(times)
    }

    rows = []
    packet_number = 0
    for timestamp, group in df.groupby("t", sort=True):
        bid, ask, bsz, asz = quote[float(timestamp)]
        visible = group[group["ty"] == 4]
        hidden = group[group["ty"] == 5]
        native = sorted(set(int(x) for x in visible["native_sign"] if x != 0))

        if len(native) == 1:
            # A unique visible aggressor sign identifies all executions caused by the
            # same-timestamp incoming order, including hidden matches.
            rows.append(_packet_row(group, native[0], "native+timestamp",
                                    bid, ask, bsz, asz, packet_number))
            packet_number += 1
            continue

        if len(native) > 1:
            # Do not merge two economically incompatible aggressors merely because the
            # data vendor reports the same timestamp resolution.
            for sign in native:
                sub = visible[visible["native_sign"] == sign]
                rows.append(_packet_row(sub, sign, "native-conflict-split",
                                        bid, ask, bsz, asz, packet_number))
                packet_number += 1
            # Hidden rows cannot inherit an ambiguous visible sign.  Apply only the
            # conservative outside-prequote rule below.

        if len(native) == 0 and not visible.empty:
            # Defensive only: valid type-4 rows should always have native direction.
            rows.append(_packet_row(visible, 0, "invalid-native-direction",
                                    bid, ask, bsz, asz, packet_number))
            packet_number += 1

        if not hidden.empty:
            hidden = hidden.copy()
            hidden["inferred_sign"] = [
                _outside_quote_sign(p, bid, ask) for p in hidden["price"]
            ]
            for sign, sub in hidden.groupby("inferred_sign", sort=True):
                source = "outside-prequote" if int(sign) != 0 else "ambiguous-hidden"
                rows.append(_packet_row(sub, int(sign), source,
                                        bid, ask, bsz, asz, packet_number))
                packet_number += 1

    packets = pd.DataFrame(rows, columns=PACKET_COLUMNS)
    packets = packets.sort_values(["time", "row_first", "sign"], kind="stable")
    packets = packets.reset_index(drop=True)
    packets["packet_id"] = np.arange(len(packets), dtype=np.int64)
    return context, packets


def deterministic_seed(*parts):
    """Stable 32-bit seed; unlike Python ``hash()``, identical across processes."""
    token = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def signed_packets(packets):
    """Directional packet view.  Ambiguous hidden-only packets remain available upstream."""
    if packets.empty:
        return packets.copy()
    return packets[packets["sign"].isin([-1, 1])].reset_index(drop=True)
