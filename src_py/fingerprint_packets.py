#!/usr/bin/env python3
"""Stage 1 of fingerprint-v1: cache one ticker-day of economic packets as a compact npz.

Licensed-data derivative: keep on the cluster, never commit. Uses the canonical
``execution_packets.reconstruct_packets`` (type-5 Direction ignored). Adds the pre-packet
state needed by stage 2. Every quote is the one prevailing strictly before the packet.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import execution_packets as EP


def packet_arrays(packets):
    p = packets.sort_values(["time", "packet_id"], kind="stable").reset_index(drop=True)
    sign = p["sign"].to_numpy(np.int8)
    volume = p["volume"].to_numpy(float)
    bid = p["pre_bid"].to_numpy(float); ask = p["pre_ask"].to_numpy(float)
    bsz = p["pre_bid_size"].to_numpy(float); asz = p["pre_ask_size"].to_numpy(float)
    good_quote = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    mid = np.where(good_quote, (bid + ask) / 2, np.nan)
    # Depth on the side the aggressor consumes (ask for buys, bid for sells).
    exec_depth = np.where(sign > 0, asz, np.where(sign < 0, bsz, np.nan))
    single_level = p["min_price"].to_numpy(float) == p["max_price"].to_numpy(float)
    # Untruncated: the packet did not exhaust displayed touch depth and did not walk the book,
    # so its volume is the incoming order's size rather than a property of the book.
    untruncated = (sign != 0) & good_quote & np.isfinite(exec_depth) & single_level & (volume < exec_depth)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread_bps = np.where(good_quote, (ask - bid) / mid * 1e4, np.nan)
        imbalance = np.where(good_quote & (bsz + asz > 0), sign * (bsz - asz) / (bsz + asz), np.nan)
    return dict(
        time=p["time"].to_numpy(float), sign=sign, volume=volume,
        untruncated=untruncated.astype(np.bool_), mid=mid, spread_bps=spread_bps,
        exec_depth=exec_depth, imbalance=imbalance,
        hidden_share=np.where(volume > 0, p["hidden_volume"].to_numpy(float) / volume, 0.0),
        n_messages=p["n_messages"].to_numpy(np.int32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(args.msg).name)
    if not match:
        raise ValueError("message filename must contain YYYY-MM-DD")
    _context, packets = EP.reconstruct_packets(args.msg)
    arrays = packet_arrays(packets) if len(packets) else {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(out)
    n = len(arrays.get("time", []))
    print(json.dumps({"ticker": args.ticker, "date": "".join(match.groups()), "packets": n,
                      "signed": int(np.sum(arrays["sign"] != 0)) if n else 0,
                      "untruncated": int(np.sum(arrays["untruncated"])) if n else 0}))


if __name__ == "__main__":
    main()
