#!/usr/bin/env python3
"""Shared daily extractor for informed-flow and latent-parent research avenues.

One row is a price-free fragment made from economic execution packets.  Columns prefixed by
formation information are available at the fragment end.  Future markouts and executable
returns are labels only.  Simulation-trained fragment/campaign scores never use real returns.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import execution_packets as EP
import fragment_reconstruction as FR
import metaorder_models as MM


def _load_models(path):
    if path is None:
        return None, None, 0.5
    with open(path) as handle:
        spec = json.load(handle)
    fragment = MM.RidgeLogit.from_dict(spec["fragment_model"])
    join = MM.RidgeLogit.from_dict(spec["join_model"])
    threshold = float(spec.get("join_threshold", 0.5))
    return fragment, join, threshold


def extract(msg_path, ticker, model_path=None, gap=1.0, min_packets=3,
            sample_per_day=0, compact=False, return_state=False):
    context, packets = EP.reconstruct_packets(msg_path)
    fragments = FR.form_fragments(packets, gap=gap, min_packets=min_packets)
    if fragments.empty:
        return (fragments, context, packets) if return_state else fragments
    fragments = FR.attach_formation_price_features(fragments, context)
    fragment_model, join_model, join_threshold = _load_models(model_path)
    if fragment_model is not None:
        fragments = MM.score_fragments(fragments, fragment_model)
        fragments = MM.stitch_campaigns(fragments, join_model, threshold=join_threshold)
    else:
        fragments["fragment_score"] = np.nan
        fragments["campaign_id"] = np.arange(len(fragments), dtype=np.int64)
        fragments["join_probability"] = np.nan
    fragments = FR.attach_prior_flow_state(fragments, packets)
    fragments = FR.attach_future_flow_outcomes(fragments, packets)
    fragments = FR.attach_price_outcomes(fragments, context)
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(msg_path))
    date = int("".join(match.groups())) if match else 0
    fragments.insert(0, "ticker", ticker)
    fragments.insert(1, "date", date)
    fragments.insert(2, "definition", "packet_sil_%g_min%d" % (gap, min_packets))
    fragments["n_day_packets"] = len(packets)
    fragments["ambiguous_packet_share_day"] = float((packets["sign"] == 0).mean())
    fragments["ambiguous_hidden_volume_share_day"] = (
        float(packets.loc[packets["sign"] == 0, "hidden_volume"].sum() /
              packets["volume"].sum()) if packets["volume"].sum() > 0 else np.nan
    )
    if sample_per_day and len(fragments) > sample_per_day:
        rng = np.random.default_rng(EP.deterministic_seed(ticker, date, "two-avenue-sample"))
        keep = np.sort(rng.choice(len(fragments), size=int(sample_per_day), replace=False))
        fragments = fragments.iloc[keep].reset_index(drop=True)
    if compact:
        columns = [
            "ticker", "date", "definition", "fragment_id", "start_time", "end_time", "sign",
            "n_packets", "n_messages", "volume", "duration", "mean_packet_volume",
            "cv_packet_volume", "mean_gap", "cv_gap", "max_gap", "hidden_share",
            "spread_start", "depth_start", "depth_imbalance_start", "intensity", "tod_sin",
            "tod_cos", "impact_to_end", "end_halfspread_bps", "fragment_score",
            "campaign_id", "join_probability", "future_count_imbalance_60s",
            "future_volume_imbalance_60s", "future_count_imbalance_300s",
            "future_volume_imbalance_300s", "prior_total_count_60s",
            "prior_count_imbalance_60s", "prior_total_volume_60s",
            "prior_volume_imbalance_60s", "prior_total_count_300s",
            "prior_count_imbalance_300s", "prior_total_volume_300s",
            "prior_volume_imbalance_300s", "permanent_proxy", "persistent_positive",
            "markout_5m", "markout_15m", "markout_30m", "mean_executable",
            "executable_5m", "executable_15m", "executable_30m", "n_day_packets",
            "ambiguous_packet_share_day", "ambiguous_hidden_volume_share_day",
        ]
        fragments = fragments[columns]
    if return_state:
        return fragments, context, packets
    return fragments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--model")
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--min-packets", type=int, default=3)
    ap.add_argument("--sample-per-day", type=int, default=0)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()
    try:
        frame = extract(args.msg, args.ticker, args.model, args.gap, args.min_packets,
                        args.sample_per_day, args.compact)
        if not frame.empty:
            frame.to_csv(sys.stdout, index=False, header=args.header)
    except Exception as error:
        print("%s,ERR,%s" % (args.ticker, error), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
