#!/usr/bin/env python3
"""Extract frozen liquidity-pause discrete hazard rows for one ticker-day."""
import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import execution_packets as EP
import fragment_reconstruction as FR
import liquidity_pause_common as LP
import metaorder_models as MM


def extract(msg_path, ticker, simulation_model, strict_frozen):
    context, packets = EP.reconstruct_packets(msg_path)
    fragments = FR.form_fragments(packets)
    if fragments.empty:
        return fragments
    with open(simulation_model) as handle:
        simulation = json.load(handle)
    with open(strict_frozen) as handle:
        strict = json.load(handle)
    fragments = MM.score_fragments(
        fragments, MM.RidgeLogit.from_dict(simulation["fragment_model"])
    )
    fragments = FR.attach_prior_flow_state(fragments, packets)
    risk = LP.build_risk_rows(fragments, packets, context, strict["score_top_decile"])
    if risk.empty:
        return risk
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(msg_path))
    date = int("".join(match.groups())) if match else 0
    risk.insert(0, "ticker", ticker)
    risk.insert(1, "date", date)
    columns = ["ticker", "date", "fragment_id", "sign", "interval_id",
               "risk_time", "event"]
    columns += LP.JOINT_FEATURES
    return risk[columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--simulation-model", required=True)
    ap.add_argument("--strict-frozen", required=True)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()
    frame = extract(args.msg, args.ticker, args.simulation_model, args.strict_frozen)
    if not frame.empty:
        frame.to_csv(sys.stdout, index=False, header=args.header)


if __name__ == "__main__":
    main()
