#!/usr/bin/env python3
"""Extract price-free fragment rows for the frozen strict-continuation experiment."""
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
import strict_continuation_common as SC


def extract(msg_path, ticker, simulation_model, sample_per_day=0):
    _context, packets = EP.reconstruct_packets(msg_path)
    fragments = FR.form_fragments(packets)
    if fragments.empty:
        return fragments
    with open(simulation_model) as handle:
        spec = json.load(handle)
    fragment_model = MM.RidgeLogit.from_dict(spec["fragment_model"])
    fragments = MM.score_fragments(fragments, fragment_model)
    fragments = FR.attach_prior_flow_state(fragments, packets)
    fragments = FR.attach_future_flow_outcomes(fragments, packets)
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(msg_path))
    date = int("".join(match.groups())) if match else 0
    fragments.insert(0, "ticker", ticker)
    fragments.insert(1, "date", date)
    if sample_per_day and len(fragments) > sample_per_day:
        rng = np.random.default_rng(EP.deterministic_seed(
            ticker, date, "strict-continuation-sample-v1"
        ))
        keep = np.sort(rng.choice(len(fragments), int(sample_per_day), replace=False))
        fragments = fragments.iloc[keep].reset_index(drop=True)
    columns = ["ticker", "date", "fragment_id", "start_time", "end_time", "sign"]
    columns += SC.CONTROL_FEATURES + ["fragment_score"]
    columns += sorted(set(column for column, _transform in SC.TARGETS.values()))
    return fragments[columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--simulation-model", required=True)
    ap.add_argument("--sample-per-day", type=int, default=0)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()
    frame = extract(args.msg, args.ticker, args.simulation_model, args.sample_per_day)
    if not frame.empty:
        frame.to_csv(sys.stdout, index=False, header=args.header)


if __name__ == "__main__":
    main()
