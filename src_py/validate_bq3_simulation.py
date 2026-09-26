#!/usr/bin/env python3
"""Ground-truth stress test for the sign-blind timing-cluster definitions.

This is a simulator diagnostic, not evidence about real NASDAQ parents.  It reports whether
timing-only blocks actually recover known simulated parents under splitting, herding,
overlap, pauses, and partial observation, alongside the same Hurst-placebo statistic used on
real packets.
"""
import argparse
import json

import numpy as np
import pandas as pd

import burst_quality_v2 as BQ2
import burst_quality_v3 as BQ3
import metaorder_simulation as MS


def block_recovery(blocks, packets, purity_cutoff=0.8):
    recovered = set(); purities = []; true = []
    parent_ids = packets["parent_id"].to_numpy(int)
    all_parents = set(int(x) for x in parent_ids if int(x) >= 0)
    for first, last in blocks:
        ids = parent_ids[first:last + 1]
        valid = ids[ids >= 0]
        if not len(valid):
            purity = 0.0; dominant = -1
        else:
            values, counts = np.unique(valid, return_counts=True)
            k = int(np.argmax(counts)); dominant = int(values[k])
            purity = float(counts[k] / len(ids))
        is_true = purity >= purity_cutoff
        purities.append(purity); true.append(float(is_true))
        if is_true and dominant >= 0:
            recovered.add(dominant)
    return {
        "n_blocks": len(blocks),
        "mean_purity": float(np.mean(purities)) if purities else np.nan,
        "true_block_rate": float(np.mean(true)) if true else np.nan,
        "parent_coverage": float(len(recovered) / len(all_parents)) if all_parents else np.nan,
    }


def one_day(seed, scenario, draws):
    packets = MS.simulate_day(seed, scenario=scenario)
    times = packets["time"].to_numpy(float)
    signs = packets["sign"].to_numpy(float)
    rng = np.random.default_rng(seed + 1000003)
    rows = []
    for quantile in BQ3.GAP_QUANTILES:
        blocks, cutoff = BQ3.timing_blocks(times, quantile, BQ3.MIN_PACKETS)
        metrics = block_recovery(blocks, packets)
        h_real = BQ2.hurst(BQ2.block_signs(blocks, signs))
        placebo = []
        for _ in range(draws):
            rotated = BQ2.rotate_within_tod(times, signs, rng)
            placebo.append(BQ2.hurst(BQ2.block_signs(blocks, rotated)))
        placebo = np.asarray(placebo, float)
        h_placebo = float(np.nanmean(placebo)) if np.isfinite(placebo).any() else np.nan
        metrics.update({
            "scenario": scenario, "seed": int(seed),
            "definition": "time_q%02d_m%d" % (round(100 * quantile), BQ3.MIN_PACKETS),
            "gap_cutoff": cutoff, "h_real": h_real, "h_placebo": h_placebo,
            "dH": h_placebo - h_real if np.isfinite(h_placebo) and np.isfinite(h_real)
                  else np.nan,
        })
        rows.append(metrics)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-per-scenario", type=int, default=30)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=90210)
    ap.add_argument("--out")
    args = ap.parse_args()
    scenarios = ["splitting", "herding", "overlap", "pauses", "partial", "full"]
    rows = []
    for s_index, scenario in enumerate(scenarios):
        for day in range(args.days_per_scenario):
            rows.extend(one_day(args.seed + 1000 * s_index + day, scenario, args.draws))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["scenario", "definition"]).agg(
        days=("seed", "size"), usable_hurst_days=("dH", "count"),
        mean_blocks=("n_blocks", "mean"), mean_purity=("mean_purity", "mean"),
        true_block_rate=("true_block_rate", "mean"),
        parent_coverage=("parent_coverage", "mean"), mean_dH=("dH", "mean"),
    ).reset_index()
    result = {"design": vars(args), "rows": summary.to_dict(orient="records")}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
