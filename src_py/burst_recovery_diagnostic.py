#!/usr/bin/env python3
"""Paired mechanism ablations. Synthetic labels are not evidence of real parent identity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TREATMENTS = ("isolated", "background", "dense_background", "overlap", "pauses",
              "partial_35pct", "ambiguous_20pct", "herding_only", "combined")


def tape(seed, treatment):
    rng = np.random.default_rng(seed)
    events = []
    for parent in range(60):
        n = int(rng.integers(12, 61))
        side = int(rng.choice([-1, 1]))
        gaps = rng.exponential(0.3, n)
        pause = rng.random(n) < 0.10
        if treatment in ("pauses", "combined"):
            gaps = gaps + 10.0 * pause
        start = 34300.0 + 300.0 * parent
        if treatment in ("overlap", "combined"):
            start = 34300.0 + 900.0 * (parent // 3) + 0.4 * (parent % 3)
        volumes = rng.lognormal(np.log(100 + parent * 5), 0.25, n)
        events.extend(zip(start + np.cumsum(gaps), [side] * n, volumes, [parent] * n))
    frame = pd.DataFrame(events, columns=["time", "sign", "volume", "parent_id"])
    if treatment == "herding_only":
        # Identical observed sequence to isolated splitting, but each trade is independent.
        # An explicit observational-equivalence example, not a calibrated herding model.
        frame["parent_id"] = -1
    if treatment in ("background", "dense_background", "combined"):
        noise = np.random.default_rng(seed + 100000)
        rate = 2.0 if treatment == "dense_background" else 0.2
        n = int(noise.poisson(23400 * rate))
        bg = pd.DataFrame({"time": noise.uniform(34200, 57600, n),
                           "sign": noise.choice([-1, 1], n),
                           "volume": noise.lognormal(4.7, 0.7, n), "parent_id": -1})
        frame = pd.concat([frame, bg], ignore_index=True)
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    total_parent = int((frame.parent_id >= 0).sum())
    if treatment in ("partial_35pct", "combined"):
        keep = np.random.default_rng(seed + 200000).random(len(frame)) < 0.35
        frame = frame[keep].reset_index(drop=True)
    if treatment in ("ambiguous_20pct", "combined"):
        hide = np.random.default_rng(seed + 300000).random(len(frame)) < 0.2
        frame.loc[hide, "sign"] = 0
    return frame, total_parent


def blocks(frame, definition):
    if frame.empty:
        return []
    t = frame.time.to_numpy(); s = frame.sign.to_numpy()
    cut = np.diff(t) >= 1.0
    if definition != "timing":
        cut |= (s[1:] != s[:-1]) | (s[1:] == 0) | (s[:-1] == 0)
    if definition == "size_run":
        cut |= np.abs(np.diff(np.log(frame.volume.to_numpy()))) > 0.7
    edges = np.r_[0, np.flatnonzero(cut) + 1, len(frame)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])
            if b - a >= 3 and (definition == "timing" or s[a] != 0)]


def recovery(frame, groups, total_parent=None):
    ids = frame.parent_id.to_numpy(int)
    parent_ids, parent_counts = np.unique(ids[ids >= 0], return_counts=True)
    count = dict(zip(parent_ids, parent_counts))
    pair_den = sum(n * (n - 1) / 2 for n in parent_counts)
    true_pairs = 0; predicted_pairs = 0; detected = 0; captured = 0
    purities = []; recalls = []; parent_groups = {int(p): 0 for p in parent_ids}
    high = 0; noise_blocks = 0
    for first, stop in groups:
        block = ids[first:stop]; n = len(block)
        p, c = np.unique(block[block >= 0], return_counts=True)
        predicted_pairs += n * (n - 1) / 2
        true_pairs += sum(k * (k - 1) / 2 for k in c)
        detected += n; captured += int((block >= 0).sum())
        purity = float(max(c) / n) if len(c) else 0.0
        purities.append(purity); high += purity >= 0.8; noise_blocks += not len(c)
        if len(c):
            winner = int(p[np.argmax(c)])
            recalls.append(float(max(c) / count[winner]))
        for parent in p:
            parent_groups[int(parent)] += 1
    n_parent = int((ids >= 0).sum())
    return {
        "n_packets": len(frame), "n_parent_packets": n_parent, "n_blocks": len(groups),
        "mean_purity": float(np.mean(purities)) if purities else None,
        "high_purity_fraction": high / len(groups) if groups else None,
        "noise_only_fraction": noise_blocks / len(groups) if groups else None,
        "pair_precision": true_pairs / predicted_pairs if predicted_pairs else None,
        "pair_recall_observed": true_pairs / pair_den if pair_den else None,
        "detected_parent_share": captured / detected if detected else None,
        "parent_packet_coverage_observed": captured / n_parent if n_parent else None,
        "parent_packet_coverage_full": captured / total_parent if total_parent else None,
        "mean_dominant_parent_recall": float(np.mean(recalls)) if recalls else None,
        "mean_fragments_per_observed_parent": float(np.mean(list(parent_groups.values())))
        if parent_groups else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for day in range(args.days):
        for treatment in TREATMENTS:
            frame, total = tape(91000 + day, treatment)
            for definition in ("run", "timing", "size_run"):
                row = recovery(frame, blocks(frame, definition), total)
                rows.append(dict(day=day, treatment=treatment, definition=definition, **row))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix(".csv"), index=False)
    metrics = [c for c in df if c not in ("day", "treatment", "definition")]
    summary = df.groupby(["treatment", "definition"])[metrics].mean().reset_index()
    result = {"experiment": "burst-recovery-ablation-v1", "days": args.days,
              "scope": "Synthetic mechanisms only; herding_only relabels an identical tape.",
              "summary": json.loads(summary.to_json(orient="records"))}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(summary[["treatment", "definition", "mean_purity", "pair_precision",
                   "pair_recall_observed", "parent_packet_coverage_full"]].to_string(index=False))


if __name__ == "__main__":
    main()
