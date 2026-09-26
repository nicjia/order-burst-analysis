#!/usr/bin/env python3
"""Audit persisted session-join labels and probabilities without importing join code."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def graph_metrics(rows, model):
    parent = {}; truth = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for r in rows:
        a = (r["left_day"], r["left_index"]); b = (r["right_day"], r["right_index"])
        for x, label in ((a, int(r["left_parent"])), (b, int(r["right_parent"]))):
            parent.setdefault(x, x); truth[x] = label
        if float(r[model + "_probability"]) >= .5:
            parent[find(b)] = find(a)
    groups = defaultdict(list)
    for x in parent:
        groups[find(x)].append(x)
    choose2 = lambda n: n * (n - 1) // 2
    total_true = sum(choose2(n) for n in Counter((x[0], label) for x, label in truth.items() if label >= 0).values())
    predicted = 0; correct = 0
    for nodes in groups.values():
        predicted += choose2(len(nodes))
        correct += sum(choose2(n) for n in Counter((x[0], truth[x]) for x in nodes if truth[x] >= 0).values())
    return dict(candidate_fragment_nodes=len(parent), campaign_components=len(groups),
                fragment_pair_precision=correct / predicted if predicted else None,
                fragment_pair_recall=correct / total_true if total_true else None,
                scope="Transitive components on candidate-eligible fragments, using dominant-parent labels; not whole raw-parent recovery.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results/burst_information_v1")
    args = ap.parse_args(); p = Path(args.root)
    summary = json.loads((p / "join_session_diagnostic.json").read_text())
    with (p / "join_session_diagnostic_training_pairs.csv").open() as f:
        train = list(csv.DictReader(f))
    cross = [r for r in train if r["left_day"] != r["right_day"]]
    assert len(cross) / len(train) == summary["legacy_cross_day_fraction"]
    assert sum(int(r["legacy_label"]) for r in cross) == summary["cross_day_positive_labels"]
    with (p / "join_session_diagnostic.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all(r["left_day"] == r["right_day"] for r in rows)
    y = [float(r["label"]) for r in rows]
    assert all(int(y[i]) == int(int(r["left_parent"]) >= 0 and r["left_parent"] == r["right_parent"])
               for i, r in enumerate(rows))
    checks = {}; graphs = {}
    for name in ("legacy", "corrected"):
        prob = [float(r[name + "_probability"]) for r in rows]
        pred = [v >= .5 for v in prob]
        tp = sum(t == 1 and z for t, z in zip(y, pred))
        actual = dict(n=len(y), brier=sum((t - v) ** 2 for t, v in zip(y, prob)) / len(y),
                      precision=tp / sum(pred), recall=tp / sum(y), selected=sum(pred))
        for k, value in actual.items():
            assert math.isclose(value, summary[name + "_on_valid_test_pairs"][k], rel_tol=1e-12, abs_tol=1e-12)
        checks[name] = actual; graphs[name] = graph_metrics(rows, name)
    result = dict(status="pass", cross_day_training_pairs=len(cross), valid_test_pairs=len(rows),
                  checks=checks, graph_diagnostic=graphs,
                  scope="Direct CSV day/parent labels and Brier/confusion counts; generator realism not validated.")
    (p / "join_session_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
