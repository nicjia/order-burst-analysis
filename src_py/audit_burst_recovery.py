#!/usr/bin/env python3
"""Independently enumerate pair labels for every saved synthetic recovery cell."""
import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from burst_recovery_diagnostic import tape, blocks


def explicit_pairs(frame, groups):
    ids = frame.parent_id.to_numpy(int)
    true = set()
    for parent in set(ids) - {-1}:
        true.update(combinations(np.flatnonzero(ids == parent).tolist(), 2))
    predicted = set()
    for a, b in groups:
        predicted.update(combinations(range(a, b), 2))
    intersection = len(true & predicted)
    return (intersection / len(predicted) if predicted else np.nan,
            intersection / len(true) if true else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/burst_information_v1")
    args = ap.parse_args(); root = Path(args.root)
    source = root / "recovery.csv"; rows = pd.read_csv(source)
    errors = []; checked = 0; maximum = 0.0
    for (day, treatment), group in rows.groupby(["day", "treatment"]):
        frame, _total = tape(91000 + int(day), treatment)
        for row in group.itertuples():
            precision, recall = explicit_pairs(frame, blocks(frame, row.definition))
            for name, actual in (("pair_precision", precision), ("pair_recall_observed", recall)):
                expected = getattr(row, name)
                if np.isnan(actual) and np.isnan(expected):
                    continue
                delta = abs(actual - expected)
                if not np.isfinite(delta) or delta > 1e-12:
                    errors.append(dict(day=int(day), treatment=treatment,
                                       definition=row.definition, metric=name))
                else:
                    maximum = max(maximum, delta)
            checked += 1
    summary = json.loads((root / "recovery.json").read_text())["summary"]
    numeric = [c for c in rows if c not in ("day", "treatment", "definition")]
    for cell in summary:
        group = rows[(rows.treatment == cell["treatment"]) & (rows.definition == cell["definition"])]
        for key in numeric:
            actual = group[key].mean(); expected = cell[key]
            if expected is None and np.isnan(actual):
                continue
            if not np.isclose(actual, expected, rtol=1e-8, atol=1e-9):
                errors.append(dict(treatment=cell["treatment"], definition=cell["definition"], metric=key))
    intervals = []
    rng = np.random.default_rng(8203)
    for (treatment, definition), group in rows.groupby(["treatment", "definition"]):
        for metric in ("mean_purity", "pair_precision", "pair_recall_observed", "parent_packet_coverage_full"):
            values = group[metric].dropna().to_numpy()
            if not len(values):
                continue
            boot = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            intervals.append(dict(treatment=treatment, definition=definition, metric=metric,
                                  mean=float(values.mean()), lo95=float(lo), hi95=float(hi), n=len(values)))
    result = dict(status="pass" if not errors else "fail", checked_cells=checked,
                  max_pair_error=maximum, errors=errors, intervals=intervals,
                  csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  note="Pair audit enumerates memberships independently; tape and detector definitions shared. Intervals are Monte Carlo day uncertainty, not NASDAQ inference.")
    (root / "recovery_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "intervals"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
