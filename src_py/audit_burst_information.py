#!/usr/bin/env python3
"""Independent CSV/weighted-day/HAC recomputation; does not import production evaluator."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIRS = {"nonlinearity": ("ridge_state", "gbt_state"),
         "regime": ("gbt_state", "gbt_state_regime"),
         "burst": ("gbt_state_regime", "gbt_state_regime_burst"),
         "simulation_score": ("gbt_state_regime_burst", "gbt_state_regime_burst_score")}


def independent_stat(records):
    name_day = defaultdict(list)
    for ticker, date, value in records:
        name_day[date, ticker].append(value)
    daily = defaultdict(list)
    for (date, _ticker), values in name_day.items():
        daily[date].append(sum(values) / len(values))
    dates = sorted(daily)
    values = np.array([sum(daily[d]) / len(daily[d]) for d in dates])
    if not len(values):
        return dict(n=0, mean=None, se=None, t=None)
    mean = float(values.mean()); centered = values - mean
    if len(values) < 2:
        return dict(n=len(values), mean=mean, se=None, t=None)
    lags = min(10, len(values) - 1)
    distance = np.abs(np.arange(len(values))[:, None] - np.arange(len(values))[None, :])
    weight = np.maximum(1 - distance / (lags + 1), 0)
    se = float(np.sqrt(max(centered @ weight @ centered, 0)) / len(values))
    return dict(n=len(values), mean=mean, se=se, t=mean / se if se else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    args = ap.parse_args(); root = Path(args.root)
    summary = json.loads((root / "summary.json").read_text())
    spec = json.loads((root / "design.json").read_text())
    seen = set(spec["seen_names"]); held = set(spec["heldout_names"])
    expected = {(r["stage"], r["target"], r["cohort"], r["contrast"]): r
                for r in summary["comparisons"]}
    expected_execution = {(r["stage"], r["cohort"], r["model"]): r
                          for r in summary["execution_diagnostics"]}
    errors = []; max_difference = 0.; checked = 0; economics = 0

    def compare(actual, reference, label):
        nonlocal max_difference
        for key in ("n", "mean", "se", "t"):
            x, y = actual[key], reference[key]
            if x is None and y is None:
                continue
            if x is None or y is None or not np.isclose(x, y, rtol=1e-9, atol=1e-9):
                errors.append(dict(label=label, metric=key, audit=x, production=y))
            else:
                max_difference = max(max_difference, abs(x - y))

    for stage in ("third", "sixth", "completion"):
        for target in ("flow_60s", "flow_300s", "return_60s", "return_300s", "wait_cost_60s"):
            path = root / "predictions" / (stage + "_" + target + ".csv.gz")
            with gzip.open(path, "rt") as handle:
                rows = list(csv.DictReader(handle))
            if len({r["row_id"] for r in rows}) != len(rows):
                raise ValueError("duplicate predictions " + str(path))
            for r in rows:
                if int(r["date"]) // 10000 != spec["evaluation_year"]:
                    raise ValueError("wrong evaluation period")
                if r["ticker"] not in (seen if r["cohort"] == "seen" else held):
                    raise ValueError("wrong name cohort")
                for field in ["target"] + sorted(set(x for pair in PAIRS.values() for x in pair)):
                    if not np.isfinite(float(r[field])):
                        raise ValueError("nonfinite prediction or target")
            for cohort in ("seen", "heldout"):
                group = [r for r in rows if r["cohort"] == cohort]
                for label, (base, aug) in PAIRS.items():
                    values = [(r["ticker"], r["date"],
                               (float(r["target"]) - float(r[base])) ** 2
                               - (float(r["target"]) - float(r[aug])) ** 2) for r in group]
                    compare(independent_stat(values), expected[stage, target, cohort, label],
                            ":".join((stage, target, cohort, label)))
                    checked += 1
                if target == "wait_cost_60s":
                    until = {}; opportunities = []
                    for r in sorted(group, key=lambda r: (r["ticker"], r["date"], float(r["decision_time"]), r["row_id"])):
                        key = (r["ticker"], r["date"])
                        if float(r["reference_depth"]) < 1 or float(r["decision_time"]) < until.get(key, -np.inf):
                            continue
                        if float(r["wait_depth"]) < 1:
                            raise ValueError("invalid delayed execution depth")
                        opportunities.append(r); until[key] = float(r["reference_time"]) + 60
                    models = sorted(set(x for pair in PAIRS.values() for x in pair))
                    for model in models:
                        actual = defaultdict(list)
                        for r in opportunities:
                            cost = float(r["target"])
                            chosen_cost = cost if float(r[model]) < 0 else 0
                            base_cost = cost if float(r["gbt_state_regime"]) < 0 else 0
                            for name, value in (("savings_vs_now", -chosen_cost),
                                                ("savings_vs_wait", cost - chosen_cost),
                                                ("savings_vs_state_regime", base_cost - chosen_cost)):
                                actual[name].append((r["ticker"], r["date"], value))
                        reference = expected_execution[stage, cohort, model]
                        if len(opportunities) != reference["n_orders"]:
                            raise ValueError("opportunity count mismatch")
                        for name in ("savings_vs_now", "savings_vs_wait", "savings_vs_state_regime"):
                            compare(independent_stat(actual[name]), reference[name],
                                    ":".join((stage, cohort, model, name)))
                            economics += 1
    result = dict(status="pass" if not errors else "fail", comparisons_checked=checked,
                  execution_statistics_checked=economics, max_absolute_difference=max_difference,
                  errors=errors, summary_sha256=hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
                  note="Independent CSV grouping and matrix-form HAC; verifies saved predictions and economics, not raw-data reconstruction or causal identification.")
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
