#!/usr/bin/env python3
"""Independent coverage, schema, and NW(10) audit for strict-continuation-v1.

This deliberately does not import the production aggregator or its NW helper.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


TARGETS = ("count_60s", "count_300s", "volume_60s", "volume_300s")
BASE_COLUMNS = (
    "ticker", "date", "name_holdout", "n_fragments", "n_selected",
    "selection_rate",
)
SUFFIXES = (
    "base_mse", "aug_mse", "delta_mse", "selected_target",
    "selected_base_residual", "selected_aug_residual",
)


def nw10(values):
    """Bartlett-kernel HAC t-statistic for a constant, implemented independently."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return {"mean": None, "t": None, "n_days": int(n)}
    mean = float(np.mean(x))
    u = x - mean
    long_run = float(np.sum(u * u) / n)
    for lag in range(1, min(10, n - 1) + 1):
        covariance = float(np.sum(u[lag:] * u[:-lag]) / n)
        long_run += 2.0 * (1.0 - lag / 11.0) * covariance
    se = np.sqrt(max(long_run, 0.0) / n)
    return {"mean": mean, "t": float(mean / se) if se > 0 else None,
            "n_days": int(n)}


def daily_stat(frame, column, eligible):
    use = frame.loc[frame[eligible] > 0, ["date", column]]
    daily = use.groupby("date", sort=True)[column].mean()
    return nw10(daily.to_numpy(dtype=float))


def read_outputs(pattern):
    frames, empty, files = [], [], sorted(glob.glob(pattern))
    for path in files:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            empty.append(os.path.basename(path))
            continue
        if frame.empty:
            empty.append(os.path.basename(path))
        else:
            frames.append(frame)
    if not frames:
        raise ValueError("no nonempty output files")
    return files, empty, pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files, empty, data = read_outputs(args.input)
    expected = [line.strip() for line in open(args.universe) if line.strip()]
    observed = [os.path.splitext(os.path.basename(path))[0] for path in files]
    required = set(BASE_COLUMNS)
    for target in TARGETS:
        required.update(target + "_" + suffix for suffix in SUFFIXES)
    missing_columns = sorted(required - set(data.columns))
    core_numeric = sorted(set(BASE_COLUMNS) - {"ticker"})
    core_numeric += [target + "_" + suffix for target in TARGETS
                     for suffix in ("base_mse", "aug_mse", "delta_mse")]
    selected_numeric = [target + "_" + suffix for target in TARGETS
                        for suffix in ("selected_target", "selected_base_residual",
                                       "selected_aug_residual")]
    nonfinite_core = None
    nonfinite_selected_eligible = None
    zero_selection_undefined = None
    if not missing_columns:
        nonfinite_core = int(
            (~np.isfinite(data[core_numeric].to_numpy(float))).sum()
        )
        eligible = data.n_selected > 0
        nonfinite_selected_eligible = int(
            (~np.isfinite(data.loc[eligible, selected_numeric].to_numpy(float))).sum()
        )
        zero_selection_undefined = int(
            (~np.isfinite(data.loc[~eligible, selected_numeric].to_numpy(float))).sum()
        )

    result = {
        "experiment": "strict-continuation-v1-independent-audit",
        "coverage": {
            "expected_files": len(expected), "observed_files": len(files),
            "nonempty_files": len(files) - len(empty), "empty_files": empty,
            "missing_files": sorted(set(expected) - set(observed)),
            "extra_files": sorted(set(observed) - set(expected)),
        },
        "schema": {
            "missing_columns": missing_columns,
            "nonfinite_core_values": nonfinite_core,
            "nonfinite_selected_eligible_values": nonfinite_selected_eligible,
            "zero_selection_rows": int((data.n_selected == 0).sum()),
            "zero_selection_undefined_values": zero_selection_undefined,
            "non_2025_rows": int(((data.date.astype(int) // 10000) != 2025).sum()),
            "duplicate_name_days": int(data.duplicated(["ticker", "date"]).sum()),
            "invalid_selection_rates": int(
                ((data.selection_rate < 0) | (data.selection_rate > 1)).sum()
            ),
        },
        "n_name_days": int(len(data)), "n_names": int(data.ticker.nunique()),
        "cohorts": {},
    }
    audit_ok = (
        len(files) == len(expected) and not result["coverage"]["missing_files"]
        and not result["coverage"]["extra_files"] and not missing_columns
        and nonfinite_core == 0 and nonfinite_selected_eligible == 0
        and result["schema"]["non_2025_rows"] == 0
        and result["schema"]["duplicate_name_days"] == 0
        and result["schema"]["invalid_selection_rates"] == 0
    )
    result["audit_ok"] = bool(audit_ok)

    for label, cohort in (
        ("seen_names", data[data.name_holdout == 0]),
        ("heldout_names", data[data.name_holdout == 1]),
    ):
        values = {"n_name_days": int(len(cohort)),
                  "n_names": int(cohort.ticker.nunique()), "targets": {}}
        for target in TARGETS:
            values["targets"][target] = {
                "delta_mse": daily_stat(cohort, target + "_delta_mse", "n_fragments"),
                "selected_base_residual": daily_stat(
                    cohort, target + "_selected_base_residual", "n_selected"
                ),
            }
        result["cohorts"][label] = values

    primary = []
    short_sign = []
    for cohort in result["cohorts"].values():
        c300 = cohort["targets"]["count_300s"]
        primary.extend([
            c300["delta_mse"]["mean"] > 0 and c300["delta_mse"]["t"] > 2,
            c300["selected_base_residual"]["mean"] > 0
            and c300["selected_base_residual"]["t"] > 2,
        ])
        c60 = cohort["targets"]["count_60s"]
        short_sign.extend([
            c60["delta_mse"]["mean"] > 0,
            c60["selected_base_residual"]["mean"] > 0,
        ])
    result["frozen_gate"] = {
        "primary_300s_conditions": primary,
        "count_60s_positive_signs": short_sign,
        "pass": bool(audit_ok and all(primary) and all(short_sign)),
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
