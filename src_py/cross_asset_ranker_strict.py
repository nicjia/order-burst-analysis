#!/usr/bin/env python3
"""
cross_asset_ranker_strict.py

Aggregate model-zoo JSON outputs across tickers with strict coverage filters.

- Reads JSONs from results/zoo_bursts_<TICKER>_<phase>/*.json
- Computes mean/std AUC by (model_name, target)
- Keeps only rows with complete ticker coverage (default: all provided tickers)
- Produces:
  1) best model per target
  2) overall model ranking across targets

Usage:
  python3 src_py/cross_asset_ranker_strict.py \
      --results-root results \
      --tickers "NVDA,TSLA,JPM,MS" \
      --phases "unfiltered,filtered" \
      --out-prefix results/strict_rankings
"""

import argparse
import glob
import json
import os
from typing import List

import pandas as pd


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def collect_rows(results_root: str, tickers: List[str], phases: List[str]):
    rows = []
    for ticker in tickers:
        for phase in phases:
            pattern = os.path.join(results_root, f"zoo_bursts_{ticker}_{phase}", "*.json")
            for path in glob.glob(pattern):
                try:
                    with open(path, "r") as f:
                        r = json.load(f)
                except Exception:
                    continue

                if r.get("task_type") != "binary":
                    continue

                auc = r.get("pooled", {}).get("AUC", None)
                if auc is None:
                    continue

                rows.append(
                    {
                        "ticker": ticker,
                        "phase": phase,
                        "model_key": r.get("model_key", ""),
                        "model_name": r.get("model_name", ""),
                        "target": r.get("target", ""),
                        "auc": float(auc),
                        "json_path": path,
                    }
                )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Strict cross-asset ranker for model-zoo JSON outputs")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--tickers", default="NVDA,TSLA,JPM,MS")
    ap.add_argument("--phases", default="unfiltered,filtered")
    ap.add_argument("--out-prefix", default="results/strict_rankings")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    tickers = parse_csv_list(args.tickers)
    phases = parse_csv_list(args.phases)
    expected_n = len(tickers)

    df = collect_rows(args.results_root, tickers, phases)
    if df.empty:
        raise SystemExit("No binary JSON results found. Check --results-root/--tickers/--phases.")

    # Aggregate at (model, target) level, enforcing full ticker coverage.
    g = (
        df.groupby(["model_key", "model_name", "target"], as_index=False)
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            min_auc=("auc", "min"),
            max_auc=("auc", "max"),
            n_tickers=("ticker", pd.Series.nunique),
        )
    )

    strict = g[g["n_tickers"] == expected_n].copy()
    if strict.empty:
        raise SystemExit("No model-target rows have full ticker coverage.")

    # Best per target.
    best_per_target = (
        strict.sort_values(["target", "mean_auc", "std_auc"], ascending=[True, False, True])
        .groupby("target", as_index=False)
        .head(1)
        .sort_values("target")
    )

    # Overall ranking by mean over targets, tie-break by lower std.
    overall = (
        strict.groupby(["model_key", "model_name"], as_index=False)
        .agg(
            overall_mean_auc=("mean_auc", "mean"),
            overall_mean_std=("std_auc", "mean"),
            n_targets=("target", "nunique"),
        )
        .sort_values(["overall_mean_auc", "overall_mean_std"], ascending=[False, True])
    )

    # Save artifacts.
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    strict.to_csv(f"{args.out_prefix}_model_target_strict.csv", index=False)
    best_per_target.to_csv(f"{args.out_prefix}_best_per_target.csv", index=False)
    overall.to_csv(f"{args.out_prefix}_overall_models.csv", index=False)

    # Print concise summary.
    print("=" * 95)
    print("STRICT BEST MODEL PER TARGET (full ticker coverage required)")
    print("=" * 95)
    for _, row in best_per_target.iterrows():
        print(
            f"Target: {row['target']:<10} | Model: {row['model_name']:<28} | "
            f"Mean AUC: {row['mean_auc']:.4f} | Std AUC: {row['std_auc']:.4f}"
        )

    print("\n" + "=" * 95)
    print(f"TOP {args.top_k} OVERALL MODELS")
    print("=" * 95)
    top = overall.head(args.top_k)
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        print(
            f"Rank {i:<2} | {row['model_name']:<30} | "
            f"Overall Mean AUC: {row['overall_mean_auc']:.4f} | "
            f"Mean Std AUC: {row['overall_mean_std']:.4f} | Targets: {int(row['n_targets'])}"
        )

    print("\nSaved:")
    print(f"  {args.out_prefix}_model_target_strict.csv")
    print(f"  {args.out_prefix}_best_per_target.csv")
    print(f"  {args.out_prefix}_overall_models.csv")


if __name__ == "__main__":
    main()
