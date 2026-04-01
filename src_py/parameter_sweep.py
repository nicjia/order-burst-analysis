#!/usr/bin/env python3
"""
parameter_sweep.py — Evaluate burst parameter sets for stability across stocks.

Workflow per parameter set + ticker:
    1) Run data_processor with burst extraction only (kappa disabled in C++)
  2) Run compute_permanence.py to add Perm_* columns (arcsinh target)
  3) Train a simple model (default: logreg_l2) via train_model_zoo.py
  4) Collect AUC (or MAE for regression) and summarize across tickers

Param file format (JSON array):
[
  {"name": "baseline", "silence": 1.0, "min_vol": 100, "dir_thresh": 0.9, "vol_ratio": 0.5, "kappa": 0.10, "tau_max": 10.0},
  {"name": "alt", "silence": 0.5, "min_vol": 200, "dir_thresh": 0.85, "vol_ratio": 0.6, "kappa": 0.20, "tau_max": 10.0}
]
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Sweep burst parameters and compare model stability across stocks.")
    ap.add_argument("--param-file", required=True, help="JSON file with parameter sets")
    ap.add_argument("--data-root", required=True, help="Root folder containing per-stock data folders")
    ap.add_argument("--tickers", required=True, help="Comma-separated tickers (e.g., NVDA,TSLA)")
    ap.add_argument("--open", required=True, help="CRSP open prices CSV (pivot)")
    ap.add_argument("--close", required=True, help="CRSP close prices CSV (pivot)")
    ap.add_argument("--outdir", default="results/param_sweep", help="Output directory")
    ap.add_argument("--data-processor", default="./data_processor", help="Path to data_processor binary")
    ap.add_argument("--target", default="cls_close", help="Target key for train_model_zoo")
    ap.add_argument("--model", default="logreg_l2", help="Model key for train_model_zoo")
    ap.add_argument("--features", default="extended", choices=["base", "extended"], help="Feature set")
    ap.add_argument("--min-train-months", type=int, default=3, help="Min training months before first test fold")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.param_file) as f:
        params_list = json.load(f)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    summary_rows = []

    for p in params_list:
        name = p.get("name") or f"s{p.get('silence')}_v{p.get('min_vol')}_d{p.get('dir_thresh')}_r{p.get('vol_ratio')}_k{p.get('kappa')}_t{p.get('tau_max')}"
        param_dir = outdir / name
        param_dir.mkdir(parents=True, exist_ok=True)

        metrics = []

        for ticker in tickers:
            # Find stock folder matching ticker
            data_root = Path(args.data_root)
            matches = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith(f"{ticker}_")]
            if not matches:
                raise FileNotFoundError(f"No folder found for ticker {ticker} under {data_root}")
            stock_folder = matches[0]

            bursts_csv = param_dir / f"bursts_{ticker}.csv"

            cmd = [
                str(args.data_processor),
                str(stock_folder),
                str(bursts_csv),
                "-s", str(p.get("silence", 1.0)),
                "-v", str(p.get("min_vol", 100)),
                "-d", str(p.get("dir_thresh", 0.9)),
                "-r", str(p.get("vol_ratio", 0.5)),
                "-k", "0",
                "-t", str(p.get("tau_max", 10.0)),
            ]
            run(cmd)

            # Permanence + CRSP lookups
            run([
                "python", "src_py/compute_permanence.py",
                str(bursts_csv),
                str(args.open),
                str(args.close),
                "--kappa", "0",
            ])

            filtered_csv = bursts_csv.with_name(bursts_csv.stem + "_filtered" + bursts_csv.suffix)

            zoo_outdir = param_dir / f"{ticker}_zoo"
            zoo_outdir.mkdir(parents=True, exist_ok=True)

            run([
                "python", "src_py/train_model_zoo.py",
                str(filtered_csv),
                "--model", args.model,
                "--target", args.target,
                "--features", args.features,
                "--outdir", str(zoo_outdir),
                "--min-train-months", str(args.min_train_months),
            ])

            result_path = zoo_outdir / f"{args.model}__{args.target}.json"
            with open(result_path) as rf:
                result = json.load(rf)

            pooled = result.get("pooled", {})
            if "AUC" in pooled:
                metric = pooled["AUC"]
                metric_name = "AUC"
            elif "MAE" in pooled:
                metric = pooled["MAE"]
                metric_name = "MAE"
            else:
                metric = None
                metric_name = "Metric"

            metrics.append(metric)
            summary_rows.append({
                "param_set": name,
                "ticker": ticker,
                "metric_name": metric_name,
                "metric_value": metric,
            })

        # Aggregate across tickers
        metric_vals = [m for m in metrics if m is not None]
        if metric_vals:
            mean_metric = sum(metric_vals) / len(metric_vals)
            var = sum((m - mean_metric) ** 2 for m in metric_vals) / len(metric_vals)
            std_metric = var ** 0.5
        else:
            mean_metric = None
            std_metric = None

        summary_rows.append({
            "param_set": name,
            "ticker": "__MEAN__",
            "metric_name": metric_name if metric_vals else "Metric",
            "metric_value": mean_metric,
        })
        summary_rows.append({
            "param_set": name,
            "ticker": "__STD__",
            "metric_name": metric_name if metric_vals else "Metric",
            "metric_value": std_metric,
        })

    # Write summary CSV
    summary_path = outdir / "sweep_summary.csv"
    import csv
    keys = sorted({k for row in summary_rows for k in row.keys()})
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
