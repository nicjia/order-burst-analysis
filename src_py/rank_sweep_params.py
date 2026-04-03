#!/usr/bin/env python3
"""
Rank parameter sweep configs by cross-asset stability.

Reads sweep outputs produced by silence_optimized_sweep.py under:
  results/silence_sweep_<TICKER>/<MODEL>/short/sweep_summary.csv
  results/silence_sweep_<TICKER>/<MODEL>/long/sweep_summary.csv

Outputs:
  results/sweep_rankings/<model>_config_target_stats.csv
  results/sweep_rankings/<model>_config_overall.csv
  results/sweep_rankings/global_top_configs.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def load_one_csv(path: Path, ticker: str, model: str, phase: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ticker"] = ticker
    df["model"] = model
    df["phase"] = phase
    return df


def main():
    ap = argparse.ArgumentParser(description="Rank sweep parameter sets by cross-asset AUC variance")
    ap.add_argument("--tickers", default="NVDA,TSLA,JPM,MS")
    ap.add_argument("--models", default="et,rf,stacking,lgb_tuned,adaboost")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--min-coverage", type=int, default=4,
                    help="Minimum number of tickers required per (model, config, target)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    root = Path(args.results_root)

    frames = []
    for t in tickers:
        for m in models:
            short_csv = root / f"silence_sweep_{t}" / m / "short" / "sweep_summary.csv"
            long_csv = root / f"silence_sweep_{t}" / m / "long" / "sweep_summary.csv"
            frames.append(load_one_csv(short_csv, t, m, "short"))
            frames.append(load_one_csv(long_csv, t, m, "long"))

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if df.empty:
        raise SystemExit("No sweep summary files found for requested tickers/models.")

    df = df[df["metric_name"].str.upper() == "AUC"].copy()
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df = df.dropna(subset=["metric_value", "config", "target", "ticker", "model"])

    # Per target stability stats across tickers.
    g = (df.groupby(["model", "config", "target"], as_index=False)
           .agg(mean_auc=("metric_value", "mean"),
                std_auc=("metric_value", "std"),
                min_auc=("metric_value", "min"),
                max_auc=("metric_value", "max"),
                n_tickers=("ticker", "nunique")))
    g["std_auc"] = g["std_auc"].fillna(0.0)
    g = g[g["n_tickers"] >= args.min_coverage].copy()

    out_root = root / "sweep_rankings"
    out_root.mkdir(parents=True, exist_ok=True)

    if g.empty:
        raise SystemExit("No configs met min coverage requirement.")

    # Overall per-config stats: maximize mean AUC, minimize cross-target std mean.
    cfg = (g.groupby(["model", "config"], as_index=False)
             .agg(overall_mean_auc=("mean_auc", "mean"),
                  overall_mean_std=("std_auc", "mean"),
                  targets=("target", "nunique")))

    # Write per-model files + console top-k.
    global_rows = []
    for m in models:
        mg = g[g["model"] == m].sort_values(["target", "mean_auc"], ascending=[True, False])
        mc = cfg[cfg["model"] == m].sort_values(
            ["overall_mean_auc", "overall_mean_std"], ascending=[False, True]
        )
        if mc.empty:
            continue

        mg.to_csv(out_root / f"{m}_config_target_stats.csv", index=False)
        mc.to_csv(out_root / f"{m}_config_overall.csv", index=False)

        print("=" * 100)
        print(f"MODEL: {m}  | TOP {args.top_k} CONFIGS")
        print("=" * 100)
        top = mc.head(args.top_k)
        for i, row in enumerate(top.itertuples(index=False), start=1):
            print(
                f"Rank {i:<2} | {row.config:<28} | Mean AUC: {row.overall_mean_auc:.4f} "
                f"| Mean Std: {row.overall_mean_std:.4f} | Targets: {int(row.targets)}"
            )
            global_rows.append({
                "model": m,
                "config": row.config,
                "overall_mean_auc": row.overall_mean_auc,
                "overall_mean_std": row.overall_mean_std,
                "targets": int(row.targets),
            })

    if global_rows:
        gf = pd.DataFrame(global_rows).sort_values(
            ["overall_mean_auc", "overall_mean_std"], ascending=[False, True]
        )
        gf.to_csv(out_root / "global_top_configs.csv", index=False)
        print("\nSaved:")
        print(f"  {out_root}")


if __name__ == "__main__":
    main()
