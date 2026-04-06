#!/usr/bin/env python3
"""
analyze_optuna_results.py

Aggregate and analyze Optuna-tuned model evaluation results.
Reads JSON result files from results/optuna_eval/ and produces:
  1. A summary table of AUC scores across tickers, models, configs, and targets
  2. Cross-asset stability analysis (variance of AUC across tickers)
  3. Comparison of lgb_tuned vs xgb_tuned performance
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_optuna_results(root="results/optuna_eval"):
    """Walk the optuna_eval directory tree and load all JSON results."""
    root = Path(root)
    rows = []
    for json_file in sorted(root.rglob("*.json")):
        # Path structure: results/optuna_eval/<TICKER>/<MODEL>/<CONFIG>/<PHASE>/<MODEL>__<TARGET>.json
        parts = json_file.relative_to(root).parts
        if len(parts) != 5:
            continue
        ticker, model, config, phase, fname = parts
        target = fname.replace(f"{model}__", "").replace(".json", "")

        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"  WARN: Could not read {json_file}")
            continue

        row = {
            "ticker": ticker,
            "model": model,
            "config": config,
            "phase": phase,
            "target": target,
            "json_path": str(json_file),
        }

        # Extract key metrics
        if "summary" in data:
            s = data["summary"]
            row["mean_auc"] = s.get("mean_primary", None)
            row["std_auc"] = s.get("std_primary", None)
            row["n_folds"] = s.get("n_folds", None)
            row["metric_name"] = s.get("primary_metric", None)
        elif "mean_primary" in data:
            row["mean_auc"] = data.get("mean_primary", None)
            row["std_auc"] = data.get("std_primary", None)
            row["n_folds"] = data.get("n_folds", None)
            row["metric_name"] = data.get("primary_metric", None)

        # Try to get per-fold details
        if "fold_results" in data:
            fold_aucs = [f.get("primary", None) for f in data["fold_results"]
                         if f.get("primary") is not None]
            if fold_aucs:
                row["median_auc"] = np.median(fold_aucs)
                row["min_auc"] = np.min(fold_aucs)
                row["max_auc"] = np.max(fold_aucs)

        rows.append(row)

    return pd.DataFrame(rows)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/optuna_eval"
    output_dir = Path("results/optuna_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_optuna_results(results_dir)

    if df.empty:
        print("No results found! Check that the optuna_eval directory exists and contains JSON files.")
        return

    # ── 1. Overview ──────────────────────────────────────────────────
    print_section("1. OVERVIEW")
    print(f"Total result files: {len(df)}")
    print(f"Tickers:  {sorted(df['ticker'].unique())}")
    print(f"Models:   {sorted(df['model'].unique())}")
    print(f"Configs:  {sorted(df['config'].unique())}")
    print(f"Targets:  {sorted(df['target'].unique())}")

    # ── 2. Full Results Table ────────────────────────────────────────
    print_section("2. ALL RESULTS (sorted by AUC)")
    cols = ["ticker", "model", "config", "target", "mean_auc", "std_auc", "n_folds"]
    display_df = df[cols].sort_values("mean_auc", ascending=False)
    print(display_df.to_string(index=False))

    # Save to CSV
    display_df.to_csv(output_dir / "optuna_all_results.csv", index=False)

    # ── 3. Best Model per Ticker × Target ────────────────────────────
    print_section("3. BEST MODEL PER TICKER × TARGET")
    best = df.loc[df.groupby(["ticker", "target"])["mean_auc"].idxmax()]
    best_cols = ["ticker", "target", "model", "config", "mean_auc"]
    print(best[best_cols].sort_values(["ticker", "target"]).to_string(index=False))

    # ── 4. Model Comparison (lgb_tuned vs xgb_tuned) ────────────────
    print_section("4. MODEL COMPARISON: lgb_tuned vs xgb_tuned")
    pivot = df.pivot_table(
        index=["ticker", "config", "target"],
        columns="model",
        values="mean_auc",
    )
    if "lgb_tuned" in pivot.columns and "xgb_tuned" in pivot.columns:
        pivot["lgb_wins"] = pivot["lgb_tuned"] > pivot["xgb_tuned"]
        lgb_win_pct = pivot["lgb_wins"].mean() * 100
        print(f"LightGBM wins: {lgb_win_pct:.1f}% of comparisons")
        print(f"\nMean AUC by model:")
        for m in ["lgb_tuned", "xgb_tuned"]:
            if m in pivot.columns:
                print(f"  {m}: {pivot[m].mean():.4f} (±{pivot[m].std():.4f})")
        print(f"\nPer-ticker mean AUC:")
        for ticker in sorted(df["ticker"].unique()):
            sub = pivot.loc[ticker]
            for m in ["lgb_tuned", "xgb_tuned"]:
                if m in sub.columns:
                    print(f"  {ticker} {m}: {sub[m].mean():.4f}")
    else:
        print("Not all models available for comparison.")

    # ── 5. Cross-Asset Stability ─────────────────────────────────────
    print_section("5. CROSS-ASSET AUC STABILITY (lower variance = better)")
    stability = df.groupby(["model", "config", "target"])["mean_auc"].agg(
        ["mean", "std", "count"]
    ).rename(columns={"mean": "cross_ticker_mean", "std": "cross_ticker_std", "count": "n_tickers"})
    stability = stability[stability["n_tickers"] >= 4].sort_values("cross_ticker_std")
    if not stability.empty:
        print("Top 10 most stable (config, target) across all 4 tickers:")
        print(stability.head(10).to_string())
    else:
        print("Not enough cross-ticker coverage for stability analysis.")

    stability.to_csv(output_dir / "optuna_cross_asset_stability.csv")

    # ── 6. Visualization ────────────────────────────────────────────
    # AUC heatmap: Ticker × Target for each model
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        if mdf.empty:
            continue

        pivot_heat = mdf.pivot_table(
            index="ticker",
            columns="target",
            values="mean_auc",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(pivot_heat.values, cmap="RdYlGn", aspect="auto",
                       vmin=0.50, vmax=0.70)

        ax.set_xticks(range(len(pivot_heat.columns)))
        ax.set_xticklabels(pivot_heat.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot_heat.index)))
        ax.set_yticklabels(pivot_heat.index)

        # Add text annotations
        for i in range(len(pivot_heat.index)):
            for j in range(len(pivot_heat.columns)):
                val = pivot_heat.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if val < 0.55 else "black")

        ax.set_title(f"Mean AUC — {model} (Optuna-Tuned)", fontsize=14)
        plt.colorbar(im, ax=ax, label="AUC")
        plt.tight_layout()
        plt.savefig(output_dir / f"optuna_heatmap_{model}.png", dpi=150)
        plt.close()
        print(f"\nSaved: {output_dir}/optuna_heatmap_{model}.png")

    # Bar chart: model comparison per ticker
    fig, axes = plt.subplots(1, len(df["ticker"].unique()), figsize=(16, 5), sharey=True)
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax, ticker in zip(axes, sorted(df["ticker"].unique())):
        tdf = df[df["ticker"] == ticker]
        model_means = tdf.groupby("model")["mean_auc"].mean().sort_values(ascending=True)
        bars = ax.barh(model_means.index, model_means.values, color=["#4ECDC4", "#FF6B6B"])
        ax.set_xlim(0.50, 0.70)
        ax.set_title(ticker, fontsize=13, fontweight="bold")
        ax.axvline(x=0.50, color="gray", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, model_means.values):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
    axes[0].set_xlabel("Mean AUC")
    plt.suptitle("Optuna-Tuned Model Comparison by Ticker", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "optuna_model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/optuna_model_comparison.png")

    # ── Summary ──────────────────────────────────────────────────────
    print_section("SUMMARY")
    print(f"All results saved to: {output_dir}/")
    print(f"  - optuna_all_results.csv")
    print(f"  - optuna_cross_asset_stability.csv")
    for model in df["model"].unique():
        print(f"  - optuna_heatmap_{model}.png")
    print(f"  - optuna_model_comparison.png")


if __name__ == "__main__":
    main()
