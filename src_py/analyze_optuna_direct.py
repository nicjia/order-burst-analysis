#!/usr/bin/env python3
"""
analyze_optuna_direct.py

Aggregate and analyze direct Optuna results from results/optuna_direct.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_optuna_direct(root="results/optuna_direct"):
    root = Path(root)
    rows = []
    for json_file in sorted(root.rglob("*.json")):
        # Path structure: results/optuna_direct/<TICKER>/<MODEL>/<STAG>/<MODEL>__<TARGET>.json
        parts = json_file.relative_to(root).parts
        
        if len(parts) != 4:
            continue
            
        ticker, model, silence_tag, fname = parts
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
            "silence": silence_tag,
            "target": target,
            "json_path": str(json_file),
        }

        # Extract metrics
        if "pooled" in data:
            row["mean_auc"] = data["pooled"].get("AUC", None)
            row["std_auc"] = None
            row["n_folds"] = data.get("n_folds", None)
        elif "summary" in data:
            s = data["summary"]
            row["mean_auc"] = s.get("mean_primary", None)
            row["std_auc"] = s.get("std_primary", None)
            row["n_folds"] = s.get("n_folds", None)
        elif "mean_primary" in data:
            row["mean_auc"] = data.get("mean_primary", None)
            row["std_auc"] = data.get("std_primary", None)
            row["n_folds"] = data.get("n_folds", None)

        rows.append(row)

    return pd.DataFrame(rows)

def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "results/optuna_direct"
    out_dir = Path("results/optuna_direct_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_optuna_direct(root_dir)
    if df.empty:
        print("No valid results found. Did you rsync the results/optuna_direct folder correctly?")
        return

    print("="*60)
    print("  DIRECT OPTUNA RESULTS SUMMARY")
    print("="*60)
    
    # 1. Best config for each ticker + target
    print("\n[Best Configs by AUC]")
    best_idx = df.groupby(["ticker", "target"])["mean_auc"].idxmax()
    best_df = df.loc[best_idx].sort_values(["ticker", "target"])
    print(best_df[["ticker", "target", "model", "silence", "mean_auc", "n_folds"]].to_string(index=False))

    # 2. Overall comparison
    print("\n[Model Performance Distribution]")
    print(df.groupby(["model"])["mean_auc"].describe().round(4))

    # Save to CSV
    best_df.to_csv(out_dir / "best_configs.csv", index=False)
    df.to_csv(out_dir / "all_results.csv", index=False)
    
    # Simple AUC Heatmap across Ticker vs Target for best metrics
    pivot = best_df.pivot_table(index="ticker", columns="target", values="mean_auc")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.5, vmax=0.7)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", 
                        color="white" if val < 0.55 else "black", fontweight="bold")
                
    ax.set_title("Best AUC Heatmap (Optuna Direct)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / "best_auc_heatmap.png", dpi=150)
    
    print(f"\nSaved analysis to {out_dir}/")

if __name__ == "__main__":
    main()
