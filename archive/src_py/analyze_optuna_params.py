#!/usr/bin/env python3
"""
analyze_optuna_params.py

Extracts and analyzes hyperparameter consistency across different tickers and targets.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def load_optuna_params(root="results/optuna_direct"):
    root = Path(root)
    rows = []
    
    for json_file in sorted(root.rglob("*.json")):
        parts = json_file.relative_to(root).parts
        if len(parts) != 4:
            continue
            
        ticker, model, silence_tag, fname = parts
        target = fname.replace(f"{model}__", "").replace(".json", "")

        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
            
        if "fold_results" not in data:
            continue
            
        # We collect the median parameters across the 21 forward-walk folds
        # Because we want to know what the 'typical' chosen parameter is!
        params_list = {}
        
        for fold in data["fold_results"]:
            if "best_params" in fold:
                for k, v in fold["best_params"].items():
                    if k not in params_list:
                        params_list[k] = []
                    params_list[k].append(v)
                    
        if not params_list:
            continue
            
        # Calculate medians across the 21 folds
        median_params = {k: np.median(v) for k, v in params_list.items()}
        # For categorical or integer params, we might want the mode or round to int
        for k in ['num_leaves', 'max_depth', 'min_child_samples', 'min_child_weight']:
            if k in median_params:
                median_params[k] = int(round(median_params[k]))
                
        # Format the parameters nicely
        row = {
            "ticker": ticker,
            "target": target,
            "model": model,
            "silence": silence_tag,
        }
        
        # Flatten parameters into the row
        row.update(median_params)
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "results/optuna_direct"
    
    df = load_optuna_params(root_dir)
    if df.empty:
        print("No hyperparameter data found. Did you rerun eval_optuna_direct.sh with the updated train_model_zoo.py?")
        return
        
    print("="*80)
    print("  HYPERPARAMETER CONSISTENCY ANALYSIS (MEDIAN ACROSS FOLDS)")
    print("="*80)
    
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        if mdf.empty: continue
            
        print(f"\n[{model.upper()} Parameter Sets]")
        
        # We only really care about s2p0, or we can just show everything
        param_cols = [c for c in mdf.columns if c not in ['ticker', 'target', 'model', 'silence']]
        
        display_df = mdf[['ticker', 'target', 'silence'] + param_cols].sort_values(['target', 'ticker'])
        print(display_df.to_string(index=False))
        
        # Analyze variance across tickers for each target
        print(f"\n[Coefficient of Variation (Std/Mean) across Tickers for each Target - {model}]")
        print("Lower is better (closer to 0 means universally identical parameters!)")
        
        cv_rows = []
        for target in mdf['target'].unique():
            tdf = mdf[mdf['target'] == target]
            if len(tdf) > 1:
                cv = (tdf[param_cols].std() / tdf[param_cols].mean()).fillna(0)
                cv['target'] = target
                cv_rows.append(cv)
                
        if cv_rows:
            cv_df = pd.DataFrame(cv_rows).set_index('target')
            # round it to 3 decimal places
            print(cv_df.round(3).to_string())

if __name__ == "__main__":
    main()
