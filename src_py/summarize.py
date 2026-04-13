#!/usr/bin/env python3
import os
import json
import pandas as pd
from pathlib import Path

def main():
    # Base directory where Optuna saved the JSON files
    base_dir = Path("results/optuna_physical")
    
    if not base_dir.exists():
        print(f"ERROR: Cannot find {base_dir}. Run this from your project root.")
        return

    data = []

    # Walk through all ticker folders (NVDA, TSLA, JPM, MS)
    for ticker_dir in base_dir.iterdir():
        if not ticker_dir.is_dir():
            continue
            
        ticker = ticker_dir.name
        
        # Find all JSON files in the ticker directory
        for json_file in ticker_dir.glob("best_physical_params_*.json"):
            # Extract the target name from the filename
            # e.g., best_physical_params_cls_clop_s0p5.json -> cls_clop
            filename = json_file.name
            target_str = filename.replace("best_physical_params_", "").replace(".json", "")
            
            # Sometimes the silence tag is appended to the filename (e.g., _s0p5)
            # Let's clean up the target name if it has a silence tag
            target_clean = target_str
            if "_s" in target_str:
                target_clean = target_str.split("_s")[0]

            with open(json_file, 'r') as f:
                try:
                    obj = json.load(f)
                    bp = obj.get("best_params", {})
                    
                    # Some sweeps might have saved the AUC score as well
                    auc = obj.get("best_auc", None) 
                    
                    data.append({
                        "Ticker": ticker,
                        "Target": target_clean,
                        "Silence": bp.get("silence_tag", "s0p5"), # Default if missing
                        "Vol_Frac": round(bp.get("vol_frac", 0.0), 6),
                        "Dir_Thresh": round(bp.get("dir_thresh", 0.0), 4),
                        "Vol_Ratio": round(bp.get("vol_ratio", 0.0), 4),
                        "Kappa": round(bp.get("kappa", 0.0), 4),
                        "Optuna_AUC": round(auc, 4) if auc else ""
                    })
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")

    if not data:
        print("No parameter JSON files found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort for clean viewing: Ticker alphabetical, then Target
    df = df.sort_values(by=["Ticker", "Target"]).reset_index(drop=True)
    
    # Save to CSV for Excel/LaTeX easy copy-pasting
    out_csv = "results/all_optuna_parameters_summary.csv"
    df.to_csv(out_csv, index=False)
    
    print("\n==========================================================================================")
    print("MASTER OPTUNA PARAMETER SUMMARY")
    print("==========================================================================================")
    # Print a beautiful console table
    print(df.to_markdown(index=False))
    print("==========================================================================================")
    print(f"\nSaved master summary to: {out_csv}\n")

if __name__ == "__main__":
    main()