#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    root = Path("results/optuna_physical")
    if not root.exists():
        print(f"Directory {root} does not exist.")
        return
        
    jsons = list(root.rglob("*.json"))
    if not jsons:
        print("No JSON results found. Did you rsync them down?")
        return
        
    records = []
    for j in jsons:
        try:
            with open(j) as f:
                data = json.load(f)
            rec = {
                "ticker": data.get("ticker", j.parent.name),
                "target": data.get("target", "cls_10m"),
                "best_auc": data.get("best_auc", 0.0)
            }
            
            # Extract parameters as pure floats for accurate math
            params = data.get("best_params", {})
            rec["silence_tag"] = params.get("silence_tag", "UNKNOWN")
            rec["vol_frac"] = params.get("vol_frac", np.nan)
            rec["dir_thresh"] = params.get("dir_thresh", np.nan)
            rec["vol_ratio"] = params.get("vol_ratio", np.nan)
            
            # Kappa is only guessed by Optuna for long targets. Default to 0.
            rec["kappa"] = params.get("kappa", 0.0) 
            
            records.append(rec)
        except Exception as e:
            pass
            
    if not records:
        print("Failed to parse JSONs.")
        return
        
    df = pd.DataFrame(records)
    
    print("="*80)
    print("  OPTUNA CONTINUOUS PHYSICAL GEOMETRIES (BEST CONFIGS FOUND)")
    print("="*80)
    
    # 1. Detailed breakdown per target
    for target in sorted(df['target'].unique()):
        tdf = df[df['target'] == target].sort_values("best_auc", ascending=False).copy()
        print(f"\n[ TARGET: {target} ]")
        
        # Create a display copy to format strings without ruining the math
        display_df = tdf.copy()
        display_df['best_auc'] = display_df['best_auc'].round(4)
        display_df['vol_frac'] = display_df['vol_frac'].apply(lambda x: f"{x:.6f}")
        display_df['dir_thresh'] = display_df['dir_thresh'].round(4)
        display_df['vol_ratio'] = display_df['vol_ratio'].round(4)
        display_df['kappa'] = display_df['kappa'].round(4)
        
        print(display_df.to_string(index=False))
        
        # Consistency Summary (Coefficient of Variation)
        print("\n  -- Cross-Asset Consistency Summary --")
        print("  (Lower CV % means the parameter is universally identical across stocks)")
        
        mode_silence = tdf['silence_tag'].mode()[0]
        
        mean_vf = tdf['vol_frac'].mean()
        cv_vf = (tdf['vol_frac'].std() / mean_vf) * 100 if mean_vf else 0
        
        mean_dt = tdf['dir_thresh'].mean()
        cv_dt = (tdf['dir_thresh'].std() / mean_dt) * 100 if mean_dt else 0
        
        mean_vr = tdf['vol_ratio'].mean()
        cv_vr = (tdf['vol_ratio'].std() / mean_vr) * 100 if mean_vr else 0
        
        print(f"  Silence Tag Mode:   {mode_silence}")
        print(f"  Vol Frac:           {mean_vf:.6f}  (CV: {cv_vf:.1f}%)")
        print(f"  Dir Thresh:         {mean_dt:.4f}  (CV: {cv_dt:.1f}%)")
        print(f"  Vol Ratio:          {mean_vr:.4f}  (CV: {cv_vr:.1f}%)")
        
        # Handle kappa logic (only meaningful if > 0)
        if tdf['kappa'].mean() > 0:
            mean_k = tdf['kappa'].mean()
            cv_k = (tdf['kappa'].std() / mean_k) * 100 if mean_k else 0
            print(f"  Kappa:              {mean_k:.4f}  (CV: {cv_k:.1f}%)")

    # 2. Grand Universal Summary Across ALL Targets
    print("\n" + "="*80)
    print("  GRAND UNIVERSAL PHYSICAL PARAMETERS (AVERAGED ACROSS ALL STOCKS & TARGETS)")
    print("="*80)
    print("Use these values as your locked-in, absolute physical data filters.")

    grand_silence = df['silence_tag'].mode()[0]
    grand_vf = df['vol_frac'].mean()
    grand_dt = df['dir_thresh'].mean()
    grand_vr = df['vol_ratio'].mean()

    print(f"\n  Universal Silence:    {grand_silence}")
    print(f"  Universal Vol Frac:   {grand_vf:.6f}")
    print(f"  Universal Dir Thresh: {grand_dt:.4f}")
    print(f"  Universal Vol Ratio:  {grand_vr:.4f}")

    # Calculate Grand Kappa (but only using rows where Kappa was actually tested)
    long_df = df[df['kappa'] > 0]
    if not long_df.empty:
        grand_k = long_df['kappa'].mean()
        print(f"  Universal Kappa:      {grand_k:.4f} (For long horizons only)")

if __name__ == "__main__":
    main()