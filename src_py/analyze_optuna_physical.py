#!/usr/bin/env python3
import json
import pandas as pd
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
                "best_auc": round(data.get("best_auc", 0.0), 4)
            }
            
            # Flatten physical parameters
            params = data.get("best_params", {})
            for k, v in params.items():
                if isinstance(v, float):
                    if k == "vol_frac":
                        # display fractional numbers with more precision
                        rec[k] = f"{v:.6f}"
                    else:
                        rec[k] = round(v, 4)
                else:
                    rec[k] = v
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
    
    for target in df['target'].unique():
        tdf = df[df['target'] == target].sort_values("best_auc", ascending=False)
        print(f"\n[ TARGET: {target} ]")
        print(tdf.to_string(index=False))
        
        # Calculate consistency metrics simply
        if "vol_frac" in tdf.columns:
            # We must convert formatted strings back to float for analysis
            vf_floats = tdf['vol_frac'].astype(float)
            print("\n  -- Consistency Summary --")
            print(f"  Silence Tag Mode:   {tdf['silence_tag'].mode()[0]}")
            print(f"  Mean Vol Frac:      {vf_floats.mean():.6f}")
            print(f"  Mean Dir Thresh:    {tdf['dir_thresh'].mean():.4f}")
            print(f"  Mean Vol Ratio:     {tdf['vol_ratio'].mean():.4f}")

if __name__ == "__main__":
    main()
