#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def main():
    root = Path("results")
    
    summaries = list(root.rglob("sweep_summary.csv"))
    if not summaries:
        print("No sweep_summary.csv files found. Did you rsync them?")
        return
        
    print("="*80)
    print("  PARTIAL SWEEP PROGRESS & UNIVERSAL STABILITY")
    print("="*80)
    
    all_rows = []
    for s in summaries:
        try:
            df = pd.read_csv(s)
            
            # Extract sweep type from path
            if "sweep_frac" in str(s):
                sweep_type = "Fractional"
            elif "silence_sweep" in str(s):
                sweep_type = "Absolute"
            else:
                sweep_type = "Unknown"
                
            df['SweepType'] = sweep_type
            all_rows.append(df)
            
        except Exception as e:
            pass
            
    if not all_rows:
        return
        
    combo_df = pd.concat(all_rows, ignore_index=True)
    
    valid = combo_df[~combo_df['metric_name'].isin(['SKIP', 'MISSING'])]
    
    # ── 1. Progress & "How Many Left" ──────────────────────────────────────
    # Assuming standard grid: 324 unique configs per stock
    EXPECTED_PER_STOCK = 324 
    
    print(f"\n[SWEEP PROGRESS PER TICKER]")
    print(f"(Assuming {EXPECTED_PER_STOCK} total configurations per stock)")
    
    progress_rows = []
    for sweep_type in combo_df['SweepType'].unique():
        sdf = combo_df[combo_df['SweepType'] == sweep_type]
        for ticker in ['NVDA', 'TSLA', 'JPM', 'MS']:
            tdf = sdf[sdf['ticker'] == ticker]
            completed = len(tdf)
            left = max(0, EXPECTED_PER_STOCK - completed)
            valid_cnt = len(tdf[~tdf['metric_name'].isin(['SKIP', 'MISSING'])])
            skip_cnt = len(tdf[tdf['metric_name'] == 'SKIP'])
            
            progress_rows.append({
                'Sweep': sweep_type,
                'Ticker': ticker,
                'Done': completed,
                'Remaining': left,
                'Valid_Models': valid_cnt,
                'Skipped_Data': skip_cnt
            })
            
    prog_df = pd.DataFrame(progress_rows)
    print(prog_df.to_string(index=False))
    
    # ── 2. Universal Stability (>= 3 Stocks) ───────────────────────────────
    print(f"\n[UNIVERSAL STABILITY: CONFIGS SURVIVING >= 3 STOCKS]")
    
    for target in valid['target'].unique():
        tdf = valid[valid['target'] == target]
        if tdf.empty: continue
            
        metric = tdf['metric_name'].iloc[0]
        ascending = False if metric.upper() == 'AUC' else True
        
        # Group by configuration across different stocks
        grp = tdf.groupby(['SweepType', 'config']).agg(
            stocks_survived=('ticker', 'nunique'),
            mean_score=('metric_value', 'mean'),
            std_score=('metric_value', 'std'),
            avg_rows=('rows', 'mean')
        ).reset_index()
        
        # FILTER: Must exist for at least 3 stocks
        univ = grp[grp['stocks_survived'] >= 3]
        
        if univ.empty:
            print(f"\nTarget: {target} -> No configs have survived on 3+ stocks yet.")
            
            # Fallback: Show the best 2-stock configs if 3 aren't available
            univ_fallback = grp[grp['stocks_survived'] == 2]
            if not univ_fallback.empty:
                print(f"Target: {target} -> Showing best configs on 2 stocks:")
                univ_fallback = univ_fallback.sort_values('mean_score', ascending=ascending).head(3)
                univ_fallback['mean_score'] = univ_fallback['mean_score'].round(4)
                print(univ_fallback[['SweepType', 'config', 'stocks_survived', 'mean_score']].to_string(index=False))
            continue
            
        # Sort by best mean score
        univ = univ.sort_values('mean_score', ascending=ascending).head(5)
        
        print(f"\nTarget: {target} (Metric: {metric})")
        univ['mean_score'] = univ['mean_score'].round(4)
        univ['std_score'] = univ['std_score'].round(4)
        univ['avg_rows'] = univ['avg_rows'].astype(int)
        
        print(univ.to_string(index=False))

if __name__ == "__main__":
    main()