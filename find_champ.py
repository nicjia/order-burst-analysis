import pandas as pd
import os

csv_path = "results/sweep_summary.csv"

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit()

df = pd.read_csv(csv_path)

# 1. Filter to only look at rows where the metric is AUC and drop NaNs
df = df[df['metric_name'] == 'AUC'].copy()
df = df.dropna(subset=['metric_value'])

print("\n" + "="*65)
print("🏆 WINNING CONFIGURATIONS BY TARGET 🏆")
print("="*65)

targets = ['cls_1m', 'cls_5m', 'cls_10m', 'cls_close']

# 2. Find the winner for each specific target
for t in targets:
    # Get all rows for this specific target
    target_df = df[df['target'] == t]
    
    if not target_df.empty:
        best_idx = target_df['metric_value'].idxmax()
        best_row = target_df.loc[best_idx]
        print(f"Best for {t:<10} | AUC: {best_row['metric_value']:.4f} | Config: {best_row['config']}")
    else:
        print(f"Best for {t:<10} | No data available")

print("\n" + "="*65)
print("👑 GRAND CHAMPION (Highest Average AUC Across All Targets) 👑")
print("="*65)

# 3. Group by config to find the overall average AUC
if not df.empty:
    avg_df = df.groupby('config')['metric_value'].mean().reset_index()
    avg_df = avg_df.rename(columns={'metric_value': 'AUC_mean'})
    
    best_mean_idx = avg_df['AUC_mean'].idxmax()
    best_mean_row = avg_df.loc[best_mean_idx]

    print(f"Winner Config : {best_mean_row['config']}")
    print(f"Average AUC   : {best_mean_row['AUC_mean']:.4f}")
else:
    print("Error: No valid AUC data found to calculate an average.")
    
print("="*65 + "\n")