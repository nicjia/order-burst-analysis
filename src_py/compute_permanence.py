import pandas as pd
import sys
import os

def compute_permanence(row):
    """Permanence = direction × (close − start) / |peak − start|"""
    peak_impact = abs(row['PeakPrice'] - row['StartPrice'])
    if peak_impact == 0:
        return float('nan')
    return row['Direction'] * (row['CloseMid'] - row['StartPrice']) / peak_impact

def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_permanence.py <bursts_file>")
        print("  bursts_file: CSV produced by data_processor (must have CloseMid column)")
        sys.exit(1)

    bursts_file = sys.argv[1]
    bursts = pd.read_csv(bursts_file)

    if 'CloseMid' not in bursts.columns:
        print(f"Error: {bursts_file} does not have a CloseMid column.")
        print("Re-run data_processor to generate the updated burst CSV.")
        sys.exit(1)

    if 'Perm_tCLOSE' in bursts.columns:
        print(f"{bursts_file} already has Perm_tCLOSE column. Skipping.")
        return

    # Filter out zero-impact bursts
    bursts = bursts[abs(bursts['PeakPrice'] - bursts['StartPrice']) > 0].copy()

    bursts['Perm_tCLOSE'] = bursts.apply(compute_permanence, axis=1)

    # Also compute forward-return permanence if mid-price columns exist
    for label, col in [('1m', 'Mid_1m'), ('3m', 'Mid_3m'),
                       ('5m', 'Mid_5m'), ('10m', 'Mid_10m')]:
        if col in bursts.columns:
            bursts[f'Perm_t{label}'] = bursts.apply(
                lambda r: r['Direction'] * (r[col] - r['StartPrice']) / abs(r['PeakPrice'] - r['StartPrice'])
                if abs(r['PeakPrice'] - r['StartPrice']) > 0 else float('nan'), axis=1)

    bursts.to_csv(bursts_file, index=False)
    print(f"Updated {bursts_file} with {len(bursts)} bursts")
    print(bursts[['Ticker', 'Date', 'BurstID', 'Direction',
                  'StartPrice', 'PeakPrice', 'Perm_tCLOSE']].head(10))

if __name__ == '__main__':
    main()
