import pandas as pd
import sys

def main():
    ticker = sys.argv[1]
    df = pd.read_csv(f'results/passive/passive_bursts_{ticker}_raw_filtered.csv')
    
    print(f"--- Passive Burst Summary for {ticker} ---")
    print(f"Total bursts: {len(df)}")
    
    print("\nDirection Distribution:")
    print(df['Direction'].value_counts(normalize=True).round(3))
    
    print("\nTarget Variable (Perm_CLOP) Summary:")
    print(df['Perm_CLOP'].describe())
    
    print("\nTarget Variable (Perm_tCLOSE) Summary:")
    print(df['Perm_tCLOSE'].describe())
    
    print("\nFeature Summary:")
    features = ['Volume', 'SubmissionCount', 'CancelCount', 'CancelRatio', 'HawkesPeakIntensity', 'PreBurstCancelRate']
    print(df[features].describe().round(3).T[['mean', '50%', 'max']])
    
    print("\nCorrelation with Perm_CLOP:")
    corrs = df[features + ['Perm_CLOP']].corr()['Perm_CLOP'].sort_values(ascending=False)
    print(corrs)

if __name__ == '__main__':
    main()
