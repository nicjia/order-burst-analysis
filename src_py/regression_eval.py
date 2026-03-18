import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ticker', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--config', required=True)
args = parser.parse_args()

# 1. LOAD DATA
df = pd.read_csv(args.data)

# 2. FEATURE ENGINEERING (Strictly matching your train_model.py)
if 'Volume' in df.columns and 'BurstVolume' not in df.columns:
    df['BurstVolume'] = df['Volume']
if 'Duration' not in df.columns:
    df['Duration'] = df['EndTime'] - df['StartTime']

df['AvgTradeSize'] = df['BurstVolume'] / df['TradeCount'].clip(lower=1)
df['PriceChange']  = df['Direction'] * (df['EndPrice'] - df['StartPrice'])

RTH_START, RTH_END = 34200.0, 57600.0
df['TimeOfDay'] = ((df['StartTime'] - RTH_START) / (RTH_END - RTH_START)).clip(0, 1)
df['LogVolume'] = np.log1p(df['BurstVolume'])
df['LogPeakImpact'] = np.log1p(df['PeakImpact'].abs() * 10000)
df['ImpactPerShare'] = df['PeakImpact'] / df['BurstVolume'].clip(lower=1)

if 'Spread' in df.columns:
    df['LogSpread'] = np.log1p(df['Spread'].fillna(0) * 10000)
    df['SpreadXVolume'] = df['Spread'].fillna(0) * df['BurstVolume']
else:
    df['LogSpread'], df['SpreadXVolume'] = 0.0, 0.0

if 'BidDepth5' in df.columns and 'AskDepth5' in df.columns:
    total = (df['BidDepth5'] + df['AskDepth5']).clip(lower=1)
    df['DepthRatio'] = df['BidDepth5'] / total
else:
    df['DepthRatio'] = 0.5

df['LogTradeIntensity'] = np.log1p(df['TradeCount5m']) if 'TradeCount5m' in df.columns else 0.0

# Base allowed features
BASE_FEATURES = [
    'Direction', 'BurstVolume', 'TradeCount', 'Duration', 'PeakImpact', 'AvgTradeSize', 
    'PriceChange', 'Spread', 'BidVolBest', 'AskVolBest', 'BidDepth5', 'AskDepth5', 
    'BookImbalance', 'Volatility60s', 'Momentum5s', 'Momentum30s', 'Momentum60s', 
    'TradeCount5m', 'TradeVolume5m', 'TimeOfDay', 'LogVolume', 'LogPeakImpact', 
    'ImpactPerShare', 'LogSpread', 'DepthRatio', 'LogTradeIntensity', 'SpreadXVolume'
]

# 3. WALK-FORWARD SPLIT (Train on 2023, Test on 2024)
df['date_parsed'] = pd.to_datetime(df['Date']).dt.date
train_df = df[df['date_parsed'] < pd.to_datetime('2024-01-01').date()].copy()
test_df = df[df['date_parsed'] >= pd.to_datetime('2024-01-01').date()].copy()

targets = ['Perm_t1m', 'Perm_t5m', 'Perm_t10m', 'Perm_tCLOSE']
results = []
print(f"Training Regressors for {args.ticker} -> {args.config}")

for target in targets:
    if target not in df.columns:
        continue
        
    # CLEANING: Drop rows with NaN targets
    tr_mask, te_mask = train_df[target].notna(), test_df[target].notna()
    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        continue

    y_train = train_df.loc[tr_mask, target].copy()
    y_test = test_df.loc[te_mask, target].copy()

    # WINSORIZING: Match train_model.py 1st/99th percentile clipping
    lo, hi = y_train.quantile(0.01), y_train.quantile(0.99)
    y_train = y_train.clip(lower=lo, upper=hi)
    y_test = y_test.clip(lower=lo, upper=hi)

    # LEAKAGE GUARD: Only allow D_b if predicting Close
    valid_features = [f for f in BASE_FEATURES if f in df.columns]
    if target == 'Perm_tCLOSE' and 'D_b' in df.columns:
        valid_features.append('D_b')

    X_train = train_df.loc[tr_mask, valid_features]
    X_test = test_df.loc[te_mask, valid_features]
    
    # TRAIN
    model = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=5, 
        tree_method='hist', random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # EVALUATE
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    # Directional Accuracy (does predicted sign match actual sign?)
    dir_acc = accuracy_score((y_test > 0).astype(int), (preds > 0).astype(int))
    
    # PnL SIMULATION: Only trade top 25% of predicted signals
    threshold = np.percentile(np.abs(preds), 75)
    traded_indices = np.abs(preds) > threshold
    
    if traded_indices.sum() > 0:
        # Since Perm_ is already direction-adjusted, the actual value IS our return
        captured_perm = y_test.values[traded_indices].mean()
    else:
        captured_perm = 0

    results.append({
        'Config': args.config,
        'Ticker': args.ticker,
        'Target': target,
        'R2': round(r2, 4),
        'MAE': round(mae, 4),
        'DirAcc': round(dir_acc, 4),
        'Top25_Avg_Perm': round(captured_perm, 4),
        'Trades_Taken': traded_indices.sum()
    })

if results:
    res_df = pd.DataFrame(results)
    out_path = 'results/diverse_regression_summary.csv'
    if os.path.exists(out_path):
        res_df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(out_path, index=False)