import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ticker', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--kappa', type=float, required=True, help="Threshold for long-horizon D_b filter")
args = parser.parse_args()

# 1. LOAD DATA
df = pd.read_csv(args.data)

# 2. FEATURE ENGINEERING
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

# 3. WALK-FORWARD SPLIT
df['date_parsed'] = pd.to_datetime(df['Date']).dt.date
train_df_base = df[df['date_parsed'] < pd.to_datetime('2024-01-01').date()].copy()
test_df_base = df[df['date_parsed'] >= pd.to_datetime('2024-01-01').date()].copy()

targets = ['Perm_t1m', 'Perm_t3m', 'Perm_t5m', 'Perm_t10m', 'Perm_tCLOSE', 'Perm_CLOP', 'Perm_CLCL']
results = []

# 4. DEFINE THE MODEL ZOO
models = {
    'XGB_Restricted': xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=2, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, reg_alpha=10.0, tree_method='hist', random_state=42),
    'HistGB_Restricted': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.01, max_depth=2, l2_regularization=10.0, random_state=42),
    'Ridge': Ridge(alpha=100.0, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'RandomForest_Shallow': RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=100, random_state=42, n_jobs=-1)
}

for target in targets:
    if target not in df.columns:
        continue

    # DYNAMIC FILTERING & FEATURE SELECTION
    if target in ['Perm_tCLOSE', 'Perm_CLOP', 'Perm_CLCL']:
        tr_filter_mask = (train_df_base['D_b'] >= args.kappa) & train_df_base['D_b'].notna()
        te_filter_mask = (test_df_base['D_b'] >= args.kappa) & test_df_base['D_b'].notna()
        train_df = train_df_base[tr_filter_mask].copy()
        test_df = test_df_base[te_filter_mask].copy()
        valid_features = [f for f in BASE_FEATURES if f in df.columns] + ['D_b']
    else:
        train_df = train_df_base.copy()
        test_df = test_df_base.copy()
        valid_features = [f for f in BASE_FEATURES if f in df.columns]
        
    tr_mask, te_mask = train_df[target].notna(), test_df[target].notna()
    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        continue

    y_train = train_df.loc[tr_mask, target].copy()
    y_test = test_df.loc[te_mask, target].copy()
    X_train_raw = train_df.loc[tr_mask, valid_features]
    X_test_raw = test_df.loc[te_mask, valid_features]

    # WINSORIZING
    lo, hi = y_train.quantile(0.01), y_train.quantile(0.99)
    y_train = y_train.clip(lower=lo, upper=hi)
    y_test = y_test.clip(lower=lo, upper=hi)
    
    # SCALING
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # LOOP OVER MODELS
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        dir_acc = accuracy_score((y_test > 0).astype(int), (test_preds > 0).astype(int))
        
        # Extract the exact dates for every valid prediction
        pred_dates = pd.to_datetime(np.concatenate([
            train_df.loc[tr_mask, 'Date'], 
            test_df.loc[te_mask, 'Date']
        ]))
        
        # Combine absolute predictions and attach the dates as the index
        all_abs_preds = pd.Series(
            np.concatenate([np.abs(train_preds), np.abs(test_preds)]), 
            index=pred_dates
        )
        
        # Calculate a 30-day rolling 75th percentile. 
        # shift(1) guarantees the threshold uses data STRICTLY BEFORE the current burst.
        rolling_thresholds = all_abs_preds.rolling('30D', min_periods=1).quantile(0.75).shift(1).bfill()
        
        # Extract ONLY the thresholds for the 2024 test data
        test_thresholds = rolling_thresholds.iloc[len(train_preds):].values
        
        # Execute trade if test prediction magnitude beats its 30-day rolling threshold
        traded_indices = np.abs(test_preds) > test_thresholds
        # -------------------------------------------
        
        captured_perm = y_test.values[traded_indices].mean() if traded_indices.sum() > 0 else 0

        results.append({
            'Config': args.config,
            'Ticker': args.ticker,
            'Model': model_name,
            'Target': target,
            'R2': round(r2, 4),
            'MAE': round(mae, 4),
            'DirAcc': round(dir_acc, 4),
            'Top25_Avg_Perm': round(captured_perm, 4),
            'Trades_Taken': traded_indices.sum()
        })

if results:
    res_df = pd.DataFrame(results)
    out_path = 'results/multi_model_regression_summary.csv'
    if os.path.exists(out_path):
        res_df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(out_path, index=False)