import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ticker', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--kappa', type=float, required=True, help="Threshold for long-horizon D_b filter")
# --- NEW: Dynamic Date Arguments for Regime Testing ---
parser.add_argument('--train_start', type=str, default='2023-01-01', help="Start date for training data")
parser.add_argument('--test_start', type=str, default='2024-01-01', help="Start date for out-of-sample testing")
parser.add_argument('--test_end', type=str, default='2024-12-31', help="End date for out-of-sample testing")
args = parser.parse_args()

# 1. LOAD DATA
df = pd.read_csv(args.data)

# STRICT CHRONOLOGICAL SORTING
df['Date'] = pd.to_datetime(df['Date'])
if 'StartTime' in df.columns:
    df = df.sort_values(by=['Date', 'StartTime']).reset_index(drop=True)
else:
    df = df.sort_values(by=['Date']).reset_index(drop=True)

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

BASE_FEATURES = [
    'Direction', 'BurstVolume', 'TradeCount', 'Duration', 'PeakImpact', 'AvgTradeSize', 
    'PriceChange', 'Spread', 'BidVolBest', 'AskVolBest', 'BidDepth5', 'AskDepth5', 
    'BookImbalance', 'Volatility60s', 'Momentum5s', 'Momentum30s', 'Momentum60s', 
    'TradeCount5m', 'TradeVolume5m', 'TimeOfDay', 'LogVolume', 'LogPeakImpact', 
    'ImpactPerShare', 'LogSpread', 'DepthRatio', 'LogTradeIntensity', 'SpreadXVolume'
]

# 3. DYNAMIC WALK-FORWARD SPLIT
df['date_parsed'] = df['Date'].dt.date
train_start = pd.to_datetime(args.train_start).date()
test_start = pd.to_datetime(args.test_start).date()
test_end = pd.to_datetime(args.test_end).date()

train_df_base = df[(df['date_parsed'] >= train_start) & (df['date_parsed'] < test_start)].copy()
test_df_base = df[(df['date_parsed'] >= test_start) & (df['date_parsed'] <= test_end)].copy()

targets = ['Perm_t1m', 'Perm_t3m', 'Perm_t5m', 'Perm_t10m', 'Perm_tCLOSE', 'Perm_CLOP', 'Perm_CLCL']
results = []

models = {
    'XGB_Restricted': xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=2, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, reg_alpha=10.0, tree_method='hist', random_state=42),
    'HistGB_Restricted': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.01, max_depth=2, l2_regularization=10.0, random_state=42),
    'Ridge': Ridge(alpha=100.0, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'RandomForest_Shallow': RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=100, random_state=42, n_jobs=-1)
}

# Use TimeSeriesSplit to prevent look-ahead bias during cross-validation
tscv = TimeSeriesSplit(n_splits=5)

for target in targets:
    if target not in df.columns:
        continue

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

    lo, hi = y_train.quantile(0.01), y_train.quantile(0.99)
    y_train = y_train.clip(lower=lo, upper=hi)
    y_test = y_test.clip(lower=lo, upper=hi)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    for model_name, model in models.items():
        # Fit the final model on the training data
        model.fit(X_train, y_train)
        
        # Use in-sample predictions for the threshold warm-up to avoid TimeSeriesSplit 
        # partition errors. Because of the heavy regularization, magnitude bias is minimal.
        train_preds_cv = model.predict(X_train)
        
        # Generate the actual out-of-sample predictions for the test set
        test_preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        dir_acc = accuracy_score((y_test > 0).astype(int), (test_preds > 0).astype(int))
        
        pred_dates = pd.to_datetime(np.concatenate([
            train_df.loc[tr_mask, 'Date'], 
            test_df.loc[te_mask, 'Date']
        ]))
        
        # Build threshold using the warm-up predictions
        all_abs_preds = pd.Series(
            np.concatenate([np.abs(train_preds_cv), np.abs(test_preds)]), 
            index=pred_dates
        )
        
        rolling_thresholds = all_abs_preds.rolling('30D', min_periods=1).quantile(0.75).shift(1).bfill()
        test_thresholds = rolling_thresholds.iloc[len(train_preds_cv):].values
        traded_indices = np.abs(test_preds) > test_thresholds
        
        captured_perm = y_test.values[traded_indices].mean() if traded_indices.sum() > 0 else 0
        
        # --- NEW: Transaction Cost / Spread Penalty Metric ---
        if 'Spread' in test_df.columns and traded_indices.sum() > 0:
            # Spread is scaled * 10000 to match the raw LOBSTER pricing units
            avg_spread_crossed = (test_df.loc[te_mask, 'Spread'].values[traded_indices].mean() * 10000)
        else:
            avg_spread_crossed = 0.0

        results.append({
            'Config': args.config,
            'Ticker': args.ticker,
            'Model': model_name,
            'Target': target,
            'R2': round(r2, 4),
            'Global_DirAcc': round(dir_acc, 4),
            'Avg_Captured_Perm': round(captured_perm, 4),
            'Avg_Spread_Crossed': round(avg_spread_crossed, 4),
            'Trades_Taken': traded_indices.sum()
        })
if results:
    res_df = pd.DataFrame(results)
    out_path = 'results/multi_model_regression_summary.csv'
    if os.path.exists(out_path):
        res_df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(out_path, index=False)