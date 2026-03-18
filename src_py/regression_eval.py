import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ticker', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--config', required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)

# Targets are now continuous returns
targets = ['ret_1m', 'ret_5m', 'ret_10m', 'ret_close']
exclude_cols = ['timestamp', 'ticker', 'date', 'time'] + targets + [t.replace('ret_', 'cls_') for t in targets]
features = [c for c in df.columns if c not in exclude_cols]

# Walk-Forward Split
df['date'] = pd.to_datetime(df['timestamp'], unit='ns').dt.date
train_df = df[df['date'] < pd.to_datetime('2024-01-01').date()]
test_df = df[df['date'] >= pd.to_datetime('2024-01-01').date()]

X_train, X_test = train_df[features], test_df[features]

results = []
print(f"Training Regressors for {args.ticker} -> {args.config}")

for target in targets:
    if target not in df.columns:
        continue
        
    y_train, y_test = train_df[target], test_df[target]
    
    # Train the Regressor
    model = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=5, 
        tree_method='hist', random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Simulated Trading: Only trade the top 25% strongest predicted magnitudes
    threshold = np.percentile(np.abs(preds), 75)
    traded_indices = np.abs(preds) > threshold
    
    if traded_indices.sum() > 0:
        # PnL logic: Direction of our trade * Actual market return
        trade_returns = np.sign(preds[traded_indices]) * y_test[traded_indices].values
        avg_trade_pnl = trade_returns.mean()
        win_rate = (trade_returns > 0).mean()
    else:
        avg_trade_pnl = 0
        win_rate = 0

    results.append({
        'Config': args.config,
        'Ticker': args.ticker,
        'Target': target,
        'MAE': round(mae, 5),
        'R2': round(r2, 4),
        'Win_Rate': round(win_rate, 4),
        'Avg_Trade_Return': round(avg_trade_pnl, 6),
        'Trades_Taken': traded_indices.sum()
    })

res_df = pd.DataFrame(results)
out_path = 'results/diverse_regression_summary.csv'

# Append to master file
if os.path.exists(out_path):
    res_df.to_csv(out_path, mode='a', header=False, index=False)
else:
    res_df.to_csv(out_path, index=False)