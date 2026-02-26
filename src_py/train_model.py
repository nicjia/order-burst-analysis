import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import joblib

def load_burst_files():
    files = glob.glob('bursts_*.csv')
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'Perm_tCLOSE' in df.columns:
            df['source'] = f
            dfs.append(df)
            print(f"Loaded {f}: {len(df)} bursts")
        else:
            print(f"Skipping {f}: no Perm_tCLOSE column")
    if not dfs:
        print("No valid burst files found.")
        return None
    return pd.concat(dfs, ignore_index=True)

def engineer_features(df):
    df = df.copy()
    df['Duration'] = df['EndTime'] - df['StartTime']
    df['PeakImpact'] = abs(df['PeakPrice'] - df['StartPrice'])
    df['PriceChange'] = df['EndPrice'] - df['StartPrice']
    df['AvgTradeSize'] = df['Volume'] / df['TradeCount'].clip(lower=1)

    # Forward-return features (mid-price change from burst end)
    for col in ['Mid_1m', 'Mid_3m', 'Mid_5m', 'Mid_10m']:
        if col in df.columns:
            label = col.replace('Mid_', 'FwdRet_')
            df[label] = df[col] - df['EndPrice']
    return df

def main():
    df = load_burst_files()
    if df is None:
        return

    df = engineer_features(df)
    df = df.dropna(subset=['Perm_tCLOSE'])
    df = df[df['Perm_tCLOSE'].abs() < 1000]  # remove outliers

    features = ['Direction', 'Volume', 'TradeCount', 'Duration', 'PeakImpact', 'PriceChange', 'AvgTradeSize']
    X = df[features]
    y = df['Perm_tCLOSE']

    print(f"\nTotal samples: {len(df)}")
    print(f"Features: {features}")
    print(f"Target: Perm_tCLOSE")
    print()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Classification accuracy: predict sign of permanence (persist vs revert)
    y_test_sign = (y_test > 0).astype(int)
    y_pred_sign = (y_pred > 0).astype(int)
    accuracy = accuracy_score(y_test_sign, y_pred_sign)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R2:   {r2:.4f}")
    print(f"Direction Accuracy (persist vs revert): {accuracy:.2%}")
    print()

    print("Feature Importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    joblib.dump(model, 'permanence_model.pkl')
    print("\nModel saved to permanence_model.pkl")

if __name__ == '__main__':
    main()
