import glob
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def process_file(fpath):
    try:
        # LOBSTER cols: Time, Type, OrderID, Size, Price, Direction
        msg_df = pd.read_csv(fpath, header=None, usecols=[0, 1, 3], names=['Time', 'Type', 'Size'], engine='c')
        # Filter for RTH (seconds from midnight: 9:30 AM to 4:00 PM)
        rth_mask = (msg_df['Time'] >= 34200.0) & (msg_df['Time'] <= 57600.0)
        msg_df = msg_df[rth_mask]
        # Sum ONLY Type 4 (Visible Execution) and Type 5 (Hidden Execution)
        traded_vol = msg_df[msg_df['Type'].isin([4, 5])]['Size'].sum()
        
        fname = os.path.basename(fpath)
        date_str = fname.split('_')[1]
        return date_str, traded_vol
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return None, None

def main():
    tickers = ["NVDA", "TSLA", "JPM", "MS", "AAPL", "LLY", "SPY"]
    base_dir = "data"
    out_file = "results/true_adv_daily.csv"
    
    all_results = []
    
    for ticker in tickers:
        stock_folder = os.path.join(base_dir, ticker)
        msg_files = sorted(glob.glob(os.path.join(stock_folder, "*_message_*.csv")))
        if not msg_files:
            continue
            
        print(f"Processing {len(msg_files)} files for {ticker}...")
        
        daily_vols = {}
        with ProcessPoolExecutor(max_workers=4) as executor:
            for date_str, vol in executor.map(process_file, msg_files):
                if date_str is not None:
                    daily_vols[date_str] = vol
                    
        for date_str, vol in daily_vols.items():
            all_results.append({
                "Ticker": ticker,
                "Date": date_str,
                "TradedVolume": vol
            })
            
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_file, index=False)
        print(f"Saved true traded volume to {out_file}")

if __name__ == "__main__":
    main()
