import glob
import os
import subprocess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def process_file_awk(fpath):
    try:
        # LOBSTER cols: Time, Type, OrderID, Size, Price, Direction
        # Time is $1, Type is $2, Size is $4
        awk_cmd = f"awk -F, '$1 >= 34200.0 && $1 <= 57600.0 && ($2 == 4 || $2 == 5) {{ sum += $4 }} END {{ if (sum == \"\") print 0; else print sum }}' {fpath}"
        result = subprocess.check_output(awk_cmd, shell=True, text=True).strip()
        traded_vol = int(float(result))
        
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
            
        print(f"Processing {len(msg_files)} files for {ticker} using AWK...")
        
        daily_vols = {}
        # using more workers since awk uses almost no memory
        with ProcessPoolExecutor(max_workers=8) as executor:
            for date_str, vol in executor.map(process_file_awk, msg_files):
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
