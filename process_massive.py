import pandas as pd
import glob
import os

def process_massive_data(data_dir, open_base_file, close_base_file):
    print(f"Loading Massive data from {data_dir}...")
    
    # 1. Grab all the downloaded CSVs for 2025 and 2026
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv*'), recursive=True)
    
    if not csv_files:
        print("No CSV files found! Did the AWS S3 sync finish?")
        return

    # Load all files into one large dataframe
    df_list = []
    for f in csv_files:
        temp_df = pd.read_csv(f)
        df_list.append(temp_df)
    
    massive_df = pd.concat(df_list, ignore_index=True)
    
    # 2. Format columns to match our expected standard
    # Assuming Massive provides 'date', 'ticker', 'open', 'close'
    # Convert dates to YYYYMMDD integers (e.g., 20250102)
    massive_df['date'] = pd.to_datetime(massive_df['date']).dt.strftime('%Y%m%d').astype(int)
    
    print("Pivoting data to Ticker-by-Date matrices...")
    # 3. Pivot to match the CRSP matrix format (Rows: Date, Columns: Ticker)
    new_open_df = massive_df.pivot_table(index='date', columns='ticker', values='open')
    new_close_df = massive_df.pivot_table(index='date', columns='ticker', values='close')
    
    # 4. Merge with the existing 2016-2024 CRSP files
    print("Merging with existing baselines...")
    old_open_df = pd.read_csv(open_base_file, index_col=0)
    old_close_df = pd.read_csv(close_base_file, index_col=0)
    
    # Concatenate vertically and sort by date
    final_open = pd.concat([old_open_df, new_open_df]).sort_index()
    final_close = pd.concat([old_close_df, new_close_df]).sort_index()
    
    # 5. Save back to disk
    final_open.to_csv('open_all_extended.csv')
    final_close.to_csv('close_all_extended.csv')
    
    print(f"Done! New max date is: {final_close.index.max()}")

if __name__ == "__main__":
    # Point this to where you ran the aws s3 sync
    MASSIVE_DATA_PATH = "massive_data/"
    
    process_massive_data(
        data_dir=MASSIVE_DATA_PATH,
        open_base_file="open_all.csv",
        close_base_file="close_all.csv"
    )
