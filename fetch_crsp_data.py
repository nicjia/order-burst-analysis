import wrds
import pandas as pd
import numpy as np

# 1. Connect to WRDS (will prompt for your WRDS username/password)
db = wrds.Connection()

# 2. Query the CRSP Daily Stock File mapped to Tickers for 2025-2026
# We join crsp.dsf with crsp.stocknames to get string tickers instead of PERMNOs
sql = """
SELECT a.date, b.ticker, a.openprc, a.prc
FROM crsp.dsf AS a
JOIN crsp.stocknames AS b
    ON a.permno = b.permno
    AND a.date >= b.namedt
    AND a.date <= b.nameendt
WHERE a.date >= '2025-01-01'
  AND b.ticker IS NOT NULL
"""
print("Fetching 2025-2026 data from WRDS...")
df = db.raw_sql(sql)

# 3. Clean and format CRSP quirks
# CRSP sets 'prc' to negative if there was no closing trade (uses bid/ask average)
df['prc'] = df['prc'].abs()

# Drop rows where price is missing
df = df.dropna(subset=['openprc', 'prc'])

# Format date to YYYYMMDD integer to match your existing matrices (e.g., 20241230)
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(int)

# 4. Pivot into Date (rows) x Ticker (columns) matrices
print("Pivoting matrices...")
open_df = df.pivot_table(index='date', columns='ticker', values='openprc')
close_df = df.pivot_table(index='date', columns='ticker', values='prc')

# 5. Save the update files
open_df.to_csv('open_update_2025_2026.csv')
close_df.to_csv('close_update_2025_2026.csv')
print("Complete. Updates saved to CSV.")