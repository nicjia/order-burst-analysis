## Burst Detection & Analysis Pipeline

### Data Format

Input is a **stock folder** containing one LOBSTER `*_message_0.csv` per trading day.
No `orderbook.csv` is needed — the pipeline reconstructs the top-of-book (Best Bid / Best Ask)
from the message events, which include pre-open orders starting at ~4 AM.

```
data/
  TSLA_2026-01-01_2026-02-14_0/
    TSLA_2026-01-02_34140000_57660000_message_0.csv
    TSLA_2026-01-05_34140000_57660000_message_0.csv
    ...
```

Message CSV columns (7 fields, no header):
`time, type, order_id, size, price, direction, extra`

### Build

```bash
# Local / any Linux with g++ ≥ 7
make clean && make

# UCLA Hoffman2
module load gcc/11.3.0
make clean && make
```

### Run

```bash
./data_processor <stock_folder> <output.csv> [options]

# Example
./data_processor data/TSLA_2026-01-01_2026-02-14_0/ bursts_tsla.csv -s 1.0 -d 0.9
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `-s` | 1.0 | Silence threshold (seconds) to end a burst |
| `-v` | 100 | Minimum burst volume (shares) |
| `-d` | 0.9 | Direction ratio threshold for buy/sell classification |
| `-k` | 0.5 | Kappa filter parameter |

### Hoffman2 (UGE/SGE)

```bash
qsub hoffman2_submit.sh
```

This compiles and processes every stock folder under `data/`.

### Top-of-Book Reconstruction

The `OrderBook` class rebuilds the visible limit order book from LOBSTER messages:

| Type | Event | Book Action |
|------|-------|-------------|
| 1 | Submission | Add order (bid or ask side) |
| 2 | Partial cancel | Reduce order size |
| 3 | Full deletion | Remove order |
| 4 | Visible execution | Reduce / remove order |
| 5 | Hidden execution | No visible-book impact |
| 6 | Cross trade | No book impact |
| 7 | Trading halt | No book impact |

Best Bid = highest resting buy price. Best Ask = lowest resting sell price.
Mid-price snapshots are stored whenever the BBO changes, enabling forward-return lookups.

### Output Columns

| Column | Description |
|--------|-------------|
| Ticker | Stock symbol |
| Date | Trading day (YYYY-MM-DD) |
| BurstID | Order ID of the first trade in the burst |
| StartTime / EndTime | Seconds after midnight |
| Direction | 1 = Buy, -1 = Sell, 0 = Mixed |
| Volume / TradeCount | Total shares & number of trades |
| StartPrice / EndPrice / PeakPrice | Mid-prices in dollars |
| CloseMid | Last mid-price of the day |
| Mid_1m / Mid_3m / Mid_5m / Mid_10m | Mid-price at end + 1/3/5/10 minutes |

### Permanence Calculation

```bash
python src_py/compute_permanence.py bursts_tsla.csv
```

- Uses `CloseMid` from the burst CSV (no orderbook file needed)
- Formula: $direction \times (Close - StartPrice) / |PeakPrice - StartPrice|$
- Also computes `Perm_t1m`, `Perm_t3m`, `Perm_t5m`, `Perm_t10m` from forward-return columns

### Burst Detection Rules

- **Message Type:** Only executions (LOBSTER type 4 or 5)
- **Burst End:** Time gap > silence threshold between consecutive trades
- **Direction:** ≥ threshold ratio buy → 1, sell → -1, else 0 (mixed)
- **Peak Price:** Max mid for buy bursts, min mid for sell bursts
