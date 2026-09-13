"""Reconstruct a small invented message sequence without licensed data."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_py"))
from burst_alt import reconstruct

MESSAGES = """34200.0,1,1,100,1000000,1
34200.1,1,2,100,1000200,-1
34200.2,1,3,100,1000300,-1
34200.3,4,2,100,1000200,-1
34200.4,5,4,25,1000150,1
"""

def main():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "synthetic_message.csv"
        path.write_text(MESSAGES)
        times, mids, bids, asks, bid_sizes, ask_sizes, ofi, trades = reconstruct(path)
    print("Synthetic messages; no historical observations.")
    for t, bid, ask, mid in zip(times, bids, asks, mids):
        print(f"{t:.1f}: bid={bid:.2f}, ask={ask:.2f}, midpoint={mid:.3f}")
    print(f"Executions: {len(trades[0])}; hidden: {int(trades[3].sum())}")
    print("Hidden-trade direction in the generic parser is not a validated aggressor sign.")

if __name__ == "__main__":
    main()
