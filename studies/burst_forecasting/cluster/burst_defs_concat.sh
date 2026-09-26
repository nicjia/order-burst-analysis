#!/bin/bash
#$ -cwd
#$ -o results/burst_defs_raw/log/
#$ -e results/burst_defs_raw/log/
# Concatenate the raw-pass per-name-day outputs into one table with a permno column (held on the shard arrays).
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
python3 - <<'PY'
import glob, pandas as pd
from pathlib import Path
parts = []
for f in glob.glob("results/burst_defs_raw/out/RAW/*/*.csv.gz"):
    try:
        d = pd.read_csv(f, dtype={"date": str})
    except Exception:
        continue
    if len(d):
        d.insert(0, "permno", int(Path(f).parent.name)); parts.append(d)
all_ = pd.concat(parts, ignore_index=True)
all_.to_csv("results/burst_defs_raw/RAW_all.csv.gz", index=False)
print("rows", len(all_), "name-days", len(parts), all_.groupby("defn").size().to_dict())
PY
