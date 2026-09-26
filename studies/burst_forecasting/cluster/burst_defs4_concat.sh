#!/bin/bash
#$ -cwd
#$ -o results/burst_defs_raw4/log/
#$ -e results/burst_defs_raw4/log/
# Concatenate the v4 outputs (cell V4B): sampled events -> V4_events.csv.gz, 5-minute bins -> V4_bins.csv.gz.
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
python3 - <<'PY'
import glob, pandas as pd
from pathlib import Path
for kind, pat, out in (("events", "*[0-9].csv.gz", "V4_events.csv.gz"), ("buckets", "*_bins.csv.gz", "V4_bins.csv.gz")):
    parts = []
    for f in glob.glob("results/burst_defs_raw4/out/V4B/*/" + pat):
        try:
            d = pd.read_csv(f, dtype={"date": str})
        except Exception:
            continue
        if len(d):
            d.insert(0, "permno", int(Path(f).parent.name)); parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    a.to_csv("results/burst_defs_raw4/" + out, index=False)
    print(kind, "rows", len(a), "files", len(parts))
PY
