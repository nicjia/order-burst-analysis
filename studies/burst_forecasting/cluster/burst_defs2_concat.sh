#!/bin/bash
#$ -cwd
#$ -o results/burst_defs_raw2/log/
#$ -e results/burst_defs_raw2/log/
# Concatenate the v2 raw-pass outputs: sampled bursts -> RAW2_bursts.csv.gz, 30-min buckets -> RAW2_buckets.csv.gz.
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
python3 - <<'PY'
import glob, pandas as pd
from pathlib import Path
for kind, pat, out in (("bursts", "*[0-9].csv.gz", "RAW2_bursts.csv.gz"), ("buckets", "*_buckets.csv.gz", "RAW2_buckets.csv.gz")):
    parts = []
    for f in glob.glob("results/burst_defs_raw2/out/RAW2/*/" + pat):
        try:
            d = pd.read_csv(f, dtype={"date": str})
        except Exception:
            continue
        if len(d):
            d.insert(0, "permno", int(Path(f).parent.name)); parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    a.to_csv("results/burst_defs_raw2/" + out, index=False)
    print(kind, "rows", len(a), "files", len(parts))
PY
