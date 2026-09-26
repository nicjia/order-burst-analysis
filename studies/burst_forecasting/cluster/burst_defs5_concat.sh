#!/bin/bash
#$ -cwd
#$ -o results/burst_defs_raw5/log/
#$ -e results/burst_defs_raw5/log/
# Concatenate the v5 outputs (cell V5) in 8 parallel parts, split by event type so the local analysis can load one
# type at a time: V5_events_<defn>_<kind>_part<k>.csv.gz, V5_bins_part<k>.csv.gz, and V5_check.jsonl (the rebuilt-book
# level-1 checks), V5_blist_part<k>.csv.gz (bursts of 5+ orders, for the cross-stock features). Submit: qsub -N bd5cat -hold_jid <arrays> -l h_rt=4:00:00,h_data=6G -pe shared 8 hoffman2/burst_defs5_concat.sh
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
python3 - <<'PY'
import glob, gzip
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
BASE = "results/burst_defs_raw5/"
files = sorted(glob.glob(BASE + "out/V5/[0-9]*/[0-9]*[0-9].csv.gz"))
parts = [files[k::8] for k in range(8)]

def work(k):
    handles, nev, nbin = {}, 0, 0
    for f in parts[k]:
        permno = int(Path(f).parent.name)
        try:
            d = pd.read_csv(f, dtype={"date": str})
            b = pd.read_csv(f.replace(".csv.gz", "_bins.csv.gz"), dtype={"date": str})
        except Exception:
            continue
        if len(d):
            d.insert(0, "permno", permno)
            for (dn, kind), g in d.groupby(["defn", "kind"]):
                key = "%s_%s" % (dn, kind)
                first = key not in handles
                if first:
                    handles[key] = gzip.open(BASE + "V5_events_%s_part%d.csv.gz" % (key, k), "wt")
                g.to_csv(handles[key], header=first, index=False, float_format="%.7g")
            nev += len(d)
        if len(b):
            b.insert(0, "permno", permno)
            first = "bins" not in handles
            if first:
                handles["bins"] = gzip.open(BASE + "V5_bins_part%d.csv.gz" % k, "wt")
            b.to_csv(handles["bins"], header=first, index=False, float_format="%.7g")
            nbin += len(b)
        try:
            bl = pd.read_csv(f.replace(".csv.gz", "_blist.csv.gz"))
        except Exception:
            bl = pd.DataFrame()
        if len(bl):
            bl.insert(0, "date", Path(f).name[:8]); bl.insert(0, "permno", permno)
            first = "blist" not in handles
            if first:
                handles["blist"] = gzip.open(BASE + "V5_blist_part%d.csv.gz" % k, "wt")
            bl.to_csv(handles["blist"], header=first, index=False)
    for h in handles.values():
        h.close()
    return k, nev, nbin

with Pool(8) as p:
    for k, nev, nbin in p.imap_unordered(work, range(8)):
        print("part", k, "events", nev, "bins", nbin, flush=True)
with open(BASE + "V5_check.jsonl", "w") as out:
    for f in sorted(glob.glob(BASE + "out/V5/_json/*.jsonl")):
        out.write(open(f).read())
print("files", len(files))
PY
