#!/bin/bash
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null; source .venv/bin/activate
export OMP_NUM_THREADS=1
L=nicjia@lobster2.math.ucla.edu; GD=results/research/poisson
rm -rf $GD; mkdir -p $GD/rows $GD/tmp
# ~40 names spread across the universe, 12 dates spread across 2023
awk 'NR%12==1' results/hidden_xsec/universe.txt > $GD/names.txt
grep -E '^2023' results/hidden_xsec/dates.txt | awk 'NR%20==1' | head -12 > $GD/dates.txt
echo "names=$(wc -l < $GD/names.txt) dates=$(wc -l < $GD/dates.txt)"
> $GD/tasks; for t in $(cat $GD/names.txt); do for dd in $(cat $GD/dates.txt); do echo "$t $dd" >> $GD/tasks; done; done
echo "tasks=$(wc -l < $GD/tasks) $(date)"
work(){ t=$1; dd=$2; L=nicjia@lobster2.math.ucla.edu; GD=results/research/poisson; yr=${dd:0:4}; d=$GD/tmp/${t}_${dd}
  row="$GD/rows/${t}_${dd}.row"; [ -s "$row" ] && return 0
  rsync -a --timeout=120 "$L:/lobster/$yr/$dd/$t.7z" "$d.7z" 2>/dev/null || return 0
  [ -s "$d.7z" ] || return 0
  ~/bin/7z x "$d.7z" -o"$d" -y >/dev/null 2>&1
  msg=$(ls "$d"/*message*.csv 2>/dev/null | head -1)
  [ -n "$msg" ] && python3 src_py/poisson_test.py --msg "$msg" --ticker "$t" > "$row" 2>/dev/null
  rm -rf "$d" "$d.7z"; }
export -f work
cat $GD/tasks | xargs -P12 -n2 bash -c 'work "$@"' _
echo "rows=$(ls $GD/rows/*.row 2>/dev/null | wc -l) $(date)"
echo "ticker,date,n_trades,fano,obs,poisson_mean,poisson_std,z" > results/research/poisson_daily.csv
cat $GD/rows/*.row 2>/dev/null >> results/research/poisson_daily.csv
python3 - <<'PY'
import pandas as pd, numpy as np
d=pd.read_csv("results/research/poisson_daily.csv")
for c in ["n_trades","fano","obs","poisson_mean","poisson_std","z"]: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=["z","fano"])
print(f"\n=== B2 POISSON NULL TEST: {len(d)} ticker-days, {d.ticker.nunique()} names ===")
print(f"  Fano factor (index of dispersion; Poisson=1): median={d.fano.median():.1f} mean={d.fano.mean():.1f} min={d.fano.min():.1f}")
print(f"  observed same-side bursts/day median={int(d.obs.median())} vs Poisson-null median={int(d.poisson_mean.median())}")
print(f"  z (obs vs homogeneous-Poisson+iid-sign null): median={d.z.median():.1f} mean={d.z.mean():.1f} min={d.z.min():.1f}")
print(f"  fraction of ticker-days with z>3 (reject Poisson): {100*(d.z>3).mean():.1f}%")
print(f"  fraction with z>5: {100*(d.z>5).mean():.1f}%")
PY
echo "DONE_POISSON rc=$? $(date)"
