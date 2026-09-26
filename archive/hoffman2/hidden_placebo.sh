#!/bin/bash
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null; source .venv/bin/activate
export OMP_NUM_THREADS=1
L=nicjia@lobster2.math.ucla.edu; GD=results/research/hidplacebo
rm -rf $GD; mkdir -p $GD/rows $GD/tmp
awk 'NR%12==1' results/hidden_xsec/universe.txt > $GD/names.txt          # ~40 names
awk 'NR%17==1' results/hidden_xsec/dates.txt | head -30 > $GD/dates.txt   # ~30 dates over 2023-2024
echo "names=$(wc -l < $GD/names.txt) dates=$(wc -l < $GD/dates.txt) $(date)"
> $GD/tasks; for t in $(cat $GD/names.txt); do for dd in $(cat $GD/dates.txt); do echo "$t $dd" >> $GD/tasks; done; done
work(){ t=$1; dd=$2; L=nicjia@lobster2.math.ucla.edu; GD=results/research/hidplacebo; yr=${dd:0:4}; d=$GD/tmp/${t}_${dd}
  row="$GD/rows/${t}_${dd}.row"; [ -s "$row" ] && return 0
  rsync -a --timeout=120 "$L:/lobster/$yr/$dd/$t.7z" "$d.7z" 2>/dev/null || return 0
  [ -s "$d.7z" ] || return 0
  ~/bin/7z x "$d.7z" -o"$d" -y >/dev/null 2>&1
  msg=$(ls "$d"/*message*.csv 2>/dev/null | head -1)
  [ -n "$msg" ] && python3 src_py/hidden_full.py --msg "$msg" --ticker "$t" > "$row" 2>/dev/null
  rm -rf "$d" "$d.7z"; }
export -f work
cat $GD/tasks | xargs -P12 -n2 bash -c 'work "$@"' _
echo "rows=$(ls $GD/rows/*.row 2>/dev/null | wc -l) $(date)"
echo "ticker,date,n,mk3,mk15,mk30,buy,sell,n_mid,n_sig,pmk3,pmk15,pmk30,n_tick,mk3_tick" > results/research/hidden_placebo.csv
cat $GD/rows/*.row 2>/dev/null >> results/research/hidden_placebo.csv
python3 - <<'PY'
import pandas as pd, numpy as np, math
d=pd.read_csv("results/research/hidden_placebo.csv")
for c in ["n","mk3","mk15","mk30","pmk3","pmk15","pmk30","mk3_tick"]: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d[d["n"].fillna(0)>0]
print(f"\n=== HIDDEN PLACEBO (Major 2): {len(d)} ticker-days, {d.ticker.nunique()} names ===")
def dct(x): x=x.dropna(); return (x.mean(), x.mean()/(x.std()/math.sqrt(len(x)))) if len(x)>30 else (np.nan,np.nan)
for h,pb in [("mk3","pmk3"),("mk15","pmk15"),("mk30","pmk30")]:
    bm,bt=dct(d[h]); pm,pt=dct(d[pb]); nm,nt=dct(d[h]-d[pb])
    print(f"  {h:5s}: burst={bm:+.3f} (t={bt:+.1f}) | placebo={pm:+.3f} (t={pt:+.1f}) | NET burst-placebo={nm:+.3f} (t={nt:+.1f}) | survives={100*nm/bm if bm else 0:.0f}%")
# tick-rule signing sensitivity (3-min)
qm,qt=dct(d["mk3"]); tm,tt=dct(d["mk3_tick"])
print(f"  signing sensitivity 3-min: quote-rule={qm:+.3f} (t={qt:+.1f}) | tick-rule(incl at-mid)={tm:+.3f} (t={tt:+.1f})")
PY
echo "DONE_HIDPLACEBO $(date)"
