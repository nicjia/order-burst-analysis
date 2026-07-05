#!/bin/bash
# hidden_xsec.sh — SGE job ARRAY: full-universe hidden-execution cross-section (referee M12).
# One array task = one ticker. Streams that ticker's 2023-2024 LOBSTER message files from
# lobster2, Lee-Ready-signs hidden (type-5) execs, clusters into bursts, computes 3/15/30-min
# markouts + daily signed hidden COI, and DELETES raw data on the fly (stays under quota).
# Resumable: a date whose .row already exists is skipped.
#
# Submit (throttled to protect lobster2):
#   qsub -t 1-483 -tc 16 -l h_data=4G,h_rt=8:00:00 -pe shared 6 -cwd \
#        -o logs/hxsec.$TASK_ID.out -e logs/hxsec.$TASK_ID.err -N hxsec hoffman2/hidden_xsec.sh
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
source .venv/bin/activate
export OMP_NUM_THREADS=1
L=nicjia@lobster2.math.ucla.edu
GD=results/hidden_xsec
PAR=6                          # concurrent dates within this ticker

TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
[ -z "$TK" ] && { echo "no ticker for task ${SGE_TASK_ID}"; exit 0; }
rowdir=$GD/rows/$TK; tmp=$GD/tmp/$TK
mkdir -p "$rowdir" "$tmp" "$GD/out"
echo "=== [$SGE_TASK_ID] $TK start $(date) ==="

work(){
  dd=$1; TK=$2; L=nicjia@lobster2.math.ucla.edu; GD=results/hidden_xsec
  rowdir=$GD/rows/$TK; tmp=$GD/tmp/$TK
  row="$rowdir/$dd.row"
  [ -s "$row" ] && return 0                       # resume: already done
  yr=${dd:0:4}; d="$tmp/$dd"
  rsync -a --timeout=120 "$L:/lobster/$yr/$dd/$TK.7z" "$d.7z" 2>/dev/null || { echo "$TK,$dd,0,nan,nan,nan,0,0,0,0" > "$row"; return 0; }
  [ -s "$d.7z" ] || { echo "$TK,$dd,0,nan,nan,nan,0,0,0,0" > "$row"; return 0; }
  ~/bin/7z x "$d.7z" -o"$d" -y >/dev/null 2>&1
  msg=$(ls "$d"/*message*.csv 2>/dev/null | head -1)
  if [ -n "$msg" ]; then
    python3 src_py/hidden_full.py --msg "$msg" --ticker "$TK" > "$row" 2>/dev/null
  else
    echo "$TK,$dd,0,nan,nan,nan,0,0,0,0" > "$row"
  fi
  rm -rf "$d" "$d.7z"
}
export -f work

xargs -P$PAR -n1 -I{} bash -c 'work "$@"' _ {} "$TK" < "$GD/dates.txt"

# consolidate this ticker's rows -> one csv
echo "ticker,date,n,mk3,mk15,mk30,buy,sell,n_mid,n_sig" > "$GD/out/$TK.csv"
cat "$rowdir"/*.row 2>/dev/null >> "$GD/out/$TK.csv"
rm -rf "$tmp"
echo "=== [$SGE_TASK_ID] $TK done $(date): $(($(wc -l < "$GD/out/$TK.csv")-1)) day-rows ==="
