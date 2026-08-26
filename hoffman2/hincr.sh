#!/bin/bash
# ideas.sh — SGE array: 15-idea x 3-param sweep, long format.
# One task = one ticker. Emits results/hincr/out/$TK.csv.
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
source .venv/bin/activate 2>/dev/null
export OMP_NUM_THREADS=1
GD=results/hincr; PAR=6; L=nicjia@lobster2.math.ucla.edu
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
[ -z "$TK" ] && { echo "no ticker for task ${SGE_TASK_ID}"; exit 0; }
rowdir=$GD/rows/$TK; tmp=$GD/tmp/$TK; mkdir -p "$rowdir" "$tmp" "$GD/out"
echo "=== [$SGE_TASK_ID] $TK start $(date) ==="
work(){
  dd=$1; TK=$2; L=nicjia@lobster2.math.ucla.edu; GD=results/hincr
  rowdir=$GD/rows/$TK; tmp=$GD/tmp/$TK; row="$rowdir/$dd.row"
  [ -s "$row" ] && return 0
  yr=${dd:0:4}; d="$tmp/$dd"
  rsync -a --timeout=120 "$L:/lobster/$yr/$dd/$TK.7z" "$d.7z" 2>/dev/null || { echo "$TK,$dd,MISSING" > "$row"; return 0; }
  [ -s "$d.7z" ] || { echo "$TK,$dd,MISSING" > "$row"; return 0; }
  ~/bin/7z x "$d.7z" -o"$d" -y >/dev/null 2>&1
  msg=$(ls "$d"/*message*.csv 2>/dev/null | head -1)
  if [ -n "$msg" ]; then python3 src_py/hidden_incr.py --msg "$msg" --ticker "$TK" > "$row" 2>/dev/null
  else echo "$TK,$dd,MISSING" > "$row"; fi
  rm -rf "$d" "$d.7z"
}
export -f work
xargs -P$PAR -n1 -I{} bash -c 'work "$@"' _ {} "$TK" < "$GD/dates.txt"
echo "ticker,date,W,nb,r2_0,r2_v,r2_vh,r2_h,r2_full,t_h_given_v,t_v_given_h,t_h_given_vvol" > "$GD/out/$TK.csv"
cat "$rowdir"/*.row 2>/dev/null >> "$GD/out/$TK.csv"
rm -rf "$tmp"
echo "=== [$SGE_TASK_ID] $TK done $(date): $(($(wc -l < "$GD/out/$TK.csv")-1)) rows ==="
