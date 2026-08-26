#!/bin/bash
# zoo_dev.sh — SGE array: alternative burst definition sweep (development set).
# One task = one ticker. Emits results/zoo_dev/out/$TK.csv.
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
source .venv/bin/activate 2>/dev/null
export OMP_NUM_THREADS=1
GD=results/zoo_dev; PAR=6; L=nicjia@lobster2.math.ucla.edu
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
[ -z "$TK" ] && { echo "no ticker for task ${SGE_TASK_ID}"; exit 0; }
rowdir=$GD/rows/$TK; tmp=$GD/tmp/$TK; mkdir -p "$rowdir" "$tmp" "$GD/out"
echo "=== [$SGE_TASK_ID] $TK start $(date) ==="
work(){
  dd=$1; TK=$2; L=nicjia@lobster2.math.ucla.edu; GD=results/zoo_dev
  rowdir=$GD/rows/$TK; tmp=$GD/tmp/$TK; row="$rowdir/$dd.row"
  [ -s "$row" ] && return 0
  yr=${dd:0:4}; d="$tmp/$dd"
  rsync -a --timeout=120 "$L:/lobster/$yr/$dd/$TK.7z" "$d.7z" 2>/dev/null || { echo "$TK,$dd,MISSING" > "$row"; return 0; }
  [ -s "$d.7z" ] || { echo "$TK,$dd,MISSING" > "$row"; return 0; }
  ~/bin/7z x "$d.7z" -o"$d" -y >/dev/null 2>&1
  msg=$(ls "$d"/*message*.csv 2>/dev/null | head -1)
  if [ -n "$msg" ]; then python3 src_py/burst_zoo.py --msg "$msg" --ticker "$TK" > "$row" 2>/dev/null
  else echo "$TK,$dd,MISSING" > "$row"; fi
  rm -rf "$d" "$d.7z"
}
export -f work
xargs -P$PAR -n1 -I{} bash -c 'work "$@"' _ {} "$TK" < "$GD/dates.txt"
echo "ticker,date,d1_hidden_time_n,d1_hidden_time_mk3,d1_hidden_time_mk10,d1_hidden_time_net3,d2_hidden_rate_n,d2_hidden_rate_mk3,d2_hidden_rate_mk10,d2_hidden_rate_net3,d3_vis_time_n,d3_vis_time_mk3,d3_vis_time_mk10,d3_vis_time_net3,d4_vis_big_n,d4_vis_big_mk3,d4_vis_big_mk10,d4_vis_big_net3,d5_cancel_n,d5_cancel_mk3,d5_cancel_mk10,d5_cancel_net3,d6_submit_n,d6_submit_mk3,d6_submit_mk10,d6_submit_net3,d7_mixed_n,d7_mixed_mk3,d7_mixed_mk10,d7_mixed_net3,d8_accel_n,d8_accel_mk3,d8_accel_mk10,d8_accel_net3,d9_oddlot_n,d9_oddlot_mk3,d9_oddlot_mk10,d9_oddlot_net3,d10_block_n,d10_block_mk3,d10_block_mk10,d10_block_net3" > "$GD/out/$TK.csv"
cat "$rowdir"/*.row 2>/dev/null >> "$GD/out/$TK.csv"
rm -rf "$tmp"
echo "=== [$SGE_TASK_ID] $TK done $(date): $(($(wc -l < "$GD/out/$TK.csv")-1)) rows ==="
