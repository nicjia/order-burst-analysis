#!/bin/bash
# Canonical packet/fragment extractor. One SGE task per ticker.
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
export OMP_NUM_THREADS=1
GD=results/two_avenue; PAR=6; L=nicjia@lobster2.math.ucla.edu
MODEL=config/metaorder_simulation_model.json
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
[ -z "$TK" ] && { echo "no ticker for task ${SGE_TASK_ID}"; exit 0; }
rowdir=$GD/rows/$TK; tmp=$GD/tmp/$TK; mkdir -p "$rowdir" "$tmp" "$GD/out"
work(){
  dd=$1; TK=$2; L=nicjia@lobster2.math.ucla.edu; GD=results/two_avenue
  rowdir=$GD/rows/$TK; tmp=$GD/tmp/$TK; row="$rowdir/$dd.row"
  [ -s "$row" ] && return 0
  yr=${dd:0:4}; d="$tmp/$dd"
  rsync -a --timeout=120 "$L:/lobster/$yr/$dd/$TK.7z" "$d.7z" 2>/dev/null || { : > "$row"; return 0; }
  [ -s "$d.7z" ] || { : > "$row"; return 0; }
  ~/bin/7z x "$d.7z" -o"$d" -y >/dev/null 2>&1
  msg=$(ls "$d"/*message*.csv 2>/dev/null | head -1)
  if [ -n "$msg" ]; then
    python3 src_py/two_avenue_extract.py --msg "$msg" --ticker "$TK" \
      --model config/metaorder_simulation_model.json --sample-per-day 25 --compact --header \
      > "$row" 2>"$row.err"
  else : > "$row"; fi
  rm -rf "$d" "$d.7z"
}
export -f work
xargs -P$PAR -n1 -I{} bash -c 'work "$@"' _ {} "$TK" < "$GD/dates.txt"
first=$(find "$rowdir" -name '*.row' -size +0c | sort | head -1)
if [ -n "$first" ]; then
  awk 'FNR==1 && NR!=1 {next} {print}' "$rowdir"/*.row > "$GD/out/$TK.csv"
else
  : > "$GD/out/$TK.csv"
fi
rm -rf "$tmp"
