#!/bin/bash
# One task per name; explicit missing/failure receipts, atomic per-day outputs.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${BI_DIR:-results/burst_information_v1}
CODE=results/burst_information_v1/code
test -f src_py/burst_alt.py
test -s "$GD/design.json"
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
test -n "$TK"
mkdir -p "$GD/rows/$TK" "$GD/status/$TK" "$GD/raw/$TK"
work(){
  set -euo pipefail
  dd=$1
  dest="$GD/rows/$TK/$dd.csv"
  status="$GD/status/$TK/$dd.txt"
  if [ -s "$status" ] && [ "$(cat "$status")" = ok ] && [ -s "$dest" ]; then return 0; fi
  if [ -s "$status" ] && [ "$(cat "$status")" = missing ]; then return 0; fi
  remote="/lobster/${dd:0:4}/$dd/$TK.7z"
  rc=0
  ssh -o BatchMode=yes -o ConnectTimeout=30 nicjia@lobster2.math.ucla.edu "test -s '$remote'" || rc=$?
  if [ "$rc" = 1 ]; then printf 'missing\n' > "$status"; return 0; fi
  if [ "$rc" != 0 ]; then printf 'connection_failure\n' > "$status"; exit "$rc"; fi
  d=$(mktemp -d "$GD/raw/$TK/${dd}.XXXXXX")
  if ! ssh -o BatchMode=yes nicjia@lobster2.math.ucla.edu "cat '$remote'" > "$d/archive.7z"; then
    printf 'download_failure\n' > "$status"; exit 3
  fi
  if ! ~/bin/7z x "$d/archive.7z" -o"$d/extracted" -y > "$d/extract.log"; then
    printf 'archive_failure\n' > "$status"; exit 4
  fi
  mapfile -t messages < <(find "$d/extracted" -name '*message*.csv')
  if [ "${#messages[@]}" != 1 ]; then printf 'message_count_failure\n' > "$status"; exit 5; fi
  if ! python3 "$CODE/burst_information_extract.py" --msg "${messages[0]}" --ticker "$TK" \
    --model "$CODE/metaorder_simulation_model.json" --sample-modulus 32 --out "$dest.part" \
    > "$GD/status/$TK/$dd.json" 2> "$GD/status/$TK/$dd.stderr"; then
    printf 'extractor_failure\n' > "$status"; exit 6
  fi
  mv "$dest.part" "$dest"
  printf 'ok\n' > "$status"
  rm -rf -- "$d"
}
export GD TK CODE
export -f work
xargs -P2 -I{} bash -c 'work "$@"' _ {} < "$GD/dates.txt"
