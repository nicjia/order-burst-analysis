#!/bin/bash
# fingerprint-v1: one task per name. Stage 1 caches packets per ticker-day (explicit
# ok/missing/failure receipts); stage 2 computes pair statistics once all receipts exist.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${FP_DIR:?set FP_DIR to the group directory}
CODE=results/fingerprint_v1/code
test -f src_py/burst_alt.py
test -s "$CODE/fingerprint_stats.py"
test -s "$GD/pairs.txt"
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
test -n "$TK"
mkdir -p "$GD/packets/$TK" "$GD/status/$TK" "$GD/raw/$TK" "$GD/stats"
if [ -s "$GD/status/$TK/stats.txt" ] && [ "$(cat "$GD/status/$TK/stats.txt")" = ok ] && [ -s "$GD/stats/$TK.npz" ]; then
  exit 0
fi
# lobster2 drops SSH connections under load ("Connection closed by remote host"): retry with
# jittered backoff. Exit status 1 from `test -s` is a genuine "missing" answer, not retried.
lob(){
  local attempt rc
  for attempt in 1 2 3 4 5 6 7 8; do
    rc=0
    ssh -o BatchMode=yes -o ConnectTimeout=30 nicjia@lobster2.math.ucla.edu "$@" || rc=$?
    if [ "$rc" = 0 ] || [ "$rc" = 1 ]; then return "$rc"; fi
    sleep $(( (RANDOM % 20) + 5 * attempt ))
  done
  return "$rc"
}
work(){
  set -euo pipefail
  dd=$1
  dest="$GD/packets/$TK/$dd.npz"
  status="$GD/status/$TK/$dd.txt"
  if [ -s "$status" ] && [ "$(cat "$status")" = ok ] && [ -s "$dest" ]; then return 0; fi
  if [ -s "$status" ] && [ "$(cat "$status")" = missing ]; then return 0; fi
  remote="/lobster/${dd:0:4}/$dd/$TK.7z"
  rc=0
  lob "test -s '$remote'" || rc=$?
  if [ "$rc" = 1 ]; then printf 'missing\n' > "$status"; return 0; fi
  if [ "$rc" != 0 ]; then printf 'connection_failure\n' > "$status"; exit "$rc"; fi
  d=$(mktemp -d "$GD/raw/$TK/${dd}.XXXXXX")
  ok=0
  for attempt in 1 2 3 4 5; do
    if lob "cat '$remote'" > "$d/archive.7z" && ~/bin/7z t "$d/archive.7z" > /dev/null 2>&1; then ok=1; break; fi
    sleep $(( (RANDOM % 20) + 10 * attempt ))
  done
  if [ "$ok" != 1 ]; then
    printf 'download_failure\n' > "$status"; rm -rf -- "$d"; exit 3
  fi
  if ! ~/bin/7z x "$d/archive.7z" -o"$d/extracted" -y > "$d/extract.log"; then
    printf 'archive_failure\n' > "$status"; rm -rf -- "$d"; exit 4
  fi
  mapfile -t messages < <(find "$d/extracted" -name '*message*.csv')
  if [ "${#messages[@]}" != 1 ]; then printf 'message_count_failure\n' > "$status"; rm -rf -- "$d"; exit 5; fi
  if ! python3 "$CODE/fingerprint_packets.py" --msg "${messages[0]}" --ticker "$TK" --out "$dest" \
    > "$GD/status/$TK/$dd.json" 2> "$GD/status/$TK/$dd.stderr"; then
    printf 'extractor_failure\n' > "$status"; rm -rf -- "$d"; exit 6
  fi
  printf 'ok\n' > "$status"
  rm -rf -- "$d"
}
export GD TK CODE
export -f work lob
PAR=2
if grep -qx "$TK" results/fingerprint_v1/heavy.txt; then PAR=1; fi
xargs -P"$PAR" -I{} bash -c 'work "$@"' _ {} < "$GD/dates.txt"
# Stage 2 only when every planned ticker-day has an ok or missing receipt.
while read -r dd; do
  s=$(cat "$GD/status/$TK/$dd.txt" 2>/dev/null || echo absent)
  if [ "$s" != ok ] && [ "$s" != missing ]; then echo "incomplete receipt $TK $dd: $s" >&2; exit 7; fi
done < "$GD/dates.txt"
python3 "$CODE/fingerprint_stats.py" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" \
  --ticker "$TK" --out "$GD/stats/$TK.npz" > "$GD/status/$TK/stats.json" 2> "$GD/status/$TK/stats.stderr"
printf 'ok\n' > "$GD/status/$TK/stats.txt"
rmdir "$GD/raw/$TK" 2>/dev/null || true
