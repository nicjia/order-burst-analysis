#!/bin/bash
# program-evidence-v1 contiguous panels: cache fingerprint-v1 packets for every trading day of a
# point-in-time name (one task per CRSP permno; job file lines "YYYYMMDD TICKER").
# One multiplexed SSH master per task avoids lobster2's handshake limit ("Connection closed").
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${CG_DIR:?set CG_DIR to the contiguous group directory}
CODE=results/fingerprint_v1/code
test -f src_py/burst_alt.py
test -s "$CODE/fingerprint_packets.py"
PERMNO=$(sed -n "${SGE_TASK_ID}p" "$GD/${CG_UNIVERSE:-universe.txt}")
test -n "$PERMNO"
JOB="$GD/jobs/$PERMNO.txt"
test -s "$JOB"
mkdir -p "$GD/packets/$PERMNO" "$GD/status/$PERMNO" "$GD/raw/$PERMNO"
CM=$(mktemp -d /tmp/cm.XXXXXX)
trap 'ssh -o ControlPath="$CM/s" -O exit nicjia@lobster2.math.ucla.edu >/dev/null 2>&1 || true; rm -rf "$CM"' EXIT
lob(){
  local attempt rc
  for attempt in 1 2 3 4 5 6 7 8; do
    rc=0
    ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto -o ControlPath="$CM/s" -o ControlPersist=900 \
      nicjia@lobster2.math.ucla.edu "$@" || rc=$?
    if [ "$rc" = 0 ] || [ "$rc" = 1 ]; then return "$rc"; fi
    sleep $(( (RANDOM % 20) + 5 * attempt ))
  done
  return "$rc"
}
work(){
  set -euo pipefail
  dd=$1; tk=$2
  dest="$GD/packets/$PERMNO/$dd.npz"
  status="$GD/status/$PERMNO/$dd.txt"
  if [ -s "$status" ] && [ "$(cat "$status")" = ok ] && [ -s "$dest" ]; then return 0; fi
  if [ -s "$status" ] && [ "$(cat "$status")" = missing ]; then return 0; fi
  remote="/lobster/${dd:0:4}/$dd/$tk.7z"
  rc=0
  lob "test -s '$remote'" || rc=$?
  if [ "$rc" = 1 ]; then printf 'missing\n' > "$status"; return 0; fi
  if [ "$rc" != 0 ]; then printf 'connection_failure\n' > "$status"; exit "$rc"; fi
  d=$(mktemp -d "$GD/raw/$PERMNO/${dd}.XXXXXX")
  ok=0
  for attempt in 1 2 3 4 5; do
    if lob "cat '$remote'" > "$d/archive.7z" && ~/bin/7z t "$d/archive.7z" > /dev/null 2>&1; then ok=1; break; fi
    sleep $(( (RANDOM % 20) + 10 * attempt ))
  done
  if [ "$ok" != 1 ]; then printf 'download_failure\n' > "$status"; rm -rf -- "$d"; exit 3; fi
  if ! ~/bin/7z x "$d/archive.7z" -o"$d/extracted" -y > "$d/extract.log"; then
    printf 'archive_failure\n' > "$status"; rm -rf -- "$d"; exit 4
  fi
  mapfile -t messages < <(find "$d/extracted" -name '*message*.csv')
  if [ "${#messages[@]}" != 1 ]; then printf 'message_count_failure\n' > "$status"; rm -rf -- "$d"; exit 5; fi
  if ! python3 "$CODE/fingerprint_packets.py" --msg "${messages[0]}" --ticker "$tk" --out "$dest" \
    > "$GD/status/$PERMNO/$dd.json" 2> "$GD/status/$PERMNO/$dd.stderr"; then
    printf 'extractor_failure\n' > "$status"; rm -rf -- "$d"; exit 6
  fi
  printf 'ok\n' > "$status"
  rm -rf -- "$d"
}
export GD PERMNO CODE CM
export -f work lob
lob true   # open the master connection once
PAR=${CG_PAR:-2}
xargs -P"$PAR" -L1 bash -c 'work "$@"' _ < "$JOB"
bad=0
while read -r dd tk; do
  s=$(cat "$GD/status/$PERMNO/$dd.txt" 2>/dev/null || echo absent)
  if [ "$s" != ok ] && [ "$s" != missing ]; then echo "incomplete receipt $PERMNO $dd: $s" >&2; bad=1; fi
done < "$JOB"
rmdir "$GD/raw/$PERMNO" 2>/dev/null || true
exit "$bad"
