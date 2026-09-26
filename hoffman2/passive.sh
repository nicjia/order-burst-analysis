#!/bin/bash
# program-evidence-v1 module H: for one fingerprint-v1 ticker, download each sampled date, extract
# non-round limit-order submissions (passive_extract.py), then compute passive fingerprints
# (passive_stats.py) against the cached aggressive packets. Multiplexed SSH to lobster2.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GD=${FP_DIR:?}; OUT=${EV_OUT:?}
CODE=results/program_evidence_v1/code
test -f src_py/burst_alt.py; test -s "$CODE/passive_extract.py"; test -s "$CODE/burst_alt.py"
TK=$(sed -n "${SGE_TASK_ID}p" "$GD/universe.txt")
test -n "$TK"
mkdir -p "$OUT/adds/$TK" "$OUT/status/$TK" "$OUT/raw/$TK" "$OUT/stats"
[ -s "$OUT/stats/$TK.npz" ] && exit 0
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
  dd=$1
  dest="$OUT/adds/$TK/$dd.npz"; status="$OUT/status/$TK/$dd.txt"
  if [ -s "$status" ] && [ "$(cat "$status")" = ok ] && [ -s "$dest" ]; then return 0; fi
  if [ -s "$status" ] && [ "$(cat "$status")" = missing ]; then return 0; fi
  if [ ! -s "$GD/packets/$TK/$dd.npz" ]; then printf 'missing\n' > "$status"; return 0; fi
  remote="/lobster/${dd:0:4}/$dd/$TK.7z"
  rc=0; lob "test -s '$remote'" || rc=$?
  if [ "$rc" = 1 ]; then printf 'missing\n' > "$status"; return 0; fi
  if [ "$rc" != 0 ]; then printf 'connection_failure\n' > "$status"; exit "$rc"; fi
  d=$(mktemp -d "$OUT/raw/$TK/${dd}.XXXXXX")
  ok=0
  for attempt in 1 2 3 4 5; do
    if lob "cat '$remote'" > "$d/archive.7z" && ~/bin/7z t "$d/archive.7z" > /dev/null 2>&1; then ok=1; break; fi
    sleep $(( (RANDOM % 20) + 10 * attempt ))
  done
  if [ "$ok" != 1 ]; then printf 'download_failure\n' > "$status"; rm -rf -- "$d"; exit 3; fi
  if ! ~/bin/7z x "$d/archive.7z" -o"$d/extracted" -y > "$d/extract.log"; then printf 'archive_failure\n' > "$status"; rm -rf -- "$d"; exit 4; fi
  mapfile -t messages < <(find "$d/extracted" -name '*message*.csv')
  if [ "${#messages[@]}" != 1 ]; then printf 'message_count_failure\n' > "$status"; rm -rf -- "$d"; exit 5; fi
  if ! python3 "$CODE/passive_extract.py" --msg "${messages[0]}" --ticker "$TK" --out "$dest" \
      > "$OUT/status/$TK/$dd.json" 2> "$OUT/status/$TK/$dd.stderr"; then
    printf 'extractor_failure\n' > "$status"; rm -rf -- "$d"; exit 6
  fi
  printf 'ok\n' > "$status"; rm -rf -- "$d"
}
export GD OUT TK CODE CM
export -f work lob
lob true
PAR=2
if grep -qx "$TK" results/fingerprint_v1/heavy.txt; then PAR=1; fi
xargs -P"$PAR" -I{} bash -c 'work "$@"' _ {} < "$GD/dates.txt"
while read -r dd; do
  s=$(cat "$OUT/status/$TK/$dd.txt" 2>/dev/null || echo absent)
  if [ "$s" != ok ] && [ "$s" != missing ]; then echo "incomplete receipt $TK $dd: $s" >&2; exit 7; fi
done < "$GD/dates.txt"
python3 "$CODE/passive_stats.py" --adds "$OUT/adds/$TK" --packets "$GD/packets/$TK" --pairs "$GD/pairs.txt" \
  --ticker "$TK" --out "$OUT/stats/$TK.npz" > "$OUT/status/$TK/stats.json" 2> "$OUT/status/$TK/stats.stderr"
rmdir "$OUT/raw/$TK" 2>/dev/null || true
