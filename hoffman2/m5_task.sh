#!/bin/bash
# Metaorder-v1 M5: one ticker-day. Download the LOBSTER archive (multiplexed SSH), run metaorder_passive.py,
# delete the raw files. Idempotent. Usage: m5_task.sh GROUP TICKER DATE RULE MODEL
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
G=$1; TK=$2; DD=$3; RULE=$4; MODEL=$5
O=results/metaorder_v1/$G/passive/$TK; mkdir -p "$O"
[ -s "$O/$DD.csv.gz" ] && exit 0
CODE=results/metaorder_v1/code
CM=$(mktemp -d /tmp/cm.XXXXXX); trap 'ssh -o ControlPath="$CM/s" -O exit nicjia@lobster2.math.ucla.edu >/dev/null 2>&1 || true; rm -rf "$CM"' EXIT
lob(){ local a rc; for a in 1 2 3 4 5 6 7 8; do rc=0; ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto -o ControlPath="$CM/s" -o ControlPersist=300 nicjia@lobster2.math.ucla.edu "$@" || rc=$?; if [ "$rc" = 0 ] || [ "$rc" = 1 ]; then return "$rc"; fi; sleep $(( (RANDOM % 20) + 5 * a )); done; return "$rc"; }
remote="/lobster/${DD:0:4}/$DD/$TK.7z"
rc=0; lob "test -s '$remote'" || rc=$?
if [ "$rc" = 1 ]; then echo missing > "$O/$DD.missing"; exit 0; fi
d=$(mktemp -d "results/metaorder_v1/raw.XXXXXX")
trap 'rm -rf -- "$d"; ssh -o ControlPath="$CM/s" -O exit nicjia@lobster2.math.ucla.edu >/dev/null 2>&1 || true; rm -rf "$CM"' EXIT
ok=0
for a in 1 2 3 4 5; do if lob "cat '$remote'" > "$d/a.7z" && ~/bin/7z t "$d/a.7z" > /dev/null 2>&1; then ok=1; break; fi; sleep $(( 10 * a )); done
[ "$ok" = 1 ] || { echo download_failure > "$O/$DD.failed"; exit 3; }
~/bin/7z x "$d/a.7z" -o"$d/x" -y > /dev/null
msg=$(find "$d/x" -name '*message*.csv' | head -1)
python3 "$CODE/metaorder_passive.py" --msg "$msg" --ticker "$TK" --date "$DD" --rule "$RULE" --model "$MODEL" \
  --out "$O/$DD.csv.gz" > "$O/$DD.json" 2> "$O/$DD.stderr"
