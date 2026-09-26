#!/bin/bash
#$ -cwd
#$ -o results/burst_pseudo/log/
#$ -e results/burst_pseudo/log/
# P4 revisit v1 stage 1 (P4_REVISIT_DESIGN.md section 8). Processes the lines of job file JOBS
# ("date ticker permno") whose line number falls in shard SGE_TASK_ID of SHARDS, PAR at a time:
# download the LOBSTER archive through one multiplexed lobster2 connection, run the MODE script
# (extract: p4_extract.py -> out/CELL/PERMNO/DATE.npz; q0: p4_q0_legacy.py -> out/CELL/PERMNO/DATE.jsonl),
# delete the raw files. One status line per name-day goes to out/CELL/_status/shardS.txt and the
# extractor's JSON to out/CELL/_json/shardS.jsonl; name-days already ok or missing are skipped.
# Submit: qsub -t 1-N -l highp,h_rt=24:00:00,h_data=6G -pe shared PAR -q bertozzi_pod.q \
#           -v JOBS=...,CELL=...,SHARDS=N,PAR=10[,MODE=q0] hoffman2/p4_shard.sh
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${JOBS:?}" "${CELL:?}" "${SHARDS:?}"
PAR=${PAR:-6}
export MODE=${MODE:-extract}
export RETRY=${RETRY:-0}   # 1: JOBS lists later-ticker candidates for missing name-days (src_py/p4_alt_tickers.py)
S=$(( ${SGE_TASK_ID:-1} - 1 ))
export BASE=results/burst_pseudo CODE=results/p4_revisit_v1/code CELL
SUFFIX=$S; [ "$RETRY" = 1 ] && SUFFIX="retry$S"
export STATUS="$BASE/out/$CELL/_status/shard$SUFFIX.txt" JSONL="$BASE/out/$CELL/_json/shard$SUFFIX.jsonl"
mkdir -p "$BASE/out/$CELL/_status" "$BASE/out/$CELL/_json" "$BASE/raw" "$BASE/log"
export CM=$(mktemp -d /tmp/p4cm.XXXXXX)
trap 'ssh -o ControlPath="$CM/s" -O exit nicjia@lobster2.math.ucla.edu >/dev/null 2>&1 || true; rm -rf "$CM"' EXIT

lob(){
  local a rc
  for a in 1 2 3 4 5 6 7 8; do
    rc=0
    ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto -o ControlPath="$CM/s" \
        -o ControlPersist=600 nicjia@lobster2.math.ucla.edu "$@" || rc=$?
    if [ "$rc" = 0 ] || [ "$rc" = 1 ]; then return "$rc"; fi
    sleep $(( (RANDOM % 20) + 5 * a ))
  done
  return "$rc"
}

note(){ printf '%s %s %s %s\n' "$1" "$2" "$3" "$4" >> "$STATUS"; }

work(){
  set -uo pipefail
  local dd=$1 tk=$2 pm=$3
  local o="$BASE/out/$CELL/$pm" ext=csv.gz
  [ "$MODE" = q0 ] && ext=jsonl
  mkdir -p "$o"
  if [ "$RETRY" = 1 ] && grep -qh "^$dd [^ ]* $pm ok$" "$BASE/out/$CELL/_status/"shard*.txt 2>/dev/null; then return 0; fi
  local remote="/lobster/${dd:0:4}/$dd/$tk.7z" rc=0 size
  size=$(lob "stat -c %s '$remote'") || rc=$?
  if [ "$rc" = 1 ]; then note "$dd" "$tk" "$pm" missing; return 0; fi
  if [ "$rc" != 0 ] || [ -z "$size" ]; then note "$dd" "$tk" "$pm" connection_failure; return 0; fi
  local d ok=0 a msg
  # node-local scratch for the archive and the decompressed CSV; 7z x below verifies CRCs
  d=$(mktemp -d "${TMPDIR:-/tmp}/p4.$dd.$tk.XXXXXX")
  for a in 1 2 3 4 5; do
    if lob "cat '$remote'" > "$d/a.7z" && [ "$(stat -c %s "$d/a.7z")" = "$size" ]; then ok=1; break; fi
    sleep $(( 10 * a ))
  done
  if [ "$ok" != 1 ]; then note "$dd" "$tk" "$pm" download_failure; rm -rf -- "$d"; return 0; fi
  if ! ~/bin/7z x "$d/a.7z" -o"$d/x" -y > /dev/null 2>&1; then note "$dd" "$tk" "$pm" archive_failure; rm -rf -- "$d"; return 0; fi
  msg=$(find "$d/x" -name '*message*.csv' | head -1)
  if [ -z "$msg" ]; then note "$dd" "$tk" "$pm" message_missing; rm -rf -- "$d"; return 0; fi
  if [ "$MODE" = q0 ]; then
    if python3 "$CODE/p4_q0_legacy.py" --msg "$msg" --ticker "$tk" --out "$o/$dd.jsonl" --helper "$CODE/p4_bbo" 2> "$d/err"; then
      note "$dd" "$tk" "$pm" ok
    else
      note "$dd" "$tk" "$pm" extractor_failure; tail -3 "$d/err" | sed "s/^/$dd $tk: /" >&2
    fi
  else
    if python3 "$CODE/burst_pseudo_raw.py" --msg "$msg" --ticker "$tk" --out "$o/$dd.csv.gz" --helper "$CODE/p4_bbo" \
         > "$d/json" 2> "$d/err"; then
      note "$dd" "$tk" "$pm" ok; cat "$d/json" >> "$JSONL"
    else
      note "$dd" "$tk" "$pm" extractor_failure; tail -3 "$d/err" | sed "s/^/$dd $tk: /" >&2
    fi
  fi
  rm -rf -- "$d"
}
export -f work lob note
lob true
done_list=$(mktemp "$BASE/raw/done.XXXXXX")
{ echo "__sentinel__ __sentinel__"   # never empty: an empty first file would make awk treat every job as done
  cat "$BASE/out/$CELL/_status/"shard*.txt 2>/dev/null | awk -v r="$RETRY" '$4=="ok"||(r!="1"&&$4=="missing"){print $1" "$3}'; } | sort -u > "$done_list"
awk -v n="$SHARDS" -v s="$S" 'NF && (NR - 1) % n == s' "$JOBS" \
  | awk 'NR==FNR{done[$0]=1; next} !(($1" "$3) in done)' "$done_list" - \
  | xargs -P "$PAR" -L1 bash -c 'work "$@" < /dev/null' _
rm -f "$done_list"
echo "shard $((S + 1)) of $SHARDS done $(date)"
