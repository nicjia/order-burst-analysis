#!/bin/bash
# program-evidence-v1 modules D/E: daily program and other flow for every point-in-time or event
# name whose packet receipts are all final (ok or missing). Safe to rerun; finished names are skipped.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=${NSLOTS:-4}
one(){
  set -euo pipefail
  d=$1; p=$2
  out="$d/flows/$p.csv"
  [ -s "$out" ] && return 0
  job="$d/jobs/$p.txt"; [ -s "$job" ] || return 0
  n=$(wc -l < "$job"); k=$(cat "$d/status/$p/"*.txt 2>/dev/null | grep -c -E "^(ok|missing)$" || true)
  [ "$k" -lt "$n" ] && return 0
  mkdir -p "$d/flows"
  if ls "$d/packets/$p/"*.npz > /dev/null 2>&1; then
    python3 results/program_evidence_v1/code/daily_flow.py --packets "$d/packets/$p" --job "$job" --permno "$p" \
      --model results/program_evidence_v1/program_model_run60.json --out "$out.part" > /dev/null 2> "$d/flows/$p.stderr" \
      && mv "$out.part" "$out"
  else
    printf 'permno,date\n' > "$out"
  fi
}
export -f one
# Optional sharding for single-core array tasks: DF_SHARDS and SGE_TASK_ID pick every N-th name.
N=${DF_SHARDS:-1}; S=$(( ${SGE_TASK_ID:-1} - 1 ))
[ "$N" -gt 1 ] && P=1
for d in results/program_evidence_v1/contig_explore_2024 results/program_evidence_v1/contig_confirm_2021 results/program_evidence_v1/events; do
  awk -v n="$N" -v s="$S" -v d="$d" 'NF && (NR - 1) % n == s {print d, $1}' "$d/universe.txt" | xargs -P"$P" -L1 bash -c 'one "$@"' _
  echo "$d flows $(ls $d/flows/*.csv 2>/dev/null | wc -l) of $(wc -l < $d/universe.txt)"
done
