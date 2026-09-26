#!/bin/bash
# program-evidence-v1 quick per-name modules (price_null, campaigns, markouts) for both fingerprint
# groups, split into EV_SHARDS single-core tasks by line number; shard 1 also runs module G.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
N=${EV_SHARDS:?}; S=$(( SGE_TASK_ID - 1 ))
for g in explore_2024 confirm_2021; do
  G=results/fingerprint_v1/$g; O=results/program_evidence_v1/$g
  n=$(wc -l < $G/universe.txt)
  for MOD in ${EV_MODULES:-price_null campaigns markouts}; do
    for i in $(seq 1 $n); do
      [ $(( (i - 1) % N )) -eq $S ] || continue
      FP_DIR=$G EV_OUT=$O/$MOD EV_MODULE=$MOD SGE_TASK_ID=$i bash hoffman2/evidence_module.sh || echo "failed $MOD $g $i" >&2
    done
  done
  if [ $S -eq 0 ] && [ -z "${EV_MODULES:-}" ]; then
    for i in $(seq 1 20); do FP_DIR=$G EV_OUT=$O/sync SGE_TASK_ID=$i bash hoffman2/evidence_sync.sh || echo "failed sync $g $i" >&2; done
  fi
done
echo shard $SGE_TASK_ID done
