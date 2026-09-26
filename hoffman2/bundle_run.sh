#!/bin/bash
# Run the commands of BUNDLE_FILE (one shell command per line) whose line number falls in shard
# SGE_TASK_ID of BUNDLE_SHARDS, sequentially, from the project root. Each command must be
# idempotent (skip if its output exists). Long tasks avoid the cluster's short-job throttle.
# Commands read stdin from /dev/null: ssh inside a command would otherwise consume the command list.
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
N=${BUNDLE_SHARDS:?}; S=$(( SGE_TASK_ID - 1 )); F=${BUNDLE_FILE:?}
awk -v n="$N" -v s="$S" 'NF && (NR - 1) % n == s' "$F" | while IFS= read -r cmd; do
  bash -c "$cmd" < /dev/null || echo "failed: $cmd" >&2
done
echo "shard $SGE_TASK_ID of $N done"
