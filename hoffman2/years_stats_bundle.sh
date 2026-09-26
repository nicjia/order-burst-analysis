#!/bin/bash
# Bundled module-J statistics: shard SGE_TASK_ID of YS_SHARDS over results/program_evidence_v1/ys_items.txt
# ("group_dir ticker" lines). Runs the same per-ticker commands as years_stats.sh sequentially, so each
# task is long (the cluster throttles accounts that submit many short jobs).
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
N=${YS_SHARDS:?}; S=$(( SGE_TASK_ID - 1 ))
awk -v n="$N" -v s="$S" 'NF && (NR - 1) % n == s' results/program_evidence_v1/ys_items.txt > "/tmp/ys_items_${JOB_ID:-0}_$SGE_TASK_ID.txt"
while read -r gd tk; do
  u=$(mktemp /tmp/ysu.XXXXXX); echo "$tk" > "$u"
  EV_GROUP=$gd EV_UNIVERSE=$u SGE_TASK_ID=1 bash hoffman2/years_stats.sh || echo "failed $gd $tk" >&2
  rm -f "$u"
done < "/tmp/ys_items_${JOB_ID:-0}_$SGE_TASK_ID.txt"
rm -f "/tmp/ys_items_${JOB_ID:-0}_$SGE_TASK_ID.txt"
echo "shard $SGE_TASK_ID done"
