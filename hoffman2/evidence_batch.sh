#!/bin/bash
# program-evidence-v1 quick modules in one multi-core job: campaigns and markouts (per name, both
# fingerprint groups) and synchrony (per date). Same per-name commands as evidence_module.sh and
# evidence_sync.sh, run with xargs -P$NSLOTS to avoid per-task scheduling delays.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=${NSLOTS:-4}
for g in explore_2024 confirm_2021; do
  G=results/fingerprint_v1/$g; O=results/program_evidence_v1/$g
  for MOD in price_null campaigns markouts; do
    export FP_DIR=$G EV_OUT=$O/$MOD EV_MODULE=$MOD
    n=$(wc -l < $G/universe.txt)
    seq 1 $n | xargs -P"$P" -I{} bash -c 'SGE_TASK_ID={} bash hoffman2/evidence_module.sh || echo "failed $EV_MODULE $FP_DIR task {}" >&2'
  done
  export FP_DIR=$G EV_OUT=$O/sync
  seq 1 20 | xargs -P"$(( P > 4 ? 4 : P ))" -I{} bash -c 'SGE_TASK_ID={} bash hoffman2/evidence_sync.sh || echo "failed sync $FP_DIR task {}" >&2'
done
echo done
