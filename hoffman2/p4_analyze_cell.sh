#!/bin/bash
#$ -cwd
#$ -o results/p4_revisit_v1/log/
#$ -e results/p4_revisit_v1/log/
# P4 revisit v1 stage 3 on the cluster (for cells whose sample file is too large to move).
# Submit: qsub -N p4anTEST -l highp,h_rt=8:00:00,h_data=8G -pe shared 4 -q bertozzi_pod.q -v CELL=TEST hoffman2/p4_analyze_cell.sh
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${CELL:?}"
B=results/p4_revisit_v1
export P4_ANALYSIS_DIR=$B/analysis
python3 $B/code/p4_analyze.py --cell "$CELL" --dir $B/agg/$CELL --boot 1000 --sample --q5 || exit 1
python3 $B/code/p4_phase2.py evaluate --cell "$CELL" --sample $B/agg/$CELL/sample_$CELL.csv.gz --models $B/phase2 || exit 1
echo "analysis $CELL done $(date)"
