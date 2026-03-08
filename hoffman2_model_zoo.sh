#!/bin/bash
#===================================================================
# hoffman2_model_zoo.sh — SGE Job Array for Model Zoo
#
# Runs every (model × target) combination as a separate SGE task.
# Each task gets its own slot and runs independently.
#
# Usage:
#   qsub hoffman2_model_zoo.sh <bursts_csv>
#
# Example:
#   qsub hoffman2_model_zoo.sh results/bursts_NVDA_filtered.csv
#
# To re-aggregate after all jobs finish:
#   python3 src_py/train_model_zoo.py dummy --model aggregate \
#       --outdir results/zoo_NVDA/
#===================================================================

#$ -cwd
#$ -j y
#$ -o logs/zoo_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=4:00:00
#$ -pe shared 4
#$ -t 1-80

# ── 1. Initialize cluster profile (enables `module` command) ─
. /etc/profile

# ── 2. Load Python environment ──────────────────────────────
module load anaconda3

# ── 3. Install missing packages (pinned for numpy 1.19 compat) ─
pip install --user --quiet "lightgbm>=3.3,<4.0" "xgboost>=1.7,<2.0" 2>/dev/null

# ── 4. Handle SGE 1-based → Python 0-based indexing ─────────
INDEX=$((SGE_TASK_ID - 1))

BURSTS_CSV="${1:-results/bursts_NVDA_filtered.csv}"
OUTDIR="results/zoo_$(basename ${BURSTS_CSV%.csv})/"

echo "=========================================="
echo "SGE Job Array: Model Zoo"
echo "  Job ID:       ${JOB_ID}"
echo "  Task ID:      ${SGE_TASK_ID}  (0-based index: ${INDEX})"
echo "  Bursts CSV:   ${BURSTS_CSV}"
echo "  Output Dir:   ${OUTDIR}"
echo "  Hostname:     $(hostname)"
echo "  Date:         $(date)"
echo "=========================================="

mkdir -p "${OUTDIR}" logs

# ── 5. Run single job from array (explicitly use python3) ────
python3 src_py/train_model_zoo.py "${BURSTS_CSV}" \
    --model all \
    --target all \
    --features extended \
    --outdir "${OUTDIR}" \
    --slurm-index ${INDEX}

echo "Task ${SGE_TASK_ID} (index ${INDEX}) finished at $(date)"


#===================================================================
# ALTERNATIVE: Run all models sequentially on a single big node
# (Submit with: qsub -l h_data=32G,h_rt=24:00:00 -pe shared 8 ...)
#===================================================================
# python3 src_py/train_model_zoo.py "${BURSTS_CSV}" \
#     --model all \
#     --target all \
#     --features extended \
#     --outdir "${OUTDIR}"
