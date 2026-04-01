#!/bin/bash
#===================================================================
# hoffman2_model_zoo.sh — SGE Job Array: Two-Phase Model Zoo
#
# Phase 1 (Tasks 1-84):  SHORT horizons on UNFILTERED data
#   → cls_1m, cls_3m, cls_5m, cls_10m  (D_b features auto-dropped)
#
# Phase 2 (Tasks 85-147): LONG horizons on FILTERED data
#   → cls_close, cls_clop, cls_clcl
#
# Usage:
#   qsub hoffman2_model_zoo.sh
#===================================================================

#$ -cwd
#$ -j y
#$ -o logs/zoo_$JOB_ID_$TASK_ID.out
#$ -l h_data=8G,h_rt=4:00:00
#$ -pe shared 4
#$ -t 1-147

# ── 1. Initialize cluster profile (enables `module` command) ─
. /etc/profile

# ── 2. Load Python environment ──────────────────────────────
module load anaconda3

# Tickers to evaluate for cross-stock stability.
TICKERS=${TICKERS:-"NVDA TSLA JPM MS"}

# ── 3. Determine phase from task ID ─────────────────────────
PHASE1_JOBS=84   # short horizons (unfiltered): cls models × 4 targets

if [ ${SGE_TASK_ID} -le ${PHASE1_JOBS} ]; then
    # Phase 1: Short-horizon prediction on unfiltered bursts
    INDEX=$((SGE_TASK_ID - 1))
    TARGET="short"
    PHASE_TAG="unfiltered"
else
    # Phase 2: Long-horizon prediction on filtered bursts
    INDEX=$((SGE_TASK_ID - PHASE1_JOBS - 1))
    TARGET="long"
    PHASE_TAG="filtered"
fi

echo "=========================================="
echo "SGE Job Array: Model Zoo (Two-Phase)"
echo "  Job ID:       ${JOB_ID}"
echo "  Task ID:      ${SGE_TASK_ID}  (0-based index: ${INDEX})"
echo "  Phase:        ${TARGET}"
echo "  Tickers:      ${TICKERS}"
echo "  Hostname:     $(hostname)"
echo "  Date:         $(date)"
echo "=========================================="

mkdir -p logs

for TICKER in ${TICKERS}; do
    if [ "${PHASE_TAG}" = "unfiltered" ]; then
        BURSTS_CSV="results/bursts_${TICKER}_unfiltered.csv"
    else
        BURSTS_CSV="results/bursts_${TICKER}_filtered.csv"
    fi
    OUTDIR="results/zoo_bursts_${TICKER}_${PHASE_TAG}/"

    echo "Running ${TICKER} | ${TARGET} | ${BURSTS_CSV}"
    mkdir -p "${OUTDIR}"

    python3 src_py/train_model_zoo.py "${BURSTS_CSV}" \
        --model all \
        --target "${TARGET}" \
        --features extended \
        --outdir "${OUTDIR}" \
        --slurm-index ${INDEX}
done

echo "Task ${SGE_TASK_ID} (index ${INDEX}, phase=${TARGET}) finished at $(date)"