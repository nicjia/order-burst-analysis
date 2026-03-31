#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/multi_regress_$JOB_ID.out
#$ -l h_data=16G,h_rt=12:00:00
#$ -pe shared 8

set -Eeo pipefail

# --- DYNAMIC REGIME DATES ---
TRAIN_START="2023-01-01"
TEST_START="2024-01-01"
TEST_END="2024-12-31"
# ----------------------------

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

mkdir -p "${ROOT}/logs" "${ROOT}/results"
rm -f "${ROOT}/results/multi_model_regression_summary.csv" 

NAMES=(
  "s0p5_v50_d0p7_r0p5_k0p2"
  "s0p5_v100_d0p8_r0p1_k0p2"
  "s0p5_v50_d0p7_r0p3_k1p0"
  "s0p5_v100_d0p9_r0p5_k0p2"
  "s2p0_v50_d0p7_r0p5_k0p5"
)
KAPPAS=("0.2" "0.2" "1.0" "0.2" "0.5")

echo "========== MULTI-MODEL REGRESSION EVALUATION =========="
echo "Regime: Train [${TRAIN_START} to ${TEST_START}] | Test [${TEST_START} to ${TEST_END}]"

for TICKER in NVDA TSLA JPM MS; do
    echo -e "\n>>> Processing ${TICKER} <<<"
    
    for i in "${!NAMES[@]}"; do
        CNAME="${NAMES[$i]}"
        KVAL="${KAPPAS[$i]}"
        FILTERED_CSV="${ROOT}/results/champ_${TICKER}_${CNAME}_bursts_filtered.csv"

        # --- AUTO-RUN C++ DATA PROCESSOR IF MISSING ---
        if [ ! -f "${FILTERED_CSV}" ]; then
            echo "Missing ${FILTERED_CSV}! Auto-running data_processor..."
            
            # Parse parameters natively in bash from the CNAME
            IFS='_' read -ra PARTS <<< "$CNAME"
            S_VAL=${PARTS[0]:1} ; S_VAL=${S_VAL//p/.}
            V_VAL=${PARTS[1]:1}
            D_VAL=${PARTS[2]:1} ; D_VAL=${D_VAL//p/.}
            R_VAL=${PARTS[3]:1} ; R_VAL=${R_VAL//p/.}
            
            # Run C++ executable (assuming raw data is in data/TICKER)
            ./data_processor "${ROOT}/data/${TICKER}" "${FILTERED_CSV}" \
                -s "${S_VAL}" \
                -v "${V_VAL}" \
                -d "${D_VAL}" \
                -r "${R_VAL}" \
                -k "${KVAL}" \
                -t 10.0 \
                -j 8
        fi
        # ----------------------------------------------

        echo "Running Python Eval for Config: ${CNAME}"
        
        python src_py/regression_eval.py \
            --ticker "${TICKER}" \
            --data "${FILTERED_CSV}" \
            --config "${CNAME}" \
            --kappa "${KVAL}" \
            --train_start "${TRAIN_START}" \
            --test_start "${TEST_START}" \
            --test_end "${TEST_END}"
    done
done

echo -e "\n========== PIPELINE COMPLETE =========="