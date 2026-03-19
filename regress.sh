#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/multi_regress_$JOB_ID.out
#$ -l h_data=16G,h_rt=12:00:00
#$ -pe shared 8

set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

mkdir -p "${ROOT}/logs" "${ROOT}/results"
# Give the output a new name so we don't accidentally overwrite the old one
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

for TICKER in NVDA TSLA JPM MS; do
    echo -e "\n>>> Processing ${TICKER} <<<"
    
    for i in "${!NAMES[@]}"; do
        CNAME="${NAMES[$i]}"
        KVAL="${KAPPAS[$i]}"
        
        # We point directly to the files that already exist in your results folder!
        FILTERED_CSV="${ROOT}/results/champ_${TICKER}_${CNAME}_bursts_filtered.csv"

        echo "Running Python Eval for Config: ${CNAME}"
        
        # We ONLY run the Python script
        python src_py/regression_eval.py \
            --ticker "${TICKER}" \
            --data "${FILTERED_CSV}" \
            --config "${CNAME}" \
            --kappa "${KVAL}"
    done
done

echo -e "\n========== PIPELINE COMPLETE =========="