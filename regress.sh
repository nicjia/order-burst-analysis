#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/diverse_reg_$JOB_ID.out
#$ -l h_data=16G,h_rt=12:00:00
#$ -pe shared 8
#$ -N diverse_regression

set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

mkdir -p "${ROOT}/logs" "${ROOT}/results"
rm -f "${ROOT}/results/diverse_regression_summary.csv" # Clear old results

# --- The Diverse 5 Configurations ---
NAMES=(
  "s0p5_v50_d0p7_r0p5_k0p2"
  "s0p5_v100_d0p8_r0p1_k0p2"
  "s0p5_v50_d0p7_r0p3_k1p0"
  "s0p5_v100_d0p9_r0p5_k0p2"
  "s2p0_v50_d0p7_r0p5_k0p5"
)
ARGS=(
  "-s 0.5 -v 50 -d 0.7 -r 0.5"
  "-s 0.5 -v 100 -d 0.8 -r 0.1"
  "-s 0.5 -v 50 -d 0.7 -r 0.3"
  "-s 0.5 -v 100 -d 0.9 -r 0.5"
  "-s 2.0 -v 50 -d 0.7 -r 0.5"
)
KAPPAS=("0.2" "0.2" "1.0" "0.2" "0.5")

echo "========== DIVERSE REGRESSION PIPELINE =========="

for TICKER in NVDA TSLA JPM MS; do
    echo -e "\n>>> Processing ${TICKER} <<<"
    
    for i in "${!NAMES[@]}"; do
        CNAME="${NAMES[$i]}"
        CARGS="${ARGS[$i]}"
        KVAL="${KAPPAS[$i]}"
        
        OUT_CSV="${ROOT}/results/champ_${TICKER}_${CNAME}_bursts.csv"
        FILTERED_CSV="${ROOT}/results/champ_${TICKER}_${CNAME}_bursts_filtered.csv"

        echo "Running Config: ${CNAME}"
        
        # 1. C++ Processor
        ./data_processor "${ROOT}/data/${TICKER}" "${OUT_CSV}" \
          ${CARGS} -k ${KVAL} -t 10.0 -j 8 -b 34200 -e 57600
        
        # 2. Python Permanence (Attach Returns)
        python src_py/compute_permanence.py "${OUT_CSV}" "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" --kappa ${KVAL}
        
        # 3. XGBoost Regression
        python src_py/regression_eval.py --ticker "${TICKER}" --data "${FILTERED_CSV}" --config "${CNAME}"
    done
done

echo -e "\n========== PIPELINE COMPLETE =========="