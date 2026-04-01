#!/bin/bash
#$ -cwd
#$ -j y
#$ -o /u/scratch/n/nicjia/order-burst-analysis/logs/funnel_step2_prepare_$JOB_ID_$TASK_ID.out
#$ -l h_data=16G,h_rt=24:00:00
#$ -pe shared 8
#$ -t 1-4
#$ -N prepare_opt

set -Eeo pipefail
trap 'echo "ERROR line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT=/u/scratch/n/nicjia/order-burst-analysis
TICKERS=(NVDA TSLA JPM MS)
IDX=$((SGE_TASK_ID - 1))
TICKER=${TICKERS[$IDX]}

if [ -z "${TICKER}" ]; then
  echo "Invalid SGE_TASK_ID=${SGE_TASK_ID}" >&2
  exit 2
fi

# Required chosen parameter tag, e.g. s0p5_v50_d0p7_r0p5_k0p2
OPT_CONFIG=${OPT_CONFIG:-}
if [ -z "${OPT_CONFIG}" ]; then
  echo "Set OPT_CONFIG (example: qsub -v OPT_CONFIG=s0p5_v50_d0p7_r0p5_k0p2 qsub_funnel_step2_prepare_optimal.sh)" >&2
  exit 3
fi

cd "${ROOT}"
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
source "${ROOT}/.venv/bin/activate"

mkdir -p "${ROOT}/logs" "${ROOT}/results"

# Parse config string.
# Expected tokens: s*, v*, d*, r*, k*
IFS='_' read -ra PARTS <<< "${OPT_CONFIG}"
S_VAL=${PARTS[0]:1}; S_VAL=${S_VAL//p/.}
V_VAL=${PARTS[1]:1}
D_VAL=${PARTS[2]:1}; D_VAL=${D_VAL//p/.}
R_VAL=${PARTS[3]:1}; R_VAL=${R_VAL//p/.}
K_VAL=${PARTS[4]:1}; K_VAL=${K_VAL//p/.}

RAW_BASE="${ROOT}/results/funnel_${TICKER}_${OPT_CONFIG}_bursts.csv"
UNFILTERED="${ROOT}/results/bursts_${TICKER}_unfiltered.csv"
FILTERED="${ROOT}/results/bursts_${TICKER}_filtered.csv"
PERM_OUT="${ROOT}/results/funnel_${TICKER}_${OPT_CONFIG}_bursts_filtered.csv"

# 1) Build raw bursts once (always k=0 at extraction) unless already present.
if [ ! -f "${RAW_BASE}" ]; then
  echo "[${TICKER}] raw bursts missing -> running data_processor"
  ./data_processor "${ROOT}/data/${TICKER}" "${RAW_BASE}" \
    -s "${S_VAL}" -v "${V_VAL}" -d "${D_VAL}" -r "${R_VAL}" -k 0 -t 10.0 -j "${NSLOTS:-1}"
else
  echo "[${TICKER}] cache hit raw bursts: ${RAW_BASE}"
fi

# 2) Build unfiltered permanence file unless already present.
if [ ! -f "${UNFILTERED}" ]; then
  echo "[${TICKER}] unfiltered permanence missing -> compute_permanence --kappa 0"
  python src_py/compute_permanence.py "${RAW_BASE}" "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" --kappa 0
  cp "${PERM_OUT}" "${UNFILTERED}"
else
  echo "[${TICKER}] cache hit unfiltered: ${UNFILTERED}"
fi

# 3) Build filtered permanence file unless already present.
if [ ! -f "${FILTERED}" ]; then
  echo "[${TICKER}] filtered permanence missing -> compute_permanence --kappa ${K_VAL}"
  python src_py/compute_permanence.py "${RAW_BASE}" "${ROOT}/open_all.csv" "${ROOT}/close_all.csv" --kappa "${K_VAL}"
  cp "${PERM_OUT}" "${FILTERED}"
else
  echo "[${TICKER}] cache hit filtered: ${FILTERED}"
fi

echo "Prepared ${TICKER} inputs from OPT_CONFIG=${OPT_CONFIG}"