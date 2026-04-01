#!/bin/bash
set -Eeo pipefail

ROOT=/u/scratch/n/nicjia/order-burst-analysis
cd "${ROOT}"

# Must be provided by user after selecting sweep winner.
OPT_CONFIG=${OPT_CONFIG:-}
if [ -z "${OPT_CONFIG}" ]; then
	echo "Set OPT_CONFIG, e.g. OPT_CONFIG=s0p5_v50_d0p7_r0p5_k0p2 bash qsub_funnel_submit.sh" >&2
	exit 2
fi

jid1=$(qsub qsub_funnel_step1_model_zoo_baseline.sh | awk '{print $3}')
echo "Submitted Step1 baseline model-zoo array: ${jid1}"

jid2=$(qsub -hold_jid "${jid1}" qsub_funnel_step1_sweep.sh | awk '{print $3}')
echo "Submitted Step2 sweep array (hold_jid=${jid1}): ${jid2}"

jid3=$(qsub -hold_jid "${jid2}" -v OPT_CONFIG="${OPT_CONFIG}" qsub_funnel_step2_prepare_optimal.sh | awk '{print $3}')
echo "Submitted Step3 prepare-opt array (hold_jid=${jid2}): ${jid3}"

jid4=$(qsub -hold_jid "${jid3}" qsub_funnel_step4_regress.sh | awk '{print $3}')
echo "Submitted Step4 regress eval (hold_jid=${jid3}): ${jid4}"

echo "Pipeline submitted. Final job id: ${jid4}"