#!/bin/bash
# Tail tasks have one fixed name/date each; the original driver and frozen code are reused.
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
tail_index=$SGE_TASK_ID
export BI_DIR="results/burst_information_v1/tail/$tail_index"
export SGE_TASK_ID=1
exec bash hoffman2/burst_information.sh
