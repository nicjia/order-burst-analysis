#!/bin/bash
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis || exit 1
. /u/local/Modules/default/init/bash 2>/dev/null
module load gcc/11.3.0 python/3.9.6 2>/dev/null
python3 src_py/aggregate_packet_scaling.py results/packet_scaling_2025
python3 src_py/audit_packet_scaling.py results/packet_scaling_2025 \
  config/packet_scaling_gate.json \
  --production results/packet_scaling_2025/summary.json
