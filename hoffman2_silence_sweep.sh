#!/bin/bash
# SGE job script for efficient silence-first parameter sweep on Hoffman2.
# Submit from /u/scratch/n/nicjia/order-burst-analysis:
#   qsub hoffman2_silence_sweep.sh

#$ -cwd
#$ -j y
#$ -o logs/silence_sweep_$JOB_ID.out
#$ -l h_data=24G,h_rt=24:00:00
#$ -pe shared 8
#$ -N silence_sweep

. /etc/profile
module load gcc/11.3.0
module load python/3.9.6

mkdir -p logs results

echo "=== Build C++ ==="
make clean && make || exit 1

echo "=== Run efficient sweep ==="
python3 src_py/silence_optimized_sweep.py \
  --stock-folder /u/scratch/n/nicjia/order-burst-analysis/data/NVDA \
  --ticker NVDA \
  --open /u/scratch/n/nicjia/order-burst-analysis/open_all.csv \
  --close /u/scratch/n/nicjia/order-burst-analysis/close_all.csv \
  --data-processor /u/scratch/n/nicjia/order-burst-analysis/data_processor \
  --outdir /u/scratch/n/nicjia/order-burst-analysis/results/silence_sweep_nvda \
  --silence-values 0.5,1.0,2.0 \
  --min-vol-values 50,100,200 \
  --dir-thresh-values 0.8,0.9 \
  --vol-ratio-values 0.3,0.5 \
  --kappa-values 0.0,0.1,0.2 \
  --tau-max 10.0 \
  --rth-start 34200 \
  --rth-end 57600 \
  --model logreg_l2 \
  --target cls_close \
  --features extended \
  --min-train-months 3 \
  --require-directional \
  --min-rows 500

echo "=== Sweep complete ==="
