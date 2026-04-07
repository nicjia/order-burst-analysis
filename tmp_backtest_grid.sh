#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/u/scratch/n/nicjia/order-burst-analysis}
cd "$ROOT"

source /u/local/Modules/default/init/bash
module load gcc/10.2.0
module load python/3.9.6
source "$ROOT/.venv/bin/activate"

DATA=${DATA:-results/nvda_archive_2019_2022_raw_s2p0_filtered.csv}
OUTDIR=${OUTDIR:-results/grid_backtest_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"

if [ ! -f "$DATA" ]; then
  echo "ERROR: DATA not found: $DATA"
  exit 1
fi

TARGETS=(reg_1m reg_3m reg_10m)
SIGNAL_MODES=(percentile cost_aware)
BUFFERS=(0.25 0.5 0.75 1.0)

SUMMARY="$OUTDIR/summary.csv"
echo "run_id,target,signal_mode,cost_buffer,trades,cum_pnl_raw,sharpe" > "$SUMMARY"

run_id=0
for tgt in "${TARGETS[@]}"; do
  for sm in "${SIGNAL_MODES[@]}"; do
    if [ "$sm" = "percentile" ]; then
      run_id=$((run_id+1))
      log="$OUTDIR/run_${run_id}_${tgt}_${sm}.log"
      python3 src_py/online_sgd_backtest.py \
        --data "$DATA" \
        --target "$tgt" \
        --silence-tag s2p0 \
        --vol-frac 0.0027 \
        --dir-thresh 0.68 \
        --vol-ratio 0.36 \
        --execution-mode burst_stream \
        --signal-mode percentile \
        --mid-col EndPrice \
        --spread-col Spread \
        --spread-multiplier 0.5 \
        --spread-exit-multiplier 0.5 \
        --position-mode shares \
        --shares-per-trade 1 \
        --pnl-space raw > "$log" 2>&1

      trades=$(grep -E "Total Trades Fired" "$log" | tail -n1 | sed -E 's/.*: *([0-9,]+).*/\1/' | tr -d ',')
      pnl=$(grep -E "Cumulative Simulated PnL \(raw\)" "$log" | tail -n1 | sed -E 's/.*: *([-0-9.]+).*/\1/')
      sharpe=$(grep -E "Annualized Sharpe Ratio" "$log" | tail -n1 | sed -E 's/.*: *([-0-9.]+).*/\1/')
      echo "$run_id,$tgt,$sm,NA,$trades,$pnl,$sharpe" >> "$SUMMARY"
    else
      for cb in "${BUFFERS[@]}"; do
        run_id=$((run_id+1))
        log="$OUTDIR/run_${run_id}_${tgt}_${sm}_cb${cb}.log"
        python3 src_py/online_sgd_backtest.py \
          --data "$DATA" \
          --target "$tgt" \
          --silence-tag s2p0 \
          --vol-frac 0.0027 \
          --dir-thresh 0.68 \
          --vol-ratio 0.36 \
          --execution-mode burst_stream \
          --signal-mode cost_aware \
          --cost-buffer-mult "$cb" \
          --mid-col EndPrice \
          --spread-col Spread \
          --spread-multiplier 0.5 \
          --spread-exit-multiplier 0.5 \
          --position-mode shares \
          --shares-per-trade 1 \
          --pnl-space raw > "$log" 2>&1

        trades=$(grep -E "Total Trades Fired" "$log" | tail -n1 | sed -E 's/.*: *([0-9,]+).*/\1/' | tr -d ',')
        pnl=$(grep -E "Cumulative Simulated PnL \(raw\)" "$log" | tail -n1 | sed -E 's/.*: *([-0-9.]+).*/\1/')
        sharpe=$(grep -E "Annualized Sharpe Ratio" "$log" | tail -n1 | sed -E 's/.*: *([-0-9.]+).*/\1/')
        echo "$run_id,$tgt,$sm,$cb,$trades,$pnl,$sharpe" >> "$SUMMARY"
      done
    fi
  done
done

echo "Grid done. Summary: $SUMMARY"
column -s, -t "$SUMMARY" | sed 's/^/  /'
