#!/bin/bash
# poll the EMO array until it clears qstat, then aggregate
SSH="ssh -o ConnectTimeout=25 -o BatchMode=yes -o RemoteCommand=none hoff"
for i in $(seq 1 45); do
  running=$($SSH 'qstat 2>/dev/null | grep -c hidden_emo' 2>/dev/null | tr -d '[:space:]')
  done_files=$($SSH 'ls /u/scratch/n/nicjia/order-burst-analysis/results/hidden_emo/out/ 2>/dev/null | wc -l' 2>/dev/null | tr -d '[:space:]')
  echo "poll $i: running_tasks=$running out_files=$done_files"
  if [ "$running" = "0" ] && [ "$done_files" -gt 0 ]; then
    echo "ARRAY DONE — aggregating"
    break
  fi
  sleep 55
done
SP="/Users/nick/order-burst-analysis/measurements"
$SSH 'cd /u/scratch/n/nicjia/order-burst-analysis/results/hidden_emo && head -1 $(ls out/*.csv | head -1) && for f in out/*.csv; do tail -n +2 "$f"; done' 2>/dev/null | grep -v Pseudo > "$SP/data/hidden_emo_rows.csv"
echo "downloaded $(wc -l < $SP/data/hidden_emo_rows.csv) rows to hidden_emo_rows.csv"
