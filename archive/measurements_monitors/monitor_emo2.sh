#!/bin/bash
SSH="ssh -o ConnectTimeout=25 -o BatchMode=yes -o RemoteCommand=none hoff"
for i in $(seq 1 40); do
  running=$($SSH 'qstat 2>/dev/null | grep -c hidden_emo' 2>/dev/null | tr -d '[:space:]')
  outf=$($SSH 'ls /u/scratch/n/nicjia/order-burst-analysis/results/hidden_emo/out/ 2>/dev/null | wc -l' 2>/dev/null | tr -d '[:space:]')
  echo "poll $i: running=$running out=$outf"
  [ "$running" = "0" ] && [ "$outf" -ge 35 ] && { echo DONE; break; }
  sleep 55
done
$SSH 'cd /u/scratch/n/nicjia/order-burst-analysis/results/hidden_emo && for f in out/*.csv; do tail -n +2 "$f"; done' 2>/dev/null | grep -v Pseudo > /Users/nick/order-burst-analysis/measurements/data/hidden_emo_decomp.csv
echo "downloaded $(wc -l < /Users/nick/order-burst-analysis/measurements/data/hidden_emo_decomp.csv) rows"
