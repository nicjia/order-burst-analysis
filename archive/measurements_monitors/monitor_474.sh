#!/bin/bash
SSH="ssh -o ConnectTimeout=25 -o BatchMode=yes -o RemoteCommand=none hoff"
JID=14132159
for i in $(seq 1 96); do   # up to ~8h at 5-min polls
  alive=$($SSH "qstat -u nicjia 2>/dev/null | grep -c $JID" 2>/dev/null | tr -d '[:space:]')
  outf=$($SSH 'ls /u/scratch/n/nicjia/order-burst-analysis/results/hidden_emo474/out/ 2>/dev/null | wc -l' 2>/dev/null | tr -d '[:space:]')
  echo "poll $i ($(date +%H:%M)): job_lines=$alive out_files=$outf"
  [ "$alive" = "0" ] && [ "$outf" -ge 400 ] && { echo DONE; break; }
  sleep 300
done
$SSH 'cd /u/scratch/n/nicjia/order-burst-analysis/results/hidden_emo474 && for f in out/*.csv; do tail -n +2 "$f"; done' 2>/dev/null | grep -v Pseudo > /Users/nick/order-burst-analysis/measurements/data/hidden_emo474_rows.csv
echo "downloaded $(wc -l < /Users/nick/order-burst-analysis/measurements/data/hidden_emo474_rows.csv) rows"
