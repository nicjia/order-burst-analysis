#!/bin/bash
SSH="ssh -o ConnectTimeout=25 -o BatchMode=yes -o RemoteCommand=none hoff"
JID=14152019
for i in $(seq 1 60); do
  alive=$($SSH "qstat -u nicjia 2>/dev/null | grep -c $JID" 2>/dev/null | tr -d '[:space:]')
  outf=$($SSH 'ls /u/scratch/n/nicjia/order-burst-analysis/results/pit_rev/out/ 2>/dev/null | wc -l' 2>/dev/null | tr -d '[:space:]')
  echo "poll $i ($(date +%H:%M)): job=$alive out=$outf/36"
  [ "$alive" = "0" ] && [ "$outf" -ge 32 ] && { echo DONE; break; }
  sleep 180
done
$SSH 'cd /u/scratch/n/nicjia/order-burst-analysis/results/pit_rev && for f in out/*.csv; do tail -n +2 "$f"; done' 2>/dev/null | grep -v Pseudo > /Users/nick/order-burst-analysis/measurements/data/pit_rev_rows.csv
echo "downloaded $(wc -l < /Users/nick/order-burst-analysis/measurements/data/pit_rev_rows.csv) rows"
