#!/bin/bash
S="ssh -o RemoteCommand=none -o BatchMode=yes hoff"
L=/Users/nick/order-burst-analysis/measurements/out/monitor_mech.log
: > $L
for i in $(seq 1 200); do
  r=$($S 'cd /u/scratch/n/nicjia/order-burst-analysis && echo -n "$(qstat -u nicjia 2>/dev/null|grep -c hidmech) $(ls results/hid_mech/out/*.csv 2>/dev/null|wc -l)"' 2>/dev/null|grep -v Pseudo)
  job=$(echo $r|cut -d' ' -f1); out=$(echo $r|cut -d' ' -f2)
  echo "poll $i ($(date +%H:%M)): job=$job out=$out" >> $L
  if [ "$job" = "0" ] && [ "$out" -gt 400 ]; then echo "DONE out=$out" >> $L; break; fi
  sleep 300
done
