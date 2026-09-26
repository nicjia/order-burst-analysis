#!/bin/bash
S="ssh -o RemoteCommand=none -o BatchMode=yes hoff"
L=/Users/nick/order-burst-analysis/measurements/out/monitor_two.log
: > $L
for i in $(seq 1 200); do
  r=$($S 'cd /u/scratch/n/nicjia/order-burst-analysis && echo -n "$(qstat -u nicjia 2>/dev/null|grep -cE "hiddepl|hidhb2") $(ls results/hid_depl/out/*.csv 2>/dev/null|wc -l) $(ls results/hid_hb2/out/*.csv 2>/dev/null|wc -l)"' 2>/dev/null|grep -v Pseudo)
  job=$(echo $r|cut -d' ' -f1); a=$(echo $r|cut -d' ' -f2); b=$(echo $r|cut -d' ' -f3)
  echo "poll $i ($(date +%H:%M)): job=$job depl=$a hb2=$b" >> $L
  if [ "$job" = "0" ] && [ "$a" -gt 400 ] && [ "$b" -gt 400 ]; then echo "DONE depl=$a hb2=$b" >> $L; break; fi
  sleep 300
done
