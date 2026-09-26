#!/bin/bash
# Concatenate per-name daily flows into one CSV per group (header once, empty files skipped).
set -euo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis/results/program_evidence_v1
for g in contig_explore_2024 contig_confirm_2021 events; do
  out=flows_$g.csv; first=1; : > "$out.part"
  while read -r p; do
    f=$g/flows/$p.csv
    [ -s "$f" ] || continue
    [ "$(wc -l < "$f")" -gt 1 ] || continue
    if [ $first = 1 ]; then cat "$f" >> "$out.part"; first=0; else tail -n +2 "$f" >> "$out.part"; fi
  done < $g/universe.txt
  mv "$out.part" "$out"; echo "$g $(($(wc -l < $out) - 1)) rows from $(ls $g/flows/*.csv | wc -l) files"
done
