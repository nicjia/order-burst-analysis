#!/bin/bash
#$ -cwd
#$ -o results/p4_revisit_v1/log/
#$ -e results/p4_revisit_v1/log/
# P4 revisit v1: real-data acceptance of the fast paths. For each "date ticker" in VERIFY_LIST, run
# p4_extract.py with fast packets + C++ book, with canonical packets + C++ book, and (when the third
# field is "py") with fast packets + the Python book; every saved array must be identical.
set -uo pipefail
cd /u/scratch/n/nicjia/order-burst-analysis
. /u/local/Modules/default/init/bash
module load gcc/11.3.0 python/3.9.6
export OMP_NUM_THREADS=1
CODE=results/p4_revisit_v1/code; W=results/p4_revisit_v1/verify; mkdir -p "$W"
CM=$(mktemp -d /tmp/p4v.XXXXXX); trap 'ssh -o ControlPath="$CM/s" -O exit nicjia@lobster2.math.ucla.edu >/dev/null 2>&1; rm -rf "$CM"' EXIT
lob(){ ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto -o ControlPath="$CM/s" -o ControlPersist=600 nicjia@lobster2.math.ucla.edu "$@"; }
while read -r dd tk py; do
  d=$(mktemp -d "$W/raw.XXXXXX")
  lob "cat /lobster/${dd:0:4}/$dd/$tk.7z" > "$d/a.7z" < /dev/null && ~/bin/7z x "$d/a.7z" -o"$d/x" -y > /dev/null
  msg=$(find "$d/x" -name '*message*.csv' | head -1)
  echo "== $tk $dd $(wc -l < "$msg") rows; first/last time: $(head -1 "$msg" | cut -d, -f1) $(tail -1 "$msg" | cut -d, -f1)"
  python3 $CODE/p4_extract.py --msg "$msg" --ticker "$tk" --out "$W/${tk}_$dd.fast.npz" --helper $CODE/p4_bbo --model $CODE/program_model_run60.json | python3 -c "import json,sys; s=json.load(sys.stdin); print('fast', s['n_T'], s['n_S'], s['seconds'])"
  python3 $CODE/p4_extract.py --msg "$msg" --ticker "$tk" --out "$W/${tk}_$dd.canon.npz" --helper $CODE/p4_bbo --model $CODE/program_model_run60.json --canonical-packets | python3 -c "import json,sys; s=json.load(sys.stdin); print('canonical', s['n_T'], s['n_S'], s['seconds'])"
  pairs="$W/${tk}_$dd.fast.npz $W/${tk}_$dd.canon.npz"
  if [ "${py:-}" = py ]; then
    python3 $CODE/p4_extract.py --msg "$msg" --ticker "$tk" --out "$W/${tk}_$dd.py.npz" --model $CODE/program_model_run60.json | python3 -c "import json,sys; s=json.load(sys.stdin); print('python-book', s['engine'], s['seconds'])"
    pairs="$pairs $W/${tk}_$dd.py.npz"
  fi
  python3 - $pairs <<'PY'
import sys, numpy as np
ref = np.load(sys.argv[1])
for other in sys.argv[2:]:
    o = np.load(other)
    keys = sorted(set(ref.files) | set(o.files))
    bad = [k for k in keys if k not in ref.files or k not in o.files or
           not (np.array_equal(ref[k], o[k], equal_nan=True) if ref[k].dtype.kind == "f" else np.array_equal(ref[k], o[k]))]
    print("compare", other.split("/")[-1], "arrays", len(keys), "mismatches", bad[:10])
PY
  rm -rf "$d"
done < "$VERIFY_LIST"
