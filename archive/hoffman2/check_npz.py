#!/usr/bin/env python3
"""Integrity check: every packet npz of the given name directories loads and has equal-length arrays."""
import sys
from pathlib import Path

import numpy as np

bad = []
for d in sys.argv[1:]:
    for p in sorted(Path(d).glob("*.npz")):
        if p.name.endswith(".part.npz"):
            continue
        try:
            with np.load(p) as z:
                lens = {len(z[k]) for k in z.files}
                if len(lens) > 1:
                    bad.append((str(p), "length mismatch"))
        except Exception as e:  # noqa: BLE001
            bad.append((str(p), type(e).__name__))
print("checked", sum(1 for d in sys.argv[1:] for _ in Path(d).glob("*.npz")), "bad", len(bad))
for b in bad:
    print(*b)
