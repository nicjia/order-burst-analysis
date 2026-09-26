"""Known-truth targets for any-program participation; never infer truth from prices.

parent_id >= 0 denotes a known split execution program, -1 known non-program flow.
Unknown identities must not be encoded as -1. This is not an ITCH order-ID classifier.
Fractions describe observed executions only, not unobserved venues or remaining inventory.
"""
import numpy as np
import pandas as pd


def participation_labels(fragments, packets, *, truth_source):
    if truth_source not in ("simulation", "identified_parent_records"):
        raise ValueError("participation truth requires simulation or identified parent records")
    if not {"parent_id", "volume"}.issubset(packets.columns):
        raise ValueError("missing known parent truth or execution volume")
    ids = packets.parent_id.to_numpy(float)
    volume = packets.volume.to_numpy(float)
    if (not np.isfinite(ids).all() or np.any(ids < -1)
            or not np.equal(ids, np.floor(ids)).all()):
        raise ValueError("unknown/invalid parent identity; never label unknown as noise")
    if not np.isfinite(volume).all() or np.any(volume <= 0):
        raise ValueError("execution volumes must be finite and positive")
    labels = []
    for r in fragments.itertuples(index=False):
        a, b = float(r.packet_first), float(r.packet_last)
        if not (a.is_integer() and b.is_integer() and 0 <= a <= b < len(packets)):
            raise ValueError("invalid inclusive fragment packet bounds")
        a, b = int(a), int(b) + 1
        parent = ids[a:b]; weight = volume[a:b]; known = parent >= 0
        count_fraction = float(known.mean())
        volume_fraction = float(weight[known].sum() / weight.sum())
        values, counts = np.unique(parent[known], return_counts=True)
        dominant_count = float(counts.max() / len(parent)) if len(counts) else 0.0
        labels.append(dict(
            parent_packet_fraction=count_fraction,
            parent_volume_fraction=volume_fraction,
            distinct_parents=int(len(values)),
            dominant_parent_packet_fraction=dominant_count,
            participation_class=("program_heavy" if volume_fraction >= .8 else
                                 "noise_heavy" if volume_fraction <= .2 else "mixed"),
            truth_source=truth_source,
        ))
    columns = ["parent_packet_fraction", "parent_volume_fraction", "distinct_parents",
               "dominant_parent_packet_fraction", "participation_class", "truth_source"]
    return pd.DataFrame(labels, columns=columns, index=fragments.index)
