"""Audit the target change on legacy synthetic tapes; no fitting or market claims."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import fragment_reconstruction as FR
import metaorder_simulation as MS
from metaorder_participation import participation_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days-per-scenario", type=int, default=5)
    args = ap.parse_args()
    if args.days_per_scenario < 1:
        raise ValueError("positive number of days required")
    scenarios = ["splitting", "herding", "overlap", "pauses", "partial", "full"]
    records = []
    for j, scenario in enumerate(scenarios):
        for day in range(args.days_per_scenario):
            seed = 713000 + j * 10000 + day
            p = MS.simulate_day(seed, scenario)
            f = FR.form_fragments(p)
            y = participation_labels(f, p, truth_source="simulation")
            # Independent direct sums from each raw slice, including noise in denominator.
            for pos, fragment in enumerate(f.itertuples(index=False)):
                raw = p.iloc[int(fragment.packet_first):int(fragment.packet_last) + 1]
                n = len(raw); parent_count = 0; parent_volume = 0.; total_volume = 0.
                counts = {}
                for packet in raw.itertuples(index=False):
                    total_volume += packet.volume
                    if packet.parent_id >= 0:
                        parent_count += 1; parent_volume += packet.volume
                        counts[packet.parent_id] = counts.get(packet.parent_id, 0) + 1
                actual = y.iloc[pos]
                assert np.isclose(actual.parent_packet_fraction, parent_count / n, atol=1e-12)
                assert np.isclose(actual.parent_volume_fraction, parent_volume / total_volume, atol=1e-12)
                assert np.isclose(actual.dominant_parent_packet_fraction,
                                  max(counts.values(), default=0) / n, atol=1e-12)
                assert actual.distinct_parents == len(counts)
            old = y.dominant_parent_packet_fraction >= .8
            any_count = y.parent_packet_fraction >= .8
            any_volume = y.parent_volume_fraction >= .8
            records.append(dict(scenario=scenario, seed=seed, fragments=len(f),
                                old_single_parent_positive=int(old.sum()),
                                any_parent_count_positive=int(any_count.sum()),
                                any_parent_volume_positive=int(any_volume.sum()),
                                count_positive_missed_by_old=int((any_count & ~old).sum()),
                                volume_positive_missed_by_old=int((any_volume & ~old).sum()),
                                mixed=int((y.participation_class == "mixed").sum()),
                                noise_heavy=int((y.participation_class == "noise_heavy").sum())))
    totals = {key: sum(r[key] for r in records) for key in records[0]
              if key not in ("scenario", "seed")}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in
              ["diagnose_participation_target.py", "metaorder_participation.py",
               "metaorder_simulation.py", "fragment_reconstruction.py"]}
    result = dict(status="pass", scope="Synthetic target audit, not classifier accuracy. "
                  "Legacy simulator has independent book snapshots and random pauses; "
                  "it does not validate book-adaptive recurrence.",
                  days_per_scenario=args.days_per_scenario, records=records, totals=totals,
                  source_sha256=hashes, fits=0)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], totals=totals, fits=0), indent=2))


if __name__ == "__main__":
    main()
