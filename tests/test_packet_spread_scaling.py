"""Formation helpers for the packet-level spread-scaling recomputation.

The run and cluster builders are the only sign-sensitive code in the extractor, and the
distinction between them is the point of the test: ``_runs`` chooses its boundaries with
the sign (the original, circular construction) while ``_clusters`` chooses boundaries on
timing alone and assigns the sign afterwards.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src_py"))
import packet_spread_scaling as PS


def test_runs_require_three_same_sign_packets():
    time = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    sign = np.array([1, 1, 1, -1, -1])
    ends, signs = PS._runs(time, sign)
    assert ends.tolist() == [0.2]
    assert signs.tolist() == [1]


def test_runs_break_on_sign_flip_and_on_gap():
    # A flip ends a run even with no time gap; a gap ends it even with no flip.
    time = np.array([0.0, 0.1, 0.2, 0.3])
    assert len(PS._runs(time, np.array([1, 1, -1, -1]))[0]) == 0
    time_gapped = np.array([0.0, 0.1, 9.0, 9.1])
    assert len(PS._runs(time_gapped, np.array([1, 1, 1, 1]))[0]) == 0


def test_clusters_are_timing_only_and_signed_by_net_volume():
    # Mixed signs stay in one cluster; the minority side does not split it.
    time = np.array([0.0, 0.1, 0.2])
    sign = np.array([1, -1, 1])
    volume = np.array([100.0, 10.0, 100.0])
    ends, signs = PS._clusters(time, sign, volume)
    assert ends.tolist() == [0.2]
    assert signs.tolist() == [1]


def test_clusters_drop_exactly_balanced_flow():
    time = np.array([0.0, 0.1, 0.2, 0.3])
    sign = np.array([1, -1, 1, -1])
    volume = np.array([50.0, 50.0, 50.0, 50.0])
    assert len(PS._clusters(time, sign, volume)[0]) == 0


def test_clusters_admit_what_runs_reject():
    """The construction difference the recomputation exists to measure."""
    time = np.array([0.0, 0.1, 0.2, 0.3])
    sign = np.array([1, 1, -1, 1])
    volume = np.ones(4)
    assert len(PS._runs(time, sign)[0]) == 0
    assert len(PS._clusters(time, sign, volume)[0]) == 1


def test_trimmed_mean_drops_nonfinite_and_absurd_markouts():
    assert PS._trimmed_mean([1.0, 3.0, np.nan, np.inf, 5000.0]) == 2.0
    assert np.isnan(PS._trimmed_mean([np.nan, 2000.0]))
