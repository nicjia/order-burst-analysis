#!/usr/bin/env python3
"""Frozen specification shared by the strict 2023-to-2025 continuation test."""
import numpy as np

import two_avenue_evaluate as EV


CONTROL_FEATURES = EV.BASE_FEATURES + [
    "prior_total_count_60s", "prior_count_imbalance_60s",
    "prior_total_volume_60s", "prior_volume_imbalance_60s",
    "prior_total_count_300s", "prior_count_imbalance_300s",
    "prior_total_volume_300s", "prior_volume_imbalance_300s",
]
AUGMENTED_FEATURES = CONTROL_FEATURES + ["fragment_score"]

TARGETS = {
    "count_60s": ("future_count_imbalance_60s", "identity"),
    "count_300s": ("future_count_imbalance_300s", "identity"),
    "volume_60s": ("future_volume_imbalance_60s", "signed_log1p"),
    "volume_300s": ("future_volume_imbalance_300s", "signed_log1p"),
}


def target_values(frame, target):
    column, transform = TARGETS[target]
    values = frame[column].to_numpy(float)
    if transform == "identity":
        return values
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError("unknown target transform %s" % transform)


def validate_specification():
    """Fail loudly if the frozen strict controls cease to cover score inputs and lagged flow."""
    missing = sorted(set(EV.BASE_FEATURES) - set(CONTROL_FEATURES))
    required_lags = {
        "prior_total_count_60s", "prior_count_imbalance_60s",
        "prior_total_volume_60s", "prior_volume_imbalance_60s",
        "prior_total_count_300s", "prior_count_imbalance_300s",
        "prior_total_volume_300s", "prior_volume_imbalance_300s",
    }
    missing_lags = sorted(required_lags - set(CONTROL_FEATURES))
    if missing or missing_lags or AUGMENTED_FEATURES[-1] != "fragment_score":
        raise ValueError("invalid strict specification: base=%s lags=%s" %
                         (missing, missing_lags))


validate_specification()
