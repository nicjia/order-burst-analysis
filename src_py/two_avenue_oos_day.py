#!/usr/bin/env python3
"""Apply frozen 2023 models to one untouched 2024 ticker-day and emit one summary row."""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import metaorder_models as MM
import two_avenue_evaluate as EV
from two_avenue_extract import extract
import burst_alt as BA


def _mean(frame, column):
    values = frame[column].to_numpy(float)
    values = values[np.isfinite(values) & (np.abs(values) <= 1000.0)]
    return float(values.mean()) if len(values) else np.nan


def _nonoverlap_mean(frame, selected, column, holding_seconds):
    use = frame.loc[selected].sort_values("end_time")
    values = []; available = -np.inf
    for row in use.itertuples(index=False):
        if float(row.end_time) < available:
            continue
        value = float(getattr(row, column))
        if np.isfinite(value) and abs(value) <= 1000.0:
            values.append(value)
            available = float(row.end_time) + holding_seconds
    return (float(np.mean(values)) if values else np.nan, int(len(values)))


def _campaign_reversal(frame, context, wait_seconds=1800.0):
    """Fade reconstructed multi-fragment campaigns after the full wait clock.

    Singleton fragments are not reconstructed campaigns.  Including them made the original
    OOS diagnostic a fade-every-fragment baseline when the simulation join model transported
    poorly.  Decisions must also be chronological before enforcing non-overlap.
    """
    bt, _bm, bb, ba, _bbsz, _basz, _ofi, _trades = context
    campaigns = frame.sort_values("end_time").groupby("campaign_id", as_index=False).agg(
        end_time=("end_time", "max"), sign=("sign", "last"),
        n_fragments=("fragment_id", "size")
    )
    campaigns = campaigns[campaigns.n_fragments >= 2].sort_values(
        "end_time", kind="stable"
    ).reset_index(drop=True)
    if campaigns.empty:
        return {**{"reversal_pnl_%dm" % m: np.nan for m in (5, 15, 30)},
                **{"reversal_trades_%dm" % m: 0 for m in (5, 15, 30)}}
    decision = campaigns.end_time.to_numpy(float) + wait_seconds
    trade_sign = -campaigns.sign.to_numpy(float)
    entry_bid, entry_ask = BA.bbo_at(bt, bb, ba, decision)
    entry = np.where(trade_sign > 0, entry_ask, entry_bid)
    result = {}
    for minutes in (5, 15, 30):
        exit_bid, exit_ask = BA.bbo_at(bt, bb, ba, decision + minutes * 60.0)
        exit_price = np.where(trade_sign > 0, exit_bid, exit_ask)
        midpoint = (entry_bid + entry_ask) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            pnl = trade_sign * (exit_price - entry) / midpoint * 1e4
        available = -np.inf; kept = []
        for when, value in zip(decision, pnl):
            if when < available or not np.isfinite(value) or abs(value) > 1000.0:
                continue
            kept.append(float(value)); available = when + minutes * 60.0
        result["reversal_pnl_%dm" % minutes] = float(np.mean(kept)) if kept else np.nan
        result["reversal_trades_%dm" % minutes] = int(len(kept))
    return result


def summarize(frame, context, ticker, date, frozen):
    result = {
        "ticker": ticker, "date": int(date), "name_holdout": int(EV.stable_holdout(ticker)),
        "n_fragments": int(len(frame)), "n_campaigns": int(frame.campaign_id.nunique()),
        "campaign_join_rate": float(1.0 - frame.campaign_id.nunique() / max(len(frame), 1)),
        "ambiguous_packet_share": float(frame.ambiguous_packet_share_day.iloc[0]),
        "ambiguous_hidden_volume_share":
            float(frame.ambiguous_hidden_volume_share_day.iloc[0]),
    }
    for label, features in [("price_free", EV.BASE_FEATURES),
                            ("post_end", EV.POST_END_FEATURES)]:
        spec = frozen["models"][label]
        X = EV._matrix(frame, features)
        ridge = EV.Ridge.from_dict(spec["ridge"])
        classifier = MM.RidgeLogit.from_dict(spec["classifier"])
        regression_score = ridge.predict(X)
        probability = classifier.predict_proba(X)
        selections = {
            "%s_reg" % label: regression_score >= float(spec["ridge_top_decile"]),
            "%s_cls" % label: probability >= float(spec["classifier_top_decile"]),
        }
        for name, selected in selections.items():
            chosen = frame.loc[selected]
            result[name + "_n"] = int(selected.sum())
            result[name + "_permanent"] = _mean(chosen, "permanent_proxy")
            result[name + "_persistent_rate"] = _mean(chosen, "persistent_positive")
            for minutes in (5, 15, 30):
                column = "executable_%dm" % minutes
                pnl, trades = _nonoverlap_mean(frame, selected, column, minutes * 60.0)
                result[name + "_pnl_%dm" % minutes] = pnl
                result[name + "_trades_%dm" % minutes] = trades

    high = frame.fragment_score.to_numpy(float) >= float(frozen["fragment_score_top_decile"])
    result["sim_high_n"] = int(high.sum())
    flow_specs = frozen.get("flow_baselines", {"300s": frozen["flow_baseline"]})
    for horizon, spec in sorted(flow_specs.items()):
        flow_model = EV.Ridge.from_dict(spec)
        baseline = flow_model.predict(EV._matrix(frame, spec.get("features", EV.FLOW_BASELINE)))
        target = "future_count_imbalance_%s" % horizon
        residual = frame[target].to_numpy(float) - baseline
        result["sim_high_flow_residual_%s" % horizon] = (
            float(np.nanmean(residual[high])) if high.any() else np.nan
        )
        result["sim_high_future_flow_%s" % horizon] = _mean(frame.loc[high], target)
        if horizon == "300s":
            result["sim_high_flow_residual"] = result["sim_high_flow_residual_300s"]
            result["sim_high_future_flow"] = result["sim_high_future_flow_300s"]
    result.update(_campaign_reversal(frame, context))
    return pd.DataFrame([result])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    ap.add_argument("--simulation-model", required=True); ap.add_argument("--frozen", required=True)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()
    with open(args.frozen) as handle:
        frozen = json.load(handle)
    frame, context, _packets = extract(
        args.msg, args.ticker, args.simulation_model, return_state=True
    )
    if frame.empty:
        return
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(args.msg))
    date = int("".join(match.groups())) if match else 0
    summarize(frame, context, args.ticker, date, frozen).to_csv(
        sys.stdout, index=False, header=args.header
    )


if __name__ == "__main__":
    main()
