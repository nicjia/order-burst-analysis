#!/usr/bin/env python3
"""Ground-truth simulator for anonymous metaorder-fragment reconstruction.

The simulator deliberately includes mechanisms that are observationally confounded in real
LOBSTER data: genuine parent splitting, common-signal herding by unrelated traders,
overlapping parents, liquidity-sensitive pauses, and partial venue observation.  It is an
evaluation laboratory, not a structural estimate of NASDAQ.
"""
import argparse
import json

import numpy as np
import pandas as pd

import fragment_reconstruction as FR
import metaorder_models as MM

RTH0, RTH1 = 34200.0, 57600.0


def _market_state(times, rng):
    times = np.asarray(times, float)
    phase = (times - RTH0) / (RTH1 - RTH0)
    u_shape = 1.0 + 1.5 * np.abs(phase - 0.5)
    spread = 0.01 * np.maximum(1.0, np.round(u_shape + rng.lognormal(-1.0, 0.5, len(times))))
    depth = rng.lognormal(7.2, 0.55, len(times)) / u_shape
    imbalance = np.clip(rng.normal(0.0, 0.35, len(times)), -0.95, 0.95)
    return spread, depth, imbalance


def simulate_day(seed=1, scenario="full", venue_probability=0.65,
                 background_rate=0.20, n_parents=45):
    """Return a packet tape with exact ``parent_id`` ground truth.

    ``scenario`` may be ``splitting``, ``herding``, ``overlap``, ``pauses``, ``partial``,
    or ``full``.  Full combines every mechanism.
    """
    rng = np.random.default_rng(int(seed))
    horizon = RTH1 - RTH0
    use_splitting = scenario in ("splitting", "overlap", "pauses", "partial", "full")
    use_herding = scenario in ("herding", "full")
    use_pauses = scenario in ("pauses", "full")
    use_partial = scenario in ("partial", "full")

    # Unrelated background traders.  In the herding treatment, their signs respond to a
    # persistent public/common signal but parent IDs remain distinct (-1), creating the key
    # splitting-vs-herding observational confound.
    n_background = int(rng.poisson(background_rate * horizon))
    bg_time = np.sort(rng.uniform(RTH0, RTH1, n_background))
    if use_herding:
        bins = np.arange(RTH0, RTH1 + 60.0, 60.0)
        latent = np.zeros(len(bins))
        for i in range(1, len(latent)):
            latent[i] = 0.94 * latent[i - 1] + rng.normal(0.0, 0.45)
        signal = latent[np.clip(np.searchsorted(bins, bg_time) - 1, 0, len(latent) - 1)]
        prob_buy = 1.0 / (1.0 + np.exp(-signal))
        bg_sign = np.where(rng.random(n_background) < prob_buy, 1, -1)
    else:
        bg_sign = rng.choice([-1, 1], n_background)
    events = [
        (float(t), int(s), float(rng.lognormal(4.5, 0.8)), -1)
        for t, s in zip(bg_time, bg_sign)
    ]

    if use_splitting:
        # Starts are unrestricted, so overlap occurs naturally.  The `overlap` scenario uses
        # more parents in the same day to stress identifiability.
        parent_count = int(n_parents * (1.8 if scenario == "overlap" else 1.0))
        for parent_id in range(parent_count):
            sign = int(rng.choice([-1, 1]))
            child_count = int(np.clip(rng.lognormal(3.1, 0.65), 6, 180))
            start = float(rng.uniform(RTH0, RTH1 - 1800.0))
            base_gap = float(rng.lognormal(-0.2, 0.7))
            times = [start]
            for _ in range(1, child_count):
                gap = float(rng.exponential(base_gap))
                if use_pauses and rng.random() < 0.07:
                    # A parent waits when liquidity is unfavorable; long silence therefore
                    # need not be a true parent boundary.
                    gap += float(rng.lognormal(4.2, 0.8))
                times.append(times[-1] + gap)
            target_scale = float(rng.lognormal(5.0, 0.55))
            for t in times:
                if t >= RTH1:
                    break
                child = float(max(1.0, rng.lognormal(np.log(target_scale), 0.35)))
                events.append((float(t), sign, child, parent_id))

    events.sort(key=lambda x: x[0])
    raw = pd.DataFrame(events, columns=["time", "sign", "volume", "parent_id"])
    if use_partial:
        raw = raw[rng.random(len(raw)) < float(venue_probability)].copy()
    raw = raw.reset_index(drop=True)

    spread, depth, imbalance = _market_state(raw["time"].to_numpy(float), rng)
    hidden_share = rng.beta(1.2, 7.0, len(raw))
    raw["packet_id"] = np.arange(len(raw), dtype=np.int64)
    raw["sign_source"] = "simulation"
    raw["vwap"] = 100.0 + np.cumsum(raw["sign"].to_numpy(float) *
                                    np.sqrt(raw["volume"].to_numpy(float)) * 1e-5)
    raw["min_price"] = raw["vwap"]
    raw["max_price"] = raw["vwap"]
    raw["n_messages"] = rng.integers(1, 5, len(raw))
    raw["n_hidden"] = np.minimum(raw["n_messages"],
                                 rng.binomial(raw["n_messages"], hidden_share))
    raw["n_visible"] = raw["n_messages"] - raw["n_hidden"]
    raw["hidden_volume"] = raw["volume"] * hidden_share
    raw["visible_volume"] = raw["volume"] - raw["hidden_volume"]
    raw["pre_bid"] = raw["vwap"] - spread / 2.0
    raw["pre_ask"] = raw["vwap"] + spread / 2.0
    raw["pre_bid_size"] = depth * (1.0 + imbalance) / 2.0
    raw["pre_ask_size"] = depth * (1.0 - imbalance) / 2.0
    raw["spread"] = spread
    raw["depth"] = depth
    raw["row_first"] = raw["packet_id"]
    raw["row_last"] = raw["packet_id"]
    return raw


def _campaign_pair_accuracy(stitched):
    pairs = MM.join_feature_frame(stitched)
    if pairs.empty:
        return np.nan
    left = stitched.loc[pairs["left_index"]]
    right = stitched.loc[pairs["right_index"]]
    truth = ((left["dominant_parent"].to_numpy(int) >= 0) &
             (left["dominant_parent"].to_numpy(int) ==
              right["dominant_parent"].to_numpy(int)))
    predicted = (left["campaign_id"].to_numpy(int) ==
                 right["campaign_id"].to_numpy(int))
    return float((truth == predicted).mean())


def calibrate(train_days=80, test_days=30, gap=1.0, min_packets=3, seed=7319):
    train = []; test = []; metrics = []
    scenarios = ["splitting", "herding", "overlap", "pauses", "partial", "full"]
    for day in range(train_days + test_days):
        scenario = scenarios[day % len(scenarios)]
        packets = simulate_day(seed + day, scenario=scenario)
        fragments = FR.form_fragments(packets, gap=gap, min_packets=min_packets)
        labeled = MM.add_simulated_fragment_labels(fragments, packets)
        labeled["simulation_day"] = day
        labeled["scenario"] = scenario
        (train if day < train_days else test).append(labeled)
        m = MM.recovery_metrics(labeled, packets)
        m.update({"simulation_day": day, "scenario": scenario})
        metrics.append(m)
    train_df = pd.concat(train, ignore_index=True)
    test_df = pd.concat(test, ignore_index=True)
    fragment_model = MM.fit_fragment_model(train_df)
    train_scored = MM.score_fragments(train_df, fragment_model)
    test_scored = MM.score_fragments(test_df, fragment_model)
    join_model = MM.fit_join_model(train_scored)
    stitched = MM.stitch_campaigns(test_scored, join_model)
    y = test_scored["true_fragment"].to_numpy(float)
    p = test_scored["fragment_score"].to_numpy(float)
    result = {
        "train_fragments": int(len(train_df)),
        "test_fragments": int(len(test_df)),
        "test_brier": float(np.mean((p - y) ** 2)),
        "test_score_true": float(p[y == 1].mean()) if (y == 1).any() else np.nan,
        "test_score_false": float(p[y == 0].mean()) if (y == 0).any() else np.nan,
        "campaign_pair_accuracy": _campaign_pair_accuracy(stitched),
        "scenario_metrics": pd.DataFrame(metrics).groupby("scenario").mean(numeric_only=True)
                            .to_dict(orient="index"),
        "fragment_model": {
            "mean": fragment_model.mean_.tolist(), "scale": fragment_model.scale_.tolist(),
            "coef": fragment_model.coef_.tolist(), "features": MM.FRAGMENT_FEATURES,
        },
        "join_model": {
            "mean": join_model.mean_.tolist(), "scale": join_model.scale_.tolist(),
            "coef": join_model.coef_.tolist(), "features": MM.JOIN_FEATURES,
        },
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=80)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7319)
    ap.add_argument("--out")
    args = ap.parse_args()
    result = calibrate(args.train_days, args.test_days, seed=args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

