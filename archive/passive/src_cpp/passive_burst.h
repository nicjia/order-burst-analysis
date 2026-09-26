#ifndef PASSIVE_BURST_H
#define PASSIVE_BURST_H

#include "../../src_cpp/types.h"
#include <cmath>
#include <algorithm>
#include <vector>

// ─────────────────────────────────────────────────────────────
// PassiveBurst — A cluster of limit order submissions (Type 1)
//   at or near the BBO, detected via the Hawkes process.
// ─────────────────────────────────────────────────────────────

struct PassiveBurst {
    long id;              // Order ID of the first submission in the burst
    double start_time;
    double end_time;
    int direction;        // 1=Net Bid Additions (Bullish), -1=Net Ask Additions (Bearish), 0=Mixed
    int volume;           // Total submitted volume (Type 1 only)
    int submission_count; // Number of Type 1 events in the burst
    int bid_sub_count;    // Bid-side submission count
    int ask_sub_count;    // Ask-side submission count
    int bid_sub_volume;   // Bid-side submitted volume
    int ask_sub_volume;   // Ask-side submitted volume
    double bid_ratio;     // bid_sub_count / submission_count
    double ask_ratio;     // ask_sub_count / submission_count
    double minmax_vol_ratio; // min(bid_vol,ask_vol)/max(...)
    double start_price;   // Mid-price BEFORE the burst started
    double end_price;     // Mid-price AFTER the burst ended
    double peak_price;    // Most extreme mid-price during burst

    // ── Passive-Specific Features ──────────────────────────────
    double submission_size_variance; // Variance of individual submission sizes
    double round_lot_pct;            // Fraction of submissions that are 100-share multiples
    double hawkes_peak_intensity;    // Maximum Hawkes intensity during burst

    // ── Cancellation Features (NOT triggers, just features) ────
    int cancel_count;             // Total Types 2/3 events during burst window
    int cancel_volume;            // Total cancelled volume during burst window
    int bid_cancel_count;         // Bid-side cancellations
    int ask_cancel_count;         // Ask-side cancellations
    int bid_cancel_volume;        // Bid-side cancelled volume
    int ask_cancel_volume;        // Ask-side cancelled volume
    double cancel_ratio;          // cancel_count / (submission_count + cancel_count)
    double preburst_cancel_rate;  // Cancellation rate in pre-burst window
};

class PassiveBurstDetector {
public:
    PassiveBurstDetector(double silence_threshold, double min_volume_threshold,
                         double direction_threshold, double volume_ratio_threshold = 0.5,
                         double hawkes_beta = 1.0, double trigger_intensity = 0.5,
                         int max_bbo_levels = 3);

    // Process a LOBSTER message. Excites the Hawkes process ONLY on Type 1
    // submissions at L1-L3. Types 2/3 are tracked as features.
    // Returns true if a burst just finished.
    bool process(const LobsterMessage& msg, double current_mid,
                 int best_bid, int best_ask, PassiveBurst& result);

    void set_preburst_cancel_rate(double rate);
    bool flush(PassiveBurst& result);
    void reset();

private:
    bool should_terminate(double time_gap);
    void classify_direction();
    void compute_fingerprint();
    bool passes_filter();

    double silence_threshold_;
    double min_volume_threshold_;
    double direction_threshold_;
    double volume_ratio_threshold_;
    int max_bbo_levels_;  // How many levels from BBO to accept (e.g., 3)

    // Hawkes state
    double hawkes_beta_;
    double trigger_intensity_;
    bool use_hawkes_;
    double hawkes_intensity_;

    bool is_active_;
    PassiveBurst current_burst_;
    double last_msg_time_;
    double last_mid_price_;

    // Submission size tracking
    std::vector<int> submission_sizes_;
    int round_lot_count_;

    // Directional accumulators
    int bid_sub_count_;
    int ask_sub_count_;
    int bid_sub_volume_;
    int ask_sub_volume_;

    // Cancellation accumulators during burst
    int cancel_count_;
    int cancel_volume_;
    int bid_cancel_count_;
    int ask_cancel_count_;
    int bid_cancel_volume_;
    int ask_cancel_volume_;

    // Price extremes
    double max_price_;
    double min_price_;

    double pending_preburst_cancel_rate_;
};

#endif
