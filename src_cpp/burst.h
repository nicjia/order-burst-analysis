#ifndef BURST_H
#define BURST_H

#include "types.h"
#include <cmath>
#include <algorithm>
#include <vector>

struct Burst {
    long id;            // The ID of the FIRST order in the burst
    double start_time;
    double end_time;
    int direction;      // 1=Buy, -1=Sell, 0=Mixed (didn't meet direction threshold)
    int volume;         // Total shares traded/added
    int trade_count;    // Number of orders in the burst
    int buy_count;      // Buy-initiated trade count in burst
    int sell_count;     // Sell-initiated trade count in burst
    int buy_volume;     // Buy-initiated volume in burst
    int sell_volume;    // Sell-initiated volume in burst
    double buy_ratio;   // buy_count / trade_count
    double sell_ratio;  // sell_count / trade_count
    double minmax_vol_ratio; // min(buy_volume,sell_volume)/max(...)
    double start_price; // Price BEFORE the burst started
    double end_price;   // Price AFTER the burst ended
    double peak_price;  // The most extreme price reached during the burst

    // ── Path 1: VWAP/TWAP Fingerprinting ──────────────────────
    double trade_size_variance;   // Variance of individual trade sizes within burst
    double round_lot_pct;         // Fraction of trades that are multiples of 100 shares

    // ── Path 2: Hawkes Process ────────────────────────────────
    double hawkes_peak_intensity; // Maximum intensity score reached during burst

    // ── Path 3: Pre-Burst Quote Depletion ─────────────────────
    double preburst_cancel_rate;  // Cancellation rate on opposing side in pre-burst window
};

class BurstDetector {
public:
    // silence_threshold: time gap (seconds) that ends a burst (legacy mode)
    // min_volume_threshold: minimum total volume for a burst to be output
    // direction_threshold: ratio (e.g., 0.7) of buy/total or sell/total to classify direction
    // volume_ratio_threshold: max minority_vol / majority_vol ratio (e.g., 0.5) for directional bursts
    // hawkes_beta: exponential decay rate for Hawkes intensity (0 = disable Hawkes, use silence)
    // trigger_intensity: burst stays active while intensity >= this threshold
    BurstDetector(double silence_threshold, double min_volume_threshold, double direction_threshold,
                  double volume_ratio_threshold = 0.5,
                  double hawkes_beta = 0.0, double trigger_intensity = 0.5);
    
    // Returns true if a burst just finished (and passed filters)
    // If true, 'result' will contain that finished burst data
    bool process(const LobsterMessage& msg, double current_mid, Burst& result);

    // Set the pre-burst cancellation rate for the NEXT burst that starts.
    // Called from main.cpp which has access to order book cancel history.
    void set_preburst_cancel_rate(double rate);

    // Finalize any active burst (call at end of each trading day).
    // Returns true if an active burst was emitted into 'result'.
    bool flush(Burst& result);

    // Reset all state for a new trading day.
    void reset();

private:
    // ── CHANGE THESE TO CHANGE BEHAVIOR ──────────────────────
    
    // Should the current burst end? Hawkes or silence-based.
    bool should_terminate(double time_gap);
    
    // Set direction & peak_price on current_burst_. Currently: buy/sell ratio.
    void classify_direction();

    // Compute Path 1 fingerprint metrics on current_burst_.
    void compute_fingerprint();
    
    // Is the finished burst worth keeping? Currently: minimum volume.
    bool passes_filter();
    
    // ─────────────────────────────────────────────────────────
    
    double silence_threshold_;
    double min_volume_threshold_;
    double direction_threshold_;
    double volume_ratio_threshold_;

    // ── Path 2: Hawkes Process state ─────────────────────────
    double hawkes_beta_;           // Exponential decay rate (0 = legacy silence mode)
    double trigger_intensity_;     // Burst stays active above this threshold
    bool   use_hawkes_;            // true = Hawkes mode, false = legacy silence mode
    double hawkes_intensity_;      // Current rolling intensity score
    
    bool is_active_;
    Burst current_burst_;
    double last_msg_time_;
    double last_mid_price_; 

    // ── Path 1: Trade size tracking within burst ─────────────
    std::vector<int> trade_sizes_; // Individual trade sizes for variance calc
    int round_lot_count_;          // Count of 100-share multiples
    
    // Track buy/sell counts and volumes to determine direction at burst end
    int buy_count_;
    int sell_count_;
    int buy_volume_;
    int sell_volume_;
    
    // Track both max and min since direction is unknown until end
    double max_price_;
    double min_price_;

    // ── Path 3: Pre-burst cancel rate (set externally) ───────
    double pending_preburst_cancel_rate_;  // Buffered value for the next burst start
};

#endif