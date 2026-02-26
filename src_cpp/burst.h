#ifndef BURST_H
#define BURST_H

#include "types.h"
#include <cmath>
#include <algorithm>

struct Burst {
    long id;            // The ID of the FIRST order in the burst
    double start_time;
    double end_time;
    int direction;      // 1=Buy, -1=Sell, 0=Mixed (didn't meet direction threshold)
    int volume;         // Total shares traded/added
    int trade_count;    // Number of orders in the burst
    double start_price; // Price BEFORE the burst started
    double end_price;   // Price AFTER the burst ended
    double peak_price;  // The most extreme price reached during the burst
};

class BurstDetector {
public:
    // silence_threshold: time gap (seconds) that ends a burst
    // min_volume: minimum total volume for a burst to be output
    // direction_threshold: ratio (e.g., 0.7) of buy/total or sell/total to classify direction
    BurstDetector(double silence_threshold, int min_volume, double direction_threshold);
    
    // Returns true if a burst just finished (and passed filters)
    // If true, 'result' will contain that finished burst data
    bool process(const LobsterMessage& msg, double current_mid, Burst& result);

    // Finalize any active burst (call at end of each trading day).
    // Returns true if an active burst was emitted into 'result'.
    bool flush(Burst& result);

    // Reset all state for a new trading day.
    void reset();

private:
    // ── CHANGE THESE TO CHANGE BEHAVIOR ──────────────────────
    
    // Should the current burst end? Currently: silence-based.
    bool should_terminate(double time_gap);
    
    // Set direction & peak_price on current_burst_. Currently: buy/sell ratio.
    void classify_direction();
    
    // Is the finished burst worth keeping? Currently: minimum volume.
    bool passes_filter();
    
    // ─────────────────────────────────────────────────────────
    
    double silence_threshold_;
    int min_volume_;
    double direction_threshold_;
    
    bool is_active_;
    Burst current_burst_;
    double last_msg_time_;
    double last_mid_price_; 
    
    // Track buy/sell counts to determine direction at burst end
    int buy_count_;
    int sell_count_;
    
    // Track both max and min since direction is unknown until end
    double max_price_;
    double min_price_;
};

#endif