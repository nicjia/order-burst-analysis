#include "burst.h"
#include <cmath> // Required for std::abs

BurstDetector::BurstDetector(double silence_threshold, int min_volume, double direction_threshold) 
    : silence_threshold_(silence_threshold), 
      min_volume_(min_volume),
      direction_threshold_(direction_threshold),
      is_active_(false), 
      last_msg_time_(0),
      last_mid_price_(0),
      buy_count_(0),
      sell_count_(0),
      max_price_(0),
      min_price_(0) {}


// ── TERMINATION: when does a burst end? ─────────────────────
bool BurstDetector::should_terminate(double time_gap) {
    return time_gap > silence_threshold_;
}

// ── CLASSIFICATION: Sets direction AND peak_price ────────────
void BurstDetector::classify_direction() {
    int total = buy_count_ + sell_count_;
    if (total == 0) return;
    
    double buy_ratio = (double)buy_count_ / total;
    double sell_ratio = (double)sell_count_ / total;
    
    if (buy_ratio >= direction_threshold_) {
        current_burst_.direction = 1;   // Buy burst
        current_burst_.peak_price = max_price_;
    } else if (sell_ratio >= direction_threshold_) {
        current_burst_.direction = -1;  // Sell burst
        current_burst_.peak_price = min_price_;
    } else {
        current_burst_.direction = 0;   // Mixed
        // Use std::abs for safer distance calculation
        double up_move = std::abs(max_price_ - current_burst_.start_price);
        double down_move = std::abs(min_price_ - current_burst_.start_price);
        current_burst_.peak_price = (up_move >= down_move) ? max_price_ : min_price_;
    }
}

// ── FILTER: is this burst worth keeping? ────────────────────
bool BurstDetector::passes_filter() {
    (void)min_volume_; 
    return true;
}

// ─────────────────────────────────────────────────────────────

// ── FLUSH: finalize active burst at end of day ──────────────
bool BurstDetector::flush(Burst& result) {
    if (!is_active_) return false;

    current_burst_.end_time = last_msg_time_;
    current_burst_.end_price = last_mid_price_;
    current_burst_.trade_count = buy_count_ + sell_count_;
    classify_direction();

    is_active_ = false;

    if (passes_filter()) {
        result = current_burst_;
        return true;
    }
    return false;
}

// ── RESET: clear all state for next day ─────────────────────
void BurstDetector::reset() {
    is_active_ = false;
    last_msg_time_ = 0;
    last_mid_price_ = 0;
    buy_count_ = 0;
    sell_count_ = 0;
    max_price_ = 0;
    min_price_ = 0;
    current_burst_ = {};
}

bool BurstDetector::process(const LobsterMessage& msg, double current_mid, Burst& result) {
    // 1. Check if this is a trade (Execution or Hidden Execution)
    bool is_trade = (msg.type == 4 || msg.type == 5);

    // 2. IF NOT A TRADE:
    // We just update the price tracker so 'start_price' will be fresh 
    // when the next trade finally happens. Then we return false.
    if (!is_trade) {
        last_mid_price_ = current_mid; 
        return false;
    }

    // ─────────────────────────────────────────────────────────────
    // FROM HERE DOWN, WE KNOW IT IS A TRADE
    // ─────────────────────────────────────────────────────────────

    bool burst_finished = false;

    if (is_active_) {
        // Calculate gap from the LAST TRADE time (not last message time)
        double time_gap = msg.time - last_msg_time_;

        if (should_terminate(time_gap)) {
            // 1. Finalize the burst stats
            current_burst_.end_time = last_msg_time_;
            current_burst_.end_price = last_mid_price_;
            current_burst_.trade_count = buy_count_ + sell_count_;
            
            // 2. Classify
            classify_direction();

            // 3. Filter and Emit
            if (passes_filter()) {
                result = current_burst_;
                burst_finished = true;
            }
            
            // 4. Reset Active State
            is_active_ = false;
        }
    }

    // Start new burst if not active
    if (!is_active_) {
        is_active_ = true;
        current_burst_.id = msg.order_id;
        current_burst_.start_time = msg.time;
        current_burst_.direction = 0;
        current_burst_.volume = 0;
        current_burst_.trade_count = 0;
        
        // last_mid_price_ is now the price from the most recent message 
        // (even if it was a quote update 1ms ago), so this is accurate.
        current_burst_.start_price = (last_mid_price_ > 0) ? last_mid_price_ : current_mid;
        
        buy_count_ = 0;
        sell_count_ = 0;
        
        // Initialize extremes to include both start_price and current_mid
        max_price_ = std::max(current_burst_.start_price, current_mid);
        min_price_ = std::min(current_burst_.start_price, current_mid);
    }

    // Accumulate Current Trade
    current_burst_.volume += msg.size;
    
    // LOBSTER Direction: -1 = Buyer-initiated, 1 = Seller-initiated
    if (msg.direction == -1) buy_count_++;
    else sell_count_++;
    
    // Track extremes
    max_price_ = std::max(max_price_, current_mid);
    min_price_ = std::min(min_price_, current_mid);
    
    // Update trackers for the NEXT loop iteration
    last_msg_time_ = msg.time;      // Only update time on trades (to measure trade silence)
    last_mid_price_ = current_mid;  // Always update price

    return burst_finished;
}