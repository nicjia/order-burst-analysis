#include "burst.h"
#include <cmath> // Required for std::abs

BurstDetector::BurstDetector(double silence_threshold, int min_volume, double direction_threshold,
                             double volume_ratio_threshold) 
    : silence_threshold_(silence_threshold), 
      min_volume_(min_volume),
      direction_threshold_(direction_threshold),
      volume_ratio_threshold_(volume_ratio_threshold),
      is_active_(false), 
      last_msg_time_(0),
      last_mid_price_(0),
      buy_count_(0),
      sell_count_(0),
      buy_volume_(0),
      sell_volume_(0),
      max_price_(0),
      min_price_(0) {}


// ── TERMINATION: when does a burst end? ─────────────────────
bool BurstDetector::should_terminate(double time_gap) {
    return time_gap > silence_threshold_;
}

// ── CLASSIFICATION: Hybrid count + volume check (Eq 2.3) ─────
//
// Two conditions must hold for a directional classification:
//   1. Count-based:  buy_ratio >= direction_threshold_  (or sell)
//   2. Volume-based: minority_volume < volume_ratio_threshold_ × majority_volume
//
// Condition 2 prevents cases like "9 buys of 10 shares + 1 sell of 1000 shares"
// from being classified as a Buy burst.
void BurstDetector::classify_direction() {
    int total = buy_count_ + sell_count_;
    if (total == 0) return;
    
    double buy_ratio = (double)buy_count_ / total;
    double sell_ratio = (double)sell_count_ / total;
    current_burst_.buy_count = buy_count_;
    current_burst_.sell_count = sell_count_;
    current_burst_.buy_volume = buy_volume_;
    current_burst_.sell_volume = sell_volume_;
    current_burst_.buy_ratio = buy_ratio;
    current_burst_.sell_ratio = sell_ratio;
    double major_vol = std::max((double)buy_volume_, (double)sell_volume_);
    double minor_vol = std::min((double)buy_volume_, (double)sell_volume_);
    current_burst_.minmax_vol_ratio = (major_vol > 0.0) ? (minor_vol / major_vol) : 1.0;
    
    if (buy_ratio >= direction_threshold_) {
        // Count says Buy – verify volume doesn't contradict
        double minority_vol = (double)sell_volume_;
        double majority_vol = (double)buy_volume_;
        if (majority_vol > 0 && minority_vol <= volume_ratio_threshold_ * majority_vol) {
            current_burst_.direction = 1;   // Buy burst
            current_burst_.peak_price = max_price_;
        } else {
            current_burst_.direction = 0;   // Counts say Buy, but volume is contradictory
            double up_move = std::abs(max_price_ - current_burst_.start_price);
            double down_move = std::abs(min_price_ - current_burst_.start_price);
            current_burst_.peak_price = (up_move >= down_move) ? max_price_ : min_price_;
        }
    } else if (sell_ratio >= direction_threshold_) {
        // Count says Sell – verify volume doesn't contradict
        double minority_vol = (double)buy_volume_;
        double majority_vol = (double)sell_volume_;
        if (majority_vol > 0 && minority_vol <= volume_ratio_threshold_ * majority_vol) {
            current_burst_.direction = -1;  // Sell burst
            current_burst_.peak_price = min_price_;
        } else {
            current_burst_.direction = 0;   // Counts say Sell, but volume is contradictory
            double up_move = std::abs(max_price_ - current_burst_.start_price);
            double down_move = std::abs(min_price_ - current_burst_.start_price);
            current_burst_.peak_price = (up_move >= down_move) ? max_price_ : min_price_;
        }
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
    buy_volume_ = 0;
    sell_volume_ = 0;
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
        current_burst_.buy_count = 0;
        current_burst_.sell_count = 0;
        current_burst_.buy_volume = 0;
        current_burst_.sell_volume = 0;
        current_burst_.buy_ratio = 0.0;
        current_burst_.sell_ratio = 0.0;
        current_burst_.minmax_vol_ratio = 1.0;
        
        // last_mid_price_ is now the price from the most recent message 
        // (even if it was a quote update 1ms ago), so this is accurate.
        current_burst_.start_price = (last_mid_price_ > 0) ? last_mid_price_ : current_mid;
        
        buy_count_ = 0;
        sell_count_ = 0;
        buy_volume_ = 0;
        sell_volume_ = 0;
        
        // Initialize extremes to include both start_price and current_mid
        max_price_ = std::max(current_burst_.start_price, current_mid);
        min_price_ = std::min(current_burst_.start_price, current_mid);
    }

    // Accumulate Current Trade
    current_burst_.volume += msg.size;
    
    // LOBSTER Direction: -1 = Buyer-initiated, 1 = Seller-initiated
    if (msg.direction == -1) {
        buy_count_++;
        buy_volume_ += msg.size;
    } else {
        sell_count_++;
        sell_volume_ += msg.size;
    }
    
    // Track extremes
    max_price_ = std::max(max_price_, current_mid);
    min_price_ = std::min(min_price_, current_mid);
    
    // Update trackers for the NEXT loop iteration
    last_msg_time_ = msg.time;      // Only update time on trades (to measure trade silence)
    last_mid_price_ = current_mid;  // Always update price

    return burst_finished;
}