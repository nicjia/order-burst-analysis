#include "passive_burst.h"
#include <cmath>
#include <numeric>

PassiveBurstDetector::PassiveBurstDetector(
    double silence_threshold, double min_volume_threshold, double direction_threshold,
    double volume_ratio_threshold, double hawkes_beta, double trigger_intensity,
    int max_bbo_levels)
    : silence_threshold_(silence_threshold),
      min_volume_threshold_(min_volume_threshold),
      direction_threshold_(direction_threshold),
      volume_ratio_threshold_(volume_ratio_threshold),
      max_bbo_levels_(max_bbo_levels),
      hawkes_beta_(hawkes_beta),
      trigger_intensity_(trigger_intensity),
      use_hawkes_(hawkes_beta > 0.0),
      hawkes_intensity_(0.0),
      is_active_(false),
      last_msg_time_(0),
      last_mid_price_(0),
      round_lot_count_(0),
      bid_sub_count_(0), ask_sub_count_(0),
      bid_sub_volume_(0), ask_sub_volume_(0),
      cancel_count_(0), cancel_volume_(0),
      bid_cancel_count_(0), ask_cancel_count_(0),
      bid_cancel_volume_(0), ask_cancel_volume_(0),
      max_price_(0), min_price_(0),
      pending_preburst_cancel_rate_(0.0) {}


bool PassiveBurstDetector::should_terminate(double time_gap) {
    if (use_hawkes_) {
        double decayed = hawkes_intensity_ * std::exp(-hawkes_beta_ * time_gap);
        return decayed < trigger_intensity_;
    }
    return time_gap > silence_threshold_;
}

void PassiveBurstDetector::classify_direction() {
    int total = bid_sub_count_ + ask_sub_count_;
    if (total == 0) return;

    double bid_ratio = (double)bid_sub_count_ / total;
    double ask_ratio = (double)ask_sub_count_ / total;
    current_burst_.bid_sub_count = bid_sub_count_;
    current_burst_.ask_sub_count = ask_sub_count_;
    current_burst_.bid_sub_volume = bid_sub_volume_;
    current_burst_.ask_sub_volume = ask_sub_volume_;
    current_burst_.bid_ratio = bid_ratio;
    current_burst_.ask_ratio = ask_ratio;

    double major_vol = std::max((double)bid_sub_volume_, (double)ask_sub_volume_);
    double minor_vol = std::min((double)bid_sub_volume_, (double)ask_sub_volume_);
    current_burst_.minmax_vol_ratio = (major_vol > 0.0) ? (minor_vol / major_vol) : 1.0;

    // Bid-heavy submissions = Bullish (direction = 1)
    if (bid_ratio >= direction_threshold_) {
        double minority_vol = (double)ask_sub_volume_;
        double majority_vol = (double)bid_sub_volume_;
        if (majority_vol > 0 && minority_vol <= volume_ratio_threshold_ * majority_vol) {
            current_burst_.direction = 1;   // Bullish passive intent
            current_burst_.peak_price = max_price_;
        } else {
            current_burst_.direction = 0;
            double up_move = std::abs(max_price_ - current_burst_.start_price);
            double down_move = std::abs(min_price_ - current_burst_.start_price);
            current_burst_.peak_price = (up_move >= down_move) ? max_price_ : min_price_;
        }
    } else if (ask_ratio >= direction_threshold_) {
        double minority_vol = (double)bid_sub_volume_;
        double majority_vol = (double)ask_sub_volume_;
        if (majority_vol > 0 && minority_vol <= volume_ratio_threshold_ * majority_vol) {
            current_burst_.direction = -1;  // Bearish passive intent
            current_burst_.peak_price = min_price_;
        } else {
            current_burst_.direction = 0;
            double up_move = std::abs(max_price_ - current_burst_.start_price);
            double down_move = std::abs(min_price_ - current_burst_.start_price);
            current_burst_.peak_price = (up_move >= down_move) ? max_price_ : min_price_;
        }
    } else {
        current_burst_.direction = 0;
        double up_move = std::abs(max_price_ - current_burst_.start_price);
        double down_move = std::abs(min_price_ - current_burst_.start_price);
        current_burst_.peak_price = (up_move >= down_move) ? max_price_ : min_price_;
    }

    // Fill in cancellation stats
    current_burst_.cancel_count = cancel_count_;
    current_burst_.cancel_volume = cancel_volume_;
    current_burst_.bid_cancel_count = bid_cancel_count_;
    current_burst_.ask_cancel_count = ask_cancel_count_;
    current_burst_.bid_cancel_volume = bid_cancel_volume_;
    current_burst_.ask_cancel_volume = ask_cancel_volume_;
    int total_events = total + cancel_count_;
    current_burst_.cancel_ratio = (total_events > 0) ? (double)cancel_count_ / total_events : 0.0;
}

void PassiveBurstDetector::compute_fingerprint() {
    int n = (int)submission_sizes_.size();
    if (n <= 1) {
        current_burst_.submission_size_variance = 0.0;
    } else {
        double sum = 0.0;
        for (int s : submission_sizes_) sum += (double)s;
        double mean = sum / n;
        double sq_sum = 0.0;
        for (int s : submission_sizes_) {
            double diff = (double)s - mean;
            sq_sum += diff * diff;
        }
        current_burst_.submission_size_variance = sq_sum / (n - 1);
    }
    current_burst_.round_lot_pct = (n > 0) ? (double)round_lot_count_ / n : 0.0;
}

bool PassiveBurstDetector::passes_filter() {
    return (double)current_burst_.volume >= min_volume_threshold_;
}

void PassiveBurstDetector::set_preburst_cancel_rate(double rate) {
    pending_preburst_cancel_rate_ = rate;
}

bool PassiveBurstDetector::flush(PassiveBurst& result) {
    if (!is_active_) return false;
    current_burst_.end_time = last_msg_time_;
    current_burst_.end_price = last_mid_price_;
    current_burst_.submission_count = bid_sub_count_ + ask_sub_count_;
    classify_direction();
    compute_fingerprint();
    is_active_ = false;
    if (passes_filter()) {
        result = current_burst_;
        return true;
    }
    return false;
}

void PassiveBurstDetector::reset() {
    is_active_ = false;
    last_msg_time_ = 0;
    last_mid_price_ = 0;
    bid_sub_count_ = 0; ask_sub_count_ = 0;
    bid_sub_volume_ = 0; ask_sub_volume_ = 0;
    cancel_count_ = 0; cancel_volume_ = 0;
    bid_cancel_count_ = 0; ask_cancel_count_ = 0;
    bid_cancel_volume_ = 0; ask_cancel_volume_ = 0;
    max_price_ = 0; min_price_ = 0;
    current_burst_ = {};
    submission_sizes_.clear();
    round_lot_count_ = 0;
    hawkes_intensity_ = 0.0;
    pending_preburst_cancel_rate_ = 0.0;
}

bool PassiveBurstDetector::process(const LobsterMessage& msg, double current_mid,
                                    int best_bid, int best_ask, PassiveBurst& result) {
    // ── Determine if this is a Type 1 submission at L1-L3 ────
    bool is_submission = (msg.type == 1);
    bool is_cancel = (msg.type == 2 || msg.type == 3);

    // Check if the price is within max_bbo_levels_ of the BBO
    bool is_near_bbo = false;
    if (best_bid > 0 && best_ask > 0) {
        int tick_size = 100; // LOBSTER prices are in $×10000, so $0.01 = 100 units
        if (msg.direction == 1) {
            // Buy-side: check if price is within L1-L3 of best bid
            is_near_bbo = (msg.price >= best_bid - max_bbo_levels_ * tick_size) &&
                          (msg.price <= best_ask);
        } else {
            // Sell-side: check if price is within L1-L3 of best ask
            is_near_bbo = (msg.price <= best_ask + max_bbo_levels_ * tick_size) &&
                          (msg.price >= best_bid);
        }
    }

    // Track cancellations during an active burst (as features, NOT triggers)
    if (is_cancel && is_active_ && is_near_bbo) {
        cancel_count_++;
        cancel_volume_ += msg.size;
        if (msg.direction == 1) {  // Buy-side cancel (bid)
            bid_cancel_count_++;
            bid_cancel_volume_ += msg.size;
        } else {
            ask_cancel_count_++;
            ask_cancel_volume_ += msg.size;
        }
    }

    // Only Type 1 submissions near the BBO excite the Hawkes process
    if (!is_submission || !is_near_bbo) {
        last_mid_price_ = current_mid;
        return false;
    }

    // ─────────────────────────────────────────────────────────────
    // FROM HERE DOWN: Type 1 submission at L1-L3
    // ─────────────────────────────────────────────────────────────

    bool burst_finished = false;

    if (is_active_) {
        double time_gap = msg.time - last_msg_time_;
        if (should_terminate(time_gap)) {
            current_burst_.end_time = last_msg_time_;
            current_burst_.end_price = last_mid_price_;
            current_burst_.submission_count = bid_sub_count_ + ask_sub_count_;
            classify_direction();
            compute_fingerprint();
            if (passes_filter()) {
                result = current_burst_;
                burst_finished = true;
            }
            is_active_ = false;
        } else if (use_hawkes_) {
            double time_gap_h = msg.time - last_msg_time_;
            hawkes_intensity_ = hawkes_intensity_ * std::exp(-hawkes_beta_ * time_gap_h) + 1.0;
            current_burst_.hawkes_peak_intensity = std::max(
                current_burst_.hawkes_peak_intensity, hawkes_intensity_);
        }
    }

    // Start new burst if not active
    if (!is_active_) {
        is_active_ = true;
        current_burst_.id = msg.order_id;
        current_burst_.start_time = msg.time;
        current_burst_.direction = 0;
        current_burst_.volume = 0;
        current_burst_.submission_count = 0;
        current_burst_.bid_sub_count = 0;
        current_burst_.ask_sub_count = 0;
        current_burst_.bid_sub_volume = 0;
        current_burst_.ask_sub_volume = 0;
        current_burst_.bid_ratio = 0.0;
        current_burst_.ask_ratio = 0.0;
        current_burst_.minmax_vol_ratio = 1.0;
        current_burst_.submission_size_variance = 0.0;
        current_burst_.round_lot_pct = 0.0;
        current_burst_.cancel_count = 0;
        current_burst_.cancel_volume = 0;
        current_burst_.bid_cancel_count = 0;
        current_burst_.ask_cancel_count = 0;
        current_burst_.bid_cancel_volume = 0;
        current_burst_.ask_cancel_volume = 0;
        current_burst_.cancel_ratio = 0.0;

        submission_sizes_.clear();
        round_lot_count_ = 0;
        hawkes_intensity_ = 1.0;
        current_burst_.hawkes_peak_intensity = 1.0;
        current_burst_.preburst_cancel_rate = pending_preburst_cancel_rate_;
        current_burst_.start_price = (last_mid_price_ > 0) ? last_mid_price_ : current_mid;

        bid_sub_count_ = 0; ask_sub_count_ = 0;
        bid_sub_volume_ = 0; ask_sub_volume_ = 0;
        cancel_count_ = 0; cancel_volume_ = 0;
        bid_cancel_count_ = 0; ask_cancel_count_ = 0;
        bid_cancel_volume_ = 0; ask_cancel_volume_ = 0;

        max_price_ = std::max(current_burst_.start_price, current_mid);
        min_price_ = std::min(current_burst_.start_price, current_mid);
    }

    // Accumulate submission
    current_burst_.volume += msg.size;
    submission_sizes_.push_back(msg.size);
    if (msg.size % 100 == 0) round_lot_count_++;

    // LOBSTER direction for Type 1: 1 = buy (bid), -1 = sell (ask)
    if (msg.direction == 1) {
        bid_sub_count_++;
        bid_sub_volume_ += msg.size;
    } else {
        ask_sub_count_++;
        ask_sub_volume_ += msg.size;
    }

    // Track price extremes
    max_price_ = std::max(max_price_, current_mid);
    min_price_ = std::min(min_price_, current_mid);

    last_msg_time_ = msg.time;
    last_mid_price_ = current_mid;

    return burst_finished;
}
