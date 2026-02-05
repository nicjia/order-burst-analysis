#include "burst.h"

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


// Returns true if a burst just finished (and met min_volume threshold)
bool BurstDetector::process(const LobsterMessage& msg, double current_mid, Burst& result) {

    //currently only order submissions
    if (msg.type != 1) return false;

    bool burst_finished = false;

    // Check if an active burst should end (time gap > silence threshold)
    if (is_active_) {
        double time_gap = msg.time - last_msg_time_;

        if (time_gap > silence_threshold_) {

            current_burst_.end_time = last_msg_time_;
            current_burst_.end_price = last_mid_price_;  
            current_burst_.trade_count = buy_count_ + sell_count_;
            
            int total = buy_count_ + sell_count_;
            if (total > 0) {
                double buy_ratio = (double)buy_count_ / total;
                double sell_ratio = (double)sell_count_ / total;
                
                if (buy_ratio >= direction_threshold_) {
                    current_burst_.direction = 1;   // Buy burst
                    current_burst_.peak_price = max_price_;
                } else if (sell_ratio >= direction_threshold_) {
                    current_burst_.direction = -1;  // Sell burst
                    current_burst_.peak_price = min_price_;
                } else {
                    current_burst_.direction = 0;   // Mixed - didn't meet threshold
                    // For mixed, use max deviation from start
                    double up_move = max_price_ - current_burst_.start_price;
                    double down_move = current_burst_.start_price - min_price_;
                    current_burst_.peak_price = (up_move >= down_move) ? max_price_ : min_price_;
                }
            }
            
            // Only output if meets minimum volume requirement
            if (current_burst_.volume >= min_volume_) {
                result = current_burst_;
                burst_finished = true;
            }
            
            is_active_ = false;
        }
    }

    // Start new burst if not active
    if (!is_active_) {
        is_active_ = true;
        current_burst_.id = msg.order_id;
        current_burst_.start_time = msg.time;
        current_burst_.direction = 0;  // Will be determined at end
        current_burst_.volume = 0;
        current_burst_.trade_count = 0;
        current_burst_.start_price = current_mid;
        
        buy_count_ = 0;
        sell_count_ = 0;
        max_price_ = current_mid;
        min_price_ = current_mid;
    }

    // Update current burst state
    current_burst_.volume += msg.size;
    
    if (msg.direction == 1) {
        buy_count_++;
    } else {
        sell_count_++;
    }
    
    // Track both max and min (direction unknown until burst ends)
    max_price_ = std::max(max_price_, current_mid);
    min_price_ = std::min(min_price_, current_mid);
    
    last_msg_time_ = msg.time;
    last_mid_price_ = current_mid;

    return burst_finished;
}