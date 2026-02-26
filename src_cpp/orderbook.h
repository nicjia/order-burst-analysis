#ifndef ORDERBOOK_H
#define ORDERBOOK_H

#include "types.h"
#include <unordered_map>
#include <map>

// ─────────────────────────────────────────────────────────────
// OrderBook: Top-of-Book Reconstruction from LOBSTER Messages
// ─────────────────────────────────────────────────────────────
//
// Maintains the visible limit order book and provides continuous
// Best Bid / Best Ask / Mid-Price.  Built from scratch each day
// starting with pre-open submissions (~4 AM).
//
// Message types handled:
//   1  Submission       → add order to book
//   2  Partial cancel   → reduce order size
//   3  Full deletion    → remove order entirely
//   4  Visible exec     → reduce order size (remove if filled)
//   5  Hidden exec      → no visible-book impact
//   6  Cross trade      → no book impact
//   7  Trading halt     → no book impact
// ─────────────────────────────────────────────────────────────

class OrderBook {
public:
    OrderBook();

    // Process a single LOBSTER message.
    // Returns true if the BBO (best bid or best ask) changed.
    bool process_message(const LobsterMessage& msg);

    // Current mid-price in dollar terms:  (best_bid + best_ask) / 2 / 10000
    double get_mid_price() const;

    // Raw LOBSTER price units (dollar * 10000)
    int get_best_bid() const;
    int get_best_ask() const;

    // True when both sides of the book have at least one resting order
    bool is_valid() const;

    // Reset for a new trading day (clears all state)
    void reset();

private:
    // Per-order tracking for O(1) lookup on cancel / exec
    struct Order {
        int price;
        int size;
        int direction;   // 1 = buy (bid),  -1 = sell (ask)
    };

    std::unordered_map<long, Order> orders_;   // order_id → details

    // Price-level aggregation (total resting size at each price)
    //   bids_: highest key = best bid
    //   asks_: lowest  key = best ask
    std::map<int, int> bids_;
    std::map<int, int> asks_;

    void add_order(long order_id, int price, int size, int direction);
    void reduce_order(long order_id, int size_delta);   // type 2 / 4
    void delete_order(long order_id);                    // type 3
};

#endif
