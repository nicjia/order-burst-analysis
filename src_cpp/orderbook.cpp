#include "orderbook.h"

OrderBook::OrderBook() {}

void OrderBook::reset() {
    orders_.clear();
    bids_.clear();
    asks_.clear();
}

// ── Internal helpers ────────────────────────────────────────

void OrderBook::add_order(long order_id, int price, int size, int direction) {
    // Guard: ignore nonsense prices / sizes
    if (price <= 0 || size <= 0) return;

    orders_[order_id] = {price, size, direction};

    if (direction == 1) {
        bids_[price] += size;
    } else {
        asks_[price] += size;
    }
}

void OrderBook::reduce_order(long order_id, int size_delta) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) return;           // unknown order – skip

    Order& order = it->second;
    auto& side = (order.direction == 1) ? bids_ : asks_;

    // Shrink the price level
    auto pl = side.find(order.price);
    if (pl != side.end()) {
        pl->second -= size_delta;
        if (pl->second <= 0) side.erase(pl);
    }

    // Shrink (or remove) the order itself
    order.size -= size_delta;
    if (order.size <= 0) {
        orders_.erase(it);
    }
}

void OrderBook::delete_order(long order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) return;           // unknown order – skip

    const Order& order = it->second;
    auto& side = (order.direction == 1) ? bids_ : asks_;

    auto pl = side.find(order.price);
    if (pl != side.end()) {
        pl->second -= order.size;
        if (pl->second <= 0) side.erase(pl);
    }

    orders_.erase(it);
}

// ── Public interface ────────────────────────────────────────

bool OrderBook::process_message(const LobsterMessage& msg) {
    int old_bid = get_best_bid();
    int old_ask = get_best_ask();

    switch (msg.type) {
        case 1: add_order(msg.order_id, msg.price, msg.size, msg.direction);   break;
        case 2: reduce_order(msg.order_id, msg.size);                          break;
        case 3: delete_order(msg.order_id);                                    break;
        case 4: reduce_order(msg.order_id, msg.size);                          break;
        // Types 5 (hidden exec), 6 (cross trade), 7 (halt) – no visible book change
        default: break;
    }

    return (get_best_bid() != old_bid || get_best_ask() != old_ask);
}

int OrderBook::get_best_bid() const {
    if (bids_.empty()) return 0;
    return bids_.rbegin()->first;     // highest buy price
}

int OrderBook::get_best_ask() const {
    if (asks_.empty()) return 0;
    return asks_.begin()->first;      // lowest sell price
}

double OrderBook::get_mid_price() const {
    int bid = get_best_bid();
    int ask = get_best_ask();
    if (bid == 0 || ask == 0) return 0.0;
    return ((double)bid + (double)ask) / 2.0 / 10000.0;
}

bool OrderBook::is_valid() const {
    return !bids_.empty() && !asks_.empty();
}
