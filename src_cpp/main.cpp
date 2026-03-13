// ─────────────────────────────────────────────────────────────
// main.cpp  –  Burst Detection with Top-of-Book Reconstruction
// ─────────────────────────────────────────────────────────────
//
// Input:  A stock folder containing one *_message_0.csv per day.
//         Each day file starts with pre-open orders (~4 AM) so the
//         full visible book can be reconstructed from scratch.
//
// Output: A single CSV with all bursts across all days, including:
//         Ticker, Date, forward-return mid-prices, and close mid.
//
// No orderbook.csv is needed – BBO is rebuilt from messages.
// ─────────────────────────────────────────────────────────────

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <dirent.h>

#include "parser.h"
#include "types.h"
#include "burst.h"
#include "orderbook.h"

// ── Helpers ─────────────────────────────────────────────────

// Regular Trading Hours in seconds-past-midnight.
// 9:30 AM = 34200, 4:00 PM = 57600.
// Conservative safe zone trims the open/close chaos:
//   9:40 AM = 34800,  3:50 PM = 57000
constexpr double RTH_DEFAULT_START = 34200.0;   // 09:30
constexpr double RTH_DEFAULT_END   = 57600.0;   // 16:00

// Collect all *message*.csv files in a directory, sorted by name (= by date).
std::vector<std::string> find_message_files(const std::string& folder) {
    std::vector<std::string> files;
    DIR* dir = opendir(folder.c_str());
    if (!dir) return files;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.find("message") != std::string::npos &&
            name.size() > 4 && name.substr(name.size() - 4) == ".csv") {
            // Ensure folder path ends with '/'
            std::string path = folder;
            if (!path.empty() && path.back() != '/') path += '/';
            files.push_back(path + name);
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

// Extract date from filename: TICKER_2026-01-02_..._message_0.csv → "2026-01-02"
std::string extract_date(const std::string& filepath) {
    // Isolate filename from path
    auto slash = filepath.rfind('/');
    std::string fname = (slash != std::string::npos) ? filepath.substr(slash + 1) : filepath;

    auto first  = fname.find('_');
    if (first == std::string::npos) return "unknown";
    auto second = fname.find('_', first + 1);
    if (second == std::string::npos) return "unknown";
    return fname.substr(first + 1, second - first - 1);
}

// Extract ticker from folder name: .../TSLA_2026-01-01_2026-02-14_0 → "TSLA"
std::string extract_ticker(const std::string& folder) {
    std::string path = folder;
    while (!path.empty() && path.back() == '/') path.pop_back();
    auto slash = path.rfind('/');
    std::string dirname = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    auto upos = dirname.find('_');
    return (upos != std::string::npos) ? dirname.substr(0, upos) : dirname;
}

// Binary-search the mid-price snapshot timeline for the value at (or just before) target_time.
double lookup_mid(const std::vector<std::pair<double, double>>& snaps, double target_time) {
    if (snaps.empty()) return 0.0;
    if (target_time <= snaps.front().first) return snaps.front().second;
    if (target_time >= snaps.back().first)  return snaps.back().second;

    // upper_bound gives the first element with time > target_time
    auto it = std::upper_bound(
        snaps.begin(), snaps.end(), target_time,
        [](double t, const std::pair<double, double>& p) { return t < p.first; });

    // Step back to the snapshot at or just before target_time
    if (it != snaps.begin()) --it;
    return it->second;
}

// Scan mid-price snapshots to find the most extreme price within [start_time, start_time + tau_max].
// This is the true forward-looking PeakImpact defined by the proposal.
double find_peak_price(const std::vector<std::pair<double, double>>& snaps,
                       double start_time, double start_price, double tau_max, int direction) {
    if (snaps.empty()) return start_price;

    // Binary search to the first snapshot at or after start_time
    auto it = std::lower_bound(snaps.begin(), snaps.end(), start_time,
        [](const std::pair<double, double>& p, double t) { return p.first < t; });

    double end_time = start_time + tau_max;
    double max_p = start_price;
    double min_p = start_price;

    while (it != snaps.end() && it->first <= end_time) {
        max_p = std::max(max_p, it->second);
        min_p = std::min(min_p, it->second);
        ++it;
    }

    if (direction == 1)  return max_p;   // Buy burst → highest price reached
    if (direction == -1) return min_p;   // Sell burst → lowest price reached

    // Mixed: whichever moved further from start
    return (std::abs(max_p - start_price) >= std::abs(min_p - start_price)) ? max_p : min_p;
}

// ── Per-day burst record with forward-return data ───────────

// ── Market state snapshot — captured at burst START time ─────
//    All of these are observable before the burst's impact
//    propagates, so they carry NO look-ahead bias.

struct MarketState {
    double spread;            // bid-ask spread in dollars at burst start
    int    bid_vol_best;      // volume at best bid
    int    ask_vol_best;      // volume at best ask
    int    bid_depth_5;       // total bid volume across top 5 levels
    int    ask_depth_5;       // total ask volume across top 5 levels
    double book_imbalance;    // (bid_depth_5 − ask_depth_5) / (bid + ask)
    double volatility_60s;    // realised volatility of mid-returns, 60 s window
    double momentum_5s;       // mid-price change over prior 5 seconds
    double momentum_30s;      //   ... 30 seconds
    double momentum_60s;      //   ... 60 seconds
    int    trade_count_5m;    // number of trades in prior 5 minutes
    int    trade_volume_5m;   // total shares traded in prior 5 minutes
};

struct BurstRecord {
    std::string ticker;
    std::string date;
    Burst burst;
    double close_mid;
    double mid_1m;      // mid at end_time + 60 s
    double mid_3m;      // mid at end_time + 180 s
    double mid_5m;      // mid at end_time + 300 s
    double mid_10m;     // mid at end_time + 600 s
    double d_b;         // short-horizon decay metric
    MarketState mkt;    // book state at burst start
};

// ── Usage ───────────────────────────────────────────────────

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <stock_folder> <output_file> [options]\n"
              << "  stock_folder: folder containing *_message_0.csv day files\n"
              << "  output_file:  output CSV path\n"
              << "Options:\n"
              << "  -s <silence>    silence threshold in seconds       (default: 1.0)\n"
              << "  -v <min_vol>    minimum burst volume in shares     (default: 100)\n"
              << "  -d <direction>  direction count-ratio threshold    (default: 0.9)\n"
              << "  -r <vol_ratio>  volume ratio cap for directional   (default: 0.5)\n"
              << "  -k <kappa>      kappa filter parameter             (default: 0.5)\n"
              << "  -t <tau_max>    peak-impact horizon in seconds     (default: 10.0)\n"
              << "  -b <rth_start>  RTH start in sec-past-midnight     (default: 34200 = 09:30)\n"
              << "  -e <rth_end>    RTH end   in sec-past-midnight     (default: 57600 = 16:00)\n";
}

// ── Main ────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string stock_folder = argv[1];
    std::string output_file  = argv[2];

    double silence_threshold   = 1.0;
    int    min_volume          = 100;
    double direction_threshold = 0.9;
    double volume_ratio_threshold = 0.5;  // minority_vol / majority_vol cap for directional classification
    double kappa               = 0.5;
    double tau_max             = 10.0;  // microstructure horizon for peak impact
    double rth_start            = RTH_DEFAULT_START;
    double rth_end              = RTH_DEFAULT_END;

    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        std::string opt = argv[i];
        if      (opt == "-s") silence_threshold   = std::stod(argv[i+1]);
        else if (opt == "-v") min_volume          = std::stoi(argv[i+1]);
        else if (opt == "-d") direction_threshold = std::stod(argv[i+1]);
        else if (opt == "-r") volume_ratio_threshold = std::stod(argv[i+1]);
        else if (opt == "-k") kappa               = std::stod(argv[i+1]);
        else if (opt == "-t") tau_max             = std::stod(argv[i+1]);
        else if (opt == "-b") rth_start           = std::stod(argv[i+1]);
        else if (opt == "-e") rth_end             = std::stod(argv[i+1]);
    }

    // ── Discover day files ──────────────────────────────────
    auto msg_files = find_message_files(stock_folder);
    if (msg_files.empty()) {
        std::cerr << "Error: No *_message_*.csv files found in " << stock_folder << "\n";
        return 1;
    }

    std::string ticker = extract_ticker(stock_folder);

    std::cout << "Ticker: " << ticker << "\n";
    std::cout << "Found " << msg_files.size() << " day file(s)\n";
    std::cout << "Settings: silence=" << silence_threshold
              << "  min_vol=" << min_volume
              << "  dir_thresh=" << direction_threshold
              << "  vol_ratio_thresh=" << volume_ratio_threshold
              << "  kappa=" << kappa
              << "  tau_max=" << tau_max
              << "  RTH=[" << rth_start << "," << rth_end << "]\n\n";

    std::vector<BurstRecord> all_records;

    // ── Process each trading day ────────────────────────────
    for (const auto& msg_file : msg_files) {
        std::string date = extract_date(msg_file);
        std::cout << "  " << date << " … " << std::flush;

        // Fresh book & detector per day (pre-open rebuilds the book)
        OrderBook     book;
        BurstDetector detector(silence_threshold, min_volume, direction_threshold, volume_ratio_threshold);
        LobsterParser parser(msg_file);

        // Mid-price snapshots: only recorded when mid actually changes.
        // Used after the day loop for forward-return lookups.
        std::vector<std::pair<double, double>> mid_snapshots;
        mid_snapshots.reserve(500000);

        // ── Rolling statistics accumulators ──────────────────
        // (a) Mid-return ring buffer for 60-second realized volatility
        //     Each entry = (time, mid_price).  We compute returns on the fly.
        std::deque<std::pair<double, double>> mid_ring;  // (time, mid)
        const double VOL_WINDOW = 60.0;

        // (b) Trade ring buffer for 5-minute trade intensity
        struct TradeStamp { double time; int size; };
        std::deque<TradeStamp> trade_ring;
        const double TRADE_WINDOW = 300.0;

        // (c) Burst start-time ring buffer for recent-burst features
        struct BurstStamp { double time; int direction; int volume; };
        std::deque<BurstStamp> burst_ring;

        LobsterMessage msg;
        Burst finished;
        std::vector<std::pair<Burst, MarketState>> day_bursts;  // burst + state at initiation
        double current_mid = 0.0;
        long   msg_count   = 0;
        bool   flushed_at_rth_end = false;

        // Helper lambda: compute realized volatility from mid_ring
        auto calc_volatility = [&](double now) -> double {
            // Prune old entries
            while (!mid_ring.empty() && mid_ring.front().first < now - VOL_WINDOW)
                mid_ring.pop_front();
            if (mid_ring.size() < 2) return 0.0;
            double sum_sq = 0.0;
            int n = 0;
            for (size_t k = 1; k < mid_ring.size(); ++k) {
                double prev = mid_ring[k-1].second;
                if (prev == 0.0) continue;
                double ret = (mid_ring[k].second - prev) / prev;
                sum_sq += ret * ret;
                ++n;
            }
            return (n > 0) ? std::sqrt(sum_sq / n) : 0.0;
        };

        // Helper lambda: momentum = (current_mid − mid_at(now − delta)) / mid_at(now − delta)
        auto calc_momentum = [&](double now, double delta) -> double {
            double target = now - delta;
            // Find the latest mid_ring entry at or before target
            double ref_mid = 0.0;
            for (auto it = mid_ring.rbegin(); it != mid_ring.rend(); ++it) {
                if (it->first <= target) { ref_mid = it->second; break; }
            }
            if (ref_mid == 0.0 || current_mid == 0.0) return 0.0;
            return (current_mid - ref_mid) / ref_mid;
        };

        // Helper lambda: trade intensity in prior TRADE_WINDOW
        auto calc_trade_stats = [&](double now) -> std::pair<int, int> {
            while (!trade_ring.empty() && trade_ring.front().time < now - TRADE_WINDOW)
                trade_ring.pop_front();
            int cnt = 0, vol = 0;
            for (auto& t : trade_ring) { ++cnt; vol += t.size; }
            return {cnt, vol};
        };

        // Helper lambda: snapshot full MarketState
        auto snapshot_market_state = [&](double now) -> MarketState {
            MarketState s{};
            s.spread        = book.get_spread();
            s.bid_vol_best  = book.get_bid_volume_at_best();
            s.ask_vol_best  = book.get_ask_volume_at_best();
            s.bid_depth_5   = book.get_bid_depth(5);
            s.ask_depth_5   = book.get_ask_depth(5);

            double total_depth = (double)(s.bid_depth_5 + s.ask_depth_5);
            s.book_imbalance = (total_depth > 0)
                ? (double)(s.bid_depth_5 - s.ask_depth_5) / total_depth
                : 0.0;

            s.volatility_60s  = calc_volatility(now);
            s.momentum_5s     = calc_momentum(now, 5.0);
            s.momentum_30s    = calc_momentum(now, 30.0);
            s.momentum_60s    = calc_momentum(now, 60.0);

            auto [tc, tv]     = calc_trade_stats(now);
            s.trade_count_5m  = tc;
            s.trade_volume_5m = tv;
            return s;
        };

        while (parser.next_message(msg)) {
            ++msg_count;

            // 1. ALWAYS update the order book — pre-open messages
            //    rebuild the full visible book before RTH opens.
            book.process_message(msg);

            // 2. Track mid-price (only when book has both sides).
            //    We record snapshots even outside RTH so that
            //    forward-return lookups (e.g. Mid_10m for a 3:55 PM
            //    burst) have prices right up to the close.
            if (book.is_valid()) {
                double new_mid = book.get_mid_price();
                if (new_mid != current_mid) {
                    current_mid = new_mid;
                    mid_snapshots.push_back({msg.time, current_mid});
                }
            }

            // 3. Burst detection is restricted to Regular Trading Hours.
            //    Pre-market, opening auction, and post-close are excluded.
            if (msg.time < rth_start) continue;     // pre-market: skip

            // ── Update rolling accumulators (RTH only) ──────
            if (current_mid > 0.0) {
                // Only push when mid changes (same condition as mid_snapshots)
                if (mid_ring.empty() || mid_ring.back().second != current_mid) {
                    mid_ring.push_back({msg.time, current_mid});
                }
            }
            // Track every trade for intensity
            bool is_trade = (msg.type == 4 || msg.type == 5);
            if (is_trade) {
                trade_ring.push_back({msg.time, msg.size});
            }

            if (msg.time > rth_end) {
                // Past RTH — flush once, then just keep reading for mid snapshots
                if (!flushed_at_rth_end) {
                    if (detector.flush(finished)) {
                        MarketState ms = snapshot_market_state(finished.start_time);
                        day_bursts.push_back({finished, ms});
                    }
                    flushed_at_rth_end = true;
                }
                continue;
            }

            // Inside RTH — feed to burst detector
            if (current_mid > 0.0) {
                if (detector.process(msg, current_mid, finished)) {
                    // Snapshot market state AT THE TIME THE BURST STARTED
                    MarketState ms = snapshot_market_state(finished.start_time);
                    day_bursts.push_back({finished, ms});
                }
            }
        }

        // Flush any burst still active at file end
        if (!flushed_at_rth_end && detector.flush(finished)) {
            MarketState ms = snapshot_market_state(finished.start_time);
            day_bursts.push_back({finished, ms});
        }

        double close_mid = current_mid;

        // 4. Compute peak impact (tau_max) and forward-return mid-prices
        for (auto& [b, ms] : day_bursts) {
            b.peak_price = find_peak_price(mid_snapshots, b.start_time, b.start_price, tau_max, b.direction);

            BurstRecord rec;
            rec.ticker    = ticker;
            rec.date      = date;
            rec.burst     = b;
            rec.close_mid = close_mid;
            rec.mid_1m    = lookup_mid(mid_snapshots, b.end_time + 60.0);
            rec.mid_3m    = lookup_mid(mid_snapshots, b.end_time + 180.0);
            rec.mid_5m    = lookup_mid(mid_snapshots, b.end_time + 300.0);
            rec.mid_10m   = lookup_mid(mid_snapshots, b.end_time + 600.0);
            // D_b = (1/4) Σ Q_b × Direction × (Mid_τ − StartPrice)
            double dsum = 0.0;
            int dcount = 0;
            auto accum = [&](double mid) {
                if (mid > 0.0) {
                    dsum += (double)b.volume * (double)b.direction * (mid - b.start_price);
                    dcount++;
                }
            };
            accum(rec.mid_1m);
            accum(rec.mid_3m);
            accum(rec.mid_5m);
            accum(rec.mid_10m);
            rec.d_b = (dcount > 0)
                ? (dsum / dcount)
                : std::numeric_limits<double>::quiet_NaN();

            // Apply kappa filter here to drop bursts before output
            if (kappa > 0.0) {
                if (std::isnan(rec.d_b) || rec.d_b < kappa) {
                    continue;
                }
            }

            rec.mkt       = ms;
            all_records.push_back(rec);
        }

        std::cout << msg_count << " msgs, "
                  << mid_snapshots.size() << " BBO updates, "
                  << day_bursts.size() << " bursts\n";
    }

    // ── Write output CSV ────────────────────────────────────
    std::ofstream out(output_file);
    out << "Ticker,Date,BurstID,StartTime,EndTime,Direction,Volume,TradeCount,D_b,"
        << "StartPrice,EndPrice,PeakPrice,CloseMid,"
        << "Mid_1m,Mid_3m,Mid_5m,Mid_10m,"
        << "Spread,BidVolBest,AskVolBest,BidDepth5,AskDepth5,BookImbalance,"
        << "Volatility60s,Momentum5s,Momentum30s,Momentum60s,"
        << "TradeCount5m,TradeVolume5m\n";

    for (const auto& r : all_records) {
        const auto& b = r.burst;
        const auto& m = r.mkt;
        out << r.ticker << "," << r.date << ","
            << b.id << ","
            << std::fixed << std::setprecision(6)
            << b.start_time << "," << b.end_time << ","
            << b.direction << "," << b.volume << ","
            << b.trade_count << ","
            << r.d_b << ","
            << std::setprecision(4)
            << b.start_price << "," << b.end_price << "," << b.peak_price << ","
            << r.close_mid << ","
            << r.mid_1m << "," << r.mid_3m << ","
            << r.mid_5m << "," << r.mid_10m << ","
            << std::setprecision(6)
            << m.spread << ","
            << m.bid_vol_best << "," << m.ask_vol_best << ","
            << m.bid_depth_5 << "," << m.ask_depth_5 << ","
            << std::setprecision(6) << m.book_imbalance << ","
            << std::setprecision(8) << m.volatility_60s << ","
            << m.momentum_5s << "," << m.momentum_30s << "," << m.momentum_60s << ","
            << m.trade_count_5m << "," << m.trade_volume_5m
            << "\n";
    }
    out.close();

    std::cout << "\nTotal bursts across all days: " << all_records.size() << "\n";
    std::cout << "Output: " << output_file << "\n";

    return 0;
}