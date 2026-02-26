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
#include <string>
#include <algorithm>
#include <dirent.h>

#include "parser.h"
#include "types.h"
#include "burst.h"
#include "orderbook.h"

// ── Helpers ─────────────────────────────────────────────────

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

// ── Per-day burst record with forward-return data ───────────

struct BurstRecord {
    std::string ticker;
    std::string date;
    Burst burst;
    double close_mid;
    double mid_1m;      // mid at end_time + 60 s
    double mid_3m;      // mid at end_time + 180 s
    double mid_5m;      // mid at end_time + 300 s
    double mid_10m;     // mid at end_time + 600 s
};

// ── Usage ───────────────────────────────────────────────────

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <stock_folder> <output_file> [options]\n"
              << "  stock_folder: folder containing *_message_0.csv day files\n"
              << "  output_file:  output CSV path\n"
              << "Options:\n"
              << "  -s <silence>    silence threshold in seconds (default: 1.0)\n"
              << "  -v <min_vol>    minimum burst volume in shares  (default: 100)\n"
              << "  -d <direction>  direction ratio threshold        (default: 0.9)\n"
              << "  -k <kappa>      kappa filter parameter            (default: 0.5)\n";
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
    double kappa               = 0.5;

    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        std::string opt = argv[i];
        if      (opt == "-s") silence_threshold   = std::stod(argv[i+1]);
        else if (opt == "-v") min_volume          = std::stoi(argv[i+1]);
        else if (opt == "-d") direction_threshold = std::stod(argv[i+1]);
        else if (opt == "-k") kappa               = std::stod(argv[i+1]);
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
              << "  kappa=" << kappa << "\n\n";

    std::vector<BurstRecord> all_records;

    // ── Process each trading day ────────────────────────────
    for (const auto& msg_file : msg_files) {
        std::string date = extract_date(msg_file);
        std::cout << "  " << date << " … " << std::flush;

        // Fresh book & detector per day (pre-open rebuilds the book)
        OrderBook     book;
        BurstDetector detector(silence_threshold, min_volume, direction_threshold);
        LobsterParser parser(msg_file);

        // Mid-price snapshots: only recorded when mid actually changes.
        // Used after the day loop for forward-return lookups.
        std::vector<std::pair<double, double>> mid_snapshots;
        mid_snapshots.reserve(500000);  // typical: a few hundred-K BBO changes/day

        LobsterMessage msg;
        Burst finished;
        std::vector<Burst> day_bursts;
        double current_mid = 0.0;
        long   msg_count   = 0;

        while (parser.next_message(msg)) {
            ++msg_count;

            // 1. Update the reconstructed order book
            book.process_message(msg);

            // 2. Track mid-price (only when book has both sides)
            if (book.is_valid()) {
                double new_mid = book.get_mid_price();
                if (new_mid != current_mid) {
                    current_mid = new_mid;
                    mid_snapshots.push_back({msg.time, current_mid});
                }
            }

            // 3. Feed into burst detector (needs valid mid to work)
            if (current_mid > 0.0) {
                if (detector.process(msg, current_mid, finished)) {
                    day_bursts.push_back(finished);
                }
            }
        }

        // Flush any burst still active at market close
        if (detector.flush(finished)) {
            day_bursts.push_back(finished);
        }

        double close_mid = current_mid;

        // 4. Compute forward-return mid-prices for each burst
        for (auto& b : day_bursts) {
            BurstRecord rec;
            rec.ticker    = ticker;
            rec.date      = date;
            rec.burst     = b;
            rec.close_mid = close_mid;
            rec.mid_1m    = lookup_mid(mid_snapshots, b.end_time + 60.0);
            rec.mid_3m    = lookup_mid(mid_snapshots, b.end_time + 180.0);
            rec.mid_5m    = lookup_mid(mid_snapshots, b.end_time + 300.0);
            rec.mid_10m   = lookup_mid(mid_snapshots, b.end_time + 600.0);
            all_records.push_back(rec);
        }

        std::cout << msg_count << " msgs, "
                  << mid_snapshots.size() << " BBO updates, "
                  << day_bursts.size() << " bursts\n";
    }

    // ── Write output CSV ────────────────────────────────────
    std::ofstream out(output_file);
    out << "Ticker,Date,BurstID,StartTime,EndTime,Direction,Volume,TradeCount,"
        << "StartPrice,EndPrice,PeakPrice,CloseMid,"
        << "Mid_1m,Mid_3m,Mid_5m,Mid_10m\n";

    for (const auto& r : all_records) {
        const auto& b = r.burst;
        out << r.ticker << "," << r.date << ","
            << b.id << ","
            << std::fixed << std::setprecision(6)
            << b.start_time << "," << b.end_time << ","
            << b.direction << "," << b.volume << ","
            << b.trade_count << ","
            << std::setprecision(4)
            << b.start_price << "," << b.end_price << "," << b.peak_price << ","
            << r.close_mid << ","
            << r.mid_1m << "," << r.mid_3m << ","
            << r.mid_5m << "," << r.mid_10m
            << "\n";
    }
    out.close();

    std::cout << "\nTotal bursts across all days: " << all_records.size() << "\n";
    std::cout << "Output: " << output_file << "\n";

    return 0;
}