//raw data reading
#include "parser.h"
#include <iostream>
#include <cstdlib>

LobsterParser::LobsterParser(std::string filename) {
    file_.open(filename);
    if (!file_.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
}

LobsterParser::~LobsterParser() {
    if (file_.is_open()) {
        file_.close();
    }
}

// Helper: fast integer parse advancing pointer past trailing comma.
// Handles optional leading '-' for signed fields (direction, price).
static inline long fast_parse_long(const char*& p) {
    long val = 0;
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
    if (*p == ',') p++;
    return val * sign;
}

bool LobsterParser::next_message(LobsterMessage& msg) {
    std::string line;
    if (!std::getline(file_, line)) {
        return false; // EOF
    }

    const char* p = line.c_str();
    char* end;

    // Field 1: timestamp (double, high-precision seconds-past-midnight).
    // Use strtod for correct IEEE 754 rounding — hand-rolled fractional
    // accumulators (frac *= 0.1) drift by ~2 ULPs on 9-digit timestamps.
    msg.time = std::strtod(p, &end);
    p = end;
    if (*p == ',') p++;

    // Fields 2-6: all integers — fast manual parsing, no allocation.
    msg.type      = (int)fast_parse_long(p);
    msg.order_id  = fast_parse_long(p);
    msg.size      = (int)fast_parse_long(p);
    msg.price     = (int)fast_parse_long(p);
    msg.direction = (int)fast_parse_long(p);

    // Field 7 (e.g. "null") is intentionally ignored — LOBSTER appends
    // an optional annotation column that the pipeline does not use.

    return true;
}