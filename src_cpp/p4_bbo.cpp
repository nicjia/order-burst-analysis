// P4 revisit v1: best bid and ask from one LOBSTER message file, as a fast replacement for the
// Python loop in src_py/burst_alt.py::reconstruct (equality is tested in tests/test_p4_extract.py).
//
// Price-level book: type 1 adds shares at its price on its side (+1 bid, -1 ask); types 2, 3 and 4
// reduce the level on the order's side when that price is present and delete it at <= 0; other
// types leave the book unchanged. A record is written whenever the best prices or the sizes at them
// change while 0 < bid < ask < 2^62, exactly as burst_alt.reconstruct appends its BBO arrays.
//
// Output, little-endian int64: n, then n records of {message row, bid, ask, bid size, ask size}.
// Prices stay in LOBSTER integer units (dollars x 10000). The row index lets the caller take the
// timestamp from its own parse of the same file, so both code paths share identical floats.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: p4_bbo <messages.csv> <out.bin>\n");
        return 2;
    }
    std::FILE* in = std::fopen(argv[1], "r");
    if (!in) { std::perror("p4_bbo: open input"); return 1; }

    const int64_t INF = int64_t(1) << 62;
    std::map<int64_t, int64_t> bid, ask;
    int64_t best_bid = 0, best_ask = INF;
    int64_t pbb = 0, pba = 0, pbs = 0, pas = 0;
    std::vector<int64_t> rec;
    rec.reserve(1 << 20);

    static char line[1024];
    int64_t row = -1;
    while (std::fgets(line, sizeof line, in)) {
        ++row;
        char* q = std::strchr(line, ',');            // skip the time field
        if (!q) { std::fprintf(stderr, "p4_bbo: malformed row %lld\n", (long long)row); return 3; }
        ++q;
        long typ = std::strtol(q, &q, 10); if (*q++ != ',') return 3;
        std::strtoll(q, &q, 10);          if (*q++ != ',') return 3;   // order id
        int64_t s = std::strtoll(q, &q, 10); if (*q++ != ',') return 3;
        int64_t p = std::strtoll(q, &q, 10); if (*q++ != ',') return 3;
        long d = std::strtol(q, &q, 10);

        if (typ == 1) {
            if (d == 1) { bid[p] += s; if (p > best_bid) best_bid = p; }
            else        { ask[p] += s; if (p < best_ask) best_ask = p; }
        } else if (typ == 2 || typ == 3 || typ == 4) {
            std::map<int64_t, int64_t>& book = (d == 1) ? bid : ask;
            auto it = book.find(p);
            if (it != book.end()) {
                it->second -= s;
                if (it->second <= 0) {
                    book.erase(it);
                    if (d == 1 && p >= best_bid) best_bid = bid.empty() ? 0 : bid.rbegin()->first;
                    else if (d == -1 && p <= best_ask) best_ask = ask.empty() ? INF : ask.begin()->first;
                }
            }
        } else {
            continue;
        }
        if (0 < best_bid && best_bid < best_ask && best_ask < INF) {
            int64_t cbs = bid[best_bid];
            int64_t cas = ask[best_ask];
            if (best_bid != pbb || best_ask != pba || cbs != pbs || cas != pas) {
                rec.push_back(row); rec.push_back(best_bid); rec.push_back(best_ask);
                rec.push_back(cbs); rec.push_back(cas);
                pbb = best_bid; pba = best_ask; pbs = cbs; pas = cas;
            }
        }
    }
    std::fclose(in);

    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) { std::perror("p4_bbo: open output"); return 1; }
    int64_t n = int64_t(rec.size() / 5);
    if (std::fwrite(&n, sizeof n, 1, out) != 1 ||
        (n && std::fwrite(rec.data(), sizeof(int64_t), rec.size(), out) != rec.size())) {
        std::perror("p4_bbo: write");
        return 1;
    }
    return std::fclose(out) == 0 ? 0 : 1;
}
