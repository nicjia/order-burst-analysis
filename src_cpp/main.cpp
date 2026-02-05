#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "parser.h"
#include "types.h"
#include "burst.h"

int main() {

    LobsterParser parser("data/AMZN_2012-06-21_34200000_57600000_message_10.csv");
    std::ifstream book_file("data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv");

    if (!book_file.is_open()) {
        std::cerr << "Error: Could not open orderbook file." << std::endl;
        return 1;
    }

    std::ofstream out_file("bursts.csv");
    out_file << "BurstID,StartTime,EndTime,Direction,Volume,TradeCount,StartPrice,EndPrice,PeakPrice\n";

    // Parameters for burst detection
    double silence_threshold = 1.0;    // Burst ends after 1 second of no activity
    int min_volume = 100;              // Minimum shares to qualify as a burst
    double direction_threshold = 0.9;  //min threshold for buy/sell ratio to classify direction
    
    BurstDetector detector(silence_threshold, min_volume, direction_threshold);

    LobsterMessage msg;
    std::string book_line;
    Burst finished_burst;

    int burst_count = 0;


    while (parser.next_message(msg) && std::getline(book_file, book_line)) {
        std::stringstream ss(book_line);
        std::string val;
        double ask_p1 = 0.0, bid_p1 = 0.0;

        std::getline(ss, val, ','); ask_p1 = std::stod(val) / 10000.0;
        std::getline(ss, val, ','); ; // skip ask size
        std::getline(ss, val, ','); bid_p1 = std::stod(val) / 10000.0;

        double mid_price = (ask_p1 + bid_p1) / 2.0; 

        if(detector.process(msg, mid_price, finished_burst)) {
            out_file << finished_burst.id << ","
                     << std::fixed << std::setprecision(3) << finished_burst.start_time << ","
                     << std::fixed << std::setprecision(3) << finished_burst.end_time << ","
                     << finished_burst.direction << ","
                     << finished_burst.volume << ","
                     << finished_burst.trade_count << ","
                     << finished_burst.start_price << ","
                     << finished_burst.end_price << ","
                     << finished_burst.peak_price << "\n"; 
            burst_count++;
        }
    }
    std::cout<< "Detected " << burst_count << " bursts." << std::endl;
    return 0;
}