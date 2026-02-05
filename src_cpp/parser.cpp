//raw data reading
#include "parser.h"
#include <iostream>
#include <sstream>

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

bool LobsterParser::next_message(LobsterMessage& msg) {
    std::string line;
    if (!std::getline(file_, line)) {
        return false; // EOF
    }

    std::stringstream ss(line);
    std::string value;

    std::getline(ss, value, ','); msg.time = std::stod(value);
    std::getline(ss, value, ','); msg.type = std::stoi(value);
    std::getline(ss, value, ','); msg.order_id = std::stol(value);
    std::getline(ss, value, ','); msg.size = std::stoi(value);
    std::getline(ss, value, ','); msg.price = std::stoi(value);
    std::getline(ss, value, ','); msg.direction = std::stoi(value);

    return true;
}