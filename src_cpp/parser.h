#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <vector>
#include <fstream>
#include "types.h"

class LobsterParser {
public:
    LobsterParser(std::string filename);
    ~LobsterParser();

    bool next_message(LobsterMessage& msg);
private:
    std::ifstream file_;
};
#endif