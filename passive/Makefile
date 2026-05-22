# ─────────────────────────────────────────────────────────────
# Makefile — Passive Limit Order Burst Detection Pipeline
# ─────────────────────────────────────────────────────────────
# Reuses parser.cpp and orderbook.cpp from the existing src_cpp/.

CXX      = g++
CXXFLAGS = -std=c++17 -O3 -Wall -pthread

SRC_DIR     = src_cpp
PARENT_SRC  = ../src_cpp

SRCS = $(SRC_DIR)/passive_main.cpp \
       $(SRC_DIR)/passive_burst.cpp \
       $(PARENT_SRC)/parser.cpp \
       $(PARENT_SRC)/orderbook.cpp

TARGET = passive_data_processor

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
