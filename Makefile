# ─────────────────────────────────────────────────────────────
# Makefile — Burst Detection Pipeline
# ─────────────────────────────────────────────────────────────
# Portable C++17.  No <filesystem> — uses POSIX dirent.h instead,
# so there's no need for -lstdc++fs on any platform.
#
# Hoffman2 (UCLA HPC):
#   module load gcc/11.3.0   # (or any gcc >= 7 with C++17)
#   make clean && make
# ─────────────────────────────────────────────────────────────

CXX      = g++
CXXFLAGS = -std=c++17 -O3 -Wall

SRC_DIR  = src_cpp
SRCS     = $(SRC_DIR)/main.cpp \
           $(SRC_DIR)/parser.cpp \
           $(SRC_DIR)/burst.cpp \
           $(SRC_DIR)/orderbook.cpp

TARGET   = data_processor

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)