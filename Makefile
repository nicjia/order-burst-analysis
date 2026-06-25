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
CXXFLAGS = -std=c++17 -O3 -Wall -pthread

SRC_DIR  = src_cpp
SRCS     = $(SRC_DIR)/main.cpp \
           $(SRC_DIR)/parser.cpp \
           $(SRC_DIR)/burst.cpp \
           $(SRC_DIR)/orderbook.cpp

TARGET   = data_processor

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# ─────────────────────────────────────────────────────────────
# Hoffman2 (UCLA HPC) convenience target.
# Compute nodes need the GCC module loaded for a C++17 toolchain;
# the login/system g++ is too old.  This re-invokes make for the
# default `all` target after loading the module in the same shell.
# Usage:  make hoffman2
# ─────────────────────────────────────────────────────────────
hoffman2:
	. /u/local/Modules/default/init/bash && \
	module load gcc/11.3.0 && \
	$(MAKE) all

clean:
	rm -f $(TARGET)

.PHONY: all hoffman2 clean