# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall

# Source files
SRC_DIR = src_cpp
SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/parser.cpp $(SRC_DIR)/burst.cpp

# Output binary name
TARGET = data_processor

# The Build Rule
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Clean Rule (to remove the binary)
clean:
	rm -f $(TARGET)