CXXFLAGS ?= -O3 -DNDEBUG -march=x86-64-v3
.PHONY: all
all: build/shopping-sessions

build/shopping-sessions: src/main.cpp src/input.hpp src/parquet.hpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++20 -pthread src/main.cpp -o $@
